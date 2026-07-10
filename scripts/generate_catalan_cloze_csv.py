#!/usr/bin/env python3
# Default run (from repo root):
# /home/twyla/Documents/vsCodeSystem/frequencyDictionaryCatalan/.venv/bin/python scripts/generate_catalan_cloze_csv.py
# Slower run (fewer 429s):
# /home/twyla/Documents/vsCodeSystem/frequencyDictionaryCatalan/.venv/bin/python scripts/generate_catalan_cloze_csv.py --input data/items-words.sorted-by-relative-frequency.tsv --output test/catalan_cloze_output.csv --cache test/cache_tatoeba.json --log test/enrichment_log.csv --limit 100 --sleep-seconds 8 --retry-sleep-seconds 12 --rate-limit-cooldown-seconds 600 --max-deferred-passes 3
"""Generate Catalan cloze CSV from a frequency-sorted TSV word list.

Pipeline:
1. Read words in sorted order from a TSV file.
2. Query Tatoeba for a Catalan example sentence for each word.
3. Prefer linked English translations from Tatoeba.
4. Fallback translation providers if needed (MyMemory, then LibreTranslate).
5. Format the Catalan sentence with cloze deletion and write output rows.

Output row format (no header by default, matching template style):
- sentence_with_cloze
- target_word
- cloze_token
- rank
- english_translation
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


TATOEBA_SEARCH_URL = "https://tatoeba.org/en/api_v0/search"
MYMEMORY_TRANSLATE_URL = "https://api.mymemory.translated.net/get"
LIBRETRANSLATE_URL_DEFAULT = "https://libretranslate.com/translate"
WIKIPEDIA_SUMMARY_URL = "https://ca.wikipedia.org/api/rest_v1/page/summary"
WIKIPEDIA_MEDIAWIKI_API_URL = "https://ca.wikipedia.org/w/api.php"
USER_AGENT = "Mozilla/5.0 (compatible; CatalanClozeBot/1.0)"


TOPIC_KEYWORDS: dict[str, list[str]] = {
    "politics": [
        "govern",
        "president",
        "parlament",
        "partit",
        "eleccions",
        "ministre",
        "alcalde",
        "política",
    ],
    "religion": [
        "déu",
        "deu",
        "església",
        "missa",
        "pregària",
        "religió",
        "fe",
        "bíblia",
    ],
    "violence": [
        "matar",
        "arma",
        "guerra",
        "sang",
        "assassinat",
        "atac",
        "violència",
        "violencia",
        "mort",
    ],
    "adult": [
        "sexe",
        "sexual",
        "porn",
        "puta",
        "hòstia",
        "hòsties",
        "merda",
        "coi",
    ],
    "drugs": [
        "droga",
        "drogues",
        "alcohol",
        "cocaïna",
        "heroïna",
        "cànnabis",
    ],
}


BLOCKED_WORD_RULES: list[tuple[str, re.Pattern[str]]] = []
BLOCKED_TOPIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Catalan cloze CSV using Tatoeba + translation fallback."
    )
    parser.add_argument(
        "--input",
        default="data/items-words.sorted-by-relative-frequency.tsv",
        help="Path to sorted TSV input.",
    )
    parser.add_argument(
        "--output",
        default="data/catalan_cloze_output.csv",
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--cache",
        default="data/cache_tatoeba.json",
        help="Path to API cache JSON file.",
    )
    parser.add_argument(
        "--log",
        default="logs/enrichment_log.csv",
        help="Path to log CSV.",
    )
    parser.add_argument(
        "--word-column",
        default="Word",
        help="Column name for the target word in input TSV.",
    )
    parser.add_argument(
        "--start-rank",
        type=int,
        default=1,
        help="Start rank (1-indexed) for processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of words to process (0 means no limit).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1,
        help="Delay between API calls to be polite to public APIs.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=10,
        help="Delay between retry attempts for the same word.",
    )
    parser.add_argument(
        "--wikipedia-sleep-seconds",
        type=float,
        default=1,
        help="Delay between successful Wikipedia API requests.",
    )
    parser.add_argument(
        "--rate-limit-cooldown-seconds",
        type=float,
        default=900.0,
        help="Global cooldown before retrying deferred rate-limited words.",
    )
    parser.add_argument(
        "--max-deferred-passes",
        type=int,
        default=2,
        help="Maximum additional passes for words deferred due to rate limits.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum number of words required in selected Catalan sentence.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=22,
        help="Preferred maximum number of words for selected Catalan sentence.",
    )
    parser.add_argument(
        "--absolute-max-words",
        type=int,
        default=50,
        help="Hard maximum number of words accepted when preferred bounds have no match.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="Minimum number of characters required in selected Catalan sentence.",
    )
    parser.add_argument(
        "--tatoeba-max-pages",
        type=int,
        default=8,
        help="Maximum number of Tatoeba result pages to scan for a quality match.",
    )
    parser.add_argument(
        "--backup-sentence-api",
        choices=["none", "wikipedia"],
        default="wikipedia",
        help="Backup sentence source when Tatoeba returns no results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume by skipping already written ranks (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior.",
    )
    parser.add_argument(
        "--with-header",
        action="store_true",
        help="Write CSV header row (default off to match example format).",
    )
    parser.add_argument(
        "--fallback-translator",
        choices=["mymemory", "libretranslate", "none"],
        default="mymemory",
        help="Fallback translator when Tatoeba has no linked English translation.",
    )
    parser.add_argument(
        "--libretranslate-url",
        default=LIBRETRANSLATE_URL_DEFAULT,
        help="LibreTranslate endpoint URL.",
    )
    parser.add_argument(
        "--manual-review-mode",
        action="store_true",
        help="Write potentially low-quality rows to a separate review CSV.",
    )
    parser.add_argument(
        "--review-output",
        default="logs/manual_review_candidates.csv",
        help="Path to manual review CSV output (used when --manual-review-mode is set).",
    )
    parser.add_argument(
        "--status-summary-only",
        action="store_true",
        help="Only print status summary from the log file and exit.",
    )
    parser.add_argument(
        "--blocked-words",
        default="",
        help="Comma-separated blocked words/phrases. Matching rows are discarded.",
    )
    parser.add_argument(
        "--blocked-words-file",
        default="data/blocked_terms.txt",
        help="Path to text file with blocked words/phrases (one per line).",
    )
    parser.add_argument(
        "--blocked-topics",
        default="",
        help=(
            "Comma-separated topic filters. Built-in topics: "
            + ", ".join(sorted(TOPIC_KEYWORDS.keys()))
        ),
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore cache for all words and force fresh lookup.",
    )
    parser.add_argument(
        "--refresh-words",
        default="",
        help=(
            "Comma-separated words to force refresh (bypass cache and resume). "
            "For these words, the script also tries to avoid reusing the existing sentence "
            "already written for the same rank."
        ),
    )
    parser.add_argument(
        "--refresh-ranks",
        default="",
        help="Comma-separated ranks or ranges to force refresh (example: 62,480-482).",
    )
    parser.add_argument(
        "--retry-empty-output",
        action="store_true",
        help="Force refresh only rows with empty/incomplete output fields.",
    )
    return parser.parse_args()


def split_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_rank_spec(raw: str) -> set[int]:
    ranks: set[int] = set()
    for item in split_csv_items(raw):
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            try:
                start = int(start_raw.strip())
                end = int(end_raw.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid rank range in --refresh-ranks: {item}") from exc
            if start < 1 or end < 1:
                raise ValueError(f"Rank ranges must be >= 1 in --refresh-ranks: {item}")
            if end < start:
                raise ValueError(f"Range end must be >= start in --refresh-ranks: {item}")
            for rank in range(start, end + 1):
                ranks.add(rank)
            continue

        try:
            rank_value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid rank in --refresh-ranks: {item}") from exc
        if rank_value < 1:
            raise ValueError(f"Rank must be >= 1 in --refresh-ranks: {item}")
        ranks.add(rank_value)

    return ranks


def load_blocked_words_file(path: str) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Blocked words file not found: {path}")
    items: list[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def compile_keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword.strip())
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)


def normalize_match_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def configure_content_filters(args: argparse.Namespace) -> None:
    global BLOCKED_WORD_RULES, BLOCKED_TOPIC_PATTERNS

    blocked_words = split_csv_items(args.blocked_words)
    blocked_words.extend(load_blocked_words_file(args.blocked_words_file))
    BLOCKED_WORD_RULES = [
        (normalize_match_key(item), compile_keyword_pattern(item)) for item in blocked_words
    ]

    blocked_topics = split_csv_items(args.blocked_topics)
    topic_patterns: list[tuple[str, re.Pattern[str]]] = []
    for topic in blocked_topics:
        key = topic.lower()
        keywords = TOPIC_KEYWORDS.get(key)
        if not keywords:
            raise ValueError(
                f"Unknown topic in --blocked-topics: {topic}. "
                f"Allowed topics: {', '.join(sorted(TOPIC_KEYWORDS.keys()))}"
            )
        for keyword in keywords:
            topic_patterns.append((key, compile_keyword_pattern(keyword)))
    BLOCKED_TOPIC_PATTERNS = topic_patterns


def blocked_content_reason(
    sentence: str,
    english: str | None = None,
    target_word: str | None = None,
) -> str | None:
    if not BLOCKED_WORD_RULES and not BLOCKED_TOPIC_PATTERNS:
        return None

    haystacks = [sentence]
    if isinstance(english, str) and english:
        haystacks.append(english)

    target_key = normalize_match_key(target_word or "") if target_word else ""

    for blocked_key, pattern in BLOCKED_WORD_RULES:
        # Avoid self-conflict: the target term itself should not auto-block this row.
        if target_key and blocked_key == target_key:
            continue
        if any(pattern.search(text) for text in haystacks):
            return "blocked_word"

    for topic, pattern in BLOCKED_TOPIC_PATTERNS:
        if any(pattern.search(text) for text in haystacks):
            return f"blocked_topic:{topic}"

    return None


def http_get_json(url: str, params: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, payload: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_sorted_input(path: Path, word_column: str) -> list[tuple[int, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or word_column not in reader.fieldnames:
            raise ValueError(f"Input missing required word column: {word_column}")

        rows: list[tuple[int, str]] = []
        for i, row in enumerate(reader, start=1):
            word = (row.get(word_column) or "").strip()
            if not word:
                continue
            rows.append((i, word))
    return rows


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def read_completed_ranks(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()

    completed: set[int] = set()
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                rank = int(row[3])
            except ValueError:
                # Skip header or malformed rows.
                continue
            # Treat only fully written rows as completed (not placeholders).
            if not row[0].strip() or not row[4].strip():
                continue
            completed.add(rank)
    return completed


def read_incomplete_output_ranks(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()

    incomplete: set[int] = set()
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                rank = int(row[3])
            except ValueError:
                # Skip header/malformed rows.
                continue
            if not row[0].strip() or not row[4].strip():
                incomplete.add(rank)
    return incomplete


def strip_cloze_markup(text: str) -> str:
    return re.sub(r"\{\{c\d+::(.*?)\}\}", r"\1", text)


def sentence_key(sentence: str) -> str:
    restored = strip_cloze_markup(sentence)
    normalized = normalize_sentence_whitespace(restored)
    return normalized.casefold()


def read_output_sentences_by_rank(output_path: Path) -> dict[int, str]:
    if not output_path.exists():
        return {}

    sentences_by_rank: dict[int, str] = {}
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                rank = int(row[3])
            except ValueError:
                continue
            candidate = (row[0] or "").strip()
            if not candidate:
                continue
            restored = normalize_sentence_whitespace(strip_cloze_markup(candidate))
            if restored:
                sentences_by_rank[rank] = restored
    return sentences_by_rank


def extract_english_candidates(translations: Any) -> list[str]:
    if not isinstance(translations, list):
        return []

    candidates: list[str] = []

    for group in translations:
        if not isinstance(group, list):
            continue
        for candidate in group:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("lang") == "eng":
                text = (candidate.get("text") or "").strip()
                if text:
                    candidates.append(text)
    return candidates


def contains_whole_word_ascii(text: str, token: str) -> bool:
    pattern = rf"\b{re.escape(token)}\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def choose_best_english_translation(candidates: list[str], target_word: str) -> str | None:
    if not candidates:
        return None

    word = target_word.strip().lower()
    priority_map = {
        "el": ["him", "he", "it", "you"],
        "la": ["her", "it"],
        "els": ["them", "the"],
        "les": ["them", "the"],
    }

    priorities = priority_map.get(word)
    if priorities:
        for preferred in priorities:
            for candidate in candidates:
                if contains_whole_word_ascii(candidate, preferred):
                    return candidate

    return candidates[0]


def count_sentence_words(sentence: str) -> int:
    # Count tokens containing at least one letter-like character.
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿ'’\-]+", sentence.strip(), flags=re.UNICODE)
    return len(tokens)


def passes_sentence_quality(
    sentence: str,
    min_words: int,
    max_words: int,
    min_chars: int,
) -> bool:
    compact = sentence.strip()
    if len(compact) < min_chars:
        return False

    word_count = count_sentence_words(compact)
    if word_count < min_words:
        return False
    if word_count > max_words:
        return False

    return True


def sentence_contains_word(sentence: str, word: str) -> bool:
    pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
    for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
        start, end = match.span()
        prev_char = sentence[start - 1] if start > 0 else ""
        next_char = sentence[end] if end < len(sentence) else ""
        next_next = sentence[end + 1] if end + 1 < len(sentence) else ""

        # Reject domain/path-like fragments such as ".eh" or "eh.com".
        if prev_char in {".", "@", "/"}:
            continue
        if next_char == "." and next_next.isalpha():
            continue

        return True
    return False


def normalize_sentence_whitespace(sentence: str) -> str:
    # Keep output single-line and avoid section-header formatting artifacts.
    return re.sub(r"\s+", " ", sentence).strip()


def has_unwanted_formatting(sentence: str) -> bool:
    if "\n" in sentence or "\r" in sentence or "\t" in sentence:
        return True
    compact = sentence.strip().lower()
    bad_prefixes = {
        "historia",
        "història",
        "inici",
        "inicis",
        "vegeu",
        "referencies",
        "referències",
    }
    return compact in bad_prefixes


def strip_trailing_quotes_and_brackets(text: str) -> str:
    return text.rstrip(" \t\n\r\"'”’»)]}")


def has_non_terminal_ending(sentence: str) -> bool:
    compact = strip_trailing_quotes_and_brackets(sentence.strip())
    return compact.endswith(":") or compact.endswith(";")


def has_target_acronym_context(sentence: str, word: str) -> bool:
    letters_only = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", word)
    if len(letters_only) < 2:
        return False
    if word.isupper():
        return False

    acronym = word.upper()
    pattern = rf"(?<!\w){re.escape(acronym)}(?!\w)"
    return re.search(pattern, sentence) is not None


def is_likely_name_token(token: str) -> bool:
    if len(token) < 2:
        return False
    if token.isupper():
        return False
    if not token[0].isupper():
        return False
    return any(ch.islower() for ch in token[1:])


def has_target_name_context(sentence: str, word: str) -> bool:
    letters_only = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", word)
    if len(letters_only) < 2:
        return False

    token_matches = list(re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*", sentence))
    if not token_matches:
        return False

    for idx, token_match in enumerate(token_matches):
        token = token_match.group(0)
        if token.lower() != word.lower():
            continue
        if not is_likely_name_token(token):
            continue

        prev_is_name = idx > 0 and is_likely_name_token(token_matches[idx - 1].group(0))
        next_is_name = idx + 1 < len(token_matches) and is_likely_name_token(token_matches[idx + 1].group(0))
        if prev_is_name or next_is_name:
            return True

    return False


def has_target_name_appositive_context(sentence: str, word: str) -> bool:
    # Reject encyclopedia-like entries such as "Màxim II, patriarca ...".
    name_literal = re.escape(word)
    appositive_pattern = (
        rf"(?<!\w){name_literal}(?!\w)"
        rf"(?:\s+[IVXLCDM]+)?\s*,\s*[a-zà-öø-ÿ]"
    )
    return re.search(appositive_pattern, sentence, flags=re.IGNORECASE) is not None


def has_target_hyphenated_name_context(sentence: str, word: str) -> bool:
    # Reject place/person names where the target is a capitalized hyphenated segment,
    # e.g. "Ver-sur-Launette" for target "ver".
    name_literal = re.escape(word)
    hyphen_name_pattern = rf"(?<!\w){name_literal}(?!\w)-[A-ZÀ-ÖØ-Þ]"
    return re.search(hyphen_name_pattern, sentence) is not None


def structural_filter_reason(sentence: str, word: str) -> str | None:
    if has_non_terminal_ending(sentence):
        return "non_terminal_ending"
    if has_target_hyphenated_name_context(sentence, word):
        return "name_hyphenated_context"
    if has_target_name_appositive_context(sentence, word):
        return "name_appositive_context"
    if has_target_name_context(sentence, word):
        return "name_context"
    if has_target_acronym_context(sentence, word):
        return "acronym_context"
    return None


def is_likely_wikipedia_visual_description(sentence: str) -> bool:
    lowered = sentence.lower().strip()

    visual_markers = (
        "taula",
        "taules",
        "gràfic",
        "grafic",
        "gràfics",
        "grafics",
        "figura",
        "fig.",
        "diagrama",
        "diagrames",
        "chart",
        "table",
        "dataset",
        "eix x",
        "eix y",
        "eix horitzontal",
        "eix vertical",
        "axis x",
        "axis y",
        "llegenda",
        "font:",
    )

    if any(marker in lowered for marker in visual_markers):
        return True

    # Captions often look like "Figura 1" / "Taula 2".
    if re.search(r"\b(figura|taula|gràfic|grafic)\s+\d+\b", lowered):
        return True

    # Axis descriptions in chart prose (e.g., "eix horitzontal (X)").
    if re.search(r"\b(eix|axis)\s+(x|y|horitzontal|vertical)\b", lowered):
        return True
    if re.search(r"\b(eix|axis)\s+(horitzontal|vertical)\s*\(([xy])\)", lowered):
        return True

    return False


def select_tatoeba_sentence(
    results: list[dict[str, Any]],
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
    excluded_sentence_keys: set[str] | None = None,
) -> tuple[str, str | None] | None:
    preferred_sentence: str | None = None
    preferred_english: str | None = None
    fallback_sentence: str | None = None
    fallback_english: str | None = None

    # Prefer the longest sentence within preferred bounds.
    for item in results:
        if not isinstance(item, dict):
            continue
        lang = str(item.get("lang") or "").strip().lower()
        if lang and lang != "cat":
            continue

        raw_sentence = (item.get("text") or "")
        if has_unwanted_formatting(raw_sentence):
            continue
        sentence = normalize_sentence_whitespace(raw_sentence)
        if not sentence or not sentence_contains_word(sentence, word):
            continue
        if excluded_sentence_keys and sentence_key(sentence) in excluded_sentence_keys:
            continue
        if not is_likely_catalan_sentence(sentence):
            continue
        if structural_filter_reason(sentence, word):
            continue

        english = choose_best_english_translation(
            extract_english_candidates(item.get("translations")),
            word,
        )

        if blocked_content_reason(sentence, english, target_word=word):
            continue

        if not passes_sentence_quality(
            sentence,
            min_words=min_words,
            max_words=max_words,
            min_chars=min_chars,
        ):
            # If no preferred candidate exists, allow any sentence up to absolute max.
            if len(sentence.strip()) < min_chars:
                continue
            if count_sentence_words(sentence) > absolute_max_words:
                continue
            if fallback_sentence is None or len(sentence) > len(fallback_sentence):
                fallback_sentence = sentence
                fallback_english = english
            continue

        if preferred_sentence is None or len(sentence) > len(preferred_sentence):
            preferred_sentence = sentence
            preferred_english = english

    if preferred_sentence is not None:
        return preferred_sentence, preferred_english
    if fallback_sentence is not None:
        return fallback_sentence, fallback_english
    return None


def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def is_likely_catalan_sentence(sentence: str) -> bool:
    lowered = sentence.lower()

    meta_prefixes = (
        "vegeu tambe",
        "vegeu també",
        "referencies",
        "referencies ",
        "referencies:",
        "referències",
        "enllacos externs",
        "enllaços externs",
        "coordenades",
    )
    compact = lowered.strip()
    if any(compact.startswith(prefix) for prefix in meta_prefixes):
        return False

    # Strong indicators the sentence is explicitly Spanish or quoted Spanish text.
    blocked_phrases = {
        "en espanyol",
        "en espanol",
        "en castellano",
        "idioma espanyol",
        "idioma espanol",
    }
    if any(phrase in lowered for phrase in blocked_phrases):
        return False
    if "¿" in sentence or "¡" in sentence:
        return False

    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ'’\-]+", lowered, flags=re.UNICODE)
    if not tokens:
        return False

    catalan_markers = {
        "els",
        "les",
        "dels",
        "aquesta",
        "aquest",
        "aixo",
        "perque",
        "doncs",
        "amb",
        "ahir",
        "avui",
        "estic",
        "vaig",
        "llavors",
        "mentre",
    }
    spanish_markers = {
        "esta",
        "este",
        "esto",
        "pero",
        "para",
        "muy",
        "donde",
        "como",
        "cuando",
        "porque",
        "sin",
        "usted",
        "ustedes",
        "espanol",
        "español",
        "castellano",
    }

    cat_score = sum(1 for token in tokens if token in catalan_markers)
    if "l'" in lowered or "d'" in lowered:
        cat_score += 1

    es_score = sum(1 for token in tokens if token in spanish_markers)

    if cat_score == 0 and es_score > 0:
        return False
    if es_score >= cat_score + 2:
        return False
    return True


def select_sentence_from_wikipedia_text(
    text: str,
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
    excluded_sentence_keys: set[str] | None = None,
) -> tuple[str | None, str]:
    _ = absolute_max_words
    extract = text.strip()
    if not extract:
        return None, "no_result"

    saw_word_match = False

    for sentence in split_into_sentences(extract):
        if has_unwanted_formatting(sentence):
            continue
        sentence = normalize_sentence_whitespace(sentence)

        if excluded_sentence_keys and sentence_key(sentence) in excluded_sentence_keys:
            continue

        if not sentence_contains_word(sentence, word):
            continue
        saw_word_match = True

        if not is_likely_catalan_sentence(sentence):
            continue
        if structural_filter_reason(sentence, word):
            continue
        if is_likely_wikipedia_visual_description(sentence):
            continue

        if blocked_content_reason(sentence, target_word=word):
            continue

        if passes_sentence_quality(
            sentence,
            min_words=min_words,
            max_words=max_words,
            min_chars=min_chars,
        ):
            return sentence, "ok"

    if saw_word_match:
        return None, "no_quality_result"
    return None, "no_result"


def fetch_wikipedia_search_payload(
    params: dict[str, Any],
    retry_sleep_seconds: float,
    wikipedia_sleep_seconds: float,
) -> dict[str, Any]:
    attempts = 3
    for attempt in range(attempts):
        try:
            payload = http_get_json(
                WIKIPEDIA_MEDIAWIKI_API_URL,
                params=params,
                timeout=25,
            )
            if wikipedia_sleep_seconds > 0:
                time.sleep(wikipedia_sleep_seconds)
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 429}:
                if attempt == attempts - 1:
                    raise
                time.sleep(max(retry_sleep_seconds, 0.0))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            time.sleep(max(retry_sleep_seconds, 0.0))
    raise RuntimeError("unreachable")


def lookup_wikipedia_sentence(
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
    retry_sleep_seconds: float,
    wikipedia_sleep_seconds: float,
    excluded_sentence_keys: set[str] | None = None,
) -> dict[str, Any]:
    attempts = 3
    payload: dict[str, Any] | None = None
    summary_missing = False
    encoded_word = urllib.parse.quote(word, safe="")

    for attempt in range(attempts):
        request = urllib.request.Request(
            f"{WIKIPEDIA_SUMMARY_URL}/{encoded_word}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if wikipedia_sleep_seconds > 0:
                time.sleep(wikipedia_sleep_seconds)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                if attempt == attempts - 1:
                    summary_missing = True
                    payload = {"extract": ""}
                    break
                time.sleep(max(retry_sleep_seconds, 0.0))
                continue
            if exc.code == 429:
                if attempt >= 1:
                    return {"status": "rate_limited"}
                time.sleep(max(retry_sleep_seconds, 0.0))
                continue
            return {"status": "request_error", "detail": f"http_{exc.code}"}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == attempts - 1:
                return {"status": "request_error", "detail": "network_or_json_error"}
            time.sleep(max(retry_sleep_seconds, 0.0))

    if not isinstance(payload, dict) and not summary_missing:
        return {"status": "request_error", "detail": "invalid_payload"}

    if not isinstance(payload, dict):
        payload = {}

    extract = (payload.get("extract") or "")
    sentence, status = select_sentence_from_wikipedia_text(
        extract,
        word,
        min_words=min_words,
        max_words=max_words,
        absolute_max_words=absolute_max_words,
        min_chars=min_chars,
        excluded_sentence_keys=excluded_sentence_keys,
    )
    if status == "ok" and sentence:
        return {
            "status": "ok",
            "sentence": sentence,
            "english": None,
            "source": "wikipedia_summary",
        }

    summary_status = status
    saw_quality_miss = summary_status == "no_quality_result"

    try:
        search_payload = fetch_wikipedia_search_payload(
            {
                "action": "query",
                "format": "json",
                "utf8": 1,
                "list": "search",
                "srsearch": f'"{word}"',
                "srwhat": "text",
                "srlimit": 8,
            },
            retry_sleep_seconds=retry_sleep_seconds,
            wikipedia_sleep_seconds=wikipedia_sleep_seconds,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return {"status": "rate_limited"}
        return {"status": "request_error", "detail": f"http_{exc.code}"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {"status": "request_error", "detail": "network_or_json_error"}

    search_results = ((search_payload.get("query") or {}).get("search") or [])
    if not isinstance(search_results, list) or not search_results:
        return {"status": summary_status}

    for item in search_results:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue

        try:
            page_payload = fetch_wikipedia_search_payload(
                {
                    "action": "query",
                    "format": "json",
                    "utf8": 1,
                    "prop": "extracts",
                    "explaintext": 1,
                    "exsectionformat": "plain",
                    "titles": title,
                },
                retry_sleep_seconds=retry_sleep_seconds,
                wikipedia_sleep_seconds=wikipedia_sleep_seconds,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                return {"status": "rate_limited"}
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue

        pages = ((page_payload.get("query") or {}).get("pages") or {})
        if not isinstance(pages, dict):
            continue

        for page_data in pages.values():
            if not isinstance(page_data, dict):
                continue
            page_extract = page_data.get("extract") or ""
            search_sentence, search_status = select_sentence_from_wikipedia_text(
                str(page_extract),
                word,
                min_words=min_words,
                max_words=max_words,
                absolute_max_words=absolute_max_words,
                min_chars=min_chars,
                excluded_sentence_keys=excluded_sentence_keys,
            )
            if search_status == "ok" and search_sentence:
                return {
                    "status": "ok",
                    "sentence": search_sentence,
                    "english": None,
                    "source": "wikipedia_search",
                }
            if search_status == "no_quality_result":
                saw_quality_miss = True

    if saw_quality_miss:
        return {"status": "no_quality_result"}
    return {"status": "no_result"}


def lookup_tatoeba(
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
    retry_sleep_seconds: float,
    max_pages: int,
    excluded_sentence_keys: set[str] | None = None,
) -> dict[str, Any]:
    saw_results = False

    for page in range(1, max_pages + 1):
        attempts = 5
        payload: dict[str, Any] | None = None

        for attempt in range(attempts):
            try:
                payload = http_get_json(
                    TATOEBA_SEARCH_URL,
                    params={"from": "cat", "query": word, "page": page},
                    timeout=25,
                )
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    if attempt == attempts - 1:
                        return {"status": "request_error", "detail": "http_404"}
                    time.sleep(max(retry_sleep_seconds, 0.0))
                    continue
                if exc.code == 429:
                    if attempt >= 1:
                        return {"status": "rate_limited"}
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait_seconds = 2.0
                    if retry_after:
                        try:
                            wait_seconds = min(float(retry_after), 5.0)
                        except ValueError:
                            wait_seconds = max(retry_sleep_seconds, 0.0)
                    else:
                        wait_seconds = max(retry_sleep_seconds, 0.0)
                    time.sleep(wait_seconds)
                    continue
                return {"status": "request_error", "detail": f"http_{exc.code}"}
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt == attempts - 1:
                    return {"status": "request_error", "detail": "network_or_json_error"}
                time.sleep(max(retry_sleep_seconds, 0.0))

        if payload is None:
            return {"status": "request_error", "detail": "empty_payload"}

        results = payload.get("results")
        if not isinstance(results, list):
            return {"status": "request_error", "detail": "invalid_results"}
        if not results:
            if not saw_results and page == 1:
                return {"status": "no_result"}
            break

        saw_results = True
        selected = select_tatoeba_sentence(
            results,
            word,
            min_words=min_words,
            max_words=max_words,
            absolute_max_words=absolute_max_words,
            min_chars=min_chars,
            excluded_sentence_keys=excluded_sentence_keys,
        )
        if selected:
            sentence, english = selected
            return {
                "status": "ok",
                "sentence": sentence,
                "english": english,
            }

    if not saw_results:
        return {"status": "no_result"}
    return {"status": "no_quality_result"}


def translate_with_mymemory(text: str) -> str | None:
    try:
        payload = http_get_json(
            MYMEMORY_TRANSLATE_URL,
            params={"q": text, "langpair": "ca|en"},
            timeout=25,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    response_data = payload.get("responseData")
    if not isinstance(response_data, dict):
        return None

    translated = (response_data.get("translatedText") or "").strip()
    return translated or None


def translate_with_libretranslate(text: str, endpoint: str) -> str | None:
    payload: dict[str, Any] = {
        "q": text,
        "source": "ca",
        "target": "en",
        "format": "text",
    }
    api_key = os.getenv("LIBRETRANSLATE_API_KEY", "").strip()
    if api_key:
        payload["api_key"] = api_key

    try:
        data = http_post_json(endpoint, payload=payload, timeout=25)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    translated = (data.get("translatedText") or "").strip()
    return translated or None


def make_cloze(sentence: str, word: str) -> tuple[str, str] | None:
    pattern = rf"(?<!\w)({re.escape(word)})(?!\w)"
    matches = list(re.finditer(pattern, sentence, flags=re.IGNORECASE))
    if not matches:
        return None

    surface = matches[0].group(1)
    cloze_token = "{{c1::" + surface + "}}"

    def replacer(match: re.Match[str]) -> str:
        return "{{c1::" + match.group(1) + "}}"

    sentence_with_cloze = re.sub(pattern, replacer, sentence, flags=re.IGNORECASE)
    return sentence_with_cloze, cloze_token


def strip_outer_quotes_for_non_dialogue(text: str) -> str:
    compact = text.strip()
    if len(compact) < 2:
        return text

    quote_pairs = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
        "«": "»",
    }
    opening = compact[0]
    closing = compact[-1]
    if quote_pairs.get(opening) != closing:
        return text

    inner = compact[1:-1].strip()
    if not inner:
        return text

    # Keep quotes when the sentence likely contains dialogue turns.
    if "—" in inner:
        return text
    if re.search(r"['\"«“].+?[?!.]['\"»”]\s+['\"«“]", inner):
        return text

    return inner


def normalize_nested_quotes(text: str) -> str:
    # Keep CSV robust by avoiding embedded double-quote characters in content.
    normalized = text.replace("“", '"').replace("”", '"')
    return normalized.replace('"', "'")


def upsert_output_row_by_rank(path: Path, row: list[str], with_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        rank = int(row[3])
    except (ValueError, IndexError) as exc:
        raise ValueError("Output row must contain numeric rank at index 3") from exc

    header = [
        "sentence_with_cloze",
        "target_word",
        "cloze_token",
        "rank",
        "english_translation",
    ]

    existing_rows: list[list[str]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_rows = [r for r in reader]

    has_header = False
    data_rows = existing_rows
    if with_header and existing_rows:
        if existing_rows[0] == header:
            has_header = True
            data_rows = existing_rows[1:]

    # Ensure there are enough lines so row index maps to rank line (rank 1 -> first data line).
    while len(data_rows) < rank:
        placeholder_rank = len(data_rows) + 1
        data_rows.append(["", "", "", str(placeholder_rank), ""])

    data_rows[rank - 1] = row

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if with_header:
            writer.writerow(header)
        writer.writerows(data_rows)


def append_log_row(path: Path, row: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(["rank", "word", "status", "detail"])
        writer.writerow(row)


def append_review_row(path: Path, row: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(
                [
                    "rank",
                    "word",
                    "catalan_sentence",
                    "english_translation",
                    "source",
                    "flags",
                ]
            )
        writer.writerow(row)


def translate_sentence(
    sentence: str,
    fallback_translator: str,
    libretranslate_url: str,
) -> tuple[str | None, str]:
    if fallback_translator == "none":
        return None, "none"
    if fallback_translator == "mymemory":
        translated = translate_with_mymemory(sentence)
        return translated, "mymemory"
    if fallback_translator == "libretranslate":
        translated = translate_with_libretranslate(sentence, libretranslate_url)
        return translated, "libretranslate"
    return None, "none"


def collect_review_flags(word: str, sentence: str, english: str, source: str) -> list[str]:
    flags: list[str] = []

    cat_tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿ'’\-]+", sentence, flags=re.UNICODE)
    eng_tokens = re.findall(r"[A-Za-z']+", english)

    if source != "tatoeba_linked_translation":
        flags.append("non_tatoeba_linked_translation")

    if len(eng_tokens) <= 2:
        flags.append("short_english_translation")

    if len(cat_tokens) <= 2:
        flags.append("very_short_catalan_sentence")

    if sentence_contains_word(sentence, "no"):
        if not re.search(r"(\bnot\b|\bno\b|\bnever\b|n['’]t\b)", english, flags=re.IGNORECASE):
            flags.append("possible_negation_mismatch")

    if word.lower() in {"el", "la", "els", "les"}:
        if re.search(r"\b(you|it)\b", english, flags=re.IGNORECASE):
            flags.append("possible_pronoun_ambiguity")

    return flags


def recommend_cooldown_seconds(rate_limited_pct: float) -> int:
    if rate_limited_pct >= 70.0:
        return 300
    if rate_limited_pct >= 50.0:
        return 240
    if rate_limited_pct >= 30.0:
        return 180
    if rate_limited_pct >= 15.0:
        return 120
    return 60


def compute_status_summary(log_path: Path) -> dict[str, Any]:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return {
            "events": 0,
            "tracked_ranks": 0,
            "completed": 0,
            "deferred": 0,
            "failed": 0,
            "rate_limited_current": 0,
            "rate_limited_pct": 0.0,
            "recommended_cooldown": 60,
        }

    latest_by_rank: dict[int, tuple[str, str]] = {}
    event_count = 0

    with log_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rank = int((row.get("rank") or "").strip())
            except ValueError:
                continue

            status = (row.get("status") or "").strip()
            detail = (row.get("detail") or "").strip()
            if not status:
                continue
            event_count += 1
            latest_by_rank[rank] = (status, detail)

    tracked = len(latest_by_rank)
    completed = sum(1 for status, _ in latest_by_rank.values() if status == "ok")
    deferred = sum(1 for status, _ in latest_by_rank.values() if status == "deferred")
    failed = sum(1 for status, _ in latest_by_rank.values() if status in {"error", "deferred_failed"})
    rate_limited_current = sum(
        1
        for status, detail in latest_by_rank.values()
        if "rate_limited" in detail or status in {"deferred", "deferred_failed"}
    )

    rate_limited_pct = (rate_limited_current / tracked * 100.0) if tracked else 0.0
    recommended = recommend_cooldown_seconds(rate_limited_pct)

    return {
        "events": event_count,
        "tracked_ranks": tracked,
        "completed": completed,
        "deferred": deferred,
        "failed": failed,
        "rate_limited_current": rate_limited_current,
        "rate_limited_pct": rate_limited_pct,
        "recommended_cooldown": recommended,
    }


def print_status_summary(log_path: Path) -> None:
    summary = compute_status_summary(log_path)
    tracked = summary["tracked_ranks"]
    completed = summary["completed"]
    completion_pct = (completed / tracked * 100.0) if tracked else 0.0

    print("Status summary")
    print(f"Log file: {log_path}")
    print(f"Events: {summary['events']}")
    print(f"Tracked ranks: {tracked}")
    print(f"Completed: {completed} ({completion_pct:.1f}%)")
    print(f"Deferred: {summary['deferred']}")
    print(f"Failed: {summary['failed']}")
    print(
        "Rate-limited current: "
        f"{summary['rate_limited_current']} ({summary['rate_limited_pct']:.1f}%)"
    )
    print(f"Recommended cooldown seconds: {summary['recommended_cooldown']}")


def main() -> None:
    args = parse_args()
    configure_content_filters(args)

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache)
    log_path = Path(args.log)
    review_output_path = Path(args.review_output)

    if args.min_words < 1:
        raise ValueError("--min-words must be at least 1")
    if args.max_words < 1:
        raise ValueError("--max-words must be at least 1")
    if args.absolute_max_words < 1:
        raise ValueError("--absolute-max-words must be at least 1")
    if args.min_words > args.max_words:
        raise ValueError("--min-words cannot exceed --max-words")
    if args.max_words > args.absolute_max_words:
        raise ValueError("--max-words cannot exceed --absolute-max-words")
    if args.tatoeba_max_pages < 1:
        raise ValueError("--tatoeba-max-pages must be at least 1")
    if args.min_chars < 1:
        raise ValueError("--min-chars must be at least 1")
    if args.retry_sleep_seconds < 0:
        raise ValueError("--retry-sleep-seconds cannot be negative")
    if args.wikipedia_sleep_seconds < 0:
        raise ValueError("--wikipedia-sleep-seconds cannot be negative")
    if args.rate_limit_cooldown_seconds < 0:
        raise ValueError("--rate-limit-cooldown-seconds cannot be negative")
    if args.max_deferred_passes < 0:
        raise ValueError("--max-deferred-passes cannot be negative")

    if args.no_resume:
        args.resume = False

    if args.status_summary_only:
        print_status_summary(log_path)
        return

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    words_with_rank = read_sorted_input(input_path, args.word_column)
    cache = load_cache(cache_path)
    refresh_words = {w.lower() for w in split_csv_items(args.refresh_words)}
    refresh_ranks = parse_rank_spec(args.refresh_ranks)
    retry_incomplete_ranks = read_incomplete_output_ranks(output_path) if args.retry_empty_output else set()
    existing_sentences_by_rank = read_output_sentences_by_rank(output_path)
    sentence_key_counts: dict[str, int] = collections.Counter(
        sentence_key(sentence) for sentence in existing_sentences_by_rank.values()
    )

    if refresh_words:
        prioritized = [
            (rank, word)
            for rank, word in words_with_rank
            if word.lower() in refresh_words
        ]
        remaining = [
            (rank, word)
            for rank, word in words_with_rank
            if word.lower() not in refresh_words
        ]
        words_with_rank = prioritized + remaining

    completed_ranks = read_completed_ranks(output_path) if args.resume else set()

    processed = 0
    written = 0
    skipped = 0
    errors = 0
    review_count = 0
    deferred_rate_limited: list[tuple[int, str]] = []

    def should_force_refresh(rank: int, word: str) -> bool:
        if args.ignore_cache:
            return True
        if rank in retry_incomplete_ranks:
            return True
        if rank in refresh_ranks:
            return True
        if word.lower() in refresh_words:
            return True
        return False

    def process_word(rank: int, word: str) -> str:
        nonlocal written, errors, review_count

        force_refresh = should_force_refresh(rank, word)
        cached = None if force_refresh else cache.get(word)
        existing_sentence = existing_sentences_by_rank.get(rank)
        existing_sentence_key = sentence_key(existing_sentence) if existing_sentence else None
        excluded_sentence_keys: set[str] = set(sentence_key_counts.keys())

        # Allow keeping the same sentence when updating the same rank unless a refresh-word
        # explicitly requests a different sentence for that rank.
        if existing_sentence_key:
            excluded_sentence_keys.discard(existing_sentence_key)
            if word.lower() in refresh_words:
                excluded_sentence_keys.add(existing_sentence_key)
        sentence: str | None = None
        english: str | None = None
        source = ""

        if isinstance(cached, dict) and cached.get("status") == "ok":
            cached_sentence = cached.get("sentence")
            english = cached.get("english")
            source = cached.get("source", "cache")

            # Re-validate old cache entries against current quality rules.
            if isinstance(cached_sentence, str):
                cleaned_cached_sentence = normalize_sentence_whitespace(cached_sentence)
                cached_is_valid = (
                    bool(cleaned_cached_sentence)
                    and (
                        not excluded_sentence_keys
                        or sentence_key(cleaned_cached_sentence) not in excluded_sentence_keys
                    )
                    and not has_unwanted_formatting(cached_sentence)
                    and sentence_contains_word(cleaned_cached_sentence, word)
                    and is_likely_catalan_sentence(cleaned_cached_sentence)
                    and structural_filter_reason(cleaned_cached_sentence, word) is None
                    and blocked_content_reason(cleaned_cached_sentence, english, target_word=word)
                    is None
                )
                sentence = cleaned_cached_sentence if cached_is_valid else None
            else:
                sentence = None

            if not sentence or not isinstance(english, str) or not english.strip():
                # Ignore stale/invalid cached record and force fresh lookup.
                cached = None

        if not (isinstance(cached, dict) and cached.get("status") == "ok" and sentence and english):
            looked_up = lookup_tatoeba(
                word,
                min_words=args.min_words,
                max_words=args.max_words,
                absolute_max_words=args.absolute_max_words,
                min_chars=args.min_chars,
                retry_sleep_seconds=args.retry_sleep_seconds,
                max_pages=args.tatoeba_max_pages,
                excluded_sentence_keys=excluded_sentence_keys,
            )

            lookup_status = str(looked_up.get("status") or "")
            lookup_detail = str(looked_up.get("detail") or "")
            should_try_backup = (
                args.backup_sentence_api != "none"
                and (
                    lookup_status in {"no_result", "no_quality_result"}
                    or (lookup_status == "request_error" and lookup_detail == "http_404")
                )
            )
            if should_try_backup:
                backup = lookup_wikipedia_sentence(
                    word,
                    min_words=args.min_words,
                    max_words=args.max_words,
                    absolute_max_words=args.absolute_max_words,
                    min_chars=args.min_chars,
                    retry_sleep_seconds=args.retry_sleep_seconds,
                    wikipedia_sleep_seconds=args.wikipedia_sleep_seconds,
                    excluded_sentence_keys=excluded_sentence_keys,
                )
                backup_status = backup.get("status")
                if backup_status == "ok":
                    looked_up = backup
                    lookup_status = "ok"
                elif backup_status == "rate_limited":
                    return "rate_limited"
                else:
                    tatoeba_detail = f"tatoeba_{lookup_status}"
                    if lookup_detail:
                        tatoeba_detail = f"{tatoeba_detail}:{lookup_detail}"
                    detail = f"{tatoeba_detail};backup_{args.backup_sentence_api}_{backup_status}"
                    if backup.get("detail"):
                        detail = detail + ":" + str(backup["detail"])
                    cache[word] = {
                        "status": "error",
                        "error": detail,
                        "updated_at": int(time.time()),
                    }
                    save_cache(cache_path, cache)
                    append_log_row(log_path, [str(rank), word, "error", detail])
                    errors += 1
                    return "error"

            if lookup_status != "ok":
                if lookup_status == "rate_limited":
                    return "rate_limited"

                detail = f"tatoeba_{lookup_status}"
                if lookup_detail:
                    detail = detail + ":" + lookup_detail

                # Cache only stable misses, not transient request/rate-limit failures.
                if lookup_status in {"no_result", "no_quality_result"}:
                    cache[word] = {
                        "status": "error",
                        "error": detail,
                        "updated_at": int(time.time()),
                    }
                    save_cache(cache_path, cache)

                append_log_row(log_path, [str(rank), word, "error", detail])
                errors += 1
                return "error"

            sentence = looked_up.get("sentence")
            english = looked_up.get("english")
            source = str(looked_up.get("source") or "")
            if not source:
                source = "tatoeba_linked_translation" if english else "tatoeba_sentence_only"

            if not english:
                english, fallback_source = translate_sentence(
                    sentence,
                    fallback_translator=args.fallback_translator,
                    libretranslate_url=args.libretranslate_url,
                )
                if english:
                    if source and source != fallback_source:
                        source = f"{source}+{fallback_source}"
                    else:
                        source = fallback_source

            if not english:
                cache[word] = {
                    "status": "error",
                    "error": "no_translation",
                    "sentence": sentence,
                    "updated_at": int(time.time()),
                }
                append_log_row(log_path, [str(rank), word, "error", "no_translation"])
                errors += 1
                save_cache(cache_path, cache)
                time.sleep(args.sleep_seconds)
                return "error"

            cache[word] = {
                "status": "ok",
                "sentence": sentence,
                "english": english,
                "source": source,
                "updated_at": int(time.time()),
            }
            save_cache(cache_path, cache)

        if not sentence or not english:
            append_log_row(log_path, [str(rank), word, "error", "cache_missing_fields"])
            errors += 1
            return "error"

        candidate_key = sentence_key(sentence)
        if candidate_key in excluded_sentence_keys:
            append_log_row(log_path, [str(rank), word, "error", "duplicate_sentence_reused"])
            errors += 1
            return "error"

        structural_reason = structural_filter_reason(sentence, word)
        if structural_reason:
            append_log_row(log_path, [str(rank), word, "error", structural_reason])
            errors += 1
            return "error"

        blocked_reason = blocked_content_reason(sentence, english, target_word=word)
        if blocked_reason:
            append_log_row(log_path, [str(rank), word, "error", blocked_reason])
            errors += 1
            return "error"

        cloze = make_cloze(sentence, word)
        if not cloze:
            append_log_row(log_path, [str(rank), word, "error", "word_not_found_in_sentence"])
            errors += 1
            return "error"

        sentence_with_cloze, cloze_token = cloze
        sentence_with_cloze = strip_outer_quotes_for_non_dialogue(sentence_with_cloze)
        sentence_with_cloze = normalize_nested_quotes(sentence_with_cloze)
        english = normalize_nested_quotes(english)

        upsert_output_row_by_rank(
            output_path,
            [sentence_with_cloze, word, cloze_token, str(rank), english],
            with_header=args.with_header,
        )

        # Keep in-memory duplicate tracking up to date for this run.
        if existing_sentence_key:
            previous_count = sentence_key_counts.get(existing_sentence_key, 0)
            if previous_count <= 1:
                sentence_key_counts.pop(existing_sentence_key, None)
            else:
                sentence_key_counts[existing_sentence_key] = previous_count - 1
        sentence_key_counts[candidate_key] = sentence_key_counts.get(candidate_key, 0) + 1
        existing_sentences_by_rank[rank] = sentence

        append_log_row(log_path, [str(rank), word, "ok", source])

        if args.manual_review_mode:
            flags = collect_review_flags(word, sentence, english, source)
            if flags:
                append_review_row(
                    review_output_path,
                    [
                        str(rank),
                        word,
                        normalize_nested_quotes(sentence),
                        english,
                        source,
                        ";".join(flags),
                    ],
                )
                review_count += 1

        written += 1
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        return "ok"

    for rank, word in words_with_rank:
        if rank < args.start_rank:
            continue
        if args.limit and processed >= args.limit:
            break

        processed += 1

        if args.resume and rank in completed_ranks and not should_force_refresh(rank, word):
            skipped += 1
            continue
        status = process_word(rank, word)
        if status == "rate_limited":
            deferred_rate_limited.append((rank, word))
            append_log_row(log_path, [str(rank), word, "deferred", "tatoeba_rate_limited"])

    for pass_idx in range(1, args.max_deferred_passes + 1):
        if not deferred_rate_limited:
            break

        if args.rate_limit_cooldown_seconds > 0:
            time.sleep(args.rate_limit_cooldown_seconds)

        current_batch = deferred_rate_limited
        deferred_rate_limited = []

        for rank, word in current_batch:
            if args.resume and rank in read_completed_ranks(output_path):
                continue
            status = process_word(rank, word)
            if status == "rate_limited":
                deferred_rate_limited.append((rank, word))
                append_log_row(
                    log_path,
                    [str(rank), word, "deferred", f"tatoeba_rate_limited_pass_{pass_idx}"],
                )

    for rank, word in deferred_rate_limited:
        append_log_row(log_path, [str(rank), word, "deferred_failed", "tatoeba_rate_limited_exhausted"])
        errors += 1

    print(f"Input rows considered: {processed}")
    print(f"Rows written: {written}")
    print(f"Rows skipped (resume): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output file: {output_path}")
    print(f"Log file: {log_path}")
    print(f"Cache file: {cache_path}")
    print(f"Deferred rate-limited remaining: {len(deferred_rate_limited)}")
    if args.manual_review_mode:
        print(f"Review rows flagged: {review_count}")
        print(f"Review file: {review_output_path}")

    print_status_summary(log_path)


if __name__ == "__main__":
    main()
