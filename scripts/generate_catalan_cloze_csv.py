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
USER_AGENT = "Mozilla/5.0 (compatible; CatalanClozeBot/1.0)"


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
        default=2,
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
    return parser.parse_args()


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
    return re.search(pattern, sentence, flags=re.IGNORECASE) is not None


def select_tatoeba_sentence(
    results: list[dict[str, Any]],
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
) -> tuple[str, str | None] | None:
    preferred_sentence: str | None = None
    preferred_english: str | None = None
    fallback_sentence: str | None = None
    fallback_english: str | None = None

    # Prefer the longest sentence within preferred bounds.
    for item in results:
        if not isinstance(item, dict):
            continue
        sentence = (item.get("text") or "").strip()
        if not sentence or not sentence_contains_word(sentence, word):
            continue
        english = choose_best_english_translation(
            extract_english_candidates(item.get("translations")),
            word,
        )

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


def lookup_wikipedia_sentence(
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    attempts = 3
    payload: dict[str, Any] | None = None
    encoded_word = urllib.parse.quote(word, safe="")

    for attempt in range(attempts):
        request = urllib.request.Request(
            f"{WIKIPEDIA_SUMMARY_URL}/{encoded_word}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"status": "no_result"}
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

    if not isinstance(payload, dict):
        return {"status": "request_error", "detail": "invalid_payload"}

    extract = (payload.get("extract") or "").strip()
    if not extract:
        return {"status": "no_result"}

    for sentence in split_into_sentences(extract):
        if not sentence_contains_word(sentence, word):
            continue
        if not passes_sentence_quality(
            sentence,
            min_words=min_words,
            max_words=max_words,
            min_chars=min_chars,
        ):
            if len(sentence.strip()) < min_chars:
                continue
            if count_sentence_words(sentence) > absolute_max_words:
                continue
            return {
                "status": "ok",
                "sentence": sentence,
                "english": None,
                "source": "wikipedia_summary",
            }
            continue
        return {
            "status": "ok",
            "sentence": sentence,
            "english": None,
            "source": "wikipedia_summary",
        }

    return {"status": "no_quality_result"}


def lookup_tatoeba(
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
    retry_sleep_seconds: float,
    max_pages: int,
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
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if not match:
        return None

    surface = match.group(1)
    cloze_token = "{{c1::" + surface + "}}"
    start, end = match.span(1)
    sentence_with_cloze = sentence[:start] + cloze_token + sentence[end:]
    return sentence_with_cloze, cloze_token


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

    completed_ranks = read_completed_ranks(output_path) if args.resume else set()

    processed = 0
    written = 0
    skipped = 0
    errors = 0
    review_count = 0
    deferred_rate_limited: list[tuple[int, str]] = []

    def process_word(rank: int, word: str) -> str:
        nonlocal written, errors, review_count

        cached = cache.get(word)
        sentence: str | None = None
        english: str | None = None
        source = ""

        if isinstance(cached, dict) and cached.get("status") == "ok":
            sentence = cached.get("sentence")
            english = cached.get("english")
            source = cached.get("source", "cache")
        else:
            looked_up = lookup_tatoeba(
                word,
                min_words=args.min_words,
                max_words=args.max_words,
                absolute_max_words=args.absolute_max_words,
                min_chars=args.min_chars,
                retry_sleep_seconds=args.retry_sleep_seconds,
                max_pages=args.tatoeba_max_pages,
            )

            lookup_status = looked_up.get("status")
            if lookup_status == "no_result" and args.backup_sentence_api != "none":
                backup = lookup_wikipedia_sentence(
                    word,
                    min_words=args.min_words,
                    max_words=args.max_words,
                    absolute_max_words=args.absolute_max_words,
                    min_chars=args.min_chars,
                    retry_sleep_seconds=args.retry_sleep_seconds,
                )
                backup_status = backup.get("status")
                if backup_status == "ok":
                    looked_up = backup
                    lookup_status = "ok"
                elif backup_status == "rate_limited":
                    return "rate_limited"
                else:
                    detail = f"tatoeba_no_result;backup_{args.backup_sentence_api}_{backup_status}"
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
                if looked_up.get("detail"):
                    detail = detail + ":" + str(looked_up["detail"])

                # Cache only stable misses, not transient request/rate-limit failures.
                if lookup_status == "no_result":
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

        cloze = make_cloze(sentence, word)
        if not cloze:
            append_log_row(log_path, [str(rank), word, "error", "word_not_found_in_sentence"])
            errors += 1
            return "error"

        sentence_with_cloze, cloze_token = cloze
        sentence_with_cloze = normalize_nested_quotes(sentence_with_cloze)
        english = normalize_nested_quotes(english)

        upsert_output_row_by_rank(
            output_path,
            [sentence_with_cloze, word, cloze_token, str(rank), english],
            with_header=args.with_header,
        )
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

        if args.resume and rank in completed_ranks:
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
