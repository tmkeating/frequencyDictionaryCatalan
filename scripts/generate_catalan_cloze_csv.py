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
APERTIUM_TRANSLATE_URL = "https://apertium.org/apy/translate"
LINGVA_URL_DEFAULT = "https://lingva.ml/api/v1"
WIKIPEDIA_SUMMARY_URL = "https://ca.wikipedia.org/api/rest_v1/page/summary"
WIKIPEDIA_MEDIAWIKI_API_URL = "https://ca.wikipedia.org/w/api.php"
WIKTIONARY_MEDIAWIKI_API_URL = "https://ca.wiktionary.org/w/api.php"
USER_AGENT = "Mozilla/5.0 (compatible; CatalanClozeBot/1.0)"


BLOCKED_WORD_RULES: list[tuple[str, re.Pattern[str]]] = []
ALLOWED_NAME_TERMS: set[str] = set()


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
        default=5,
        help="Delay between API calls to be polite to public APIs.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=15,
        help="Delay between retry attempts for the same word.",
    )
    parser.add_argument(
        "--wikipedia-sleep-seconds",
        type=float,
        default=5,
        help="Delay between successful Wikipedia API requests.",
    )
    parser.add_argument(
        "--wiktionary-sleep-seconds",
        type=float,
        default=5,
        help="Delay between successful Viccionari (Wiktionary) API requests.",
    )
    parser.add_argument(
        "--opensubtitles-corpus-path",
        default="data/opensubtitles_ca.txt",
        help=(
            "Path to a local, one-sentence-per-line Catalan subtitles corpus "
            "(e.g. an OPUS OpenSubtitles ca.txt export). Used only if the file exists; "
            "missing file is treated as an empty backup source, not an error."
        ),
    )
    parser.add_argument(
        "--rate-limit-cooldown-seconds",
        type=float,
        default=1800.0,
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
        default=33,
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
        default=10,
        help="Maximum number of Tatoeba result pages to scan for a quality match.",
    )
    parser.add_argument(
        "--backup-sentence-api",
        default="wikipedia,wiktionary",
        help=(
            "Comma-separated, ordered chain of backup sentence sources to try when "
            "Tatoeba returns no results. Choices: none, wikipedia, wiktionary, "
            "opensubtitles. Use 'none' alone to disable all backups. OpenSubtitles is "
            "not enabled by default (low-quality subtitle lines cause more problems "
            "than they solve) — add it explicitly if desired."
        ),
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
        default="mymemory,libretranslate,lingva",
        help=(
            "Comma-separated, ordered chain of fallback translators to try when Tatoeba "
            "has no linked English translation. Choices: none, mymemory, libretranslate, "
            "apertium, lingva. Apertium is excluded from the default chain because it is a "
            "rule-based, word-by-word engine that produces low-quality, sometimes "
            "literally-untranslated (asterisk-marked) output; pass it explicitly "
            "(e.g. 'mymemory,libretranslate,lingva,apertium') to re-enable it as a last "
            "resort. Use 'none' alone to disable all fallback translation."
        ),
    )
    parser.add_argument(
        "--libretranslate-url",
        default=LIBRETRANSLATE_URL_DEFAULT,
        help="LibreTranslate endpoint URL.",
    )
    parser.add_argument(
        "--lingva-url",
        default=LINGVA_URL_DEFAULT,
        help="Lingva Translate endpoint base URL (self-hostable).",
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
        "--allowed-terms-file",
        default="data/allowed_terms.txt",
        help=(
            "Path to text file listing well-known person/place names (one per line, "
            "multi-word phrases allowed) that are exempt from the name_mention "
            "structural filter."
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
            "already written for the same rank. When set, the script processes ONLY these "
            "words (plus any --refresh-ranks matches) — it does not iterate through every "
            "other rank, so unrelated rows are never touched. --start-rank is ignored in "
            "this mode."
        ),
    )
    parser.add_argument(
        "--refresh-ranks",
        default="",
        help=(
            "Comma-separated ranks or ranges to force refresh (example: 62,480-482). "
            "When set, the script processes ONLY these ranks (plus any --refresh-words "
            "matches) — it does not iterate through every rank in between, so unrelated "
            "rows are never touched. --start-rank is ignored in this mode."
        ),
    )
    parser.add_argument(
        "--retry-empty-output",
        action="store_true",
        help="Force refresh only rows with empty/incomplete output fields.",
    )
    args = parser.parse_args()

    valid_backup_sources = {"none", "wikipedia", "wiktionary", "opensubtitles"}
    backup_chain = [
        item.strip().lower()
        for item in args.backup_sentence_api.split(",")
        if item.strip()
    ]
    unknown = [item for item in backup_chain if item not in valid_backup_sources]
    if unknown:
        parser.error(
            f"Invalid --backup-sentence-api entries: {', '.join(unknown)}. "
            f"Valid choices: {', '.join(sorted(valid_backup_sources))}."
        )
    if "none" in backup_chain:
        backup_chain = []
    args.backup_sentence_chain = backup_chain

    valid_translators = {"none", "mymemory", "libretranslate", "apertium", "lingva"}
    translator_chain = [
        item.strip().lower()
        for item in args.fallback_translator.split(",")
        if item.strip()
    ]
    unknown_translators = [item for item in translator_chain if item not in valid_translators]
    if unknown_translators:
        parser.error(
            f"Invalid --fallback-translator entries: {', '.join(unknown_translators)}. "
            f"Valid choices: {', '.join(sorted(valid_translators))}."
        )
    if "none" in translator_chain:
        translator_chain = []
    args.fallback_translator_chain = translator_chain

    return args


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
    # Treat the interpunct (·) as part of a word for boundary purposes, e.g. so the
    # blocked brand name "Intel" does not false-positive match inside the unrelated
    # Catalan word "intel·ligent" (and likewise for other l·l digraph words).
    pattern = rf"(?<![\w·]){escaped}(?![\w·])"
    return re.compile(pattern, flags=re.IGNORECASE)


def normalize_match_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def configure_content_filters(args: argparse.Namespace) -> None:
    global BLOCKED_WORD_RULES, ALLOWED_NAME_TERMS

    blocked_words = split_csv_items(args.blocked_words)
    blocked_words.extend(load_blocked_words_file(args.blocked_words_file))
    BLOCKED_WORD_RULES = [
        (normalize_match_key(item), compile_keyword_pattern(item)) for item in blocked_words
    ]

    allowed_terms_file = getattr(args, "allowed_terms_file", "")
    ALLOWED_NAME_TERMS = {
        normalize_match_key(item) for item in load_blocked_words_file(allowed_terms_file)
    }


_PHONETIC_BRACKET_RE = re.compile(r"\[.+?\]")

# Catalan enclitic pronouns that can follow a verb with a hyphen, e.g. "intentar-ho",
# "casar-me", "convertir-se". A hyphen after a word is only valid if one of these follows.
# The trailing (?![\w-]) ensures "-le-Noble" does NOT match "-le" as a clitic.
_CATALAN_ENCLITIC_RE = re.compile(
    r"-(?:ho|hi|li|me|te|se|los?|les|nos|vos|ne|en|m|t|s|l|n)(?:'|(?![\w-]))",
    re.IGNORECASE,
)

# Characters outside the Catalan/Spanish alphabet.
# Allowed:
#   U+0000-U+00FF  Basic Latin + Latin-1 Supplement (covers all Catalan/Spanish letters,
#                  accented vowels à á â è é ê ë í ï ò ó ô ö ú ü ç ñ ·, guillemets « »)
#   U+2000-U+206F  General Punctuation (em/en dash, curly quotes, ellipsis …)
#   U+20AC         Euro sign €
# Everything else — Hebrew, Arabic, Greek, Cyrillic, Turkish-specific (ş ğ ı), combining
# diacritics (r̄), etc. — triggers the filter.
_NON_LATIN_CHAR_RE = re.compile(r"[^\x00-\xFF\u2000-\u206F\u20AC]")

# Sentences where the target word is the sole predicate after a copula with just an
# article: "Sóc una puta.", "Ell és un professor.", "Som uns idiotes."
# These provide no context beyond the identity label itself.
# The check strips trailing punctuation, confirms the sentence ends with the target word,
# then verifies that what immediately precedes it is copula + article.
_COPULA_BARE_PREDICATE_RE = re.compile(
    r"\b(?:sóc|ets|és|som|sou|són|era|eres|érem|éreu|eren|"
    r"seré|seràs|serà|serem|sereu|seran|ser)\s+"
    r"(?:un|una|uns|unes|el|la|l'|els|les)\s*$",
    re.IGNORECASE,
)

# Sentences that describe a geographic/administrative entity, e.g.
# "Ena fou un districte de la Prefectura de Gifu"
_GEO_ADMIN_RE = re.compile(
    r"\b(?:és|fou|era|va\s+ser)\s+(?:un|una)\s+"
    r"(?:municipi|districte|vila|poble|localitat|departament|prefectura|comarca|"
    r"província|cantó|comtat|parròquia|arrondissement|ciutat|commune)",
    re.IGNORECASE,
)

# Sentences that open with "X o Y és/fou/era/va ser" — encyclopedic disambiguation,
# e.g. "Essa (occità) o Esse (francès) és un municipi" or "Un pam o palm és una unitat"
_DEFINITION_ALT_FORM_RE = re.compile(
    r"^(?:(?:un|una|el|la|l'|els|les)\s+)?"
    r"\w[\w\-]*(?:\s*\([^)]+\))?\s+o\s+\w[\w\-]*(?:\s*\([^)]+\))?\s+"
    r"(?:és|fou|era|va\s+ser)\b",
    re.IGNORECASE,
)

# Full Catalan-style calendar dates ("DD de <mes> de AAAA", optionally a day range like
# "13 i 14 de setembre de 1515"), whether inside a parenthetical or in running text.
# Covers Wikipedia-lede birth/death date ranges, e.g.
# "Oriol de Bolòs i Capdevila (Olot, 16 de març de 1924 - Barcelona, 22 de març de 2007)
# fou un botànic català." as well as bare historical-fact dates, e.g.
# "La batalla de Martignano tíngué lloc el 13 i 14 de setembre de 1515."
_CATALAN_MONTHS_RE = (
    r"(?:gener|febrer|març|abril|maig|juny|juliol|agost|"
    r"setembre|octubre|novembre|desembre)"
)
_FULL_DATE_RE = re.compile(
    r"\b\d{1,2}\s*(?:i\s+\d{1,2}\s*)?de\s+" + _CATALAN_MONTHS_RE + r"\s+de\s+\d{3,4}\b",
    re.IGNORECASE,
)

# Sentences that are actually the caption/legend of a diagram or formula image, e.g.
# "es llegeix: el conjunt C és igual a la unió dels conjunts A i B." — these describe how
# to read a figure rather than using the target word in a natural sentence.
_CAPTION_LEAD_RE = re.compile(r"^\s*es\s+llegeix\s*:", re.IGNORECASE)

# Sentences that are actually in English, not Catalan — e.g. malformed Tatoeba entries
# where the "Catalan" side is really an English quote/fragment with the target word
# embedded, such as "I'm in What're we doing?" (target word "re" matched inside "What're").
# English contracted negation ("don't", "isn't", "wasn't"...) never occurs in Catalan, and
# apostrophe-contractions attached to English pronouns ("I'm", "we're", "what's"...) are
# likewise not a Catalan pattern (Catalan clitics like "escolta'm" attach to verbs, not
# pronouns/question words).
_ENGLISH_NEGATION_CONTRACTION_RE = re.compile(r"\b\w*n't\b", re.IGNORECASE)
_ENGLISH_PRONOUN_CONTRACTION_RE = re.compile(
    r"\b(?:i|you|we|they|he|she|it|what|who|that|there|here)'(?:m|re|s|ll|ve|d)\b",
    re.IGNORECASE,
)

# Sentences that open with a dictionary/reference-style numeric citation lead, e.g.
# "12,12b), rotllo anglès de c." or "3a) Definició del terme." — malformed source entries
# where a section/sense-number citation (optionally comma-separated, with a trailing
# letter like "12b") stands in for an actual example sentence.
_CITATION_REFERENCE_RE = re.compile(r"^\s*\d+[a-z]?(?:,\s*\d+[a-z]?)*\)", re.IGNORECASE)

# Sentences containing parenthetical asides. In practice these are almost always
# encyclopedic/reference clutter (disambiguation glosses like "(handbol)", etymology
# breakdowns, taxonomic names, bibliographic citations like "(23a edició, Madrid: 2014)",
# acronym expansions, etc.) rather than natural example sentences — they cause more
# translation/quality problems than they solve. The one common, legitimate exception is
# a leading pro-drop subject pronoun, e.g. "(Jo) estava a punt de saltar per damunt del
# mur.", a standard Tatoeba convention for showing the implied subject.
_PARENTHETICAL_RE = re.compile(r"\(([^)]*)\)")
_PARENTHETICAL_SUBJECT_PRONOUNS = {
    "jo", "tu", "ell", "ella", "vós", "vostè", "nosaltres", "vosaltres", "vostès",
    "ells", "elles",
}


def _skip_parenthetical_and_alt(rest: str) -> str:
    """Skip an optional parenthetical and/or 'o AlternateForm' in a lowercased string."""
    rest = rest.lstrip()
    if rest.startswith("("):
        close = rest.find(")")
        if close != -1:
            rest = rest[close + 1:].lstrip()
    if rest.startswith("o "):
        parts = rest[2:].lstrip().split()
        if parts:
            rest = rest[2:].lstrip()[len(parts[0]):].lstrip()
            if rest.startswith("("):
                close = rest.find(")")
                if close != -1:
                    rest = rest[close + 1:].lstrip()
    return rest


def _is_definition_lead(sentence: str, target_word: str) -> bool:
    """Return True when the sentence opens with the target word as grammatical subject
    followed by a copular verb introducing a noun-phrase predicate — i.e. the sentence
    defines the word rather than using it.

    Two sub-cases are handled:
    1. No article before the word (typical for proper nouns acting as encyclopedia
       subjects): any copula matches, e.g. "Ena fou un districte…"
    2. Preceded by an article: only matches when the copula is itself followed by an
       article (definitional noun phrase), e.g. "Una vall és una depressió…".
       This avoids filtering predicative adjective sentences like "La justícia és cega."
    """
    lower = sentence.lower().lstrip()
    tw = target_word.lower()

    # ── Case 1: no leading article, word starts the sentence ──────────────────
    if lower.startswith(tw):
        rest = lower[len(tw):]
        if rest and rest[0] == "-":  # hyphenated suffix, e.g. "sin-le-noble"
            end = 0
            while end < len(rest) and (rest[end].isalnum() or rest[end] == "-"):
                end += 1
            rest = rest[end:]
        rest = _skip_parenthetical_and_alt(rest)
        if re.match(r"(?:és|fou|era|va\s+ser)\b", rest):
            return True

    # ── Case 2: word preceded by article ──────────────────────────────────────
    for art in ("un ", "una ", "el ", "la ", "l'", "els ", "les "):
        if lower.startswith(art):
            after_art = lower[len(art):]
            if after_art.startswith(tw):
                rest = after_art[len(tw):]
                rest = _skip_parenthetical_and_alt(rest)
                # Require copula + article → definitional noun phrase ("X és un/una Y")
                # This excludes predicative adjectives ("La justícia és cega").
                if re.match(
                    r"(?:és|fou|era|va\s+ser)\s+(?:un|una|el|la|l'|els|les)\b",
                    rest,
                ):
                    return True
            break

    return False


def blocked_content_reason(
    sentence: str,
    english: str | None = None,
    target_word: str | None = None,
) -> str | None:
    # Sentences with square-bracket phonetic/IPA notation, e.g. [r̄] or [r]
    if _PHONETIC_BRACKET_RE.search(sentence):
        return "phonetic_notation"

    # Sentences containing characters outside the Catalan/Spanish alphabet
    # (Hebrew, Arabic, Greek, Cyrillic, Turkish-specific letters, combining diacritics…)
    if _NON_LATIN_CHAR_RE.search(sentence):
        return "non_latin_characters"

    # Sentences that define or describe a geographic/administrative entity
    if _GEO_ADMIN_RE.search(sentence):
        return "definition_sentence"

    # Sentences that open with "X o AltForm és/fou/era/va ser" (encyclopedic entries)
    if _DEFINITION_ALT_FORM_RE.search(sentence):
        return "definition_sentence"

    # Sentences containing a full Catalan calendar date, e.g.
    # "... (Olot, 16 de març de 1924 - Barcelona, 22 de març de 2007) ..." or
    # "La batalla de Martignano tíngué lloc el 13 i 14 de setembre de 1515."
    if _FULL_DATE_RE.search(sentence):
        return "historical_date"

    # Sentences that are diagram/formula-image captions, e.g. "es llegeix: ..."
    if _CAPTION_LEAD_RE.search(sentence):
        return "caption_sentence"

    # Sentences that are actually in English, not Catalan (malformed source data)
    if _ENGLISH_NEGATION_CONTRACTION_RE.search(sentence) or _ENGLISH_PRONOUN_CONTRACTION_RE.search(sentence):
        return "english_sentence"

    # General English/Spanish word-marker check (catches plain English sentences with
    # no contractions, e.g. "The green web : a union for world conservation.")
    if not is_likely_catalan_sentence(sentence):
        return "non_catalan_sentence"

    # Sentences that open with a dictionary/reference-style numeric citation, e.g.
    # "12,12b), rotllo anglès de c."
    if _CITATION_REFERENCE_RE.search(sentence):
        return "citation_reference"

    # Sentences containing a parenthetical aside, e.g. "(23a edició, Madrid: 2014)" or
    # "Central (handbol)" — usually encyclopedic clutter. Allow the single, common
    # exception of a leading pro-drop subject pronoun like "(Jo) estava...".
    paren_matches = _PARENTHETICAL_RE.findall(sentence)
    if paren_matches:
        is_allowed_leading_pronoun = (
            len(paren_matches) == 1
            and paren_matches[0].strip().lower() in _PARENTHETICAL_SUBJECT_PRONOUNS
            and sentence.lstrip().startswith("(" + paren_matches[0] + ")")
        )
        if not is_allowed_leading_pronoun:
            return "parenthetical_content"

    # Unbalanced/orphan parenthesis, e.g. "Les interjeccions poden expressar sorpresa
    # (caram!" — a truncated source sentence missing its closing paren.
    if sentence.count("(") != sentence.count(")"):
        return "parenthetical_content"

    # Sentence opens with the target word as grammatical subject + copula
    if target_word and _is_definition_lead(sentence, target_word):
        return "definition_sentence"

    # Sentences where the target word is the bare predicate after a copula:
    # "Sóc una puta.", "Ell és un professor." — no surrounding context for the word.
    # Kept: "Llegeix el diari.", "La clau és equiparar…" (more content follows).
    if target_word:
        bare = sentence.lower().rstrip(" .,!?;:…")
        tw = target_word.lower()
        if bare.endswith(tw):
            before = bare[: -len(tw)].rstrip()
            if _COPULA_BARE_PREDICATE_RE.search(before):
                return "minimal_context"

    if not BLOCKED_WORD_RULES:
        return None

    # Only scan the Catalan sentence. Scanning the English translation too caused false
    # positives: e.g. blocking the brand name "Apple" also blocked the common noun
    # "apple" (the fruit) whenever it showed up in the English translation of an
    # unrelated Catalan sentence like "Després menjaré la poma."
    target_key = normalize_match_key(target_word or "") if target_word else ""

    for blocked_key, pattern in BLOCKED_WORD_RULES:
        # Avoid self-conflict: the target term itself should not auto-block this row.
        if target_key and blocked_key == target_key:
            continue
        if pattern.search(sentence):
            return "blocked_word"

    return None


def http_get_json(url: str, params: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}" if query else url
    request = urllib.request.Request(
        full_url,
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
    # Count tokens containing at least one letter-like character. The apostrophe is
    # treated as a word boundary (not kept inside the token) because in Catalan it
    # marks elision/enclisis between two real words, e.g. "d'aigua" = "de" + "aigua",
    # "l'infern" = "el" + "infern", "salva't" = "salva" + "et" (clitic pronoun). Keeping
    # the apostrophe attached previously undercounted such sentences as having fewer
    # real words than they do (e.g. "Salva't." was counted as a single token).
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿ\-]+", sentence.strip(), flags=re.UNICODE)
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

        # Reject matches that are embedded inside a hyphenated compound
        # (e.g. "sin" in "Sin-le-Noble" or "Noble" in "Sin-le-Noble").
        # Exception: a trailing hyphen followed by a Catalan enclitic is valid
        # (e.g. "intentar-ho", "casar-me", "convertir-se").
        if prev_char == "-":
            continue
        if next_char == "-" and not _CATALAN_ENCLITIC_RE.match(sentence[end:]):
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


def contains_unrelated_name_mention(sentence: str) -> bool:
    # Reject sentences that mention what looks like a person or place name anywhere
    # in the sentence, e.g. "En Joan va anar a Barcelona." (contains "Joan" and
    # "Barcelona"), even when the cloze target word itself has nothing to do with
    # the name. Catalan doesn't capitalize common nouns mid-sentence, so a
    # capitalized, non-all-caps token that isn't the first word of the sentence is
    # very likely a proper noun.
    #
    # ALLOWED_NAME_TERMS (data/allowed_terms.txt) whitelists well-known, common
    # names (people, cities, provinces, countries) so ordinary sentences that
    # mention them aren't discarded. Multi-word entries (e.g. "Estats Units") are
    # matched against runs of adjacent capitalized tokens.
    token_matches = list(re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*", sentence))
    if len(token_matches) < 2:
        return False

    idx = 1  # sentence-initial capitalization isn't indicative of a name
    while idx < len(token_matches):
        token_match = token_matches[idx]
        token = token_match.group(0)
        if not is_likely_name_token(token):
            idx += 1
            continue

        # Capitals introduced by a quotation, aside, or question/exclamation mark
        # usually start a new clause rather than a proper noun, e.g. 'Va dir: "Vine."'
        # or a dialogue exchange like "'Gràcies.' 'De res.'".
        preceding = sentence[: token_match.start()].rstrip()
        if preceding and preceding[-1] in ':"\'\u2018\u2019\u201c\u00ab\u00bf\u00a1(':
            idx += 1
            continue

        # Grow the run to include any immediately-adjacent (whitespace-only gap)
        # name-like tokens, so multi-word names/places (e.g. "Estats Units",
        # "Nova York") are checked against the whitelist as a whole phrase.
        run_end = idx + 1
        while (
            run_end < len(token_matches)
            and is_likely_name_token(token_matches[run_end].group(0))
            and sentence[token_matches[run_end - 1].end() : token_matches[run_end].start()].strip() == ""
        ):
            run_end += 1

        run_tokens = [token_matches[i].group(0) for i in range(idx, run_end)]
        run_phrase = normalize_match_key(" ".join(run_tokens))
        if run_phrase in ALLOWED_NAME_TERMS or all(
            normalize_match_key(t) in ALLOWED_NAME_TERMS for t in run_tokens
        ):
            idx = run_end
            continue

        return True

    return False


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
    if contains_unrelated_name_mention(sentence):
        return "name_mention"
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
            # If no preferred candidate exists, allow a longer-than-preferred
            # sentence (up to absolute_max_words) as a fallback, but never one
            # that is shorter than min_words — a sentence with fewer words than
            # the configured minimum is too thin a flashcard regardless of length
            # in characters (e.g. a single-word exclamation like "Magnífic!").
            if len(sentence.strip()) < min_chars:
                continue
            word_count = count_sentence_words(sentence)
            if word_count < min_words or word_count > absolute_max_words:
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
    # Unambiguous English function words — none of these are Catalan words, so any
    # occurrence is a strong signal the "Catalan" side of the entry is really English
    # (e.g. malformed Tatoeba/Wikipedia entries like "The green web : a union for world
    # conservation.").
    english_markers = {
        "the",
        "and",
        "with",
        "this",
        "that",
        "from",
        "are",
        "was",
        "were",
        "have",
        "will",
        "would",
        "should",
        "which",
        "their",
        "your",
        "our",
        "these",
        "those",
        "into",
        "about",
        "world",
        "union",
    }

    cat_score = sum(1 for token in tokens if token in catalan_markers)
    if "l'" in lowered or "d'" in lowered:
        cat_score += 1

    es_score = sum(1 for token in tokens if token in spanish_markers)
    en_score = sum(1 for token in tokens if token in english_markers)

    if cat_score == 0 and es_score > 0:
        return False
    if es_score >= cat_score + 2:
        return False
    if cat_score == 0 and en_score > 0:
        return False
    if en_score >= cat_score + 2:
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


def fetch_wiktionary_query_payload(
    params: dict[str, Any],
    retry_sleep_seconds: float,
    wiktionary_sleep_seconds: float,
) -> dict[str, Any]:
    attempts = 3
    for attempt in range(attempts):
        try:
            payload = http_get_json(
                WIKTIONARY_MEDIAWIKI_API_URL,
                params=params,
                timeout=25,
            )
            if wiktionary_sleep_seconds > 0:
                time.sleep(wiktionary_sleep_seconds)
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


def lookup_wiktionary_sentence(
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
    retry_sleep_seconds: float,
    wiktionary_sleep_seconds: float,
    excluded_sentence_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Look up a Catalan usage sentence from Viccionari (ca.wiktionary.org).

    Viccionari has no REST 'summary' endpoint like Wikipedia, so this goes straight to
    the MediaWiki query API: first the word's own page (titles=<word>, since Viccionari
    entries are usually titled exactly as the lemma), then falls back to a text search
    for other pages that quote/use the word (e.g. usage examples on related entries).
    """
    saw_quality_miss = False

    try:
        page_payload = fetch_wiktionary_query_payload(
            {
                "action": "query",
                "format": "json",
                "utf8": 1,
                "prop": "extracts",
                "explaintext": 1,
                "exsectionformat": "plain",
                "titles": word,
            },
            retry_sleep_seconds=retry_sleep_seconds,
            wiktionary_sleep_seconds=wiktionary_sleep_seconds,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return {"status": "rate_limited"}
        return {"status": "request_error", "detail": f"http_{exc.code}"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {"status": "request_error", "detail": "network_or_json_error"}

    pages = ((page_payload.get("query") or {}).get("pages") or {})
    if isinstance(pages, dict):
        for page_data in pages.values():
            if not isinstance(page_data, dict) or "missing" in page_data:
                continue
            extract = page_data.get("extract") or ""
            sentence, status = select_sentence_from_wikipedia_text(
                str(extract),
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
                    "source": "wiktionary_page",
                }
            if status == "no_quality_result":
                saw_quality_miss = True

    try:
        search_payload = fetch_wiktionary_query_payload(
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
            wiktionary_sleep_seconds=wiktionary_sleep_seconds,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            return {"status": "rate_limited"}
        return {"status": "request_error", "detail": f"http_{exc.code}"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {"status": "request_error", "detail": "network_or_json_error"}

    search_results = ((search_payload.get("query") or {}).get("search") or [])
    if not isinstance(search_results, list) or not search_results:
        return {"status": "no_quality_result" if saw_quality_miss else "no_result"}

    for item in search_results:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue

        try:
            title_payload = fetch_wiktionary_query_payload(
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
                wiktionary_sleep_seconds=wiktionary_sleep_seconds,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                return {"status": "rate_limited"}
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue

        title_pages = ((title_payload.get("query") or {}).get("pages") or {})
        if not isinstance(title_pages, dict):
            continue

        for page_data in title_pages.values():
            if not isinstance(page_data, dict):
                continue
            extract = page_data.get("extract") or ""
            search_sentence, search_status = select_sentence_from_wikipedia_text(
                str(extract),
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
                    "source": "wiktionary_search",
                }
            if search_status == "no_quality_result":
                saw_quality_miss = True

    if saw_quality_miss:
        return {"status": "no_quality_result"}
    return {"status": "no_result"}


# In-memory cache of loaded local subtitle corpora, keyed by file path, so the (possibly
# large) file is only read from disk once per script run rather than once per word.
_OPENSUBTITLES_CORPUS_CACHE: dict[str, list[str]] = {}


def _load_opensubtitles_corpus(corpus_path: str) -> list[str]:
    cached = _OPENSUBTITLES_CORPUS_CACHE.get(corpus_path)
    if cached is not None:
        return cached

    path = Path(corpus_path)
    lines: list[str] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    lines.append(line)

    _OPENSUBTITLES_CORPUS_CACHE[corpus_path] = lines
    return lines


def lookup_opensubtitles_sentence(
    word: str,
    min_words: int,
    max_words: int,
    absolute_max_words: int,
    min_chars: int,
    corpus_path: str,
    excluded_sentence_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Look up a natural, conversational Catalan sentence from a local subtitles corpus.

    Expects a plain-text file at `corpus_path` with one subtitle line/sentence per line
    (e.g. an OPUS OpenSubtitles Catalan monolingual export — see
    https://opus.nlpl.eu/OpenSubtitles.php). If the file is not present, this backup
    source is simply a no-op ('no_result'), never an error, since it is optional.
    """
    _ = absolute_max_words
    lines = _load_opensubtitles_corpus(corpus_path)
    if not lines:
        return {"status": "no_result"}

    saw_word_match = False
    for raw_line in lines:
        if has_unwanted_formatting(raw_line):
            continue
        sentence = normalize_sentence_whitespace(raw_line)

        if excluded_sentence_keys and sentence_key(sentence) in excluded_sentence_keys:
            continue
        if not sentence_contains_word(sentence, word):
            continue
        saw_word_match = True

        if not is_likely_catalan_sentence(sentence):
            continue
        if structural_filter_reason(sentence, word):
            continue
        if blocked_content_reason(sentence, target_word=word):
            continue

        if passes_sentence_quality(
            sentence,
            min_words=min_words,
            max_words=max_words,
            min_chars=min_chars,
        ):
            return {
                "status": "ok",
                "sentence": sentence,
                "english": None,
                "source": "opensubtitles",
            }

    if saw_word_match:
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


def translate_with_mymemory(text: str) -> tuple[str | None, str]:
    try:
        payload = http_get_json(
            MYMEMORY_TRANSLATE_URL,
            params={"q": text, "langpair": "ca|en"},
            timeout=25,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None, "request_error"

    response_data = payload.get("responseData")
    if not isinstance(response_data, dict):
        return None, "invalid_payload"

    translated = (response_data.get("translatedText") or "").strip()
    if not translated:
        return None, "empty_translation"
    return translated, "ok"


def translate_with_libretranslate(text: str, endpoint: str) -> tuple[str | None, str]:
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
        return None, "request_error"

    translated = (data.get("translatedText") or "").strip()
    if not translated:
        return None, "empty_translation"
    return translated, "ok"


def translate_with_apertium(text: str) -> tuple[str | None, str]:
    try:
        payload = http_get_json(
            APERTIUM_TRANSLATE_URL,
            params={"langpair": "cat|eng", "q": text},
            timeout=25,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None, "request_error"

    response_data = payload.get("responseData")
    if not isinstance(response_data, dict):
        return None, "invalid_payload"

    translated = (response_data.get("translatedText") or "").strip()
    if not translated:
        return None, "empty_translation"

    # Apertium is a rule-based (non-neural) engine that translates word-by-word and
    # marks any word missing from its bilingual dictionary with a leading "*", e.g.
    # "*The *green web : at *union *for *world *conservation." Such output is
    # unusably literal/garbled, so treat it as a failed attempt and let the chain
    # fall through to the next fallback translator instead of accepting it.
    if "*" in translated:
        return None, "untranslated_word_marker"

    return translated, "ok"


def translate_with_lingva(text: str, endpoint: str) -> tuple[str | None, str]:
    url = f"{endpoint.rstrip('/')}/ca/en/{urllib.parse.quote(text, safe='')}"
    try:
        payload = http_get_json(url, params={}, timeout=25)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None, "request_error"

    translated = (payload.get("translation") or "").strip()
    if not translated:
        return None, "empty_translation"
    return translated, "ok"


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
    if re.search(rf"{re.escape(closing)}\s+{re.escape(opening)}", inner):
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
    fallback_translator_chain: list[str],
    libretranslate_url: str,
    lingva_url: str,
) -> tuple[str | None, str, str]:
    if not fallback_translator_chain:
        return None, "none", "translation_disabled"

    attempt_details: list[str] = []
    for provider in fallback_translator_chain:
        if provider == "mymemory":
            translated, detail = translate_with_mymemory(sentence)
        elif provider == "libretranslate":
            translated, detail = translate_with_libretranslate(sentence, libretranslate_url)
        elif provider == "apertium":
            translated, detail = translate_with_apertium(sentence)
        elif provider == "lingva":
            translated, detail = translate_with_lingva(sentence, lingva_url)
        else:
            continue
        if translated:
            return translated, provider, f"{provider}:ok"
        attempt_details.append(f"{provider}:{detail}")

    return None, "none", ";".join(attempt_details) or "no_translation_provider_attempted"


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

    if refresh_ranks or refresh_words:
        # Restrict iteration to exactly the requested ranks and/or words instead of
        # sweeping through every rank in the --start-rank/--limit window. This avoids
        # unrelated rows being incidentally rewritten by cache re-validation or
        # blank-placeholder fill-in as the loop passes over them.
        words_with_rank = [
            (rank, word)
            for rank, word in words_with_rank
            if rank in refresh_ranks or word.lower() in refresh_words
        ]

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
        # explicitly requests a different sentence for that rank. Only relax the exclusion
        # when this rank is the sole holder of that sentence across the whole output file;
        # if another rank already shares the same underlying sentence, keep it excluded so a
        # refresh picks a non-duplicate replacement instead of perpetuating the collision.
        if existing_sentence_key:
            if sentence_key_counts.get(existing_sentence_key, 0) <= 1:
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
                bool(args.backup_sentence_chain)
                and (
                    lookup_status in {"no_result", "no_quality_result"}
                    or (lookup_status == "request_error" and lookup_detail == "http_404")
                )
            )
            if should_try_backup:
                backup_detail_parts: list[str] = []
                backup_found = False
                for backup_source in args.backup_sentence_chain:
                    if backup_source == "wikipedia":
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
                    elif backup_source == "wiktionary":
                        backup = lookup_wiktionary_sentence(
                            word,
                            min_words=args.min_words,
                            max_words=args.max_words,
                            absolute_max_words=args.absolute_max_words,
                            min_chars=args.min_chars,
                            retry_sleep_seconds=args.retry_sleep_seconds,
                            wiktionary_sleep_seconds=args.wiktionary_sleep_seconds,
                            excluded_sentence_keys=excluded_sentence_keys,
                        )
                    elif backup_source == "opensubtitles":
                        backup = lookup_opensubtitles_sentence(
                            word,
                            min_words=args.min_words,
                            max_words=args.max_words,
                            absolute_max_words=args.absolute_max_words,
                            min_chars=args.min_chars,
                            corpus_path=args.opensubtitles_corpus_path,
                            excluded_sentence_keys=excluded_sentence_keys,
                        )
                    else:
                        continue

                    backup_status = backup.get("status")
                    if backup_status == "ok":
                        looked_up = backup
                        lookup_status = "ok"
                        backup_found = True
                        break
                    if backup_status == "rate_limited":
                        return "rate_limited"

                    part = f"backup_{backup_source}_{backup_status}"
                    if backup.get("detail"):
                        part = part + ":" + str(backup["detail"])
                    backup_detail_parts.append(part)

                if not backup_found:
                    tatoeba_detail = f"tatoeba_{lookup_status}"
                    if lookup_detail:
                        tatoeba_detail = f"{tatoeba_detail}:{lookup_detail}"
                    detail = ";".join([tatoeba_detail, *backup_detail_parts])
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

            translation_detail = ""
            if not english:
                english, fallback_source, translation_detail = translate_sentence(
                    sentence,
                    fallback_translator_chain=args.fallback_translator_chain,
                    libretranslate_url=args.libretranslate_url,
                    lingva_url=args.lingva_url,
                )
                if english:
                    if source and source != fallback_source:
                        source = f"{source}+{fallback_source}"
                    else:
                        source = fallback_source

            if not english:
                detail = f"no_translation:{translation_detail}" if translation_detail else "no_translation"
                cache[word] = {
                    "status": "error",
                    "error": detail,
                    "sentence": sentence,
                    "updated_at": int(time.time()),
                }
                append_log_row(log_path, [str(rank), word, "error", detail])
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
        # --start-rank only bounds the default sequential sweep; when --refresh-ranks or
        # --refresh-words is set, words_with_rank has already been narrowed to exactly the
        # requested ranks/words, so every entry here should be processed regardless of
        # --start-rank.
        if not (refresh_ranks or refresh_words) and rank < args.start_rank:
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
