#!/usr/bin/env python3
"""Probe Tatoeba API rate limiting behavior.

This script sends repeated requests to the Tatoeba search endpoint and reports:
- HTTP status distribution
- first HTTP 429 index/time (if any)
- achieved requests/second
- longest streak without 429

Examples:
  /path/to/.venv/bin/python scripts/probe_tatoeba_rate_limit.py \
      --query que --requests 120 --interval-seconds 0.5

  /path/to/.venv/bin/python scripts/probe_tatoeba_rate_limit.py \
      --query que --requests 200 --interval-seconds 0.0 --timeout 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


TATOEBA_SEARCH_URL = "https://tatoeba.org/en/api_v0/search"
USER_AGENT = "Mozilla/5.0 (compatible; TatoebaRateProbe/1.0)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Tatoeba API rate limits.")
    parser.add_argument(
        "--query-mode",
        choices=["word-list", "fixed"],
        default="word-list",
        help="Use words from a sorted TSV list, or repeat one fixed query.",
    )
    parser.add_argument(
        "--query",
        default="que",
        help="Search query used in --query-mode fixed.",
    )
    parser.add_argument(
        "--word-list-path",
        default="data/items-words.sorted-by-relative-frequency.tsv",
        help="TSV path used in --query-mode word-list.",
    )
    parser.add_argument(
        "--word-column",
        default="Word",
        help="Column name for words in --word-list-path.",
    )
    parser.add_argument(
        "--word-list-start-rank",
        type=int,
        default=1,
        help="Start rank (1-indexed) when iterating words from the list.",
    )
    parser.add_argument(
        "--from-lang",
        default="cat",
        help="Value for the 'from' query parameter (default: cat).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=120,
        help="How many requests to send.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.25,
        help="Delay between requests. Set 0 for max burst.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=25,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Progress print frequency. Use 0 to disable.",
    )
    parser.add_argument(
        "--sweep-intervals",
        action="store_true",
        help="Run a descending interval sweep (long to short) and stop when 429 appears.",
    )
    parser.add_argument(
        "--sweep-start-interval",
        type=float,
        default=5.0,
        help="Sweep start interval in seconds (longer / safer).",
    )
    parser.add_argument(
        "--sweep-end-interval",
        type=float,
        default=0.1,
        help="Sweep end interval in seconds (shorter / faster).",
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=0.1,
        help="Amount to decrease interval each sweep step.",
    )
    parser.add_argument(
        "--sweep-requests",
        type=int,
        default=40,
        help="Requests to send at each interval in sweep mode.",
    )
    parser.add_argument(
        "--cooldown-between-intervals",
        type=float,
        default=2.0,
        help="Cooldown between sweep steps to reduce cross-step carryover effects.",
    )
    parser.add_argument(
        "--json-log-path",
        default="logs/tatoeba_rate_probe.review.jsonl",
        help="Write per-request and summary records to this JSONL file.",
    )
    parser.add_argument(
        "--no-json-log",
        action="store_true",
        help="Disable JSONL logging for this run.",
    )
    parser.add_argument(
        "--review-json-log",
        action="store_true",
        help="After run, print a quick summary from the JSONL log file.",
    )
    parser.add_argument(
        "--review-max-lines",
        type=int,
        default=20,
        help="Max 429/error request lines to print during quick JSON review.",
    )
    return parser.parse_args()


def request_once(from_lang: str, query: str, timeout: int) -> tuple[int, float, str]:
    params = urllib.parse.urlencode({"from": from_lang, "query": query})
    request = urllib.request.Request(
        f"{TATOEBA_SEARCH_URL}?{params}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            duration = time.perf_counter() - started
            results = payload.get("results")
            detail = "ok"
            if isinstance(results, list):
                detail = f"results={len(results)}"
            return int(response.status), duration, detail
    except urllib.error.HTTPError as exc:
        duration = time.perf_counter() - started
        return int(exc.code), duration, f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        duration = time.perf_counter() - started
        return 0, duration, type(exc).__name__


def longest_non_429_streak(statuses: list[int]) -> int:
    longest = 0
    current = 0
    for status in statuses:
        if status == 429:
            current = 0
            continue
        current += 1
        if current > longest:
            longest = current
    return longest


def load_word_queries(path: Path, word_column: str, start_rank: int) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or word_column not in reader.fieldnames:
            raise ValueError(f"Word list missing required column: {word_column}")

        words: list[str] = []
        for rank, row in enumerate(reader, start=1):
            if rank < start_rank:
                continue
            word = (row.get(word_column) or "").strip()
            if word:
                words.append(word)

    if not words:
        raise ValueError("No words found for probing after applying start rank")
    return words


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20)[18]


def run_probe(
    *,
    from_lang: str,
    query_sequence: list[str],
    timeout: int,
    requests: int,
    interval_seconds: float,
    print_every: int,
    json_log_file: Path | None,
    run_id: str,
    run_mode: str,
    run_step: int,
) -> dict[str, object]:
    statuses: list[int] = []
    durations: list[float] = []
    first_429_index: int | None = None
    queries_used: list[str] = []

    if json_log_file is not None:
        json_log_file.parent.mkdir(parents=True, exist_ok=True)

    wall_start_epoch = time.time()

    wall_start = time.perf_counter()
    for i in range(1, requests + 1):
        query = query_sequence[(i - 1) % len(query_sequence)]
        status, duration, detail = request_once(
            from_lang=from_lang,
            query=query,
            timeout=timeout,
        )
        statuses.append(status)
        durations.append(duration)
        queries_used.append(query)

        if json_log_file is not None:
            record = {
                "record_type": "request",
                "run_id": run_id,
                "run_mode": run_mode,
                "run_step": run_step,
                "request_index": i,
                "query": query,
                "from_lang": from_lang,
                "status": status,
                "latency_seconds": round(duration, 6),
                "detail": detail,
                "interval_seconds": interval_seconds,
                "timestamp_epoch": time.time(),
            }
            with json_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if status == 429 and first_429_index is None:
            first_429_index = i

        if print_every and (i % print_every == 0 or i == requests):
            print(
                f"[{i}/{requests}] query={query} status={status} "
                f"latency={duration:.3f}s detail={detail}"
            )

        if i < requests and interval_seconds > 0:
            time.sleep(interval_seconds)

    wall_elapsed = time.perf_counter() - wall_start
    counts = Counter(statuses)
    ok_count = counts.get(200, 0)
    limited_count = counts.get(429, 0)
    error_count = requests - ok_count - limited_count

    return {
        "requests": requests,
        "interval_seconds": interval_seconds,
        "wall_elapsed": wall_elapsed,
        "counts": counts,
        "ok_count": ok_count,
        "limited_count": limited_count,
        "error_count": error_count,
        "durations": durations,
        "first_429_index": first_429_index,
        "longest_non_429_streak": longest_non_429_streak(statuses),
        "queries_used": queries_used,
        "run_id": run_id,
        "run_mode": run_mode,
        "run_step": run_step,
        "wall_start_epoch": wall_start_epoch,
    }


def write_summary_json_record(
    *,
    json_log_file: Path,
    query_source: str,
    from_lang: str,
    summary: dict[str, object],
) -> None:
    counts: Counter = summary["counts"]  # type: ignore[assignment]
    durations: list[float] = summary["durations"]  # type: ignore[assignment]
    queries_used: list[str] = summary["queries_used"]  # type: ignore[assignment]

    record = {
        "record_type": "summary",
        "run_id": summary["run_id"],
        "run_mode": summary["run_mode"],
        "run_step": summary["run_step"],
        "query_source": query_source,
        "from_lang": from_lang,
        "requests": int(summary["requests"]),
        "interval_seconds": float(summary["interval_seconds"]),
        "wall_elapsed_seconds": round(float(summary["wall_elapsed"]), 6),
        "status_counts": dict(sorted(counts.items())),
        "ok_count": int(summary["ok_count"]),
        "limited_count": int(summary["limited_count"]),
        "error_count": int(summary["error_count"]),
        "first_429_index": summary["first_429_index"],
        "longest_non_429_streak": int(summary["longest_non_429_streak"]),
        "unique_queries_tested": len(set(queries_used)),
        "latency_avg_seconds": round(statistics.mean(durations), 6) if durations else 0.0,
        "latency_p95_seconds": round(p95(durations), 6) if durations else 0.0,
        "timestamp_epoch": time.time(),
    }

    with json_log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def quick_review_json_log(json_log_file: Path, run_id: str, max_lines: int) -> None:
    if max_lines < 0:
        raise ValueError("--review-max-lines cannot be negative")
    if not json_log_file.exists():
        print(f"\n=== JSON Log Quick Review ===\nlog_not_found: {json_log_file}")
        return

    request_total = 0
    count_429 = 0
    count_error = 0
    shown = 0
    flagged_lines: list[str] = []
    summaries: list[dict[str, Any]] = []

    with json_log_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("run_id") != run_id:
                continue

            rec_type = rec.get("record_type")
            if rec_type == "request":
                request_total += 1
                status = int(rec.get("status", 0) or 0)
                if status == 429:
                    count_429 += 1
                elif status != 200:
                    count_error += 1

                if shown < max_lines and (status == 429 or status not in (0, 200)):
                    flagged_lines.append(
                        "req_index={idx} status={status} query={query} "
                        "latency={latency:.3f}s detail={detail}".format(
                            idx=int(rec.get("request_index", 0) or 0),
                            status=status,
                            query=str(rec.get("query", "")),
                            latency=float(rec.get("latency_seconds", 0.0) or 0.0),
                            detail=str(rec.get("detail", "")),
                        )
                    )
                    shown += 1
            elif rec_type == "summary":
                summaries.append(rec)

    print("\n=== JSON Log Quick Review ===")
    print(f"log_file: {json_log_file}")
    print(f"run_id: {run_id}")
    print(f"requests_logged: {request_total}")
    print(f"rate_limited_429_logged: {count_429}")
    print(f"other_errors_logged: {count_error}")

    if summaries:
        latest = summaries[-1]
        print(
            "latest_summary: "
            f"mode={latest.get('run_mode')} step={latest.get('run_step')} "
            f"ok={latest.get('ok_count')} 429={latest.get('limited_count')} "
            f"errors={latest.get('error_count')}"
        )

    if flagged_lines:
        print("flagged_request_lines:")
        for line in flagged_lines:
            print(f"  {line}")
    else:
        print("flagged_request_lines: none")


def print_summary(query_source: str, from_lang: str, summary: dict[str, object]) -> None:
    requests = int(summary["requests"])
    interval_seconds = float(summary["interval_seconds"])
    wall_elapsed = float(summary["wall_elapsed"])
    counts: Counter = summary["counts"]  # type: ignore[assignment]
    ok_count = int(summary["ok_count"])
    limited_count = int(summary["limited_count"])
    error_count = int(summary["error_count"])
    durations: list[float] = summary["durations"]  # type: ignore[assignment]
    first_429_index: int | None = summary["first_429_index"]  # type: ignore[assignment]
    streak = int(summary["longest_non_429_streak"])
    queries_used: list[str] = summary["queries_used"]  # type: ignore[assignment]

    print("\n=== Tatoeba Rate Probe Summary ===")
    print(f"query_source: {query_source}")
    print(f"from_lang: {from_lang}")
    print(f"requests_sent: {requests}")
    print(f"interval_seconds: {interval_seconds}")
    print(f"wall_time_seconds: {wall_elapsed:.3f}")
    print(f"achieved_rps: {requests / wall_elapsed:.3f}")
    print(f"status_counts: {dict(sorted(counts.items()))}")
    print(f"ok_200: {ok_count}")
    print(f"rate_limited_429: {limited_count}")
    print(f"other_errors: {error_count}")
    print(f"unique_queries_tested: {len(set(queries_used))}")

    if durations:
        print(f"latency_avg_seconds: {statistics.mean(durations):.3f}")
        print(f"latency_p95_seconds: {p95(durations):.3f}")

    if first_429_index is None:
        print("first_429_request_index: none")
    else:
        print(f"first_429_request_index: {first_429_index}")

    print(f"longest_non_429_streak: {streak}")


def main() -> None:
    args = parse_args()

    if args.requests <= 0:
        raise ValueError("--requests must be greater than 0")
    if args.word_list_start_rank <= 0:
        raise ValueError("--word-list-start-rank must be greater than 0")
    if args.interval_seconds < 0:
        raise ValueError("--interval-seconds cannot be negative")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if args.print_every < 0:
        raise ValueError("--print-every cannot be negative")
    if args.sweep_start_interval < 0:
        raise ValueError("--sweep-start-interval cannot be negative")
    if args.sweep_end_interval < 0:
        raise ValueError("--sweep-end-interval cannot be negative")
    if args.sweep_step <= 0:
        raise ValueError("--sweep-step must be greater than 0")
    if args.sweep_requests <= 0:
        raise ValueError("--sweep-requests must be greater than 0")
    if args.cooldown_between_intervals < 0:
        raise ValueError("--cooldown-between-intervals cannot be negative")
    if args.review_max_lines < 0:
        raise ValueError("--review-max-lines cannot be negative")

    json_log_file: Path | None = None
    if not args.no_json_log:
        json_log_file = Path(args.json_log_path)
    run_id = f"probe-{int(time.time())}-{os.getpid()}"

    query_sequence: list[str]
    query_source: str
    if args.query_mode == "fixed":
        query_sequence = [args.query]
        query_source = f"fixed:{args.query}"
    else:
        word_list_path = Path(args.word_list_path)
        if not word_list_path.exists():
            raise FileNotFoundError(f"Word list not found: {word_list_path}")
        query_sequence = load_word_queries(
            word_list_path,
            word_column=args.word_column,
            start_rank=args.word_list_start_rank,
        )
        query_source = (
            f"word-list:{word_list_path}"
            f" column={args.word_column} start_rank={args.word_list_start_rank}"
        )

    if args.sweep_intervals:
        if args.sweep_start_interval < args.sweep_end_interval:
            raise ValueError("--sweep-start-interval must be >= --sweep-end-interval")

        print("=== Tatoeba Descending Interval Sweep ===")
        print(f"query_source: {query_source}")
        print(f"from_lang: {args.from_lang}")
        print(
            f"intervals: {args.sweep_start_interval} -> {args.sweep_end_interval} "
            f"step {args.sweep_step}"
        )
        print(f"requests_per_interval: {args.sweep_requests}")

        tested: list[tuple[float, int, int]] = []
        safe_interval: float | None = None
        first_limited_interval: float | None = None

        interval = args.sweep_start_interval
        while interval >= args.sweep_end_interval - 1e-9:
            print(f"\n--- interval={interval:.3f}s ---")
            summary = run_probe(
                from_lang=args.from_lang,
                query_sequence=query_sequence,
                timeout=args.timeout,
                requests=args.sweep_requests,
                interval_seconds=interval,
                print_every=args.print_every,
                json_log_file=json_log_file,
                run_id=run_id,
                run_mode="sweep",
                run_step=len(tested) + 1,
            )

            limited = int(summary["limited_count"])
            ok = int(summary["ok_count"])
            tested.append((interval, ok, limited))
            print_summary(query_source, args.from_lang, summary)
            if json_log_file is not None:
                write_summary_json_record(
                    json_log_file=json_log_file,
                    query_source=query_source,
                    from_lang=args.from_lang,
                    summary=summary,
                )

            if limited > 0:
                first_limited_interval = interval
                break
            safe_interval = interval

            interval = round(interval - args.sweep_step, 6)
            if interval >= args.sweep_end_interval and args.cooldown_between_intervals > 0:
                time.sleep(args.cooldown_between_intervals)

        print("\n=== Sweep Result ===")
        if safe_interval is not None:
            print(f"recommended_safe_interval_seconds: {safe_interval:.3f}")
        else:
            print("recommended_safe_interval_seconds: none")
        if first_limited_interval is not None:
            print(f"first_interval_with_429_seconds: {first_limited_interval:.3f}")
        else:
            print("first_interval_with_429_seconds: none")

        print("tested_intervals:")
        for interval_value, ok_count, limited_count in tested:
            print(f"  interval={interval_value:.3f}s ok={ok_count} 429={limited_count}")
        if args.review_json_log and json_log_file is not None:
            quick_review_json_log(json_log_file, run_id=run_id, max_lines=args.review_max_lines)
        return

    summary = run_probe(
        from_lang=args.from_lang,
        query_sequence=query_sequence,
        timeout=args.timeout,
        requests=args.requests,
        interval_seconds=args.interval_seconds,
        print_every=args.print_every,
        json_log_file=json_log_file,
        run_id=run_id,
        run_mode="single",
        run_step=1,
    )
    print_summary(query_source, args.from_lang, summary)
    if json_log_file is not None:
        write_summary_json_record(
            json_log_file=json_log_file,
            query_source=query_source,
            from_lang=args.from_lang,
            summary=summary,
        )

    if int(summary["limited_count"]) > 0:
        print("\nSuggestion: increase --interval-seconds and rerun until 429 count is near zero.")
    if args.review_json_log and json_log_file is not None:
        quick_review_json_log(json_log_file, run_id=run_id, max_lines=args.review_max_lines)


if __name__ == "__main__":
    main()
