# Pseudocode: Generate Catalan Cloze CSV With API Enrichment

## Goal
Read the frequency-sorted word list, call an API for example sentences and English translations, and write an output CSV that preserves order and matches the template style.

## Inputs
- sorted_input_path: data/sorted_words_by_relative_frequency.csv
- output_csv_path: data/catalan_cloze_output.csv
- cache_path: data/cache_api.json
- log_path: logs/enrichment_log.csv
- api_key: environment variable
- batch_size: optional, default 200
- resume: true by default
- sweep_min_words: true by default
- sweep_min_words_low: lower bound for sentence-length sweep
- sweep_min_words_high: upper bound for sentence-length sweep

## Target Output Schema
Order should match the example template rows:
1. sentence_with_cloze
2. target_word
3. cloze_token
4. rank
5. english_translation

## API Expectations
For each Catalan word, request:
- one natural Catalan example sentence containing the target word
- one English translation of that sentence

If provider supports a structured response, request JSON keys:
- catalan_sentence
- english_translation
- matched_surface_form

## High-Level Flow
1. Parse arguments and load config.
2. Load sorted word list in rank order.
3. Load cache and existing output (if resume enabled).
4. For each pending word:
  - build min-word attempts from sweep bounds (high down to low), or use fixed min-words
   - fetch sentence and translation from API
   - build cloze sentence
   - validate output row
   - save row to output and cache
5. Save logs and summary metrics.

## Detailed Pseudocode
BEGIN

  args = parse_args()

  input_rows = read_csv(args.sorted_input_path, encoding = UTF-8)
  ensure_required_columns(input_rows, ["rank", "word", "relative_frequency"])

  cache = load_json_if_exists(args.cache_path, default = {})
  completed_ranks = set()

  IF args.resume AND file_exists(args.output_csv_path)
    existing_rows = read_csv(args.output_csv_path, encoding = UTF-8)
    completed_ranks = set(existing_rows[*]["rank"])
  ENDIF

  ensure_parent_dir_exists(args.output_csv_path)
  ensure_parent_dir_exists(args.log_path)

  open output_writer in append_or_create_mode(args.output_csv_path)
  IF output file was newly created
    write_header(output_writer, [
      "sentence_with_cloze",
      "target_word",
      "cloze_token",
      "rank",
      "english_translation"
    ])
  ENDIF

  open log_writer for args.log_path
  write_log_header_if_new(log_writer)

  processed_count = 0
  error_count = 0

  FOR each row in input_rows in ascending rank order
    rank = row["rank"]
    word = trim(row["word"])

    IF rank in completed_ranks
      CONTINUE
    ENDIF

    IF word in cache
      api_result = cache[word]
    ELSE
      api_result = call_example_api_with_retry(word)
      IF api_result failed
        write_log(log_writer, rank, word, "api_error", api_result.error)
        error_count = error_count + 1
        CONTINUE
      ENDIF
      cache[word] = api_result
      save_json_atomic(args.cache_path, cache)
    ENDIF

    catalan_sentence = api_result.catalan_sentence
    english_translation = api_result.english_translation

    IF catalan_sentence is empty OR english_translation is empty
      write_log(log_writer, rank, word, "missing_fields", "empty sentence or translation")
      error_count = error_count + 1
      CONTINUE
    ENDIF

    cloze_result = make_cloze(
      sentence = catalan_sentence,
      target_word = word,
      matched_surface_form = api_result.matched_surface_form
    )

    IF cloze_result.status != "ok"
      write_log(log_writer, rank, word, "no_match", cloze_result.reason)
      error_count = error_count + 1
      CONTINUE
    ENDIF

    sentence_with_cloze = cloze_result.sentence_with_cloze
    cloze_token = cloze_result.cloze_token

    out_row = {
      "sentence_with_cloze": sentence_with_cloze,
      "target_word": word,
      "cloze_token": cloze_token,
      "rank": rank,
      "english_translation": english_translation
    }

    IF not validate_output_row(out_row)
      write_log(log_writer, rank, word, "validation_failed", "invalid output row")
      error_count = error_count + 1
      CONTINUE
    ENDIF

    write_row(output_writer, out_row)
    flush(output_writer)

    write_log(log_writer, rank, word, "ok", "")
    processed_count = processed_count + 1

    IF processed_count % args.batch_size == 0
      print("Processed " + processed_count + " words so far")
    ENDIF

  ENDFOR

  close output_writer
  close log_writer

  print("Done")
  print("Processed: " + processed_count)
  print("Errors: " + error_count)
  print("Output: " + args.output_csv_path)

END

## Helper: call_example_api_with_retry
FUNCTION call_example_api_with_retry(word)
  max_attempts = 5
  backoff_seconds = [1, 2, 4, 8, 16]

  FOR attempt in 1..max_attempts
    response = call_api(word)

    IF response.status == success
      RETURN normalize_response(response)
    ENDIF

    IF response.status in [rate_limited, transient_error]
      sleep(backoff_seconds[attempt])
      CONTINUE
    ENDIF

    RETURN failure(response.error)
  ENDFOR

  RETURN failure("Max retries exceeded")
END

## Helper: make_cloze
FUNCTION make_cloze(sentence, target_word, matched_surface_form)
  # Preferred replacement term is exact surface form from API, fallback to target_word
  candidate = first_non_empty(matched_surface_form, target_word)

  # Case-insensitive full-word match with Unicode-aware boundaries
  match = regex_find_first_full_word(sentence, candidate)

  IF match not found AND candidate != target_word
    match = regex_find_first_full_word(sentence, target_word)
  ENDIF

  IF match not found
    RETURN { status: "no_match", reason: "word not found in sentence" }
  ENDIF

  surface = matched_text(match)
  cloze_token = "{{c1::" + surface + "}}"
  sentence_with_cloze = replace_first_match(sentence, match, cloze_token)

  RETURN {
    status: "ok",
    sentence_with_cloze: sentence_with_cloze,
    cloze_token: cloze_token
  }
END

## Validation Rules
- sentence_with_cloze must contain exactly one {{c1::...}} token.
- cloze_token must match token inserted in sentence_with_cloze.
- target_word must be non-empty.
- rank must be numeric and unique.
- english_translation must be non-empty.

## Optional Enhancements
- Add provider abstraction for multiple APIs.
- Add prompt guardrails to enforce simple, learner-friendly sentences.
- Add deterministic temperature and top_p for repeatability.
- Add manual review export for failed words.
