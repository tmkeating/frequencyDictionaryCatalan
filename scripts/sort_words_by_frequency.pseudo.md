# Pseudocode: Sort Words By Relative Frequency

## Goal
Create a copy of the source word list ordered by the Relative Frequency column.

## Inputs
- source_path: supplementary_materials/Final data/items-words.txt
- output_path: data/sorted_words_by_relative_frequency.csv
- sort_direction: desc (default for language learning) or asc

## Assumptions
- Source file is tab-separated (TSV), even though extension is .txt.
- Source headers include at least:
  - Id
  - Word
  - Relative Frequency
  - Zipf

## High-Level Flow
1. Parse command line arguments.
2. Validate input file exists.
3. Read TSV rows from source_path.
4. Normalize and validate required columns.
5. Convert Relative Frequency to numeric.
6. Sort rows by Relative Frequency in selected direction.
7. Assign rank after sorting (1..N).
8. Write a new CSV file preserving UTF-8 text.
9. Print summary: total rows, output path, top and bottom sample rows.

## Detailed Pseudocode
BEGIN

  args = parse_args()
  source_path = args.source_path
  output_path = args.output_path
  sort_direction = args.sort_direction  # desc or asc

  IF file_not_found(source_path)
    RAISE error("Input file not found")
  ENDIF

  rows = read_delimited_file(
    path = source_path,
    delimiter = TAB,
    encoding = UTF-8
  )

  required_headers = ["Id", "Word", "Relative Frequency"]
  missing = required_headers - headers(rows)
  IF missing is not empty
    RAISE error("Missing required columns: " + join(missing, ", "))
  ENDIF

  cleaned_rows = []

  FOR each row in rows
    word = trim(row["Word"])
    rel_freq_raw = trim(row["Relative Frequency"])

    IF word is empty
      CONTINUE  # skip invalid rows with no word
    ENDIF

    rel_freq = try_parse_float(rel_freq_raw)
    IF rel_freq is invalid
      log_warning("Skipping row with invalid Relative Frequency", row)
      CONTINUE
    ENDIF

    cleaned_rows.append({
      "original_id": row["Id"],
      "word": word,
      "relative_frequency": rel_freq,
      "zipf": safe_parse_float(row.get("Zipf", "")),
      "source_row": row
    })
  ENDFOR

  IF sort_direction == "desc"
    sorted_rows = sort(cleaned_rows, key = relative_frequency, descending = TRUE)
  ELSE
    sorted_rows = sort(cleaned_rows, key = relative_frequency, descending = FALSE)
  ENDIF

  rank = 1
  FOR each row in sorted_rows
    row["rank"] = rank
    rank = rank + 1
  ENDFOR

  ensure_parent_dir_exists(output_path)

  write_csv(
    path = output_path,
    encoding = UTF-8,
    headers = [
      "rank",
      "word",
      "relative_frequency",
      "zipf",
      "original_id"
    ],
    rows = sorted_rows
  )

  print("Wrote sorted file to: " + output_path)
  print("Total rows: " + count(sorted_rows))
  print("Top row: " + stringify(sorted_rows[0]))
  print("Bottom row: " + stringify(sorted_rows[last]))

END

## Validation Checks
- Confirm output is strictly ordered by relative_frequency.
- Confirm rank increments by 1 with no gaps.
- Confirm all words are non-empty.
- Confirm row count equals valid input rows.

## Optional Enhancements
- Add --keep-all-columns flag to preserve original columns.
- Add --min-frequency threshold to filter very rare words.
- Add stable tie-breaker by Word for equal frequencies.
