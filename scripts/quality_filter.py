import json
import re
from pathlib import Path


INPUT_FILE = "data/processed/normalized_arabic_enterprise.jsonl"
OUTPUT_FILE = "data/processed/quality_filtered_arabic_enterprise.jsonl"


def arabic_ratio(text):
    if not text:
        return 0.0

    arabic = sum(
        "\u0600" <= c <= "\u06FF"
        for c in text
    )

    letters = sum(c.isalpha() for c in text)

    if letters == 0:
        return 0.0

    return arabic / letters


def normalize_for_duplicate(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    records = []

    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines():

        if line.strip():
            records.append(json.loads(line))

    original_count = len(records)

    seen = set()
    filtered = []

    duplicate_count = 0
    short_count = 0
    long_count = 0
    arabic_count = 0

    for record in records:

        instruction = record["instruction"].strip()
        input_text = record["input"].strip()
        output = record["output"].strip()

        combined = (
            instruction
            + " "
            + input_text
            + " "
            + output
        )

        # Remove very short examples
        if len(output) < 10:
            short_count += 1
            continue

        # Remove excessively long examples
        if len(combined) > 12000:
            long_count += 1
            continue

        # Require meaningful Arabic content
        if arabic_ratio(combined) < 0.30:
            arabic_count += 1
            continue

        # Duplicate detection
        fingerprint = normalize_for_duplicate(
            combined
        )

        if fingerprint in seen:
            duplicate_count += 1
            continue

        seen.add(fingerprint)
        filtered.append(record)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in filtered:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n=== QUALITY FILTER ===")
    print(f"Original records: {original_count}")
    print(f"Final records: {len(filtered)}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Short examples removed: {short_count}")
    print(f"Long examples removed: {long_count}")
    print(f"Low-Arabic examples removed: {arabic_count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()