import json
from pathlib import Path


def validate_dataset(dataset_path: str):
    path = Path(dataset_path)

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    total = len(records)
    empty = 0
    arabic_records = 0
    total_chars = 0

    for record in records:
        text = record.get("text", "").strip()

        if not text:
            empty += 1
            continue

        total_chars += len(text)

        arabic_chars = sum(
            "\u0600" <= char <= "\u06FF"
            for char in text
        )

        if arabic_chars > 0:
            arabic_records += 1

    print("=== DATASET VALIDATION ===")
    print(f"Records: {total}")
    print(f"Empty records: {empty}")
    print(f"Arabic records: {arabic_records}")
    print(f"Total characters: {total_chars}")

    if total > 0:
        print(f"Arabic coverage: {arabic_records / total:.2%}")

    print("Validation: PASS" if total > 0 and empty == 0 else "Validation: CHECK")


if __name__ == "__main__":
    validate_dataset(
        "data/processed/arabic_dataset.jsonl"
    )