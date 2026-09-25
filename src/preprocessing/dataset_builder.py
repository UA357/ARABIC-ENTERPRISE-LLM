from pathlib import Path
import json

from src.preprocessing.arabic_cleaner import clean_text


def build_dataset(input_dir: str, output_file: str):
    input_path = Path(input_dir)
    output_path = Path(output_file)

    records = []

    for file in sorted(input_path.glob("*.txt")):
        raw_text = file.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue

        records.append(
            {
                "source": file.name,
                "text": cleaned_text,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Documents processed: {len(records)}")
    print(f"Dataset saved: {output_path}")


if __name__ == "__main__":
    build_dataset(
        "data/ocr",
        "data/processed/arabic_dataset.jsonl",
    )