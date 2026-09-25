import json
import random
from pathlib import Path


INPUT_FILE = "data/processed/quality_filtered_arabic_enterprise.jsonl"
OUTPUT_DIR = Path("data/processed/splits")

SEED = 42


def load_records():
    return [
        json.loads(line)
        for line in Path(INPUT_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def save_records(records, filename):
    path = OUTPUT_DIR / filename

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )


def main():

    random.seed(SEED)

    records = load_records()
    random.shuffle(records)

    total = len(records)

    train_end = int(total * 0.80)
    val_end = int(total * 0.90)

    train = records[:train_end]
    validation = records[train_end:val_end]
    test = records[val_end:]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_records(train, "train.jsonl")
    save_records(validation, "validation.jsonl")
    save_records(test, "test.jsonl")

    print("\n=== DATASET SPLIT ===")
    print(f"Total:       {total}")
    print(f"Train:       {len(train)}")
    print(f"Validation:  {len(validation)}")
    print(f"Test:        {len(test)}")
    print(f"Seed:        {SEED}")
    print(f"Output:      {OUTPUT_DIR}")


if __name__ == "__main__":
    main()