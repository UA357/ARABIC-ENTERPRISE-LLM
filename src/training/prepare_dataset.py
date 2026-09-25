from datasets import load_dataset
from pathlib import Path


INPUT_FILE = "data/processed/arabic_instruction_dataset.jsonl"
OUTPUT_DIR = "data/processed/hf_dataset"


def prepare_dataset():
    dataset = load_dataset(
        "json",
        data_files=INPUT_FILE,
        split="train",
    )

    print("=== HUGGING FACE DATASET ===")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    if len(dataset) > 0:
        print("First record:")
        print(dataset[0])

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_path))

    print(f"Saved: {output_path}")
    print("DATASET PREPARATION: PASS")


if __name__ == "__main__":
    prepare_dataset()