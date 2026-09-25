from datasets import load_dataset

DATA_FILES = {
    "train": "data/processed/splits/train.jsonl",
    "validation": "data/processed/splits/validation.jsonl",
    "test": "data/processed/splits/test.jsonl",
}

OUTPUT_DIR = "data/processed/hf_enterprise_dataset"


def main():

    dataset = load_dataset(
        "json",
        data_files=DATA_FILES,
    )

    print("\n=== HUGGING FACE DATASET ===")

    for split in dataset:
        print(f"{split}: {len(dataset[split])}")
        print(f"Columns: {dataset[split].column_names}")

    dataset.save_to_disk(OUTPUT_DIR)

    print(f"\nSaved: {OUTPUT_DIR}")
    print("HF DATASET PREPARATION: PASS")


if __name__ == "__main__":
    main()