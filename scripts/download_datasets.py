from datasets import load_dataset


DATASETS = {
    "arabic_qa": (
        "bobez999/arabic-qa-dataset-sigir2024",
        "data/raw/arabic_qa",
    ),
    "cidar": (
        "arbml/CIDAR",
        "data/raw/cidar",
    ),
    "saudi_legal": (
        "WafaaFraih/saudi-legal-moj",
        "data/raw/saudi_legal",
    ),
}


def download_dataset(name, dataset_id, output_dir):

    print(f"\n=== {name.upper()} ===")
    print(f"Dataset: {dataset_id}")

    dataset = load_dataset(
        dataset_id,
        split="train",
    )

    print(f"Rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    dataset.save_to_disk(output_dir)

    print(f"Saved: {output_dir}")


if __name__ == "__main__":

    for name, (dataset_id, output_dir) in DATASETS.items():
        download_dataset(
            name,
            dataset_id,
            output_dir,
        )

    print("\nALL DATASETS DOWNLOADED")