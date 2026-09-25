from datasets import load_from_disk


DATASETS = {
    "arabic_qa": "data/raw/arabic_qa",
    "cidar": "data/raw/cidar",
    "saudi_legal": "data/raw/saudi_legal",
}


for name, path in DATASETS.items():

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    dataset = load_from_disk(path)

    print("Rows:", len(dataset))
    print("Columns:", dataset.column_names)

    print("\nFIRST RECORD:")
    print(dataset[0])