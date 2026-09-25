from datasets import load_from_disk, Dataset
from pathlib import Path
import json
import re


def clean_text(text):
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_arabic_qa(dataset):
    records = []

    for row in dataset:
        instruction = clean_text(row["instruction"])
        input_text = clean_text(row["input"])
        output = clean_text(row["output"])

        if not instruction or not output:
            continue

        records.append({
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "source": clean_text(row["source"]),
            "domain": "arabic_qa",
        })

    return records


def normalize_cidar(dataset):
    records = []

    for row in dataset:
        instruction = clean_text(row["instruction"])
        output = clean_text(row["output"])

        if not instruction or not output:
            continue

        records.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "source": "CIDAR",
            "domain": "arabic_instruction",
        })

    return records


def normalize_saudi_legal(dataset):
    records = []

    for row in dataset:
        text = clean_text(row["text"])

        if not text:
            continue

        law_name = clean_text(row["law_name"])
        law_type = clean_text(row["law_type"])
        article_number = clean_text(row["article_number"])

        instruction = (
            "استخرج المعلومات القانونية الأساسية من المادة التالية."
        )

        input_text = (
            f"اسم النظام: {law_name}\n"
            f"نوع النظام: {law_type}\n"
            f"رقم المادة: {article_number}\n"
            f"النص: {text}"
        )

        output = (
            f"اسم النظام: {law_name}. "
            f"نوع النظام: {law_type}. "
            f"رقم المادة: {article_number}. "
            f"المعلومة الأساسية: {text}"
        )

        records.append({
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "source": clean_text(row["source"]),
            "domain": "saudi_legal",
        })

    return records


def save_jsonl(records, output_file):

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    print("Loading datasets...")

    arabic_qa = load_from_disk(
        "data/raw/arabic_qa"
    )

    cidar = load_from_disk(
        "data/raw/cidar"
    )

    saudi_legal = load_from_disk(
        "data/raw/saudi_legal"
    )

    print("Normalizing Arabic QA...")
    qa_records = normalize_arabic_qa(
        arabic_qa
    )

    print("Normalizing CIDAR...")
    cidar_records = normalize_cidar(
        cidar
    )

    print("Normalizing Saudi Legal...")
    legal_records = normalize_saudi_legal(
        saudi_legal
    )

    all_records = (
        qa_records
        + cidar_records
        + legal_records
    )

    print("\n=== NORMALIZATION RESULTS ===")
    print(f"Arabic QA: {len(qa_records)}")
    print(f"CIDAR: {len(cidar_records)}")
    print(f"Saudi Legal: {len(legal_records)}")
    print(f"Total: {len(all_records)}")

    save_jsonl(
        all_records,
        "data/processed/normalized_arabic_enterprise.jsonl",
    )

    print(
        "\nSaved: "
        "data/processed/normalized_arabic_enterprise.jsonl"
    )


if __name__ == "__main__":
    main()