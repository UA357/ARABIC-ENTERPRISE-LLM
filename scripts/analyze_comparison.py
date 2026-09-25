import json
from pathlib import Path


INPUT_FILE = "data/evaluation/base_vs_finetuned.jsonl"


def main():

    records = [
        json.loads(line)
        for line in Path(INPUT_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    base_arabic = [
        r["base_arabic_ratio"]
        for r in records
    ]

    finetuned_arabic = [
        r["finetuned_arabic_ratio"]
        for r in records
    ]

    base_lengths = [
        len(r["base_response"])
        for r in records
    ]

    finetuned_lengths = [
        len(r["finetuned_response"])
        for r in records
    ]

    avg_base_arabic = (
        sum(base_arabic) / len(base_arabic)
    )

    avg_finetuned_arabic = (
        sum(finetuned_arabic)
        / len(finetuned_arabic)
    )

    avg_base_length = (
        sum(base_lengths)
        / len(base_lengths)
    )

    avg_finetuned_length = (
        sum(finetuned_lengths)
        / len(finetuned_lengths)
    )

    identical = sum(
        r["base_response"].strip()
        == r["finetuned_response"].strip()
        for r in records
    )

    print("\n=== BASE VS FINE-TUNED ANALYSIS ===")

    print(
        f"Samples: {len(records)}"
    )

    print(
        f"Average base Arabic ratio: "
        f"{avg_base_arabic:.3f}"
    )

    print(
        f"Average fine-tuned Arabic ratio: "
        f"{avg_finetuned_arabic:.3f}"
    )

    print(
        f"Average base response length: "
        f"{avg_base_length:.1f}"
    )

    print(
        f"Average fine-tuned response length: "
        f"{avg_finetuned_length:.1f}"
    )

    print(
        f"Identical responses: "
        f"{identical}/{len(records)}"
    )

    print(
        "\nArabic ratio change: "
        f"{avg_finetuned_arabic - avg_base_arabic:+.3f}"
    )

    print(
        "Response length change: "
        f"{avg_finetuned_length - avg_base_length:+.1f}"
    )


if __name__ == "__main__":
    main()