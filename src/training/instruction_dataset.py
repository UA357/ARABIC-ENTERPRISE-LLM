import json
from pathlib import Path


SYSTEM_PROMPT = (
    "أنت مساعد ذكاء اصطناعي مؤسسي متخصص في معالجة "
    "الوثائق العربية والإجابة بدقة وأمان."
)


def build_instruction_dataset(
    input_file: str,
    output_file: str,
):
    input_path = Path(input_file)
    output_path = Path(output_file)

    records = []

    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        record = json.loads(line)
        text = record["text"].strip()

        if not text:
            continue

        records.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": "لخص المعلومات التالية باختصار ودقة.",
                    },
                    {
                        "role": "assistant",
                        "content": text,
                    },
                ],
                "source": record["source"],
            }
        )

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

    print("=== INSTRUCTION DATASET ===")
    print(f"Records created: {len(records)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    build_instruction_dataset(
        "data/processed/arabic_dataset.jsonl",
        "data/processed/arabic_instruction_dataset.jsonl",
    )