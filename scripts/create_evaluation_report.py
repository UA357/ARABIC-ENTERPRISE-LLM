import json
from pathlib import Path


QUALITY_FILE = (
    "data/evaluation/quality_metrics.json"
)

COMPARISON_FILE = (
    "data/evaluation/base_vs_finetuned_v2.jsonl"
)

OUTPUT_FILE = (
    "data/evaluation/QLORA_EVALUATION_REPORT.md"
)


def main():

    quality = json.loads(
        Path(QUALITY_FILE).read_text(
            encoding="utf-8"
        )
    )

    comparison = [
        json.loads(line)
        for line in Path(COMPARISON_FILE)
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    base = quality["base_model"]
    fine = quality["finetuned_model"]

    report = f"""# Arabic Enterprise LLM
# QLoRA Evaluation Report

## Model

- Base model: Qwen/Qwen2.5-1.5B-Instruct
- Fine-tuning method: QLoRA
- Evaluation samples: {quality["samples"]}

## Training Configuration

- Quantization: 4-bit NF4
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Learning rate: 2e-4
- Epochs: 1
- Batch size: 1
- Gradient accumulation: 8

## Base vs Fine-Tuned

| Metric | Base | Fine-Tuned | Delta |
|---|---:|---:|---:|
| Token F1 | {base["token_f1"]:.4f} | {fine["token_f1"]:.4f} | {fine["token_f1"] - base["token_f1"]:+.4f} |
| Exact Match | {base["exact_match"]:.4f} | {fine["exact_match"]:.4f} | {fine["exact_match"] - base["exact_match"]:+.4f} |
| Arabic Letter Ratio | {base["arabic_letter_ratio"]:.4f} | {fine["arabic_letter_ratio"]:.4f} | {fine["arabic_letter_ratio"] - base["arabic_letter_ratio"]:+.4f} |
| Average Response Length | {base["average_response_length"]:.1f} | {fine["average_response_length"]:.1f} | {fine["average_response_length"] - base["average_response_length"]:+.1f} |
| Repetition Ratio | {base["repetition_ratio"]:.4f} | {fine["repetition_ratio"]:.4f} | {fine["repetition_ratio"] - base["repetition_ratio"]:+.4f} |

## Behavioral Comparison

- Evaluation samples: {len(comparison)}
- Identical responses: {sum(r["identical"] for r in comparison)}
- Different responses: {sum(not r["identical"] for r in comparison)}

## Interpretation

The fine-tuned adapter produced different responses across the evaluation
set compared with the base model.

Token-level F1 increased from
{base["token_f1"]:.4f} to {fine["token_f1"]:.4f}.

The measured repetition ratio changed from
{base["repetition_ratio"]:.4f} to {fine["repetition_ratio"]:.4f}.

Arabic letter ratio remained high for both models.

These results are based on a 20-example evaluation subset and should
be treated as engineering evaluation evidence rather than a complete
assessment of Arabic language quality.

## Artifacts

- `data/evaluation/quality_metrics.json`
- `data/evaluation/base_vs_finetuned_v2.jsonl`
- `models/arabic-qlora-final`

"""

    Path(OUTPUT_FILE).write_text(
        report,
        encoding="utf-8"
    )

    print(
        "Evaluation report created:"
    )

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()