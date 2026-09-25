import json
import re
from pathlib import Path


INPUT_FILE = (
    "data/evaluation/base_vs_finetuned_v2.jsonl"
)

OUTPUT_FILE = (
    "data/evaluation/quality_metrics.json"
)


def normalize_text(text):

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def tokenize(text):

    return normalize_text(text).split()


def token_f1(prediction, reference):

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}

    for token in pred_tokens:
        pred_counts[token] = (
            pred_counts.get(token, 0) + 1
        )

    ref_counts = {}

    for token in ref_tokens:
        ref_counts[token] = (
            ref_counts.get(token, 0) + 1
        )

    overlap = 0

    for token in pred_counts:

        if token in ref_counts:

            overlap += min(
                pred_counts[token],
                ref_counts[token],
            )

    if overlap == 0:
        return 0.0

    precision = (
        overlap / len(pred_tokens)
    )

    recall = (
        overlap / len(ref_tokens)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def exact_match(prediction, reference):

    return (
        normalize_text(prediction)
        == normalize_text(reference)
    )


def arabic_letter_ratio(text):

    if not text:
        return 0.0

    letters = [
        c
        for c in text
        if c.isalpha()
    ]

    if not letters:
        return 0.0

    arabic_letters = [
        c
        for c in letters
        if "\u0600" <= c <= "\u06FF"
    ]

    return (
        len(arabic_letters)
        / len(letters)
    )


def repetition_ratio(text):

    tokens = tokenize(text)

    if len(tokens) < 5:
        return 0.0

    return (
        1
        - len(set(tokens))
        / len(tokens)
    )


def evaluate_model(
    records,
    response_key,
):

    f1_scores = []
    exact_matches = []
    arabic_ratios = []
    lengths = []
    repetition_scores = []

    for record in records:

        prediction = record[
            response_key
        ]

        reference = record[
            "reference"
        ]

        f1_scores.append(
            token_f1(
                prediction,
                reference,
            )
        )

        exact_matches.append(
            exact_match(
                prediction,
                reference,
            )
        )

        arabic_ratios.append(
            arabic_letter_ratio(
                prediction
            )
        )

        lengths.append(
            len(prediction)
        )

        repetition_scores.append(
            repetition_ratio(
                prediction
            )
        )

    count = len(records)

    return {
        "token_f1": (
            sum(f1_scores) / count
        ),
        "exact_match": (
            sum(exact_matches) / count
        ),
        "arabic_letter_ratio": (
            sum(arabic_ratios) / count
        ),
        "average_response_length": (
            sum(lengths) / count
        ),
        "repetition_ratio": (
            sum(repetition_scores) / count
        ),
    }


def main():

    print(
        "=== QUALITY EVALUATION ==="
    )

    records = [
        json.loads(line)
        for line in Path(INPUT_FILE)
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    print(
        f"Evaluation samples: "
        f"{len(records)}"
    )

    base_metrics = evaluate_model(
        records,
        "base_response",
    )

    finetuned_metrics = evaluate_model(
        records,
        "finetuned_response",
    )

    results = {
        "samples": len(records),
        "base_model": base_metrics,
        "finetuned_model": finetuned_metrics,
    }

    output_path = Path(
        OUTPUT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== BASE MODEL ==="
    )

    for key, value in base_metrics.items():

        print(
            f"{key}: {value:.4f}"
        )

    print(
        "\n=== FINE-TUNED MODEL ==="
    )

    for key, value in finetuned_metrics.items():

        print(
            f"{key}: {value:.4f}"
        )

    print(
        "\n=== METRIC DELTAS ==="
    )

    for key in base_metrics:

        delta = (
            finetuned_metrics[key]
            - base_metrics[key]
        )

        print(
            f"{key}: {delta:+.4f}"
        )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )

    print(
        "\nQUALITY EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()