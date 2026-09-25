import json
import re
import torch

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "models/arabic-qlora-final"

TEST_FILE = "data/processed/splits/test.jsonl"

OUTPUT_FILE = (
    "data/evaluation/finetuned_results.jsonl"
)

MAX_NEW_TOKENS = 128

NUM_SAMPLES = 20


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

    return len(arabic_letters) / len(letters)


def repetition_ratio(text):

    words = text.split()

    if len(words) < 5:
        return 0.0

    unique_words = len(set(words))

    return 1 - (
        unique_words / len(words)
    )


def load_test_data():

    records = []

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records[:NUM_SAMPLES]


def main():

    print("=== FINE-TUNED MODEL EVALUATION ===")

    records = load_test_data()

    print(
        f"Evaluation samples: {len(records)}"
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

    model.eval()

    results = []

    valid_count = 0
    arabic_count = 0

    for i, record in enumerate(records):

        instruction = record["instruction"]
        input_text = record["input"]

        user_content = instruction

        if input_text:
            user_content += (
                "\n\n" + input_text
            )

        messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        generated = outputs[
            0
        ][inputs["input_ids"].shape[1]:]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        is_valid = len(response) > 0
        ar_ratio = arabic_letter_ratio(response)
        rep_ratio = repetition_ratio(response)

        if is_valid:
            valid_count += 1

        if ar_ratio >= 0.50:
            arabic_count += 1

        result = {
            "index": i,
            "instruction": instruction,
            "reference": record["output"],
            "response": response,
            "arabic_letter_ratio": ar_ratio,
            "repetition_ratio": rep_ratio,
            "response_length": len(response),
            "valid": is_valid,
            
        }

        results.append(result)

        print(
            f"\n[{i + 1}/{len(records)}]"
        )

        print(
            "Response:",
            response[:300]
        )
         
        print(
            "Arabic letter ratio:",
            round(ar_ratio, 3)
        )

        print(
           "Repetition ratio:",
           round(rep_ratio, 3)
        )
        

    output_path = Path(OUTPUT_FILE)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n=== EVALUATION SUMMARY ===")

    print(
        f"Valid responses: "
        f"{valid_count}/{len(records)}"
    )

    print(
        f"Arabic responses: "
        f"{arabic_count}/{len(records)}"
    )

    print(
        f"Results saved: {OUTPUT_FILE}"
    )

    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()