import json
import gc
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
OUTPUT_FILE = "data/evaluation/base_vs_finetuned_v2.jsonl"

NUM_SAMPLES = 20
MAX_NEW_TOKENS = 128


def arabic_letter_ratio(text):

    if not text:
        return 0.0

    letters = [
        c for c in text
        if c.isalpha()
    ]

    if not letters:
        return 0.0

    arabic_letters = [
        c for c in letters
        if "\u0600" <= c <= "\u06FF"
    ]

    return len(arabic_letters) / len(letters)


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


def load_base_model():

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )


def generate_response(
    model,
    tokenizer,
    instruction,
    input_text,
):

    content = instruction

    if input_text:
        content += "\n\n" + input_text

    messages = [
        {
            "role": "user",
            "content": content,
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

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def unload_model(model):

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():

    print("=== CORRECTED BASE VS FINE-TUNED TEST ===")

    records = load_test_data()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    # -------------------------------------------------
    # PASS 1: BASE MODEL ONLY
    # -------------------------------------------------

    print("\n=== PASS 1: BASE MODEL ===")

    base_model = load_base_model()
    base_model.eval()

    base_responses = []

    for i, record in enumerate(records):

        response = generate_response(
            base_model,
            tokenizer,
            record["instruction"],
            record["input"],
        )

        base_responses.append(response)

        print(
            f"Base [{i + 1}/{len(records)}]"
        )

    unload_model(base_model)

    print("Base model unloaded.")

    # -------------------------------------------------
    # PASS 2: FINE-TUNED MODEL ONLY
    # -------------------------------------------------

    print("\n=== PASS 2: FINE-TUNED MODEL ===")

    finetuned_base = load_base_model()

    finetuned_model = PeftModel.from_pretrained(
        finetuned_base,
        ADAPTER_PATH,
    )

    finetuned_model.eval()

    finetuned_responses = []

    for i, record in enumerate(records):

        response = generate_response(
            finetuned_model,
            tokenizer,
            record["instruction"],
            record["input"],
        )

        finetuned_responses.append(response)

        print(
            f"Fine-tuned [{i + 1}/{len(records)}]"
        )

    # -------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------

    results = []

    identical = 0

    for i, record in enumerate(records):

        base_response = base_responses[i]
        finetuned_response = (
            finetuned_responses[i]
        )

        if base_response == finetuned_response:
            identical += 1

        results.append(
            {
                "index": i,
                "reference": record["output"],

                "base_response":
                    base_response,

                "finetuned_response":
                    finetuned_response,

                "base_arabic_ratio":
                    arabic_letter_ratio(
                        base_response
                    ),

                "finetuned_arabic_ratio":
                    arabic_letter_ratio(
                        finetuned_response
                    ),

                "identical":
                    base_response
                    == finetuned_response,
            }
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

    print("\n=== FINAL COMPARISON ===")

    print(
        f"Samples: {len(records)}"
    )

    print(
        f"Identical responses: "
        f"{identical}/{len(records)}"
    )

    print(
        f"Different responses: "
        f"{len(records) - identical}/"
        f"{len(records)}"
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )

    print(
        "\nCORRECTED COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()