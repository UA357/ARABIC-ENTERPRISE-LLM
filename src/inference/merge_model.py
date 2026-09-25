import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = (
    "models/arabic-qlora-final"
)

OUTPUT_PATH = (
    "models/arabic-merged"
)


def main():

    print("=== MERGING QLORA MODEL ===")

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH
    )

    print("Loading base model...")

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    )

    print("Loading QLoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    print("Merging adapter into base model...")

    merged_model = model.merge_and_unload()

    print("Saving merged model...")

    merged_model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        OUTPUT_PATH
    )

    print(
        f"\nMerged model saved to: "
        f"{OUTPUT_PATH}"
    )

    print("\n=== MERGE COMPLETE ===")


if __name__ == "__main__":
    main()