import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from configs.qlora_config import QLoRAConfig


def load_qlora_model():
    config = QLoRAConfig()

    compute_dtype = torch.bfloat16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    print("Loading 4-bit model...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map="auto",
        dtype=compute_dtype,
    )

    print("\n=== QLoRA MODEL TEST ===")
    print(f"Model: {config.model_name}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"4-bit: {model.is_loaded_in_4bit}")
    print(f"CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"VRAM allocated: {allocated:.2f} GB")
        print(f"VRAM reserved: {reserved:.2f} GB")

    print("QLORA MODEL LOAD: PASS")

    return tokenizer, model


if __name__ == "__main__":
    load_qlora_model()