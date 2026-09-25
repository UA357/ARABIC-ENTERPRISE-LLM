import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "models/arabic-qlora-final"


def main():

    print("=== LOADING FINE-TUNED ARABIC MODEL ===")

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

    print("Base model:", BASE_MODEL)
    print("Adapter:", ADAPTER_PATH)
    print("Device:", model.device)
    print("Fine-tuned model: LOAD PASS")

    messages = [
        {
            "role": "user",
            "content": (
                "السلام عليكم. "
                "ما هي أهمية حماية البيانات في المؤسسات؟"
            ),
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

    print("\n=== GENERATING ARABIC RESPONSE ===")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = outputs[
        0
    ][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    print("\nMODEL RESPONSE:")
    print(response)

    print("\n=== INFERENCE TEST COMPLETE ===")


if __name__ == "__main__":
    main()