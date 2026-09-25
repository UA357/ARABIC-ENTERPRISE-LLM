from peft import LoraConfig, get_peft_model


def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model


if __name__ == "__main__":
    from src.training.load_qlora_model import load_qlora_model

    tokenizer, model = load_qlora_model()

    model = apply_lora(model)

    print("LoRA adapter setup: PASS")