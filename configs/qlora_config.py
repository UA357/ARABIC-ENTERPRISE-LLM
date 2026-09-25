from dataclasses import dataclass


@dataclass
class QLoRAConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    max_seq_length: int = 1024

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    learning_rate: float = 2e-4
    num_train_epochs: int = 1

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    logging_steps: int = 5
    save_steps: int = 50

    output_dir: str = "models/arabic-qlora"