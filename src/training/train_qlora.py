import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "data/processed/hf_enterprise_dataset"
OUTPUT_DIR = "models/arabic-qlora"
SMOKE_OUTPUT_DIR = "models/arabic-qlora-smoke-test"


def main():

    print("Loading dataset...")

    dataset = load_from_disk(DATASET_PATH)

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_example(example):

        messages = [
            {
                "role": "user",
                "content": example["instruction"],
            }
        ]

        if example["input"]:
            messages[0]["content"] += (
                "\n\n" + example["input"]
            )

        messages.append(
            {
                "role": "assistant",
                "content": example["output"],
            }
        )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": text}

    print("Formatting training dataset...")

    train_dataset = train_dataset.map(
        format_example,
        remove_columns=train_dataset.column_names,
    )

    print("Formatting validation dataset...")

    validation_dataset = validation_dataset.map(
        format_example,
        remove_columns=validation_dataset.column_names,
    )

    def tokenize(example):

        return tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
        )

    print("Tokenizing training dataset...")

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    print("Tokenizing validation dataset...")

    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading 4-bit model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    model.config.use_cache = False

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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    print("\n=== QLORA TRAINING CONFIGURATION ===")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(
        f"Batch size: "
        f"{training_args.per_device_train_batch_size}"
    )
    print(
        f"Gradient accumulation: "
        f"{training_args.gradient_accumulation_steps}"
    )
    print(
        f"Learning rate: "
        f"{training_args.learning_rate}"
    )

    print("\n=== STARTING QLORA SMOKE TEST ===")

    trainer.args.max_steps = -1

    trainer.train()

    print("\n=== QLORA SMOKE TEST COMPLETE ===")

    model.save_pretrained(SMOKE_OUTPUT_DIR)
    tokenizer.save_pretrained(SMOKE_OUTPUT_DIR)

    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()