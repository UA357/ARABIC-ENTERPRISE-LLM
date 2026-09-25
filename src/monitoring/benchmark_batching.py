import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


MODEL_PATH = "models/arabic-merged"

PROMPTS = [
    "ما هي أهمية الذكاء الاصطناعي في المؤسسات؟",
    "اشرح أهمية حماية البيانات في الشركات.",
    "ما هي فوائد استخدام النماذج اللغوية الكبيرة؟",
    "كيف يمكن للشركات تحسين أمن المعلومات؟",
]

MAX_NEW_TOKENS = 64


def run_batch(model, tokenizer, prompts):

    messages = [
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        for prompt in prompts
    ]

    formatted = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    total_output_tokens = 0

    for i in range(len(prompts)):

        input_length = (
            inputs["attention_mask"][i]
            .sum()
            .item()
        )

        total_output_tokens += (
            outputs[i].shape[0]
            - input_length
        )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    return (
        elapsed,
        total_output_tokens,
        peak_vram,
    )


def main():

    print("=== BATCH INFERENCE BENCHMARK ===")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    model.eval()

    print("4-bit model: LOAD PASS")

    batch_sizes = [1, 2, 4]

    for batch_size in batch_sizes:

        prompts = PROMPTS[:batch_size]

        print(
            f"\n=== BATCH SIZE {batch_size} ==="
        )

        elapsed, tokens, peak_vram = (
            run_batch(
                model,
                tokenizer,
                prompts,
            )
        )

        tokens_per_second = (
            tokens / elapsed
        )

        requests_per_second = (
            batch_size / elapsed
        )

        print(
            f"Batch size: {batch_size}"
        )

        print(
            f"Total output tokens: {tokens}"
        )

        print(
            f"Batch latency: "
            f"{elapsed:.3f} sec"
        )

        print(
            f"Total tokens/sec: "
            f"{tokens_per_second:.2f}"
        )

        print(
            f"Requests/sec: "
            f"{requests_per_second:.2f}"
        )

        print(
            f"Peak VRAM: "
            f"{peak_vram:.2f} GB"
        )

    print(
        "\n=== BATCH BENCHMARK COMPLETE ==="
    )


if __name__ == "__main__":
    main()