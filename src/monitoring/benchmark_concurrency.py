import time
import concurrent.futures
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


MODEL_PATH = "models/arabic-merged"

PROMPT = (
    "اشرح أهمية الذكاء الاصطناعي "
    "في المؤسسات والشركات."
)

MAX_NEW_TOKENS = 64


def generate(model, tokenizer):

    messages = [
        {
            "role": "user",
            "content": PROMPT,
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

    return outputs


def main():

    print("=== GPU CONCURRENCY BENCHMARK ===")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    model.eval()

    print("4-bit model: LOAD PASS")

    concurrency_levels = [1, 2, 4]

    for workers in concurrency_levels:

        print(
            f"\n=== CONCURRENCY {workers} ==="
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = [
                executor.submit(
                    generate,
                    model,
                    tokenizer,
                )
                for _ in range(workers)
            ]

            results = [
                future.result()
                for future in futures
            ]

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - start
        )

        total_requests = len(results)

        requests_per_second = (
            total_requests / elapsed
        )

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Concurrent requests: "
            f"{total_requests}"
        )

        print(
            f"Total time: "
            f"{elapsed:.3f} sec"
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
        "\n=== CONCURRENCY BENCHMARK COMPLETE ==="
    )


if __name__ == "__main__":
    main()