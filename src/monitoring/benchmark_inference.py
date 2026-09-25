import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


MODEL_PATH = "models/arabic-merged"

PROMPTS = [
    "ما هي أهمية الذكاء الاصطناعي في المؤسسات؟",
    "اشرح أهمية حماية البيانات في الشركات.",
    "ما هي فوائد استخدام النماذج اللغوية الكبيرة في المؤسسات؟",
]


MAX_NEW_TOKENS = 128


def get_vram():

    return (
        torch.cuda.memory_allocated()
        / 1024**3
    )


def get_peak_vram():

    return (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )


def main():

    print("=== GPU INFERENCE BENCHMARK ===")

    print(
        "Loading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    print(
        "Loading merged model..."
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    model.eval()

    print(
        "Model device:",
        next(model.parameters()).device,
    )

    print(
        "Model loaded successfully."
    )

    results = []

    for i, prompt in enumerate(PROMPTS):

        print(
            f"\n=== TEST {i + 1}/{len(PROMPTS)} ==="
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(model.device)

        input_tokens = (
            inputs["input_ids"].shape[1]
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        vram_before = get_vram()

        # Warm-up CUDA synchronization
        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        torch.cuda.synchronize()

        end = time.perf_counter()

        generation_time = end - start

        generated_tokens = (
            outputs.shape[1]
            - input_tokens
        )

        tokens_per_second = (
            generated_tokens
            / generation_time
        )

        vram_after = get_vram()
        peak_vram = get_peak_vram()

        response = tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True,
        )

        print(
            "Input tokens:",
            input_tokens,
        )

        print(
            "Output tokens:",
            generated_tokens,
        )

        print(
            "Generation time:",
            f"{generation_time:.3f} sec",
        )

        print(
            "Tokens/sec:",
            f"{tokens_per_second:.2f}",
        )

        print(
            "VRAM before:",
            f"{vram_before:.2f} GB",
        )

        print(
            "VRAM after:",
            f"{vram_after:.2f} GB",
        )

        print(
            "Peak VRAM:",
            f"{peak_vram:.2f} GB",
        )

        print(
            "Response:",
            response[:300],
        )

        results.append(
            {
                "test": i + 1,
                "input_tokens": input_tokens,
                "output_tokens": generated_tokens,
                "generation_time_seconds":
                    generation_time,
                "tokens_per_second":
                    tokens_per_second,
                "vram_before_gb":
                    vram_before,
                "vram_after_gb":
                    vram_after,
                "peak_vram_gb":
                    peak_vram,
            }
        )

    print(
        "\n=== BENCHMARK SUMMARY ==="
    )

    avg_latency = sum(
        r["generation_time_seconds"]
        for r in results
    ) / len(results)

    avg_tokens_per_second = sum(
        r["tokens_per_second"]
        for r in results
    ) / len(results)

    max_peak_vram = max(
        r["peak_vram_gb"]
        for r in results
    )

    print(
        f"Average latency: "
        f"{avg_latency:.3f} sec"
    )

    print(
        f"Average tokens/sec: "
        f"{avg_tokens_per_second:.2f}"
    )

    print(
        f"Peak VRAM: "
        f"{max_peak_vram:.2f} GB"
    )

    print(
        "\n=== GPU BENCHMARK COMPLETE ==="
    )


if __name__ == "__main__":
    main()