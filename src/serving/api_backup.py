from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests


VLLM_URL = "http://127.0.0.1:8000"

app = FastAPI(
    title="Arabic Enterprise LLM API",
    version="1.0.0",
    description="Production API gateway for the Arabic Enterprise LLM served by vLLM.",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


@app.get("/health")
def health():
    try:
        response = requests.get(
            f"{VLLM_URL}/health",
            timeout=5,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail="vLLM server is unhealthy",
            )

        return {
            "status": "healthy",
            "service": "arabic-enterprise-llm-api",
            "vllm": "healthy",
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM unavailable: {exc}",
        )


@app.get("/model")
def model():
    try:
        response = requests.get(
            f"{VLLM_URL}/v1/models",
            timeout=5,
        )
        response.raise_for_status()

        data = response.json()

        if not data.get("data"):
            raise HTTPException(
                status_code=503,
                detail="No vLLM model available",
            )

        return {
            "model": data["data"][0]["id"],
            "max_model_len": data["data"][0].get("max_model_len"),
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to query vLLM: {exc}",
        )


@app.post("/generate")
def generate(request: GenerateRequest):
    start = perf_counter()

    try:
        model_response = requests.get(
            f"{VLLM_URL}/v1/models",
            timeout=5,
        )
        model_response.raise_for_status()

        model_data = model_response.json()

        if not model_data.get("data"):
            raise HTTPException(
                status_code=503,
                detail="No vLLM model available",
            )

        model_name = model_data["data"][0]["id"]

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("choices"):
            raise HTTPException(
                status_code=502,
                detail="vLLM returned no generation",
            )

        generated_text = result["choices"][0]["message"]["content"]

        elapsed = perf_counter() - start

        usage = result.get("usage", {})

        return {
            "response": generated_text,
            "model": model_name,
            "latency_seconds": round(elapsed, 4),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"vLLM request failed: {exc}",
        )
