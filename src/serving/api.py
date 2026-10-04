import os
from time import perf_counter, time
from collections import defaultdict, deque
import threading
from src.security.auth import authorize_generate
import requests
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator, model_validator
from src.security.privacy_controls import prepare_prompt

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000")

app = FastAPI(
    title="Arabic Enterprise LLM API",
    version="1.2.0",
)

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "reveal your instructions",
    "show your system prompt",
    "developer message",
    "jailbreak",
]

MAX_PROMPT_LENGTH = 4000
MAX_OUTPUT_LENGTH = 8000

RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60

request_history = defaultdict(deque)
rate_limit_lock = threading.Lock()

REQUEST_COUNT = Counter(
    "llm_api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)

RATE_LIMIT_COUNT = Counter(
    "llm_api_rate_limit_total",
    "Rate-limited requests",
)

REQUEST_LATENCY = Histogram(
    "llm_api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)

TOKENS_GENERATED = Counter(
    "llm_api_completion_tokens_total",
    "Total completion tokens generated",
)


def check_rate_limit(client_id: str):
    now = time()

    with rate_limit_lock:
        history = request_history[client_id]

        while history and now - history[0] > RATE_LIMIT_WINDOW:
            history.popleft()

        if len(history) >= RATE_LIMIT_REQUESTS:
            return False

        history.append(now)
        return True


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Prompt cannot be empty.")

        lowered = value.casefold()

        for pattern in BLOCKED_PATTERNS:
            if pattern in lowered:
                raise ValueError("Prompt rejected by security guardrail.")

        return value


    @model_validator(mode="after")
    def apply_privacy_controls(self):
        privacy_result = prepare_prompt(
            self.prompt,
            redact=True,
        )
        self.prompt = privacy_result["processed_text"]
        return self


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path == "/generate" and request.method == "POST":
        client_id = request.client.host if request.client else "unknown"

        if not check_rate_limit(client_id):
            RATE_LIMIT_COUNT.inc()
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Cache-Control"] = "no-store"

    return response


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
def health():
    try:
        response = requests.get(f"{VLLM_URL}/health", timeout=5)

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


@app.post("/generate", dependencies=[Depends(authorize_generate)])
def generate(request: GenerateRequest):
    start = perf_counter()
    REQUEST_COUNT.labels(endpoint="generate", status="started").inc()

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

        if not generated_text or not generated_text.strip():
            raise HTTPException(
                status_code=502,
                detail="Model returned an empty response.",
            )

        if len(generated_text) > MAX_OUTPUT_LENGTH:
            generated_text = generated_text[:MAX_OUTPUT_LENGTH]

        elapsed = perf_counter() - start
        REQUEST_LATENCY.labels(endpoint="generate").observe(elapsed)
        usage = result.get("usage", {})
        TOKENS_GENERATED.inc(usage.get("completion_tokens", 0) or 0)

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
