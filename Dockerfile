FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

COPY requirements.txt .
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/arabic-merged

EXPOSE 8080

ENTRYPOINT []

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
