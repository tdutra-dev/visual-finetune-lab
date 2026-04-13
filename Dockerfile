FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[serving]" 2>/dev/null || pip install --no-cache-dir .

COPY src/ src/

ENV MODEL_CHECKPOINT_PATH=/checkpoints/best
ENV PORT=8000

EXPOSE 8000

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "visual_finetune_lab.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
