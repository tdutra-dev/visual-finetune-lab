from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline

logger = structlog.get_logger()

# ------------------------------------------------------------------ #
# Shared state
# ------------------------------------------------------------------ #

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.environ.get("MODEL_CHECKPOINT_PATH", "./checkpoints/best")
    logger.info("loading_model", checkpoint=checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    base_id = processor.tokenizer.name_or_path
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    _state["pipe"] = pipeline(
        "text-generation",
        model=model,
        tokenizer=processor.tokenizer,
        max_new_tokens=512,
    )
    logger.info("model_ready")
    yield
    _state.clear()


app = FastAPI(
    title="visual-finetune-lab",
    description="Serve predictions from a LoRA fine-tuned vision-language model",
    version="0.1.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #

class PredictResponse(BaseModel):
    question: str
    answer: str
    model_checkpoint: str


class HealthResponse(BaseModel):
    status: str


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse)
async def health():
    if "pipe" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    question: str = Form(...),
    image: UploadFile = File(...),
):
    if "pipe" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate content type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await image.read()
    if len(raw) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    prompt = f"User: {question}\nAssistant:"
    output = _state["pipe"](prompt)[0]["generated_text"]

    if "Assistant:" in output:
        answer = output.split("Assistant:")[-1].strip()
    else:
        answer = output.strip()

    logger.info("prediction_served", question=question[:80])
    return PredictResponse(
        question=question,
        answer=answer,
        model_checkpoint=os.environ.get("MODEL_CHECKPOINT_PATH", "unknown"),
    )


@app.post("/predict/text", response_model=PredictResponse)
async def predict_text(question: str = Form(...)):
    """Text-only endpoint for questions that don't require an image."""
    if "pipe" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prompt = f"User: {question}\nAssistant:"
    output = _state["pipe"](prompt)[0]["generated_text"]
    answer = output.split("Assistant:")[-1].strip() if "Assistant:" in output else output.strip()
    return PredictResponse(
        question=question,
        answer=answer,
        model_checkpoint=os.environ.get("MODEL_CHECKPOINT_PATH", "unknown"),
    )
