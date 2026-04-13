from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
from openai import OpenAI
from pydantic import BaseModel

from visual_finetune_lab.preprocessing import ProcessedImage

logger = structlog.get_logger()


# ------------------------------------------------------------------ #
# Data models
# ------------------------------------------------------------------ #

class QAPair(BaseModel):
    question: str
    answer: str


class ImageAnalysis(BaseModel):
    description: str
    qa_pairs: list[QAPair]


@dataclass
class DatasetSample:
    image_path: str
    question: str
    answer: str
    source_description: str = ""

    def to_chat_format(self) -> dict:
        """
        Returns the sample in the multi-modal chat format expected by
        transformers AutoProcessor (Phi-3.5-Vision, Qwen2-VL, etc.)
        """
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image_path},
                        {"type": "text", "text": self.question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": self.answer,
                },
            ]
        }


# ------------------------------------------------------------------ #
# Generator
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are an expert document analyst. Given an image, generate a rich set of
question-answer pairs that would be useful for training a specialized vision model.

Focus on:
- Factual extraction (numbers, dates, names, amounts)
- Structural understanding (layout, sections, relationships)
- Anomaly detection (unusual values, errors, warnings)

Return JSON matching this schema:
{
  "description": "<one-sentence image summary>",
  "qa_pairs": [
    {"question": "...", "answer": "..."},
    ...
  ]
}
Generate between 5 and 10 qa_pairs per image."""


class SyntheticDatasetGenerator:
    """
    Uses GPT-4o Vision to automatically generate Q&A training pairs from images.

    Workflow:
      1. Receives a list of ProcessedImage objects
      2. Sends each image (base64) to GPT-4o with a structured prompt
      3. Parses the structured JSON response
      4. Returns a list of DatasetSample objects ready for fine-tuning

    Example:
        gen = SyntheticDatasetGenerator()
        samples = gen.generate(processed_images)
        gen.save(samples, Path("data/datasets/invoices.jsonl"))
    """

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 2048):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, images: list[ProcessedImage]) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for img in images:
            logger.info("generating_qa_pairs", image=str(img.path))
            try:
                analysis = self._analyze_image(img)
                for pair in analysis.qa_pairs:
                    samples.append(DatasetSample(
                        image_path=str(img.path),
                        question=pair.question,
                        answer=pair.answer,
                        source_description=analysis.description,
                    ))
            except Exception:
                logger.exception("failed_to_generate_qa", image=str(img.path))
        logger.info("dataset_generation_complete", total_samples=len(samples))
        return samples

    def save(self, samples: list[DatasetSample], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
        logger.info("dataset_saved", path=str(output_path), count=len(samples))

    def load(self, path: Path) -> list[DatasetSample]:
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                samples.append(DatasetSample(**data))
        return samples

    # ------------------------------------------------------------------ #

    def _analyze_image(self, img: ProcessedImage) -> ImageAnalysis:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img.to_base64()}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "Analyze this image and generate training Q&A pairs."},
                    ],
                },
            ],
        )
        raw = response.choices[0].message.content
        return ImageAnalysis.model_validate_json(raw)
