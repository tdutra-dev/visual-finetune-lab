#!/usr/bin/env python3
"""
End-to-end pipeline script:
  1. Preprocess all images in data/raw/
  2. Generate synthetic Q&A dataset via GPT-4o Vision
  3. Fine-tune with LoRA (tracked in MLflow)
  4. Evaluate on held-out samples

Usage:
    python scripts/run_pipeline.py --images data/raw/invoices/ --epochs 3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from visual_finetune_lab.dataset import SyntheticDatasetGenerator
from visual_finetune_lab.evaluation import ModelEvaluator
from visual_finetune_lab.preprocessing import ImageProcessor
from visual_finetune_lab.tracking import ExperimentTracker
from visual_finetune_lab.training import LoRATrainer, TrainingConfig


def main(images_dir: Path, epochs: int, lora_rank: int) -> None:
    tracker = ExperimentTracker("visual-finetune-lab")

    with tracker.run(f"run-r{lora_rank}-e{epochs}") as run:
        # 1. Preprocessing
        print(f"[1/4] Preprocessing images from {images_dir}")
        processor = ImageProcessor(max_size=1344)
        processed = processor.process_batch(images_dir)
        print(f"      → {len(processed)} images processed")

        # 2. Dataset generation
        print("[2/4] Generating synthetic Q&A dataset (GPT-4o Vision)")
        generator = SyntheticDatasetGenerator()
        samples = generator.generate(processed)
        dataset_path = Path("data/datasets/dataset.jsonl")
        generator.save(samples, dataset_path)
        print(f"      → {len(samples)} samples saved to {dataset_path}")
        tracker.log_params({
            "num_images": len(processed),
            "num_samples": len(samples),
            "lora_rank": lora_rank,
            "epochs": epochs,
        })

        # 3. Fine-tuning
        print("[3/4] Fine-tuning with LoRA…")
        config = TrainingConfig(
            num_epochs=epochs,
            lora_rank=lora_rank,
            lora_alpha=lora_rank * 2,
        )
        trainer = LoRATrainer(config)
        checkpoint = trainer.train(dataset_path)
        print(f"      → Best checkpoint saved to {checkpoint}")
        tracker.log_artifact(str(checkpoint))

        # 4. Evaluation
        print("[4/4] Evaluating on held-out samples")
        eval_samples = [{"question": s.question, "answer": s.answer, "image_path": s.image_path}
                        for s in samples[-20:]]   # last 20 as quick eval set
        evaluator = ModelEvaluator(checkpoint_path=checkpoint)
        results = evaluator.evaluate(eval_samples)
        evaluator.print_summary(results)
        tracker.log_eval_results(results)

    print(f"\nDone. MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/raw"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("NUM_EPOCHS", "3")))
    parser.add_argument("--lora-rank", type=int, default=int(os.getenv("LORA_RANK", "16")))
    args = parser.parse_args()
    main(args.images, args.epochs, args.lora_rank)
