#!/usr/bin/env python3
"""
Script di test rapido: preprocessing + generazione Q&A su una fattura.

Uso:
    python scripts/test_invoice.py data/raw/paypal_invoice.png
    python scripts/test_invoice.py  # usa data/raw/paypal_invoice.png di default
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from visual_finetune_lab.preprocessing import ImageProcessor
from visual_finetune_lab.dataset import SyntheticDatasetGenerator

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/paypal_invoice.png")

if not image_path.exists():
    print(f"Immagine non trovata: {image_path}")
    sys.exit(1)

print(f"Immagine: {image_path}")

processor = ImageProcessor()
result = processor.process(image_path)
print(f"Processata: {result.width}x{result.height}px — {len(result.regions)} regioni rilevate")

print("\nGenerazione Q&A con GPT-4o Vision...")
generator = SyntheticDatasetGenerator()
samples = generator.generate([result])

print(f"\n{len(samples)} Q&A pairs generati:\n")
for i, s in enumerate(samples, 1):
    print(f"[{i}] Q: {s.question}")
    print(f"     A: {s.answer}")
