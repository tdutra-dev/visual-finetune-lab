"""
Integration test: preprocessing + dataset generation on a real invoice image.

Requires:
  - OPENAI_API_KEY set in .env or environment
  - An image file at data/raw/paypal_invoice.png (or override via INVOICE_IMAGE env var)

Run with:
    pytest tests/test_invoice_integration.py -v -s
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
IMAGE_PATH = Path(os.environ.get("INVOICE_IMAGE", REPO_ROOT / "data/raw/paypal_invoice.png"))


@pytest.fixture(scope="module")
def processed_image():
    from visual_finetune_lab.preprocessing import ImageProcessor
    processor = ImageProcessor()
    result = processor.process(IMAGE_PATH)
    return result


@pytest.mark.skipif(not IMAGE_PATH.exists(), reason=f"Invoice image not found: {IMAGE_PATH}")
def test_preprocessing(processed_image):
    assert processed_image.width > 0
    assert processed_image.height > 0
    assert len(processed_image.regions) > 0
    print(f"\nImmagine processata: {processed_image.width}x{processed_image.height}px")
    print(f"Regioni rilevate: {len(processed_image.regions)}")


@pytest.mark.skipif(not IMAGE_PATH.exists(), reason=f"Invoice image not found: {IMAGE_PATH}")
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY non impostata nel .env")
def test_dataset_generation(processed_image):
    from visual_finetune_lab.dataset import SyntheticDatasetGenerator
    generator = SyntheticDatasetGenerator()
    samples = generator.generate([processed_image])

    assert len(samples) > 0, "Nessun sample generato"

    print(f"\nGenerati {len(samples)} Q&A pairs:")
    for s in samples:
        print(f"\n  Q: {s.question}")
        print(f"  A: {s.answer}")

    questions = [s.question.lower() for s in samples]
    answers = [s.answer.lower() for s in samples]

    # Verifica che almeno alcune informazioni chiave della fattura siano presenti
    all_text = " ".join(questions + answers)
    assert any(kw in all_text for kw in ["paypal", "openai", "payment", "amount", "total", "pagamento"]), \
        "I Q&A non sembrano riferirsi alla fattura"
