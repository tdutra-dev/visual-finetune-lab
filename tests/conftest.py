import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def fake_image_path(tmp_path: Path) -> Path:
    img = Image.fromarray(np.zeros((400, 600, 3), dtype=np.uint8))
    p = tmp_path / "test.jpg"
    img.save(p)
    return p


@pytest.fixture
def fake_dataset_path(tmp_path: Path) -> Path:
    samples = [
        {"image_path": "img1.jpg", "question": "What is the total?", "answer": "€100", "source_description": "invoice"},
        {"image_path": "img2.jpg", "question": "What is the date?", "answer": "2026-01-01", "source_description": "invoice"},
    ]
    p = tmp_path / "dataset.jsonl"
    with open(p, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return p
