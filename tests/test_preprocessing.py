from pathlib import Path

import numpy as np
import pytest

from visual_finetune_lab.preprocessing import ImageProcessor, ProcessedImage


def test_process_returns_processed_image(fake_image_path: Path):
    processor = ImageProcessor()
    result = processor.process(fake_image_path)
    assert isinstance(result, ProcessedImage)
    assert result.width > 0
    assert result.height > 0
    assert result.path == fake_image_path


def test_resize_limits_longest_side(fake_image_path: Path):
    processor = ImageProcessor(max_size=200)
    result = processor.process(fake_image_path)
    assert max(result.width, result.height) <= 200


def test_regions_are_sorted_by_position(fake_image_path: Path):
    processor = ImageProcessor()
    result = processor.process(fake_image_path)
    ys = [r["bbox"][1] for r in result.regions]
    assert ys == sorted(ys)


def test_to_base64_is_non_empty(fake_image_path: Path):
    processor = ImageProcessor()
    result = processor.process(fake_image_path)
    b64 = result.to_base64()
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_invalid_path_raises():
    processor = ImageProcessor()
    with pytest.raises(ValueError, match="Cannot load image"):
        processor.process(Path("/nonexistent/image.jpg"))


def test_process_batch(tmp_path: Path, fake_image_path: Path):
    import shutil
    shutil.copy(fake_image_path, tmp_path / "a.jpg")
    shutil.copy(fake_image_path, tmp_path / "b.jpg")
    processor = ImageProcessor()
    results = processor.process_batch(tmp_path)
    assert len(results) == 2
