from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class ProcessedImage:
    path: Path
    array: np.ndarray          # BGR numpy array (OpenCV format)
    regions: list[dict]        # detected bounding boxes [{"label": str, "bbox": [x,y,w,h]}]
    width: int
    height: int

    def to_pil(self) -> Image.Image:
        rgb = cv2.cvtColor(self.array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def to_base64(self) -> str:
        pil = self.to_pil()
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()


class ImageProcessor:
    """
    Preprocessing pipeline for documents and screenshots before fine-tuning.

    Steps:
      1. Load & validate image
      2. Resize to model's maximum input size (default 1344px longest side)
      3. Deskew (correct scan rotation)
      4. Detect high-contrast regions (bounding boxes) via contour analysis
      5. Optional: crop to a specific region

    Example:
        processor = ImageProcessor(max_size=1344)
        result = processor.process(Path("invoices/001.jpg"))
        print(result.regions)   # [{"label": "region_0", "bbox": [x, y, w, h]}, ...]
    """

    def __init__(self, max_size: int = 1344, min_region_area: int = 5000):
        self.max_size = max_size
        self.min_region_area = min_region_area

    def process(self, image_path: Path) -> ProcessedImage:
        img = self._load(image_path)
        img = self._resize(img)
        img = self._deskew(img)
        regions = self._detect_regions(img)
        h, w = img.shape[:2]
        return ProcessedImage(path=image_path, array=img, regions=regions, width=w, height=h)

    def process_batch(self, folder: Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[ProcessedImage]:
        paths = [p for p in folder.iterdir() if p.suffix.lower() in extensions]
        return [self.process(p) for p in sorted(paths)]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= self.max_size:
            return img
        scale = self.max_size / longest
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 10:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect returns angles in [-90, 0); normalize to [-45, 45)
        if angle < -45:
            angle += 90
        if abs(angle) < 0.5:   # skip negligible skew
            return img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _detect_regions(self, img: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < self.min_region_area:
                continue
            regions.append({"label": f"region_{i}", "bbox": [int(x), int(y), int(w), int(h)]})

        # Sort top-to-bottom, left-to-right (natural reading order)
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return regions
