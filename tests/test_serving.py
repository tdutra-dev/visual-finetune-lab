import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visual_finetune_lab.serving.api import app, _state


@pytest.fixture(autouse=True)
def mock_model():
    pipe = MagicMock()
    pipe.return_value = [{"generated_text": "User: Q\nAssistant: The answer is 42"}]
    _state["pipe"] = pipe
    yield
    _state.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_no_model(client):
    _state.clear()
    response = client.get("/health")
    assert response.status_code == 503


def test_predict_returns_answer(client, sample_image_bytes):
    response = client.post(
        "/predict",
        data={"question": "What is the total?"},
        files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "The answer is 42"


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict",
        data={"question": "Q"},
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_text_only(client):
    response = client.post("/predict/text", data={"question": "Hello?"})
    assert response.status_code == 200
    assert "answer" in response.json()
