import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from visual_finetune_lab.dataset import DatasetSample, SyntheticDatasetGenerator


def _mock_openai_response(qa_pairs: list[dict]) -> MagicMock:
    content = json.dumps({
        "description": "A test invoice document",
        "qa_pairs": qa_pairs,
    })
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def generator():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with patch("visual_finetune_lab.dataset.synthetic_generator.OpenAI"):
            gen = SyntheticDatasetGenerator()
            return gen


def test_generate_returns_samples(generator, fake_image_path: Path):
    qa = [{"question": "Total?", "answer": "€50"}]
    generator.client.chat.completions.create.return_value = _mock_openai_response(qa)

    processed = MagicMock()
    processed.path = fake_image_path
    processed.to_base64.return_value = "base64data"

    samples = generator.generate([processed])
    assert len(samples) == 1
    assert samples[0].question == "Total?"
    assert samples[0].answer == "€50"


def test_generate_skips_failed_images(generator, fake_image_path: Path):
    generator.client.chat.completions.create.side_effect = Exception("API error")
    processed = MagicMock()
    processed.path = fake_image_path
    processed.to_base64.return_value = "base64data"
    samples = generator.generate([processed])
    assert samples == []


def test_save_and_load_roundtrip(generator, tmp_path: Path, fake_image_path: Path):
    qa = [{"question": "Q1?", "answer": "A1"}, {"question": "Q2?", "answer": "A2"}]
    generator.client.chat.completions.create.return_value = _mock_openai_response(qa)

    processed = MagicMock()
    processed.path = fake_image_path
    processed.to_base64.return_value = "base64data"

    samples = generator.generate([processed])
    out = tmp_path / "out.jsonl"
    generator.save(samples, out)

    loaded = generator.load(out)
    assert len(loaded) == len(samples)
    assert loaded[0].question == samples[0].question


def test_to_chat_format():
    sample = DatasetSample(image_path="img.jpg", question="What is X?", answer="Y")
    chat = sample.to_chat_format()
    assert chat["messages"][0]["role"] == "user"
    assert chat["messages"][1]["role"] == "assistant"
    assert chat["messages"][1]["content"] == "Y"
