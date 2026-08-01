"""Tests for the TAMU AI Chat Preview streaming image provider."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import plugins.image_gen.tamu_preview as tamu_image


_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


class _StreamResponse:
    status_code = 200

    def __init__(self, content: str):
        payload = {"choices": [{"delta": {"content": content}}]}
        self._lines = [f"data: {json.dumps(payload)}", "data: [DONE]"]

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def _b64_png() -> str:
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def test_metadata_lists_five_documented_preview_models(monkeypatch):
    monkeypatch.setattr(tamu_image, "_load_config", lambda: {})
    provider = tamu_image.TamuPreviewImageGenProvider()
    assert provider.name == "tamu-preview"
    assert provider.default_model() == "protected.gemini-3.1-flash-lite-image"
    assert len(provider.list_models()) == 5
    assert provider.capabilities() == {
        "modalities": ["text", "image"],
        "max_reference_images": 1,
    }


def test_streaming_data_url_is_saved_and_accounted(tmp_path, monkeypatch):
    monkeypatch.setattr(tamu_image, "_load_config", lambda: {})
    monkeypatch.setattr(tamu_image, "find_tamu_api_key", lambda config, endpoint: "key")
    usage = []
    monkeypatch.setattr(tamu_image, "record_tamu_image_usage", lambda **row: usage.append(row))
    response = _StreamResponse(f"data:image/png;base64,{_b64_png()}")

    with patch("requests.post", return_value=response) as post:
        result = tamu_image.TamuPreviewImageGenProvider().generate(
            "A white circle on navy", aspect_ratio="square"
        )

    assert result["success"] is True
    assert result["provider"] == "tamu-preview"
    assert Path(result["image"]).read_bytes() == bytes.fromhex(_PNG_HEX)
    assert usage == [
        {
            "model": "protected.gemini-3.1-flash-lite-image",
            "success": True,
            "input_images": 0,
            "output_images": 1,
            "error_type": "",
        }
    ]
    request = post.call_args
    assert request.args[0].endswith("/openai/chat/completions")
    assert request.kwargs["json"]["stream"] is True
    assert "square image" in request.kwargs["json"]["messages"][0]["content"]


def test_local_edit_image_becomes_data_url(tmp_path, monkeypatch):
    monkeypatch.setattr(tamu_image, "_load_config", lambda: {})
    monkeypatch.setattr(tamu_image, "find_tamu_api_key", lambda config, endpoint: "key")
    monkeypatch.setattr(tamu_image, "record_tamu_image_usage", lambda **row: None)
    source = tmp_path / "source.png"
    source.write_bytes(bytes.fromhex(_PNG_HEX))
    response = _StreamResponse(f"data:image/png;base64,{_b64_png()}")

    with patch("requests.post", return_value=response) as post:
        result = tamu_image.TamuPreviewImageGenProvider().generate(
            "Make the circle larger", image_url=str(source)
        )

    assert result["success"] is True
    assert result["modality"] == "image"
    content = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_unknown_image_model_is_rejected_without_network(monkeypatch):
    monkeypatch.setattr(tamu_image, "record_tamu_image_usage", lambda **row: None)
    with patch("requests.post") as post:
        result = tamu_image.TamuPreviewImageGenProvider().generate(
            "test", model="protected.not-an-image-model"
        )
    assert result["success"] is False
    assert result["error_type"] == "invalid_argument"
    post.assert_not_called()
