"""TAMU AI Chat Preview image generation backend.

TAMUS currently returns generated images as ``data:image/...`` URLs inside
streaming chat-completion content.  The direct OpenAI Images passthrough is not
enabled, so this provider deliberately implements the documented chat route.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)
from hermes_cli.tamu import (
    TAMU_ENDPOINTS,
    TAMU_PREVIEW_IMAGE_MODELS,
    find_tamu_api_key,
    record_tamu_image_usage,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "protected.gemini-3.1-flash-lite-image"
_REQUEST_TIMEOUT = 180
_ASPECT_INSTRUCTIONS = {
    "landscape": "Create a landscape image, approximately 3:2 aspect ratio.",
    "square": "Create a square image, 1:1 aspect ratio.",
    "portrait": "Create a portrait image, approximately 2:3 aspect ratio.",
}


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def _resolve_model(explicit: Any = None) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    config = _load_config()
    image_gen = config.get("image_gen")
    if isinstance(image_gen, dict):
        scoped = image_gen.get("tamu_preview")
        if isinstance(scoped, dict):
            value = scoped.get("model")
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = image_gen.get("model")
        if isinstance(value, str) and value in TAMU_PREVIEW_IMAGE_MODELS:
            return value
    return DEFAULT_MODEL


def _available_model_ids() -> list[str]:
    config = _load_config()
    providers = config.get("providers")
    preview = providers.get("tamu-preview") if isinstance(providers, dict) else None
    saved = preview.get("models") if isinstance(preview, dict) else None
    if isinstance(saved, dict):
        visible = [model for model in TAMU_PREVIEW_IMAGE_MODELS if model in saved]
        if visible:
            return visible
    return list(TAMU_PREVIEW_IMAGE_MODELS)


def _source_to_url(ref: str) -> str:
    ref = ref.strip()
    if ref.lower().startswith(("http://", "https://", "data:")):
        return ref
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    path = Path(ref)
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _content_fragment(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    delta = first.get("delta") if isinstance(first, dict) else None
    message = first.get("message") if isinstance(first, dict) else None
    container = delta if isinstance(delta, dict) else message
    content = container.get("content") if isinstance(container, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if isinstance(part, str):
                fragments.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("image_url") or ""
                if isinstance(text, dict):
                    text = text.get("url") or ""
                if isinstance(text, str):
                    fragments.append(text)
        return "".join(fragments)
    return ""


def _find_image(content: str) -> tuple[str, str, str]:
    match = re.search(
        r"data:image/([a-zA-Z0-9.+-]+);base64,([a-zA-Z0-9+/=]+)", content
    )
    if match:
        subtype = match.group(1).lower()
        extension = "jpg" if subtype in {"jpeg", "pjpeg"} else subtype
        return match.group(2), "", extension
    url_match = re.search(r"https?://[^\s<>\]\)\"']+", content)
    return ("", url_match.group(0), "") if url_match else ("", "", "")


class TamuPreviewImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "tamu-preview"

    @property
    def display_name(self) -> str:
        return "TAMU AI Chat — Preview"

    def is_available(self) -> bool:
        return bool(
            find_tamu_api_key(_load_config(), TAMU_ENDPOINTS["preview"])
        )

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": model, **TAMU_PREVIEW_IMAGE_MODELS[model]}
            for model in _available_model_ids()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "preview",
            "tag": "TAMUS early-access image generation and editing",
            "env_vars": [
                {
                    "key": "HERMES_TAMU_PREVIEW_API_KEY",
                    "prompt": "TAMU AI Chat Preview API key",
                    "url": "https://chat.preview.tamu.ai/",
                }
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 1}

    def _finish(
        self,
        result: Dict[str, Any],
        *,
        model: str,
        input_images: int,
    ) -> Dict[str, Any]:
        record_tamu_image_usage(
            model=model,
            success=bool(result.get("success")),
            input_images=input_images,
            output_images=1 if result.get("success") else 0,
            error_type=str(result.get("error_type") or ""),
        )
        return result

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import requests

        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model = _resolve_model(kwargs.get("model"))
        if not prompt:
            return self._finish(
                error_response(
                    error="Prompt is required and must be a non-empty string",
                    error_type="invalid_argument",
                    provider=self.name,
                    model=model,
                    aspect_ratio=aspect,
                ),
                model=model,
                input_images=0,
            )
        if model not in TAMU_PREVIEW_IMAGE_MODELS:
            return self._finish(
                error_response(
                    error=f"{model!r} is not a documented TAMU Preview image model",
                    error_type="invalid_argument",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                ),
                model=model,
                input_images=0,
            )

        config = _load_config()
        api_key = find_tamu_api_key(config, TAMU_ENDPOINTS["preview"])
        if not api_key:
            return self._finish(
                error_response(
                    error=(
                        "No TAMU Preview API key found. Run "
                        "`hermes tamu setup --access preview`."
                    ),
                    error_type="auth_required",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                ),
                model=model,
                input_images=0,
            )

        references: list[str] = []
        if isinstance(image_url, str) and image_url.strip():
            references.append(image_url.strip())
        references.extend(normalize_reference_images(reference_image_urls) or [])
        references.extend(normalize_reference_images(kwargs.get("reference_images")) or [])
        references = references[:1]
        full_prompt = f"{prompt}\n\n{_ASPECT_INSTRUCTIONS[aspect]}"
        content: Any = full_prompt
        if references:
            try:
                content = [
                    {"type": "text", "text": full_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _source_to_url(references[0])},
                    },
                ]
            except Exception as exc:
                return self._finish(
                    error_response(
                        error=f"Could not read source image: {exc}",
                        error_type="invalid_image",
                        provider=self.name,
                        model=model,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    ),
                    model=model,
                    input_images=len(references),
                )

        try:
            response = requests.post(
                f"{TAMU_ENDPOINTS['preview'].base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "stream": True,
                },
                stream=True,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.Timeout:
            return self._finish(
                error_response(
                    error=f"TAMU Preview image generation timed out ({_REQUEST_TIMEOUT}s)",
                    error_type="timeout",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                ),
                model=model,
                input_images=len(references),
            )
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            detail = ""
            error_response_obj = getattr(exc, "response", None)
            if error_response_obj is not None:
                try:
                    body = error_response_obj.json()
                    detail = str(body.get("error", {}).get("message") or body)[:400]
                except Exception:
                    detail = str(getattr(error_response_obj, "text", ""))[:400]
            message = f"TAMU Preview image generation failed"
            if status:
                message += f" (HTTP {status})"
            if detail:
                message += f": {detail}"
            return self._finish(
                error_response(
                    error=message,
                    error_type="api_error",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                ),
                model=model,
                input_images=len(references),
            )

        fragments: list[str] = []
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    fragments.append(_content_fragment(payload))
        except Exception as exc:
            return self._finish(
                error_response(
                    error=f"Could not read the TAMU Preview image stream: {exc}",
                    error_type="invalid_response",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                ),
                model=model,
                input_images=len(references),
            )

        combined = "".join(fragments)
        b64_data, url, extension = _find_image(combined)
        try:
            if b64_data:
                image_ref = str(
                    save_b64_image(
                        b64_data,
                        prefix=f"tamu_preview_{_resolve_model(model).replace('.', '_')}",
                        extension=extension or "png",
                    )
                )
            elif url:
                image_ref = str(save_url_image(url, prefix="tamu_preview"))
            else:
                clean = combined.strip()
                detail = clean[:400] if clean else "no image content"
                return self._finish(
                    error_response(
                        error=f"TAMU Preview returned {detail}",
                        error_type="empty_response",
                        provider=self.name,
                        model=model,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    ),
                    model=model,
                    input_images=len(references),
                )
        except Exception as exc:
            return self._finish(
                error_response(
                    error=f"Could not cache the generated image: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                ),
                model=model,
                input_images=len(references),
            )

        return self._finish(
            success_response(
                image=image_ref,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
                provider=self.name,
                modality="image" if references else "text",
            ),
            model=model,
            input_images=len(references),
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(TamuPreviewImageGenProvider())
