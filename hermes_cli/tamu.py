"""TAMU AI Chat quick setup and local usage reporting.

This module deliberately builds on Hermes' existing custom-provider and
``session_model_usage`` paths. It does not add telemetry or send usage data to
TAMU, Nous, or any other service.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


@dataclass(frozen=True)
class TamuEndpoint:
    slug: str
    name: str
    access_label: str
    access_description: str
    base_url: str
    key_env: str

    @property
    def host(self) -> str:
        return (urlparse(self.base_url).hostname or "").lower()

    @property
    def provider_key(self) -> str:
        return f"tamu-{self.slug}"


TAMU_ENDPOINTS: dict[str, TamuEndpoint] = {
    "production": TamuEndpoint(
        slug="production",
        name="TAMU AI Chat — Standard access",
        access_label="Standard access",
        access_description="Normal supported TAMU AI Chat service",
        base_url="https://chat-api.tamu.ai/openai",
        key_env="HERMES_TAMU_PRODUCTION_API_KEY",
    ),
    "preview": TamuEndpoint(
        slug="preview",
        name="TAMU AI Chat — Preview access",
        access_label="Preview access",
        access_description="Early-access models and preview-only capabilities",
        base_url="https://chat-api.preview.tamu.ai/openai",
        key_env="HERMES_TAMU_PREVIEW_API_KEY",
    ),
}

# Current TAMUS documentation is authoritative for TAMUS routes.  These
# normalized keys intentionally take precedence over Hermes' provider-neutral
# metadata, which may advertise a different limit for the same model family.
TAMU_DOCUMENTED_CONTEXT_LENGTHS: dict[str, int] = {
    "claude-3-5-haiku": 200_000,
    "claude-opus-4-1": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-sonnet-4": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "gemini-2-5-flash": 1_048_576,
    "gemini-2-5-flash-lite": 1_048_576,
    "gemini-2-5-pro": 1_048_576,
    "gemini-3-1-flash-lite": 1_048_576,
    "gemini-3-5-flash": 1_048_576,
    "gpt-4-1": 1_047_576,
    "gpt-4-1-mini": 1_047_576,
    "gpt-4-1-nano": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-5-1": 400_000,
    "gpt-5-2": 400_000,
    "gpt-5-4": 1_050_000,
    "gpt-5-4-mini": 400_000,
    "gpt-5-4-nano": 400_000,
    "gpt-5-5": 1_050_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "devstral-2-123b": 262_144,
    "gemma-4-31b-it": 262_144,
    "gpt-oss-120b": 131_072,
    "laguna-s-2-1": 1_048_576,
    "gemini-3-1-flash-lite-image": 65_536,
    "gemini-3-1-flash-image": 131_072,
}

TAMU_DOCUMENTED_OUTPUT_LIMITS: dict[str, int] = {
    "claude-3-5-haiku": 8_000,
    "claude-opus-4-1": 32_000,
    "claude-opus-4-5": 64_000,
    "claude-opus-4-6": 128_000,
    "claude-opus-4-7": 128_000,
    "claude-opus-4-8": 128_000,
    "claude-sonnet-4": 64_000,
    "claude-sonnet-4-5": 64_000,
    "claude-sonnet-4-6": 64_000,
    "claude-haiku-4-5": 64_000,
    "gemini-2-5-flash": 65_535,
    "gemini-2-5-flash-lite": 65_536,
    "gemini-2-5-pro": 65_536,
    "gemini-3-1-flash-lite": 65_536,
    "gemini-3-5-flash": 65_536,
    "gpt-4-1": 32_768,
    "gpt-4-1-mini": 32_768,
    "gpt-4-1-nano": 32_768,
    "gpt-4o": 16_384,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "gpt-5-1": 128_000,
    "gpt-5-2": 128_000,
    "gpt-5-4": 128_000,
    "gpt-5-4-mini": 128_000,
    "gpt-5-4-nano": 128_000,
    "gpt-5-5": 128_000,
    "o3": 100_000,
    "o3-mini": 100_000,
    "o4-mini": 100_000,
}

# Preview currently exposes image output through the OpenAI-compatible chat
# completions stream.  Direct /images passthrough is not enabled by TAMUS.
TAMU_PREVIEW_IMAGE_MODELS: dict[str, dict[str, str]] = {
    "protected.gemini-3.1-flash-lite-image": {
        "display": "Gemini 3.1 Flash Lite Image",
        "speed": "fastest",
        "strengths": "Low-latency generation and editing",
        "price": "$0.25 input / $30 output per 1M tokens",
    },
    "protected.gemini-3.1-flash-image": {
        "display": "Gemini 3.1 Flash Image",
        "speed": "fast",
        "strengths": "Higher-quality generation and conversational editing",
        "price": "$0.50 input / $60 output per 1M tokens",
    },
    "protected.gpt-image-1-mini": {
        "display": "GPT Image 1 Mini",
        "speed": "balanced",
        "strengths": "Cost-efficient generation and editing",
        "price": "$2 input / $8 output per 1M tokens",
    },
    "protected.gpt-image-1.5": {
        "display": "GPT Image 1.5",
        "speed": "balanced",
        "strengths": "Strong instruction following and visual adherence",
        "price": "$5 input / $32 output per 1M tokens",
    },
    "protected.gpt-image-2": {
        "display": "GPT Image 2",
        "speed": "fast",
        "strengths": "Highest-fidelity generation and editing",
        "price": "$5 input / $30 output per 1M tokens",
    },
}

_CONTEXT_FIELDS = (
    "context_length",
    "context_window",
    "max_context_length",
    "max_model_len",
    "max_input_tokens",
)


class TamuSetupError(RuntimeError):
    """A user-facing TAMU setup or catalog error."""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _context_from_catalog_item(item: dict[str, Any]) -> int | None:
    """Read common context-window fields without guessing from output limits."""
    candidates: list[dict[str, Any]] = [item]
    for key in ("openai", "metadata", "limits", "capabilities"):
        nested = item.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for candidate in candidates:
        for field in _CONTEXT_FIELDS:
            parsed = _positive_int(candidate.get(field))
            if parsed:
                return parsed
    return None


def _model_id(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    for field in ("id", "model", "name"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_tamu_models(payload: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI-style and TAMU-style model catalog responses."""
    raw_items: Any = payload
    if isinstance(payload, dict):
        for key in ("data", "models", "items"):
            if isinstance(payload.get(key), list):
                raw_items = payload[key]
                break
    if not isinstance(raw_items, list):
        raise TamuSetupError("The TAMU model catalog returned an unexpected format.")

    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_items:
        model_id = _model_id(raw)
        if not model_id or not model_id.startswith("protected."):
            continue
        row: dict[str, Any] = {"id": model_id}
        if isinstance(raw, dict):
            context_length = _context_from_catalog_item(raw)
            if context_length:
                row["context_length"] = context_length
        by_id.setdefault(model_id, row)
    return [by_id[key] for key in sorted(by_id, key=str.casefold)]


def fetch_tamu_models(
    endpoint: TamuEndpoint,
    api_key: str,
    *,
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Fetch the authenticated OpenAI-compatible model catalog."""
    if not api_key:
        raise TamuSetupError(f"No API key is configured for {endpoint.name}.")
    try:
        import requests

        response = requests.get(
            f"{endpoint.base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403}:
            detail = "The TAMU API key was rejected. Check that it belongs to this endpoint."
        elif status:
            detail = f"The TAMU model catalog returned HTTP {status}."
        else:
            detail = f"Could not reach the TAMU model catalog: {exc}"
        raise TamuSetupError(detail) from exc

    models = parse_tamu_models(payload)
    if not models:
        raise TamuSetupError(
            "The catalog was reachable, but it did not expose any protected.* models."
        )
    return models


def _normalized_model_name(model: str) -> str:
    bare = re.sub(r"^protected\.", "", str(model or ""), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "-", bare.lower()).strip("-")


def _documented_limit(model: str, limits: dict[str, int]) -> int | None:
    normalized = _normalized_model_name(model)
    if normalized in limits:
        return limits[normalized]
    # Permit a dated/versioned suffix while preferring the most specific key.
    matches = [
        (len(name), value)
        for name, value in limits.items()
        if normalized.startswith(f"{name}-")
    ]
    return max(matches)[1] if matches else None


def known_context_length(model: str) -> int | None:
    """Return TAMUS' documented context first, then a generic Hermes match."""
    documented = _documented_limit(model, TAMU_DOCUMENTED_CONTEXT_LENGTHS)
    if documented:
        return documented
    try:
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
    except Exception:
        return None
    normalized = _normalized_model_name(model)
    matches: list[tuple[int, int]] = []
    for raw_name, raw_length in DEFAULT_CONTEXT_LENGTHS.items():
        name = _normalized_model_name(raw_name)
        length = _positive_int(raw_length)
        if name and length and name in normalized:
            matches.append((len(name), length))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def known_output_limit(model: str) -> int | None:
    """Return TAMUS' documented maximum output-token count when available."""
    return _documented_limit(model, TAMU_DOCUMENTED_OUTPUT_LIMITS)


def is_tamu_image_model(model: str) -> bool:
    return model in TAMU_PREVIEW_IMAGE_MODELS


def is_embedding_model(model: str) -> bool:
    normalized = _normalized_model_name(model)
    return "embedding" in normalized or normalized.startswith("embed-")


def main_agent_models(models: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude output-only image and embedding models from the agent picker."""
    return [
        item
        for item in models
        if not is_tamu_image_model(_model_id(item))
        and not is_embedding_model(_model_id(item))
    ]


def preview_image_models(models: Iterable[dict[str, Any]]) -> list[str]:
    available = {_model_id(item) for item in models}
    return [model for model in TAMU_PREVIEW_IMAGE_MODELS if model in available]


def enrich_model_contexts(
    models: Iterable[dict[str, Any]],
    *,
    existing: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, dict[str, int]]:
    """Build the configured model map while preserving operator overrides."""
    existing = existing if isinstance(existing, dict) else {}
    configured: dict[str, dict[str, int]] = {}
    for item in models:
        model_id = _model_id(item)
        if not model_id:
            continue
        context_length = None
        old = existing.get(model_id)
        if isinstance(old, dict):
            context_length = _positive_int(old.get("context_length"))
        if context_length is None and enabled:
            context_length = _context_from_catalog_item(item)
        if context_length is None and enabled:
            context_length = known_context_length(model_id)
        configured[model_id] = (
            {"context_length": context_length} if context_length else {}
        )
    return configured


def _matching_custom_entries(config: dict[str, Any], endpoint: TamuEndpoint):
    """Yield legacy custom-provider entries for this exact TAMU route."""
    entries = config.get("custom_providers")
    if not isinstance(entries, list):
        return
    target = endpoint.base_url.rstrip("/").lower()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        base_url = str(entry.get("base_url") or entry.get("api") or "")
        if base_url.rstrip("/").lower() == target:
            yield entry


def _existing_provider_entry(
    config: dict[str, Any], endpoint: TamuEndpoint
) -> dict[str, Any]:
    providers = config.get("providers")
    if isinstance(providers, dict):
        entry = providers.get(endpoint.provider_key)
        if isinstance(entry, dict):
            return entry
    return next(_matching_custom_entries(config, endpoint), {})


def find_tamu_api_key(config: dict[str, Any], endpoint: TamuEndpoint) -> str:
    """Resolve a saved TAMU key without displaying it or persisting plaintext."""
    from hermes_cli.config import get_env_value

    direct = get_env_value(endpoint.key_env) or ""
    if direct:
        return direct

    candidates: list[dict[str, Any]] = []
    providers = config.get("providers")
    if isinstance(providers, dict):
        entry = providers.get(endpoint.provider_key)
        if isinstance(entry, dict):
            candidates.append(entry)
    candidates.extend(_matching_custom_entries(config, endpoint))

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model_url = str(model_cfg.get("base_url") or "").rstrip("/").lower()
        if model_url == endpoint.base_url.rstrip("/").lower():
            candidates.append(model_cfg)

    for candidate in candidates:
        key_env = str(candidate.get("key_env") or "").strip()
        if key_env:
            value = get_env_value(key_env) or ""
            if value:
                return value
        api_key = candidate.get("api_key")
        if isinstance(api_key, str) and api_key and not api_key.startswith("${"):
            return api_key
    return ""


def persist_tamu_setup(
    endpoint: TamuEndpoint,
    api_key: str,
    models: list[dict[str, Any]],
    selected_model: str,
    *,
    max_output_tokens: int = 32768,
    context_overrides: bool = True,
    image_model: str | None = None,
    refresh_context_limits: bool = False,
) -> dict[str, Any]:
    """Persist an additive TAMU provider and select it as the default."""
    from hermes_cli.auth import deactivate_provider
    from hermes_cli.config import load_config, save_config, save_env_value

    if selected_model not in {_model_id(item) for item in models}:
        raise TamuSetupError(
            f"Model {selected_model!r} is not present in the current TAMU catalog."
        )
    requested_output_tokens = _positive_int(max_output_tokens) or 32768
    documented_output_limit = known_output_limit(selected_model)
    max_output_tokens = min(
        requested_output_tokens,
        documented_output_limit or requested_output_tokens,
    )
    if image_model:
        if endpoint.slug != "preview":
            raise TamuSetupError(
                "Preview image models can only be configured with Preview access."
            )
        if image_model not in preview_image_models(models):
            raise TamuSetupError(
                f"Image model {image_model!r} is not in the current Preview catalog."
            )

    config = load_config()
    prior_entry = _existing_provider_entry(config, endpoint)
    old_models = prior_entry.get("models") if isinstance(prior_entry, dict) else {}
    configured_models = enrich_model_contexts(
        models,
        existing=(
            old_models
            if isinstance(old_models, dict) and not refresh_context_limits
            else {}
        ),
        enabled=context_overrides,
    )

    save_env_value(endpoint.key_env, api_key)

    providers = config.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        config["providers"] = providers
    providers[endpoint.provider_key] = {
        "name": endpoint.name,
        "api": endpoint.base_url,
        "key_env": endpoint.key_env,
        "transport": "chat_completions",
        "default_model": selected_model,
        "discover_models": True,
        "models": configured_models,
    }

    # Keep a matching legacy entry functional and visually deduplicated. The
    # setup is additive: no existing provider or unrelated option is removed.
    for legacy in _matching_custom_entries(config, endpoint):
        legacy["name"] = endpoint.name
        legacy["key_env"] = endpoint.key_env
        legacy.pop("api_key", None)
        legacy["api_mode"] = "chat_completions"
        legacy["model"] = selected_model
        legacy["discover_models"] = True
        legacy["models"] = configured_models

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {"default": model_cfg} if model_cfg else {}
        config["model"] = model_cfg
    model_cfg["provider"] = f"custom:{endpoint.provider_key}"
    model_cfg["default"] = selected_model
    model_cfg["max_tokens"] = max_output_tokens
    model_cfg.pop("base_url", None)
    model_cfg.pop("api_key", None)
    model_cfg.pop("api_mode", None)
    selected_meta = configured_models.get(selected_model) or {}
    selected_context = _positive_int(selected_meta.get("context_length"))
    if selected_context:
        model_cfg["context_length"] = selected_context
    else:
        model_cfg.pop("context_length", None)

    auxiliary = config.get("auxiliary")
    if not isinstance(auxiliary, dict):
        auxiliary = {}
        config["auxiliary"] = auxiliary
    stream_only = auxiliary.get("stream_only_base_urls")
    if not isinstance(stream_only, list):
        stream_only = []
    if endpoint.host not in {str(value).lower() for value in stream_only}:
        stream_only.append(endpoint.host)
    auxiliary["stream_only_base_urls"] = stream_only

    if image_model:
        image_gen = config.get("image_gen")
        if not isinstance(image_gen, dict):
            image_gen = {}
            config["image_gen"] = image_gen
        image_gen["provider"] = "tamu-preview"
        image_gen["model"] = image_model
        image_gen["use_gateway"] = False
        scoped = image_gen.get("tamu_preview")
        if not isinstance(scoped, dict):
            scoped = {}
            image_gen["tamu_preview"] = scoped
        scoped["model"] = image_model

    save_config(config)
    deactivate_provider()
    return {
        "environment": endpoint.slug,
        "endpoint": endpoint.base_url,
        "provider": f"custom:{endpoint.provider_key}",
        "model": selected_model,
        "models_discovered": len(models),
        "models_with_context": sum(
            1 for metadata in configured_models.values() if metadata.get("context_length")
        ),
        "context_length": selected_context,
        "max_output_tokens": max_output_tokens,
        "requested_output_tokens": requested_output_tokens,
        "output_limit_clamped": max_output_tokens != requested_output_tokens,
        "image_model": image_model,
        "key_env": endpoint.key_env,
        "stream_only": True,
    }


def _prompt_choice(title: str, choices: list[str], default: int = 0) -> int | None:
    try:
        from hermes_cli.curses_ui import curses_radiolist

        selected = curses_radiolist(
            title,
            choices,
            selected=default,
            cancel_returns=-1,
            searchable=len(choices) > 12,
        )
        print()
        return selected if selected >= 0 else None
    except Exception:
        print(title)
        for index, choice in enumerate(choices, 1):
            marker = "→" if index - 1 == default else " "
            print(f"  {marker} {index}. {choice}")
        print()
        try:
            value = input(f"Choice [1-{len(choices)}] ({default + 1}): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return None
        if not value:
            return default
        try:
            selected = int(value) - 1
        except ValueError:
            return None
        return selected if 0 <= selected < len(choices) else None


def _prompt_yes_no(prompt: str, *, default: bool = True) -> bool | None:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        value = input(f"{prompt} {suffix}: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return None
    if not value:
        return default
    if value in {"y", "yes"}:
        return True
    if value in {"n", "no"}:
        return False
    return None


def _choose_endpoint(
    environment: str | None = None,
    access: str | None = None,
) -> TamuEndpoint | None:
    if access:
        environment = "production" if access == "standard" else "preview"
    if environment:
        return TAMU_ENDPOINTS[environment]
    choices = [
        "Standard access — normal supported service",
        "Preview access — early-access models and capabilities",
    ]
    selected = _prompt_choice("Choose your TAMU AI Chat access:", choices)
    if selected is None:
        return None
    return TAMU_ENDPOINTS[("production", "preview")[selected]]


def _active_tamu_environment(config: dict[str, Any]) -> str | None:
    model_cfg = config.get("model")
    provider = ""
    base_url = ""
    if isinstance(model_cfg, dict):
        provider = str(model_cfg.get("provider") or "").lower()
        base_url = str(model_cfg.get("base_url") or "").rstrip("/").lower()
    for slug, endpoint in TAMU_ENDPOINTS.items():
        if provider in {
            f"custom:{endpoint.provider_key}",
            endpoint.provider_key,
        } or base_url == endpoint.base_url.rstrip("/").lower():
            return slug
    return None


def _prompt_api_key(config: dict[str, Any], endpoint: TamuEndpoint) -> str | None:
    from hermes_cli.secret_prompt import masked_secret_prompt

    existing = find_tamu_api_key(config, endpoint)
    if existing:
        print(f"  API key: already saved as {endpoint.key_env}")
        try:
            choice = input("  Keep it? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return None
        if choice in {"", "y", "yes"}:
            return existing
    try:
        api_key = masked_secret_prompt("  Paste TAMU API key: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return None
    return api_key or None


def run_tamu_setup(args) -> int:
    from hermes_cli.config import load_config

    print()
    print("TAMU AI Chat quick setup")
    print("=" * 50)
    print("Your key is stored locally in the Hermes .env file.")
    print("Usage reporting stays in the local Hermes state database.")
    print()

    endpoint = _choose_endpoint(
        getattr(args, "environment", None),
        getattr(args, "access", None),
    )
    if endpoint is None:
        print("Setup cancelled.")
        return 1

    print(f"  Access:   {endpoint.access_label}")
    print(f"  Service:  {endpoint.access_description}")
    print(f"  URL:      {endpoint.base_url}")
    print()
    api_key = _prompt_api_key(load_config(), endpoint)
    if not api_key:
        print("Setup cancelled; no key was saved.")
        return 1

    print("  Discovering protected.* models...")
    try:
        models = fetch_tamu_models(endpoint, api_key)
    except TamuSetupError as exc:
        print(f"  Error: {exc}")
        return 1
    print(f"  Found {len(models)} model(s).")
    print()

    requested_model = str(getattr(args, "model", "") or "").strip()
    agent_models = main_agent_models(models)
    model_ids = [item["id"] for item in agent_models]
    if not model_ids:
        print("  Error: the catalog did not expose a text model usable as an agent.")
        return 1
    if requested_model:
        if is_tamu_image_model(requested_model):
            print(
                "  Error: image-output models cannot be the Hermes agent model. "
                "Use --image-model for Preview image generation."
            )
            return 1
        if is_embedding_model(requested_model) or requested_model not in model_ids:
            print(f"  Error: {requested_model!r} is not in the current catalog.")
            return 1
        selected_model = requested_model
    else:
        selected = _prompt_choice("Choose the default TAMU model:", model_ids)
        if selected is None:
            print("Setup cancelled; no configuration was changed.")
            return 1
        selected_model = model_ids[selected]

    requested_image_model = str(
        getattr(args, "image_model", "") or ""
    ).strip()
    image_model: str | None = None
    available_images = preview_image_models(models)
    if endpoint.slug != "preview" and requested_image_model:
        print("  Error: --image-model requires --access preview.")
        return 1
    if endpoint.slug == "preview" and requested_image_model:
        if requested_image_model not in available_images:
            print(
                f"  Error: {requested_image_model!r} is not an available "
                "documented Preview image model."
            )
            return 1
        image_model = requested_image_model
    elif (
        endpoint.slug == "preview"
        and available_images
        and not getattr(args, "no_image_setup", False)
    ):
        print()
        print(
            f"  Preview image generation is available ({len(available_images)} model(s))."
        )
        configure_images = _prompt_yes_no("  Set up image generation?", default=True)
        if configure_images:
            choices = [
                f"{TAMU_PREVIEW_IMAGE_MODELS[model]['display']} — {model}"
                for model in available_images
            ]
            selected = _prompt_choice(
                "Choose the Preview image model:", choices, default=0
            )
            if selected is None:
                print("Setup cancelled; no configuration was changed.")
                return 1
            image_model = available_images[selected]

    try:
        result = persist_tamu_setup(
            endpoint,
            api_key,
            models,
            selected_model,
            max_output_tokens=getattr(args, "max_output_tokens", 32768),
            context_overrides=not getattr(args, "no_context_overrides", False),
            image_model=image_model,
            refresh_context_limits=getattr(args, "refresh_context_limits", False),
        )
    except TamuSetupError as exc:
        print(f"  Error: {exc}")
        return 1

    print()
    print("✓ TAMU AI Chat is ready in Hermes")
    print(f"  Default model:       {result['model']}")
    if result["context_length"]:
        print(f"  Context window:      {result['context_length']:,} tokens")
    else:
        print("  Context window:      provider/runtime auto-detect")
    print(f"  Response-token cap:  {result['max_output_tokens']:,}")
    if result["output_limit_clamped"]:
        print(
            f"  Note: requested {result['requested_output_tokens']:,}; clamped "
            "to the TAMUS documented model limit."
        )
    print(f"  Models saved:        {result['models_discovered']}")
    if result["image_model"]:
        print(f"  Preview image model: {result['image_model']}")
    print(f"  Stream compatibility: enabled for {endpoint.host}")
    print()
    print("Next:")
    print("  hermes tamu status --check")
    print("  hermes tamu models")
    print("  hermes tamu usage --days 30")
    print("  hermes")
    return 0


def tamu_status(*, live_check: bool = False) -> dict[str, Any]:
    from hermes_cli.config import load_config

    config = load_config()
    active = _active_tamu_environment(config)
    auxiliary = config.get("auxiliary")
    stream_only_values = (
        auxiliary.get("stream_only_base_urls")
        if isinstance(auxiliary, dict)
        else []
    )
    if not isinstance(stream_only_values, list):
        stream_only_values = []
    stream_only_hosts = {str(value).lower() for value in stream_only_values}
    image_gen = config.get("image_gen")
    image_provider = (
        str(image_gen.get("provider") or "")
        if isinstance(image_gen, dict)
        else ""
    )
    image_model = (
        str(image_gen.get("model") or "")
        if isinstance(image_gen, dict)
        else ""
    )
    rows: list[dict[str, Any]] = []
    for slug, endpoint in TAMU_ENDPOINTS.items():
        entry = _existing_provider_entry(config, endpoint)
        configured = bool(entry)
        models = entry.get("models") if isinstance(entry, dict) else {}
        row: dict[str, Any] = {
            "environment": slug,
            "access": endpoint.access_label,
            "active": active == slug,
            "configured": configured,
            "endpoint": endpoint.base_url,
            "key_saved": bool(find_tamu_api_key(config, endpoint)),
            "default_model": (
                str(entry.get("default_model") or entry.get("model") or "")
                if isinstance(entry, dict)
                else ""
            ),
            "saved_models": len(models) if isinstance(models, dict) else 0,
            "stream_compatibility": endpoint.host in stream_only_hosts,
        }
        if slug == "preview":
            row["image_provider_active"] = image_provider == "tamu-preview"
            row["image_model"] = image_model if row["image_provider_active"] else ""
        if live_check and row["key_saved"]:
            try:
                live_models = fetch_tamu_models(
                    endpoint, find_tamu_api_key(config, endpoint)
                )
                row["live_ok"] = True
                row["live_models"] = len(live_models)
                row["default_model_visible"] = row["default_model"] in {
                    item["id"] for item in live_models
                }
                if slug == "preview" and row.get("image_model"):
                    row["image_model_visible"] = row["image_model"] in {
                        item["id"] for item in live_models
                    }
            except TamuSetupError as exc:
                row["live_ok"] = False
                row["live_error"] = str(exc)
        rows.append(row)
    return {"active_environment": active, "endpoints": rows}


def _format_status(report: dict[str, Any]) -> str:
    lines = ["TAMU AI Chat status", "=" * 50]
    for row in report["endpoints"]:
        marker = " (active)" if row["active"] else ""
        lines.append(f"\n{row['access']}{marker}")
        lines.append(f"  Configured:          {'yes' if row['configured'] else 'no'}")
        lines.append(f"  Endpoint:            {row['endpoint']}")
        lines.append(f"  API key saved:       {'yes' if row['key_saved'] else 'no'}")
        lines.append(f"  Default model:       {row['default_model'] or '-'}")
        lines.append(f"  Saved models:        {row['saved_models']}")
        lines.append(
            "  Stream compatibility: "
            + ("enabled" if row["stream_compatibility"] else "not configured")
        )
        if row["environment"] == "preview":
            lines.append(
                "  Preview images:      "
                + (
                    row.get("image_model", "")
                    if row.get("image_provider_active")
                    else "not configured"
                )
            )
        if "live_ok" in row:
            if row["live_ok"]:
                lines.append(f"  Live check:          ok ({row['live_models']} models)")
                lines.append(
                    "  Default visible:     "
                    + ("yes" if row["default_model_visible"] else "no")
                )
                if "image_model_visible" in row:
                    lines.append(
                        "  Image model visible: "
                        + ("yes" if row["image_model_visible"] else "no")
                    )
            else:
                lines.append(f"  Live check:          failed — {row['live_error']}")
    return "\n".join(lines)


def tamu_usage_db_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "tamu_usage.db"


def record_tamu_image_usage(
    *,
    model: str,
    success: bool,
    input_images: int = 0,
    output_images: int = 0,
    error_type: str = "",
    db_path: Path | None = None,
) -> None:
    """Record a prompt-free, credential-free local Preview image call."""
    path = Path(db_path) if db_path is not None else tamu_usage_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_error_type = re.sub(r"[^a-zA-Z0-9_.-]+", "_", error_type)[:80]
    try:
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_usage (
                    created_at REAL NOT NULL,
                    environment TEXT NOT NULL,
                    model TEXT NOT NULL,
                    api_calls INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    input_images INTEGER NOT NULL,
                    output_images INTEGER NOT NULL,
                    error_type TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO image_usage VALUES (?, 'preview', ?, 1, ?, ?, ?, ?)",
                (
                    time.time(),
                    model,
                    int(bool(success)),
                    max(int(input_images), 0),
                    max(int(output_images), 0),
                    safe_error_type,
                ),
            )
    except Exception:
        # Accounting must never prevent the requested image operation.
        return


def query_tamu_image_usage(
    *, days: int = 30, db_path: Path | None = None
) -> list[dict[str, Any]]:
    """Aggregate local Preview image calls without reading prompts or files."""
    path = Path(db_path) if db_path is not None else tamu_usage_db_path()
    if not path.exists():
        return []
    cutoff = time.time() - max(int(days), 1) * 86400
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT environment, model, 'image_generate' AS task,
                       SUM(api_calls) AS api_calls,
                       0 AS input_tokens, 0 AS output_tokens,
                       0 AS cache_read_tokens, 0 AS cache_write_tokens,
                       0 AS reasoning_tokens, 0 AS sessions,
                       SUM(api_calls) AS image_calls,
                       SUM(output_images) AS images_generated,
                       SUM(input_images) AS input_images,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed_calls,
                       MAX(created_at) AS last_seen
                FROM image_usage
                WHERE created_at >= ?
                GROUP BY environment, model
                ORDER BY SUM(api_calls) DESC, model COLLATE NOCASE
                """,
                (cutoff,),
            ).fetchall()
    except Exception:
        return []
    result = [dict(row) for row in rows]
    for row in result:
        row["total_tokens"] = 0
    return result


def merge_tamu_image_usage(
    report: dict[str, Any], image_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Merge non-token image call counts into the regular local report."""
    report["rows"].extend(image_rows)
    report["rows"].sort(
        key=lambda row: (
            -int(row.get("total_tokens") or 0),
            -int(row.get("api_calls") or 0),
            str(row.get("model") or "").casefold(),
        )
    )
    totals = report["totals"]
    totals["image_calls"] = sum(
        int(row.get("image_calls") or 0) for row in image_rows
    )
    totals["images_generated"] = sum(
        int(row.get("images_generated") or 0) for row in image_rows
    )
    totals["api_calls"] += totals["image_calls"]
    return report


def query_tamu_usage(conn, *, days: int = 30, source: str | None = None) -> dict:
    """Aggregate local usage by endpoint, model, and auxiliary task."""
    days = max(int(days), 1)
    cutoff = time.time() - days * 86400
    params: list[Any] = [cutoff]
    source_sql = ""
    if source:
        source_sql = " AND s.source = ?"
        params.append(source)
    host_checks = " OR ".join(
        "LOWER(COALESCE(NULLIF(u.billing_base_url, ''), s.billing_base_url, '')) LIKE ?"
        for _ in TAMU_ENDPOINTS.values()
    )
    params.extend(f"%{endpoint.host}%" for endpoint in TAMU_ENDPOINTS.values())
    sql = f"""
        SELECT
            CASE
                WHEN LOWER(COALESCE(NULLIF(u.billing_base_url, ''), s.billing_base_url, ''))
                     LIKE '%chat-api.preview.tamu.ai%' THEN 'preview'
                ELSE 'production'
            END AS environment,
            u.model AS model,
            CASE WHEN u.task = '' THEN 'main' ELSE u.task END AS task,
            SUM(u.api_call_count) AS api_calls,
            SUM(u.input_tokens) AS input_tokens,
            SUM(u.output_tokens) AS output_tokens,
            SUM(u.cache_read_tokens) AS cache_read_tokens,
            SUM(u.cache_write_tokens) AS cache_write_tokens,
            SUM(u.reasoning_tokens) AS reasoning_tokens,
            COUNT(DISTINCT u.session_id) AS sessions,
            MAX(u.last_seen) AS last_seen
        FROM session_model_usage u
        JOIN sessions s ON s.id = u.session_id
        WHERE COALESCE(u.last_seen, s.started_at) >= ?
          {source_sql}
          AND ({host_checks})
        GROUP BY environment, u.model, u.task
        ORDER BY (SUM(u.input_tokens) + SUM(u.output_tokens)
                  + SUM(u.cache_read_tokens) + SUM(u.cache_write_tokens)) DESC,
                 u.model COLLATE NOCASE, u.task COLLATE NOCASE
    """
    try:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    except Exception as exc:
        if "no such table" in str(exc).lower():
            rows = []
        else:
            raise
    for row in rows:
        row["total_tokens"] = sum(
            int(row.get(field) or 0)
            for field in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            )
        )
    totals = {
        "sessions": 0,
        "api_calls": sum(int(row.get("api_calls") or 0) for row in rows),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "cache_read_tokens": sum(
            int(row.get("cache_read_tokens") or 0) for row in rows
        ),
        "cache_write_tokens": sum(
            int(row.get("cache_write_tokens") or 0) for row in rows
        ),
        "reasoning_tokens": sum(
            int(row.get("reasoning_tokens") or 0) for row in rows
        ),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
    }
    # SUM(DISTINCT per grouped row cannot produce a cross-group session total.
    session_sql = f"""
        SELECT COUNT(DISTINCT u.session_id)
        FROM session_model_usage u
        JOIN sessions s ON s.id = u.session_id
        WHERE COALESCE(u.last_seen, s.started_at) >= ?
          {source_sql}
          AND ({host_checks})
    """
    try:
        totals["sessions"] = int(
            conn.execute(session_sql, params).fetchone()[0] or 0
        )
    except Exception as exc:
        if "no such table" not in str(exc).lower():
            raise
    return {
        "days": days,
        "source_filter": source,
        "local_only": True,
        "rows": rows,
        "totals": totals,
    }


def _format_usage(report: dict[str, Any]) -> str:
    lines = [
        f"TAMU AI Chat usage — last {report['days']} day(s)",
        "Local Hermes accounting only; nothing is transmitted by this report.",
        "",
    ]
    rows = report["rows"]
    if not rows:
        lines.append("No TAMU model usage was recorded in this period.")
        return "\n".join(lines)
    lines.append(
        f"{'Environment':<11} {'Model':<38} {'Task':<20} "
        f"{'Calls':>7} {'Tokens':>13} {'Images':>8}"
    )
    lines.append("-" * 105)
    for row in rows:
        model = str(row["model"])
        if len(model) > 38:
            model = model[:35] + "..."
        task = str(row["task"])
        if len(task) > 20:
            task = task[:17] + "..."
        lines.append(
            f"{row['environment']:<11} {model:<38} {task:<20} "
            f"{int(row['api_calls'] or 0):>7,} {int(row['total_tokens'] or 0):>13,} "
            f"{int(row.get('images_generated') or 0):>8,}"
        )
    totals = report["totals"]
    lines.extend(
        [
            "",
            f"Sessions: {totals['sessions']:,}  API calls: {totals['api_calls']:,}  "
            f"Tokens: {totals['total_tokens']:,}",
            f"Input: {totals['input_tokens']:,}  Output: {totals['output_tokens']:,}  "
            f"Cache read: {totals['cache_read_tokens']:,}  "
            f"Cache write: {totals['cache_write_tokens']:,}",
            f"Image calls: {int(totals.get('image_calls') or 0):,}  "
            f"Images generated: {int(totals.get('images_generated') or 0):,}",
            *(
                ["Image calls are not source-attributed and are omitted when --source is used."]
                if report.get("image_usage_omitted_for_source_filter")
                else []
            ),
            "Dollar cost is not estimated because TAMU access is quota-based and "
            "Hermes has no authoritative TAMU price table.",
        ]
    )
    return "\n".join(lines)


def cmd_tamu(args) -> int:
    command = getattr(args, "tamu_command", None)
    if command in {None, ""}:
        print(
            "usage: hermes tamu <setup|models|status|usage>\n\n"
            "  setup    Configure Standard or Preview access\n"
            "  models   List protected.* models\n"
            "  status   Check saved configuration\n"
            "  usage    Report local per-model usage"
        )
        return 0
    if command == "setup":
        return run_tamu_setup(args)
    if command == "status":
        report = tamu_status(live_check=bool(getattr(args, "check", False)))
        print(
            json.dumps(report, indent=2)
            if getattr(args, "json", False)
            else _format_status(report)
        )
        return 0
    if command == "models":
        from hermes_cli.config import load_config

        config = load_config()
        environment = getattr(args, "environment", None) or _active_tamu_environment(config)
        if not environment:
            print("No active TAMU endpoint. Run `hermes tamu setup` first.")
            return 1
        endpoint = TAMU_ENDPOINTS[environment]
        try:
            models = fetch_tamu_models(endpoint, find_tamu_api_key(config, endpoint))
        except TamuSetupError as exc:
            print(f"Error: {exc}")
            return 1
        if getattr(args, "json", False):
            print(json.dumps({"environment": environment, "models": models}, indent=2))
        else:
            print(f"{endpoint.name}: {len(models)} protected model(s)\n")
            for item in models:
                context = _context_from_catalog_item(item) or known_context_length(item["id"])
                suffix = f"  ({context:,} context)" if context else ""
                capability = ""
                if is_tamu_image_model(item["id"]):
                    capability = "  [image output]"
                elif is_embedding_model(item["id"]):
                    capability = "  [embedding]"
                print(f"  {item['id']}{suffix}{capability}")
        return 0
    if command == "usage":
        from hermes_state import DEFAULT_DB_PATH, SessionDB

        # This report is intentionally read-only and local. Avoid taking a
        # write lock or requiring write access merely to inspect usage. A new
        # profile has no state.db until its first session, so return an empty
        # report instead of asking SQLite to open a missing file read-only.
        days = max(int(getattr(args, "days", 30)), 1)
        source = getattr(args, "source", None)
        if not DEFAULT_DB_PATH.exists():
            report = {
                "days": days,
                "source_filter": source,
                "local_only": True,
                "rows": [],
                "totals": {
                    "sessions": 0,
                    "api_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                },
            }
        else:
            db = SessionDB(read_only=True)
            try:
                report = query_tamu_usage(db._conn, days=days, source=source)
            finally:
                db.close()
        if source:
            report["image_usage_omitted_for_source_filter"] = True
        else:
            merge_tamu_image_usage(
                report,
                query_tamu_image_usage(days=days),
            )
        print(
            json.dumps(report, indent=2)
            if getattr(args, "json", False)
            else _format_usage(report)
        )
        return 0
    return 1
