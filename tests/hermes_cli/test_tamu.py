from __future__ import annotations

import argparse
import copy
import sqlite3
import time
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands.tamu import build_tamu_parser
from hermes_cli.subcommands.setup import build_setup_parser
from hermes_cli.tamu import (
    TAMU_ENDPOINTS,
    TamuSetupError,
    enrich_model_contexts,
    fetch_tamu_models,
    parse_tamu_models,
    persist_tamu_setup,
    query_tamu_usage,
)


def test_tamu_subcommands_parse_and_dispatch():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    handler = lambda args: args.tamu_command  # noqa: E731
    build_tamu_parser(subparsers, cmd_tamu=handler)

    setup = parser.parse_args(
        ["tamu", "setup", "--environment", "preview", "--model", "protected.test"]
    )
    assert setup.func is handler
    assert setup.environment == "preview"
    assert setup.model == "protected.test"

    usage = parser.parse_args(["tamu", "usage", "--days", "7", "--json"])
    assert usage.days == 7
    assert usage.json is True


@pytest.mark.parametrize("argv", [["setup", "tamu"], ["setup", "--tamu"]])
def test_main_setup_parser_accepts_tamu_quick_path(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    handler = lambda args: args  # noqa: E731
    build_setup_parser(subparsers, cmd_setup=handler)

    parsed = parser.parse_args(argv)
    assert parsed.func is handler
    assert parsed.section == ("tamu" if argv[-1] == "tamu" else None)
    assert parsed.tamu is (argv[-1] == "--tamu")


def test_parse_tamu_models_keeps_only_protected_and_context_metadata():
    result = parse_tamu_models(
        {
            "data": [
                {"id": "unprotected.internal"},
                {"id": "protected.beta", "context_window": 200_000},
                {"id": "protected.alpha", "openai": {"context_length": 1_000_000}},
                {"id": "protected.alpha"},
            ]
        }
    )
    assert result == [
        {"id": "protected.alpha", "context_length": 1_000_000},
        {"id": "protected.beta", "context_length": 200_000},
    ]


def test_parse_tamu_models_rejects_an_unexpected_payload():
    with pytest.raises(TamuSetupError, match="unexpected format"):
        parse_tamu_models({"data": {"id": "protected.test"}})


def test_fetch_tamu_models_uses_openai_models_route(monkeypatch):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "protected.test"}]}

    def fake_get(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return Response()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    result = fetch_tamu_models(TAMU_ENDPOINTS["preview"], "secret", timeout=3)

    assert result == [{"id": "protected.test"}]
    assert calls == [
        (
            "https://chat-api.preview.tamu.ai/openai/models",
            {"Authorization": "Bearer secret"},
            3,
        )
    ]


def test_enrich_contexts_preserves_existing_operator_value(monkeypatch):
    monkeypatch.setattr("hermes_cli.tamu.known_context_length", lambda model: 400_000)
    result = enrich_model_contexts(
        [{"id": "protected.alpha"}, {"id": "protected.beta"}],
        existing={"protected.alpha": {"context_length": 777_777}},
    )
    assert result == {
        "protected.alpha": {"context_length": 777_777},
        "protected.beta": {"context_length": 400_000},
    }


def test_persist_tamu_setup_is_additive_and_uses_env_reference(monkeypatch):
    endpoint = TAMU_ENDPOINTS["preview"]
    initial = {
        "model": {"default": "old-model", "provider": "openrouter"},
        "providers": {"other": {"api": "https://example.test/v1"}},
        "custom_providers": [
            {
                "name": "Old preview label",
                "base_url": endpoint.base_url,
                "api_key": "plaintext-that-must-not-survive",
                "model": "protected.old",
            }
        ],
        "auxiliary": {"stream_only_base_urls": ["other.example"]},
    }
    saved = {}
    env_writes = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: copy.deepcopy(initial)
    )
    monkeypatch.setattr(
        "hermes_cli.config.save_config", lambda config: saved.update(copy.deepcopy(config))
    )
    monkeypatch.setattr(
        "hermes_cli.config.save_env_value",
        lambda key, value: env_writes.append((key, value)),
    )
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.tamu.known_context_length", lambda model: 200_000)

    report = persist_tamu_setup(
        endpoint,
        "new-secret",
        [{"id": "protected.alpha"}, {"id": "protected.beta"}],
        "protected.alpha",
    )

    assert env_writes == [(endpoint.key_env, "new-secret")]
    assert saved["providers"]["other"] == initial["providers"]["other"]
    tamu = saved["providers"][endpoint.provider_key]
    assert tamu["key_env"] == endpoint.key_env
    assert "api_key" not in tamu
    assert tamu["models"]["protected.alpha"]["context_length"] == 200_000
    assert saved["model"] == {
        "default": "protected.alpha",
        "provider": "custom:tamu-preview",
        "max_tokens": 32768,
        "context_length": 200_000,
    }
    legacy = saved["custom_providers"][0]
    assert legacy["key_env"] == endpoint.key_env
    assert "api_key" not in legacy
    assert saved["auxiliary"]["stream_only_base_urls"] == [
        "other.example",
        "chat-api.preview.tamu.ai",
    ]
    assert report["models_discovered"] == 2


def _usage_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            started_at REAL,
            billing_base_url TEXT
        );
        CREATE TABLE session_model_usage (
            session_id TEXT,
            model TEXT,
            billing_base_url TEXT,
            task TEXT,
            api_call_count INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_write_tokens INTEGER,
            reasoning_tokens INTEGER,
            last_seen REAL
        );
        """
    )
    return conn


def test_query_tamu_usage_groups_each_model_and_agent_task():
    conn = _usage_connection()
    now = time.time()
    conn.executemany(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        [
            ("a", "cli", now, "https://chat-api.preview.tamu.ai/openai"),
            ("b", "cli", now, "https://unrelated.example/v1"),
        ],
    )
    conn.executemany(
        "INSERT INTO session_model_usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("a", "protected.main", "", "", 2, 100, 20, 5, 0, 3, now),
            (
                "a",
                "protected.reference",
                "https://chat-api.preview.tamu.ai/openai",
                "moa_reference",
                1,
                50,
                10,
                0,
                0,
                0,
                now,
            ),
            ("b", "protected.not-tamu", "", "", 9, 900, 90, 0, 0, 0, now),
        ],
    )

    report = query_tamu_usage(conn, days=1, source="cli")

    assert [(row["model"], row["task"]) for row in report["rows"]] == [
        ("protected.main", "main"),
        ("protected.reference", "moa_reference"),
    ]
    assert report["totals"] == {
        "sessions": 1,
        "api_calls": 3,
        "input_tokens": 150,
        "output_tokens": 30,
        "cache_read_tokens": 5,
        "cache_write_tokens": 0,
        "reasoning_tokens": 3,
        "total_tokens": 185,
    }


def test_usage_command_is_empty_on_a_new_profile(monkeypatch, tmp_path, capsys):
    from hermes_cli.tamu import cmd_tamu

    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", tmp_path / "state.db")

    result = cmd_tamu(
        SimpleNamespace(
            tamu_command="usage",
            days=30,
            source=None,
            json=False,
        )
    )

    assert result == 0
    assert "No TAMU model usage was recorded" in capsys.readouterr().out
