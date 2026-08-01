"""``hermes tamu`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_tamu_parser(subparsers, *, cmd_tamu: Callable) -> None:
    """Attach TAMU AI Chat setup and local usage commands."""
    tamu_parser = subparsers.add_parser(
        "tamu",
        help="Set up and inspect TAMU AI Chat",
        description=(
            "Configure TAMU AI Chat as an OpenAI-compatible Hermes provider, "
            "list its protected.* models, and inspect local per-model usage."
        ),
    )
    tamu_subparsers = tamu_parser.add_subparsers(dest="tamu_command")

    setup_parser = tamu_subparsers.add_parser(
        "setup",
        help="Run the TAMU quick setup wizard",
        description=(
            "Choose Standard or Preview access, securely save the API key, "
            "discover models, and make one the Hermes default. Preview can "
            "also configure a documented TAMUS image model."
        ),
    )
    access_group = setup_parser.add_mutually_exclusive_group()
    access_group.add_argument(
        "--access",
        choices=["standard", "preview"],
        help="User-facing access track: standard or preview (prompted when omitted)",
    )
    access_group.add_argument(
        "--environment",
        choices=["production", "preview"],
        help=(
            "Endpoint name retained for scripting compatibility; prefer "
            "--access standard|preview"
        ),
    )
    setup_parser.add_argument(
        "--model",
        help="Model id to select after discovery (otherwise show a searchable picker)",
    )
    image_group = setup_parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image-model",
        help="Preview image model id to configure for image_generate",
    )
    image_group.add_argument(
        "--no-image-setup",
        action="store_true",
        help="Skip Preview image-generation setup without prompting",
    )
    setup_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=32768,
        metavar="N",
        help="Response-token cap saved for the selected model (default: 32768)",
    )
    setup_parser.add_argument(
        "--no-context-overrides",
        action="store_true",
        help="Do not save context-window metadata known to Hermes",
    )
    setup_parser.add_argument(
        "--refresh-context-limits",
        action="store_true",
        help=(
            "Replace previously saved context values with current catalog or "
            "TAMUS-documented limits"
        ),
    )

    models_parser = tamu_subparsers.add_parser(
        "models",
        help="List models currently visible to the configured key",
    )
    models_parser.add_argument(
        "--environment",
        choices=["production", "preview"],
        help="Endpoint to query (defaults to the active TAMU endpoint)",
    )
    models_parser.add_argument("--json", action="store_true", help="Print JSON")

    status_parser = tamu_subparsers.add_parser(
        "status",
        help="Show the saved TAMU configuration without exposing secrets",
    )
    status_parser.add_argument(
        "--check",
        action="store_true",
        help="Also make a live authenticated model-catalog check",
    )
    status_parser.add_argument("--json", action="store_true", help="Print JSON")

    usage_parser = tamu_subparsers.add_parser(
        "usage",
        help="Show locally recorded usage by TAMU model and agent task",
    )
    usage_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to include (default: 30)"
    )
    usage_parser.add_argument(
        "--source", help="Filter by Hermes source (cli, telegram, discord, etc.)"
    )
    usage_parser.add_argument("--json", action="store_true", help="Print JSON")

    tamu_parser.set_defaults(func=cmd_tamu)
