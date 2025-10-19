"""Command-line interface for the AI Document Generator."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from config import Config, ModelType
from generator import DocumentGenerator

try:
    from rich.prompt import Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RICH_AVAILABLE = False
    Prompt = None  # type: ignore
    Table = None  # type: ignore


def _parse_formats(raw: str | Sequence[str] | None) -> Sequence[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return list(raw)


async def run_cli(generator: DocumentGenerator) -> None:
    if RICH_AVAILABLE and generator.console:
        generator.console.print("[bold]AI Document Generator[/bold] [dim]v3.0.0[/dim]")
        generator.console.print("Supports Gemini, OpenAI GPT family, xAI Grok, local Diffusers")
        generator.console.print("Keyboard shortcuts: Ctrl+C to exit, use your EDITOR env for revisions.")
    else:
        print("AI Document Generator v3.0.0")
        print("Supports Gemini, OpenAI GPT family, xAI Grok, local Diffusers")
        print("Press Ctrl+C to exit. Configure EDITOR env variable for inline revisions.")

    while True:
        try:
            if RICH_AVAILABLE and generator.console:
                user_input = Prompt.ask("\nPrompt")
            else:
                user_input = input("\nPrompt: ")

            if user_input.strip().lower() in {"exit", "quit", "q"}:
                if RICH_AVAILABLE and generator.console:
                    generator.console.print("[dim]Exiting...[/dim]")
                else:
                    print("Exiting...")
                break

            await generator.generate_bundle(user_input)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            break
        except Exception as exc:  # pragma: no cover - runtime guard
            generator.logger.error("Error: %s", exc)


async def async_main(args: argparse.Namespace) -> None:
    config_path = Path(args.config) if args.config else None
    generator = DocumentGenerator(config_path)

    if args.list_models:
        _print_models(args.list_models)
        return

    if args.history:
        _print_history(generator.config, args.history)
        return

    if args.prompt or args.prompt_file:
        formats = _parse_formats(args.format)
        await generator.generate_bundle(
            args.prompt or "",
            args.model,
            formats=formats,
            prompt_file=args.prompt_file,
            image_input=args.image_input,
            image_prompt=args.image_prompt,
            enable_editing=not args.no_edit,
        )
        return

    await run_cli(generator)


def _print_models(provider: str | None) -> None:
    catalog = Config.list_models(provider)
    if RICH_AVAILABLE:
        table = Table(title="Available Models")
        table.add_column("Provider", style="bold cyan")
        table.add_column("Models", style="white")
        for key, models in catalog.items():
            table.add_row(key, "\n".join(models) or "(none)")
        from rich.console import Console

        Console().print(table)
    else:
        for key, models in catalog.items():
            print(f"{key}: {', '.join(models) if models else '(none)'}")


def _print_history(config: Config, limit: int) -> None:
    entries = config.get_recent_history(limit)
    if not entries:
        print("History is empty.")
        return
    if RICH_AVAILABLE:
        table = Table(title="Recent Documents")
        table.add_column("Timestamp", style="bold")
        table.add_column("Model")
        table.add_column("Formats")
        table.add_column("Path")
        for entry in entries:
            table.add_row(entry.timestamp, entry.model, ", ".join(entry.formats), entry.filepath)
        from rich.console import Console

        Console().print(table)
    else:
        for entry in entries:
            print(f"[{entry.timestamp}] {entry.model} -> {entry.filepath} ({', '.join(entry.formats)})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Document Generator")
    parser.add_argument("-p", "--prompt", help="Generate a document with this prompt and exit")
    parser.add_argument("--prompt-file", help="Path to a file containing the prompt text")
    parser.add_argument("-m", "--model", choices=[m.value for m in ModelType], help="Model to use")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--format", help="Comma separated list of output formats (docx,pdf,html,md,txt,pptx)")
    parser.add_argument("--image-prompt", help="Generate an image and include it in the document")
    parser.add_argument("--image-input", help="Analyse an image file with supported models")
    parser.add_argument("--no-edit", action="store_true", help="Skip the interactive editing step")
    parser.add_argument("--list-models", nargs="?", const="", help="List available models (optionally filtered by provider)")
    parser.add_argument("--history", nargs="?", const=10, type=int, help="Display recent generation history")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.debug:
        import logging
        import os

        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["LOG_LEVEL"] = "DEBUG"

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(async_main(args))


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
