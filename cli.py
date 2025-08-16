import argparse
import asyncio
import sys
from pathlib import Path

from .generator import DocumentGenerator
from .config import ModelType

try:
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

async def run_cli(generator: DocumentGenerator) -> None:
    if RICH_AVAILABLE:
        generator.console.print("[bold]AI Document Generator[/bold] [dim]v2.0.0[/dim]")
        generator.console.print("Supports Gemini, OpenAI, xAI (Grok)")
        generator.console.print("Type [bold cyan]exit[/] to quit or enter a prompt to generate a document.")
        generator.console.print("Use --model <model> in args or set default in config.")
    else:
        print("AI Document Generator v2.0.0")
        print("Supports Gemini, OpenAI, xAI (Grok)")
        print("Type 'exit' to quit or enter a prompt to generate a document.")

    while True:
        try:
            if RICH_AVAILABLE:
                user_input = Prompt.ask("\nPrompt")
            else:
                user_input = input("\nPrompt: ")

            if user_input.lower() in ("exit", "quit", "q"):
                if RICH_AVAILABLE:
                    generator.console.print("[dim]Exiting...[/dim]")
                else:
                    print("Exiting...")
                break

            await generator.process_prompt(user_input)

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            break
        except Exception as e:
            generator.logger.error(f"Error: {e}")

async def async_main(args: argparse.Namespace) -> None:
    config_path = Path(args.config) if args.config else None
    generator = DocumentGenerator(config_path)

    if args.prompt:
        model = args.model if hasattr(args, 'model') else None
        await generator.process_prompt(
            args.prompt,
            model,
            prompt_file=args.prompt_file if hasattr(args, 'prompt_file') else None,
            image_input=args.image_input if hasattr(args, 'image_input') else None,
            image_prompt=args.image_prompt if hasattr(args, 'image_prompt') else None,
        )
        if hasattr(args, 'format') and args.format == 'pdf':
            history_path = generator.config.history_path
            try:
                import json
                if history_path.exists():
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    if history:
                        last = history[-1]
                        docx_path = last.get('filepath')
                        pdf_path = generator.convert_docx_to_pdf(docx_path)
                        if pdf_path:
                            print(f"Converted to PDF: {pdf_path}")
            except Exception as e:
                generator.logger.debug(f"PDF conversion error: {e}")
        return

    await run_cli(generator)