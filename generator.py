"""Core generation workflow for the AI Document Generator."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from docx import Document

from ai_providers import AIProvider, ProviderCapabilities, get_provider
from config import Config, ProviderType, SupportedFormat
from templates import DocumentTemplate

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RICH_AVAILABLE = False
    Console = None  # type: ignore
    RichHandler = None  # type: ignore
    Progress = None  # type: ignore


@dataclass
class GenerationResult:
    prompt: str
    model: str
    content: str
    files: Dict[str, str]
    provider_capabilities: ProviderCapabilities


class RevisionManager:
    """Tracks user edits and revision history."""

    def __init__(self, config: Config, console: Optional[Console] = None):
        self._config = config
        self._console = console

    def apply_revisions(self, content: str, prompt: str, model: str, interactive: bool = True) -> str:
        if not interactive:
            self._config.add_revision(prompt, content, model)
            return content

        message = "Would you like to edit the generated content before saving? [y/N]: "
        if self._console:
            decision = self._console.input(message).strip().lower()
        else:
            decision = input(message).strip().lower()

        if decision not in {"y", "yes"}:
            self._config.add_revision(prompt, content, model)
            return content

        editor = os.environ.get("EDITOR")
        if editor:
            edited = self._edit_with_external_editor(editor, content)
        else:
            edited = self._edit_inline(content)

        cleaned = edited.strip() or content
        self._config.add_revision(prompt, cleaned, model)
        return cleaned

    @staticmethod
    def _edit_with_external_editor(editor: str, content: str) -> str:
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".md") as temp:
            temp.write(content)
            temp.flush()
            temp_path = temp.name
        try:
            subprocess.run([editor, temp_path], check=False)
            with open(temp_path, "r", encoding="utf-8") as file:
                return file.read()
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _edit_inline(self, content: str) -> str:
        prompt_text = (
            "\nEnter revised content. Submit with Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows).\n"
            "Leave empty to keep the original text.\n\n"
        )
        if self._console:
            self._console.print(prompt_text)
        else:
            print(prompt_text)
        try:
            edited = sys.stdin.read()
        except Exception:  # pragma: no cover - stdin quirks
            edited = content
        return edited or content


class DocumentGenerator:
    def __init__(self, config_path: Optional[Path] = None):
        self._setup_logging()
        self.config = Config(config_path)
        self.template = DocumentTemplate()
        self.console: Optional[Console] = Console() if RICH_AVAILABLE else None
        self.providers: Dict[str, AIProvider] = {}
        self._last_generated_image: Optional[bytes] = None
        self.revision_manager = RevisionManager(self.config, self.console)

    def _setup_logging(self) -> None:
        log_level = os.environ.get("LOG_LEVEL", "INFO")
        if RICH_AVAILABLE:
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True)],
            )
        else:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        self.logger = logging.getLogger("ai-docs")

    def _get_provider(self, model: str) -> Tuple[AIProvider, str]:
        if ":" not in model:
            raise ValueError("Model must be provided as 'provider:model_name'.")
        provider_str, _ = model.split(":", 1)
        provider_type = ProviderType(provider_str)
        api_key = self.config.settings["api_keys"].get(provider_str, "")
        if not api_key:
            raise ValueError(f"API key for provider '{provider_str}' is not configured.")
        if provider_str not in self.providers:
            self.providers[provider_str] = get_provider(provider_type, api_key)
        return self.providers[provider_str], provider_str

    async def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        *,
        image_input_path: Optional[str] = None,
        image_prompt: Optional[str] = None,
    ) -> Tuple[Optional[str], ProviderCapabilities]:
        model_name = model or self.config.settings["default_model"]
        temperature = float(self.config.settings.get("default_temperature", 0.6))
        provider, provider_key = self._get_provider(model_name)
        self._last_generated_image = None
        if not provider.capabilities.supports_chat:
            raise ValueError(f"Provider '{provider_key}' does not support text generation.")

        if image_input_path and provider.capabilities.supports_vision:
            try:
                image_bytes = Path(image_input_path).read_bytes()
                analysis = await provider.generate_from_image(image_bytes, prompt)
                if analysis:
                    return analysis, provider.capabilities
            except Exception as exc:
                self.logger.debug("Failed to process image input: %s", exc)

        if image_prompt and provider.capabilities.supports_images:
            try:
                image_bytes = await provider.generate_image(image_prompt)
                if image_bytes:
                    self._last_generated_image = image_bytes
            except Exception as exc:
                self.logger.debug("Image generation failed: %s", exc)

        if RICH_AVAILABLE and self.console:
            with Progress(SpinnerColumn(), TextColumn("[bold blue]Generating content..."), console=self.console) as progress:
                task = progress.add_task("generate", total=None)
                content = await provider.generate_content(prompt, model_name, temperature)
                progress.update(task, completed=True)
        else:
            self.logger.info("Generating content...")
            content = await provider.generate_content(prompt, model_name, temperature)

        return content, provider.capabilities

    def _create_document(self, content: str, prompt: str, model: str) -> str:
        timestamp = int(time.time())
        filename = f"document_{timestamp}_{uuid.uuid4().hex[:6]}.docx"
        output_dir = self.config.get_output_dir()
        filepath = output_dir / filename
        doc = Document()
        doc = self.template.apply_template(doc, content, prompt, model)
        if self._last_generated_image:
            try:
                doc = self.template.add_image(doc, self._last_generated_image, width_inches=6.0, caption="Generated illustration")
            except Exception:  # pragma: no cover - optional feature
                self.logger.debug("Failed to append generated image to document")
        doc.save(filepath)
        return str(filepath)

    def _create_markdown(self, content: str, prompt: str, model: str) -> str:
        output_dir = self.config.get_output_dir()
        filename = f"document_{int(time.time())}_{uuid.uuid4().hex[:6]}.md"
        filepath = output_dir / filename
        header = ["# AI Generated Document", "", f"**Prompt:** {prompt}", f"**Model:** {model}", ""]
        filepath.write_text("\n".join(header + [content]), encoding="utf-8")
        return str(filepath)

    def _create_html(self, content: str, prompt: str, model: str) -> str:
        output_dir = self.config.get_output_dir()
        filename = f"document_{int(time.time())}_{uuid.uuid4().hex[:6]}.html"
        filepath = output_dir / filename
        html = self.template.render_html(content, prompt, model, self._last_generated_image)
        filepath.write_text(html, encoding="utf-8")
        return str(filepath)

    def _create_text(self, content: str) -> str:
        output_dir = self.config.get_output_dir()
        filename = f"document_{int(time.time())}_{uuid.uuid4().hex[:6]}.txt"
        filepath = output_dir / filename
        filepath.write_text(content, encoding="utf-8")
        return str(filepath)

    def _create_presentation(self, content: str, prompt: str, model: str) -> Optional[str]:
        presentation = self.template.create_presentation_from_text(content, prompt, model, self._last_generated_image)
        if not presentation:
            return None
        output_dir = self.config.get_output_dir()
        filename = f"presentation_{int(time.time())}_{uuid.uuid4().hex[:6]}.pptx"
        filepath = output_dir / filename
        filepath.write_bytes(presentation)
        return str(filepath)

    def _convert_to_pdf(self, docx_path: str) -> Optional[str]:
        try:
            from docx2pdf import convert

            pdf_path = str(Path(docx_path).with_suffix(".pdf"))
            convert(docx_path, pdf_path)
            return pdf_path
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.debug("docx2pdf conversion failed: %s", exc)
            return None

    def export_formats(self, content: str, prompt: str, model: str, formats: Iterable[str]) -> Dict[str, str]:
        filepaths: Dict[str, str] = {}
        normalized_formats = {fmt.lower() for fmt in formats}

        if SupportedFormat.DOCX.value in normalized_formats or not normalized_formats:
            docx_path = self._create_document(content, prompt, model)
            filepaths[SupportedFormat.DOCX.value] = docx_path

        if SupportedFormat.MARKDOWN.value in normalized_formats:
            filepaths[SupportedFormat.MARKDOWN.value] = self._create_markdown(content, prompt, model)

        if SupportedFormat.HTML.value in normalized_formats:
            filepaths[SupportedFormat.HTML.value] = self._create_html(content, prompt, model)

        if SupportedFormat.TEXT.value in normalized_formats:
            filepaths[SupportedFormat.TEXT.value] = self._create_text(content)

        if SupportedFormat.PPTX.value in normalized_formats:
            pptx_path = self._create_presentation(content, prompt, model)
            if pptx_path:
                filepaths[SupportedFormat.PPTX.value] = pptx_path

        if SupportedFormat.PDF.value in normalized_formats:
            docx_path = filepaths.get(SupportedFormat.DOCX.value) or self._create_document(content, prompt, model)
            pdf_path = self._convert_to_pdf(docx_path)
            if pdf_path:
                filepaths[SupportedFormat.PDF.value] = pdf_path

        return filepaths

    async def generate_bundle(
        self,
        prompt: str,
        model: Optional[str] = None,
        *,
        formats: Optional[Sequence[str]] = None,
        prompt_file: Optional[str] = None,
        image_input: Optional[str] = None,
        image_prompt: Optional[str] = None,
        enable_editing: Optional[bool] = None,
    ) -> Optional[GenerationResult]:
        if prompt_file:
            path = Path(prompt_file)
            if not path.exists():
                self.logger.error("Prompt file not found: %s", prompt_file)
                return None
            prompt_text = path.read_text(encoding="utf-8")
        else:
            prompt_text = prompt

        content, capabilities = await self.generate_content(
            prompt_text,
            model,
            image_input_path=image_input,
            image_prompt=image_prompt,
        )
        if not content:
            return None

        enable_editing = self.config.settings.get("auto_open_editor", True) if enable_editing is None else enable_editing
        edited_content = self.revision_manager.apply_revisions(
            content,
            prompt_text,
            model or self.config.settings["default_model"],
            interactive=enable_editing,
        )

        selected_formats = formats or self.config.settings.get("default_formats", [SupportedFormat.DOCX.value])
        filepaths = self.export_formats(edited_content, prompt_text, model or self.config.settings["default_model"], selected_formats)

        self.config.add_to_history(prompt_text, filepaths, model or self.config.settings["default_model"], selected_formats)

        if self.console:
            for fmt, path in filepaths.items():
                self.console.print(f"[bold green]\u2713[/] {fmt.upper()} saved: [link=file://{path}]{path}[/link]")
        else:
            for fmt, path in filepaths.items():
                print(f"\u2713 {fmt.upper()} saved: {path}")

        return GenerationResult(
            prompt=prompt_text,
            model=model or self.config.settings["default_model"],
            content=edited_content,
            files=filepaths,
            provider_capabilities=capabilities,
        )
