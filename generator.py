import asyncio
import logging
import time
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from docx import Document
from .ai_providers import AIProvider, get_provider, ProviderType
from .config import Config
from .templates import DocumentTemplate

class DocumentGenerator:
    def __init__(self, config_path: Optional[Path] = None):
        self._setup_logging()
        self.config = Config(config_path)
        self.template = DocumentTemplate()
        self.console = Console() if RICH_AVAILABLE else None
        self.providers: Dict[str, AIProvider] = {}
        self._last_generated_image: Optional[bytes] = None

    def _setup_logging(self):
        log_level = os.environ.get("LOG_LEVEL", "INFO")
        if RICH_AVAILABLE:
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True)]
            )
        else:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        self.logger = logging.getLogger("ai-docs")

    def _get_provider(self, model: str) -> Tuple[AIProvider, str]:
        provider_str, _ = model.split(':')
        provider_type = ProviderType(provider_str)
        api_key = self.config.settings["api_keys"].get(provider_str, "")
        if not api_key:
            raise ValueError(f"API key for {provider_str} is not set")
        if provider_str not in self.providers:
            self.providers[provider_str] = get_provider(provider_type, api_key)
        return self.providers[provider_str], provider_str

    def create_document(self, content: str, prompt: str, model: str) -> Tuple[str, str]:
        timestamp = int(time.time())
        filename = f"document_{timestamp}_{uuid.uuid4().hex[:6]}.docx"
        output_dir = self.config.get_output_dir()
        filepath = output_dir / filename
        doc = Document()
        doc = self.template.apply_template(doc, content, prompt, model)
        last_image = getattr(self, '_last_generated_image', None)
        if last_image:
            try:
                doc = self.template.add_image(doc, last_image, width_inches=6.0, caption="Generated cover image")
            except Exception:
                pass
        doc.save(filepath)
        try:
            rel_path = filepath.relative_to(Path.cwd())
            display_path = str(rel_path)
        except ValueError:
            display_path = str(filepath)
        return str(filepath), display_path

    def create_table_document(self, table_data: list, headers: Optional[list] = None, title: Optional[str] = "Table Document") -> Tuple[Optional[str], Optional[str]]:
        try:
            doc = Document()
            if title:
                doc.add_heading(title, level=1)
            self.template.add_table(doc, table_data, headers)
            timestamp = int(time.time())
            filename = f"table_{timestamp}_{uuid.uuid4().hex[:6]}.docx"
            output_dir = self.config.get_output_dir()
            filepath = output_dir / filename
            doc.save(filepath)
            return str(filepath), str(filepath)
        except Exception as e:
            self.logger.error(f"Failed to create table document: {e}")
            return None, None

    def create_presentation_file(self, title: str, slides: list) -> Tuple[Optional[str], Optional[str]]:
        try:
            pres_bytes = self.template.create_presentation(title, slides)
            if not pres_bytes:
                raise RuntimeError("python-pptx not available")
            timestamp = int(time.time())
            filename = f"presentation_{timestamp}_{uuid.uuid4().hex[:6]}.pptx"
            output_dir = self.config.get_output_dir()
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(pres_bytes)
            return str(filepath), str(filepath)
        except Exception as e:
            self.logger.error(f"Failed to create presentation: {e}")
            return None, None

    def convert_docx_to_pdf(self, docx_path: str) -> Optional[str]:
        try:
            from docx2pdf import convert
            pdf_path = str(Path(docx_path).with_suffix('.pdf'))
            convert(docx_path, pdf_path)
            return pdf_path
        except Exception as e:
            self.logger.debug(f"docx2pdf conversion failed: {e}")
            return None

    async def generate_content(self, prompt: str, model: Optional[str] = None, *, image_input_path: Optional[str] = None, image_prompt: Optional[str] = None) -> Optional[str]:
        model = model or self.config.settings["default_model"]
        temperature = float(self.config.settings.get("default_temperature", 0.7))
        provider, provider_key = self._get_provider(model)
        self._last_generated_image = None
        if image_input_path:
            try:
                image_bytes = Path(image_input_path).read_bytes()
                analysis = await provider.generate_from_image(image_bytes, prompt)
                if analysis:
                    return analysis
            except Exception as e:
                self.logger.debug(f"Failed to process image input: {e}")
        if image_prompt:
            try:
                image_bytes = await provider.generate_image(image_prompt)
                if image_bytes:
                    self._last_generated_image = image_bytes
            except Exception as e:
                self.logger.debug(f"Image generation failed: {e}")
        if RICH_AVAILABLE and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Generating content..."),
                console=self.console
            ) as progress:
                task = progress.add_task("Generating", total=None)
                content = await provider.generate_content(prompt, model, temperature)
                progress.update(task, completed=True)
        else:
            print("Generating content...")
            content = await provider.generate_content(prompt, model, temperature)
        return content

    async def process_prompt(self, prompt: str, model: Optional[str] = None, *, prompt_file: Optional[str] = None, image_input: Optional[str] = None, image_prompt: Optional[str] = None) -> bool:
        try:
            if prompt_file:
                p = Path(prompt_file)
                if p.exists():
                    prompt_text = p.read_text(encoding='utf-8')
                else:
                    self.logger.error(f"Prompt file not found: {prompt_file}")
                    return False
            else:
                prompt_text = prompt
            content = await self.generate_content(prompt_text, model, image_input_path=image_input, image_prompt=image_prompt)
            if not content:
                return False
            used_model = model or self.config.settings["default_model"]
            filepath, display_path = self.create_document(content, prompt_text, used_model)
            self.config.add_to_history(prompt_text, filepath, used_model)
            if RICH_AVAILABLE and self.console:
                self.console.print(f"[bold green]\u2713[/] Document saved: [link=file://{filepath}]{display_path}[/link]")
            else:
                print(f"\u2713 Document saved: {display_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to process prompt: {e}")
            return False