import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

from ai_providers import GenerationParameters, ProviderFactory
from config import ConfigManager
from outputs import OutputManager, OutputRequest
from security import SecurityValidator
from templates import DocumentTemplate


@dataclass
class GenerationRequest:
    prompt: str
    model: Optional[str] = None
    formats: Optional[Sequence[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    image_prompt: Optional[str] = None
    image_input: Optional[str] = None
    video_prompt: Optional[str] = None
    video_format: Optional[str] = None
    video_duration: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None


@dataclass
class GenerationResult:
    success: bool
    content: Optional[str]
    artifacts: Dict[str, str]
    error: Optional[str] = None


class DocumentGenerator:
    def __init__(self, config_path: Optional[Path] = None):
        self.config = ConfigManager(config_path)
        self._setup_logging()
        self.template = DocumentTemplate()
        self.console = Console() if RICH_AVAILABLE else None
        security_cfg = self.config.get_security_config()
        self.security = SecurityValidator(
            security_cfg.deny_patterns,
            security_cfg.max_prompt_length,
            security_cfg.require_ascii,
            security_cfg.forbidden_paths,
            security_cfg.blocked_extensions,
        )
        self.provider_factory = ProviderFactory()
        self.output_manager = OutputManager(self.config.get_output_dir(), self.template)
        self.logger = logging.getLogger("ai-docs")

    def _setup_logging(self) -> None:
        log_level = os.environ.get("LOG_LEVEL", self.config.snapshot.log_level)
        level = getattr(logging, log_level.upper(), logging.INFO)
        handlers = None
        if RICH_AVAILABLE:
            handler = RichHandler(rich_tracebacks=True)
            handlers = [handler]
        logging.basicConfig(level=level, format="%(message)s" if RICH_AVAILABLE else "%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    async def process_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
        *,
        prompt_file: Optional[str] = None,
        prompt_dir: Optional[str] = None,
        formats: Optional[Sequence[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        image_input: Optional[str] = None,
        image_prompt: Optional[str] = None,
        video_prompt: Optional[str] = None,
        video_format: Optional[str] = None,
        video_duration: Optional[int] = None,
    ) -> bool:
        requests = self._collect_requests(
            prompt,
            model=model,
            prompt_file=prompt_file,
            prompt_dir=prompt_dir,
            formats=formats,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            image_input=image_input,
            image_prompt=image_prompt,
            video_prompt=video_prompt,
            video_format=video_format,
            video_duration=video_duration,
        )
        results: List[GenerationResult] = []
        for request in requests:
            result = await self._execute_request(request)
            results.append(result)
        for result in results:
            if not result.success:
                return False
        return True if results else False

    def _collect_requests(
        self,
        prompt: str,
        *,
        model: Optional[str],
        prompt_file: Optional[str],
        prompt_dir: Optional[str],
        formats: Optional[Sequence[str]],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        image_input: Optional[str],
        image_prompt: Optional[str],
        video_prompt: Optional[str],
        video_format: Optional[str],
        video_duration: Optional[int],
    ) -> List[GenerationRequest]:
        items: List[GenerationRequest] = []
        if prompt_dir:
            directory = Path(prompt_dir)
            for file_path in directory.glob("**/*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {".txt", ".md", ".json"}:
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception as exc:
                    self.logger.debug(f"Skipping file {file_path}: {exc}")
                    continue
                items.append(
                    self._build_request(
                        content,
                        model,
                        formats,
                        temperature,
                        top_p,
                        max_tokens,
                        image_input,
                        image_prompt,
                        video_prompt,
                        video_format,
                        video_duration,
                        str(file_path),
                    )
                )
        elif prompt_file:
            file_items = self._extract_prompts_from_file(Path(prompt_file))
            for payload in file_items:
                items.append(
                    self._build_request(
                        payload,
                        model,
                        formats,
                        temperature,
                        top_p,
                        max_tokens,
                        image_input,
                        image_prompt,
                        video_prompt,
                        video_format,
                        video_duration,
                        prompt_file,
                    )
                )
        else:
            items.append(
                self._build_request(
                    prompt,
                    model,
                    formats,
                    temperature,
                    top_p,
                    max_tokens,
                    image_input,
                    image_prompt,
                    video_prompt,
                    video_format,
                    video_duration,
                    None,
                )
            )
        return items

    def _build_request(
        self,
        prompt: Optional[str],
        model: Optional[str],
        formats: Optional[Sequence[str]],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        image_input: Optional[str],
        image_prompt: Optional[str],
        video_prompt: Optional[str],
        video_format: Optional[str],
        video_duration: Optional[int],
        source: Optional[str],
    ) -> GenerationRequest:
        defaults = self.config.get_generation_defaults()
        resolved_formats = self.security.allowed_formats(self.config.resolve_formats(formats))
        if not resolved_formats:
            resolved_formats = ["docx"]
        text = (prompt or "").strip()
        if not text:
            raise ValueError("Prompt content is required for generation")
        resolved_video_prompt = (video_prompt or "").strip() or None
        resolved_video_format: Optional[str] = None
        resolved_video_duration: Optional[int] = None
        wants_video = bool(resolved_video_prompt or video_format or video_duration)
        if wants_video:
            resolved_video_format = self.config.resolve_video_format(video_format)
            resolved_video_duration = video_duration or self.config.get_default_video_duration()
            if not resolved_video_prompt:
                resolved_video_prompt = text
        return GenerationRequest(
            prompt=text,
            model=model,
            formats=resolved_formats,
            temperature=temperature if temperature is not None else defaults[0],
            top_p=top_p if top_p is not None else defaults[1],
            max_tokens=max_tokens if max_tokens is not None else defaults[2],
            image_prompt=image_prompt,
            image_input=image_input,
            video_prompt=resolved_video_prompt,
            video_format=resolved_video_format,
            video_duration=resolved_video_duration,
            metadata={},
            source=source,
        )

    def _extract_prompts_from_file(self, path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        content = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in {".json"}:
            try:
                payload = json.loads(content)
                if isinstance(payload, list):
                    return [str(item) for item in payload if str(item).strip()]
                if isinstance(payload, dict):
                    prompts = payload.get("prompts")
                    if isinstance(prompts, list):
                        return [str(item) for item in prompts if str(item).strip()]
                    text = payload.get("prompt")
                    if text:
                        return [str(text)]
            except Exception as exc:
                self.logger.error(f"Failed to parse JSON prompts: {exc}")
                return []
        return [content]

    async def _execute_request(self, request: GenerationRequest) -> GenerationResult:
        validation = self.security.validate_prompt(request.prompt)
        if not validation.valid:
            return GenerationResult(False, None, {}, validation.reason)
        try:
            provider_key, provider_model = self.config.resolve_model(request.model)
        except Exception as exc:
            return GenerationResult(False, None, {}, str(exc))
        api_key = self.config.get_api_key(provider_key)
        if not api_key and provider_key not in {"ollama", "diffusers"}:
            return GenerationResult(False, None, {}, f"Missing API key for provider {provider_key}")
        metadata = self._build_metadata(request, provider_key, provider_model)
        image_bytes = None
        video_bytes = None
        try:
            provider = self.provider_factory.get(provider_key, api_key)
        except Exception as exc:
            return GenerationResult(False, None, {}, str(exc))
        if request.image_prompt:
            try:
                image_bytes = await provider.generate_image(request.image_prompt)
            except Exception as exc:
                self.logger.debug(f"Image generation failed: {exc}")
        if request.image_input:
            analysis = await self._analyze_image(provider, request.image_input, request.prompt)
            if analysis:
                request.prompt = analysis
        if request.video_prompt and hasattr(provider, "generate_video"):
            duration = request.video_duration or self.config.get_default_video_duration()
            video_format = request.video_format or self.config.get_default_video_format()
            try:
                video_bytes = await provider.generate_video(request.video_prompt, duration_seconds=duration, format_hint=video_format)
                if video_bytes:
                    metadata["video_format"] = video_format
                    metadata["video_duration_seconds"] = str(duration)
                    metadata["video_prompt"] = request.video_prompt
            except Exception as exc:
                self.logger.debug(f"Video generation failed: {exc}")
        params = GenerationParameters(
            provider_model=provider_model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )
        content = await self._generate_text(provider, request.prompt, params)
        if not content:
            return GenerationResult(False, None, {}, "Generation returned no content")
        metadata["slug"] = self.security.sanitize_filename(metadata["title"]) if "title" in metadata else self.security.sanitize_filename("document")
        outputs = self.output_manager.create_outputs(
            OutputRequest(
                formats=request.formats or self.config.resolve_formats(None),
                prompt=request.prompt,
                model=provider_model,
                provider=provider_key,
                metadata=metadata,
                include_image=image_bytes,
                video_bytes=video_bytes,
                video_format=request.video_format or self.config.get_default_video_format() if video_bytes else None,
            ),
            content,
        )
        self.config.record_history(request.prompt, outputs, f"{provider_key}:{provider_model}")
        if RICH_AVAILABLE and self.console:
            for fmt, path in outputs.items():
                self.console.print(f"[bold green]✓[/] {fmt.upper()} saved: [link=file://{path}]{path}[/link]")
        else:
            for fmt, path in outputs.items():
                self.logger.info(f"{fmt.upper()} saved: {path}")
        return GenerationResult(True, content, outputs)

    async def _generate_text(self, provider, prompt: str, params: GenerationParameters) -> Optional[str]:
        if RICH_AVAILABLE and self.console:
            with Progress(SpinnerColumn(), TextColumn("[bold blue]Generating content"), console=self.console) as progress:
                task = progress.add_task("generation", total=None)
                response = await provider.generate_text(prompt, params)
                progress.update(task, completed=True)
                return response
        return await provider.generate_text(prompt, params)

    async def _analyze_image(self, provider, path: str, prompt: str) -> Optional[str]:
        try:
            file_path = Path(path)
            result = self.security.validate_path(file_path)
            if not result.valid:
                self.logger.warning(result.reason)
                return None
            data = file_path.read_bytes()
            return await provider.analyze_image(data, prompt)
        except Exception as exc:
            self.logger.debug(f"Image analysis failed: {exc}")
            return None

    def _build_metadata(self, request: GenerationRequest, provider: str, provider_model: str) -> Dict[str, str]:
        timestamp = datetime.utcnow().isoformat() + "Z"
        title = request.metadata.get("title") if request.metadata else None
        if not title:
            title = request.prompt.split("\n", 1)[0][:80] or "AI Generated Document"
        return {
            "title": title,
            "timestamp": timestamp,
            "model": f"{provider}:{provider_model}",
            "prompt": request.prompt,
            "source": request.source or "cli",
        }