"""Provider adapters for the AI document generator."""
from __future__ import annotations

import asyncio
import base64
import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from config import ProviderType

logger = logging.getLogger(__name__)


def _model_name(identifier: str) -> str:
    return identifier.split(":", 1)[1] if ":" in identifier else identifier


@dataclass
class ProviderCapabilities:
    supports_chat: bool = True
    supports_images: bool = False
    supports_vision: bool = False
    native_formats: Sequence[str] = ()


class AIProvider:
    """Base provider adapter."""

    capabilities: ProviderCapabilities = ProviderCapabilities()

    async def generate_content(self, prompt: str, model: str, temperature: float) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:  # pragma: no cover - interface
        return None

    async def generate_from_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Optional[str]:  # pragma: no cover - interface
        return None


def _lazy_import(module: str, attr: Optional[str] = None):
    """Import a module or attribute lazily without wrapping in try/except."""
    if importlib.util.find_spec(module) is None:
        raise ImportError(f"Module '{module}' is required but not installed.")
    mod = importlib.import_module(module)
    return getattr(mod, attr) if attr else mod


class GeminiProvider(AIProvider):
    capabilities = ProviderCapabilities(supports_chat=True, supports_images=True, supports_vision=True)

    def __init__(self, api_key: str):
        genai = _lazy_import("google.generativeai")
        genai.configure(api_key=api_key)
        self._genai = genai

    async def generate_content(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        try:
            model_name = _model_name(model)
            gen_model = self._genai.GenerativeModel(
                model_name=model_name,
                generation_config=self._genai.GenerationConfig(temperature=temperature),
            )
            response = await gen_model.generate_content_async(prompt)
            return response.text
        except Exception as exc:  # pragma: no cover - network interaction
            logger.error("Gemini content generation failed: %s", exc)
            return None

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        if not hasattr(self._genai, "ImageGenerationModel"):
            return None
        try:
            _, width, height = size.partition("x")
            generation_model = self._genai.ImageGenerationModel("imagen-3.0-generate-001")
            image = await generation_model.generate_image_async(prompt, width=int(width or 1024), height=int(height or 1024))
            return image.image_bytes if hasattr(image, "image_bytes") else None
        except Exception as exc:  # pragma: no cover - network interaction
            logger.debug("Gemini image generation unavailable: %s", exc)
            return None

    async def generate_from_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Optional[str]:
        try:
            model = self._genai.GenerativeModel(model_name="gemini-1.5-pro")
            response = await model.generate_content_async([
                {"mime_type": "image/png", "data": image_bytes},
                prompt or "Describe the image",
            ])
            return response.text
        except Exception as exc:  # pragma: no cover
            logger.debug("Gemini vision unavailable: %s", exc)
            return None


class OpenAIProvider(AIProvider):
    capabilities = ProviderCapabilities(supports_chat=True, supports_images=True, supports_vision=False)

    def __init__(self, api_key: str):
        AsyncOpenAI = _lazy_import("openai", "AsyncOpenAI")
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_content(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        try:
            model_name = _model_name(model)
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pragma: no cover - network interaction
            logger.error("OpenAI content generation failed: %s", exc)
            return None

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        try:
            response = await self.client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
            payload = response.data[0].b64_json
            return base64.b64decode(payload)
        except Exception as exc:  # pragma: no cover - network interaction
            logger.debug("OpenAI image generation unavailable: %s", exc)
            return None


class XAIProvider(AIProvider):
    capabilities = ProviderCapabilities(supports_chat=True, supports_images=False, supports_vision=False)

    def __init__(self, api_key: str):
        AsyncXAIClient = _lazy_import("xai_sdk", "AsyncClient")
        self.client = AsyncXAIClient(api_key=api_key)

    async def generate_content(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        try:
            model_name = _model_name(model)
            chat = await self.client.chat.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a meticulous technical writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            response = await chat.asample()
            return response.content
        except Exception as exc:  # pragma: no cover - network interaction
            logger.error("xAI content generation failed: %s", exc)
            return None


class DiffusersProvider(AIProvider):
    capabilities = ProviderCapabilities(supports_chat=False, supports_images=True)

    def __init__(self, api_key: str = ""):
        pipe_cls = _lazy_import("diffusers", "StableDiffusionPipeline")
        torch_mod = _lazy_import("torch")
        device = "cuda" if torch_mod.cuda.is_available() else "cpu"
        self.pipe = pipe_cls.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipe.to(device)

    async def generate_image(self, prompt: str, size: str = "512x512") -> Optional[bytes]:
        width_str, _, height_str = size.partition("x")
        width = int(width_str or 512)
        height = int(height_str or 512)
        loop = asyncio.get_event_loop()

        def _generate() -> bytes:
            image = self.pipe(prompt, num_inference_steps=25, width=width, height=height).images[0]
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        try:
            return await loop.run_in_executor(None, _generate)
        except Exception as exc:  # pragma: no cover - heavy dependency
            logger.debug("Diffusers image generation failed: %s", exc)
            return None


_PROVIDER_MAP: Dict[ProviderType, type[AIProvider]] = {
    ProviderType.GEMINI: GeminiProvider,
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.XAI: XAIProvider,
    ProviderType.DIFFUSERS: DiffusersProvider,
}


def get_provider(provider_type: ProviderType, api_key: str) -> AIProvider:
    provider_cls = _PROVIDER_MAP.get(provider_type)
    if not provider_cls:
        raise ValueError(f"Unknown provider: {provider_type}")
    return provider_cls(api_key)
