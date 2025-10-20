import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

try:
    from xai_sdk import AsyncClient as AsyncXAIClient
except Exception:
    AsyncXAIClient = None

try:
    from anthropic import AsyncAnthropic
except Exception:
    AsyncAnthropic = None

try:
    from mistralai.async_client import MistralAsyncClient
except Exception:
    MistralAsyncClient = None

try:
    from groq import AsyncGroq
except Exception:
    AsyncGroq = None

try:
    from diffusers import StableDiffusionPipeline
    import torch
except Exception:
    StableDiffusionPipeline = None
    torch = None

from config import ProviderType

logger = logging.getLogger(__name__)


@dataclass
class GenerationParameters:
    provider_model: str
    temperature: float
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIProvider:
    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        raise NotImplementedError

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        return None

    async def analyze_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Optional[str]:
        return None

    async def generate_video(self, prompt: str, duration_seconds: int = 8, format_hint: str = "mp4") -> Optional[bytes]:
        return None


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str):
        if not genai:
            raise ImportError("google-generativeai is not installed")
        genai.configure(api_key=api_key)

    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        try:
            model = genai.GenerativeModel(
                model_name=params.provider_model,
                generation_config=genai.GenerationConfig(
                    temperature=params.temperature,
                    top_p=params.top_p,
                    max_output_tokens=params.max_tokens,
                ),
            )
            response = await model.generate_content_async(prompt)
            if hasattr(response, "text"):
                return response.text
            parts = getattr(response, "candidates", [])
            for candidate in parts:
                for part in getattr(candidate, "content", []):
                    text = getattr(part, "text", None)
                    if text:
                        return text
            return None
        except Exception as exc:
            logger.error(f"Gemini error: {exc}")
            return None

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        try:
            model = genai.GenerativeModel(model_name="imagen-3.0")
            response = await model.generate_images_async(prompt, size=size)
            data = getattr(response, "images", [])
            if not data:
                return None
            encoded = getattr(data[0], "base64_data", None)
            if not encoded:
                return None
            return base64.b64decode(encoded)
        except Exception as exc:
            logger.debug(f"Gemini image unavailable: {exc}")
            return None

    async def analyze_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Optional[str]:
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            response = await model.generate_content_async([
                genai.content_types.Image(image_bytes),
                prompt or "Describe the image",
            ])
            return getattr(response, "text", None)
        except Exception as exc:
            logger.debug(f"Gemini multimodal unavailable: {exc}")
            return None


class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str):
        if not AsyncOpenAI:
            raise ImportError("openai is not installed")
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        try:
            response = await self.client.chat.completions.create(
                model=params.provider_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
            )
            choice = response.choices[0]
            message = choice.message
            return getattr(message, "content", None)
        except Exception as exc:
            logger.error(f"OpenAI error: {exc}")
            return None

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        try:
            response = await self.client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
            payload = response.data[0]
            encoded = getattr(payload, "b64_json", None)
            if not encoded:
                return None
            return base64.b64decode(encoded)
        except Exception as exc:
            logger.debug(f"OpenAI image unavailable: {exc}")
            return None

    async def generate_video(self, prompt: str, duration_seconds: int = 8, format_hint: str = "mp4") -> Optional[bytes]:
        try:
            response = await self.client.responses.create(
                model="gpt-4o-realtime-preview",
                input=[{"role": "user", "content": prompt}],
                modalities=["video"],
                max_output_tokens=3072,
            )
            if not getattr(response, "output", None):
                return None
            for item in response.output:
                data = getattr(item, "video", None)
                if data and getattr(data, "data", None):
                    return base64.b64decode(data.data)
            return None
        except Exception as exc:
            logger.debug(f"OpenAI video unavailable: {exc}")
            return None


class XAIProvider(AIProvider):
    def __init__(self, api_key: str):
        if not AsyncXAIClient:
            raise ImportError("xai-sdk is not installed")
        self.client = AsyncXAIClient(api_key=api_key)

    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        try:
            chat = await self.client.chat.create(
                model=params.provider_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=params.temperature,
            )
            completion = await chat.asample()
            return getattr(completion, "content", None)
        except Exception as exc:
            logger.error(f"xAI error: {exc}")
            return None


class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str):
        if not AsyncAnthropic:
            raise ImportError("anthropic is not installed")
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        try:
            response = await self.client.messages.create(
                model=params.provider_model,
                max_tokens=params.max_tokens or 1024,
                temperature=params.temperature,
                top_p=params.top_p,
                messages=[{"role": "user", "content": prompt}],
            )
            text = []
            for block in getattr(response, "content", []):
                if getattr(block, "type", "") == "text":
                    value = getattr(block, "text", "")
                    if value:
                        text.append(value)
            return "".join(text) if text else None
        except Exception as exc:
            logger.error(f"Anthropic error: {exc}")
            return None


class MistralProvider(AIProvider):
    def __init__(self, api_key: str):
        if not MistralAsyncClient:
            raise ImportError("mistralai is not installed")
        self.client = MistralAsyncClient(api_key=api_key)

    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        try:
            response = await self.client.chat(
                model=params.provider_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
            )
            choices = getattr(response, "choices", [])
            if not choices:
                return None
            message = choices[0].message
            return getattr(message, "content", None)
        except Exception as exc:
            logger.error(f"Mistral error: {exc}")
            return None


class GroqProvider(AIProvider):
    def __init__(self, api_key: str):
        if not AsyncGroq:
            raise ImportError("groq is not installed")
        self.client = AsyncGroq(api_key=api_key)

    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        try:
            response = await self.client.chat.completions.create(
                model=params.provider_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
            )
            choice = response.choices[0]
            return getattr(choice.message, "content", None)
        except Exception as exc:
            logger.error(f"Groq error: {exc}")
            return None


class OllamaProvider(AIProvider):
    def __init__(self, api_key: str):
        self.base_url = api_key or "http://localhost:11434"

    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        payload = {"model": params.provider_model, "prompt": prompt, "options": {"temperature": params.temperature}}
        if params.max_tokens:
            payload["options"]["num_predict"] = params.max_tokens
        if params.top_p:
            payload["options"]["top_p"] = params.top_p
        try:
            def call() -> Optional[str]:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120,
                    stream=True,
                )
                response.raise_for_status()
                lines = []
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("response")
                    if chunk:
                        lines.append(chunk)
                    if data.get("done"):
                        break
                return "".join(lines) if lines else None

            return await asyncio.to_thread(call)
        except Exception as exc:
            logger.error(f"Ollama error: {exc}")
            return None


class DiffusersProvider(AIProvider):
    def __init__(self, api_key: str):
        if not StableDiffusionPipeline:
            raise ImportError("diffusers is not installed")
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.pipeline = StableDiffusionPipeline.from_pretrained(api_key or "runwayml/stable-diffusion-v1-5")
        self.pipeline.to(device)

    async def generate_image(self, prompt: str, size: str = "512x512") -> Optional[bytes]:
        try:
            image = await asyncio.to_thread(self.pipeline, prompt)
            picture = image.images[0]
            from io import BytesIO

            buffer = BytesIO()
            picture.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as exc:
            logger.debug(f"Diffusers error: {exc}")
            return None


class ProviderFactory:
    def __init__(self):
        self._cache: Dict[str, AIProvider] = {}

    def get(self, provider: str, api_key: str) -> AIProvider:
        key = f"{provider}:{api_key}"
        if key in self._cache:
            return self._cache[key]
        instance = self._create(provider, api_key)
        self._cache[key] = instance
        return instance

    def _create(self, provider: str, api_key: str) -> AIProvider:
        provider_type = ProviderType(provider)
        if provider_type == ProviderType.GEMINI:
            return GeminiProvider(api_key)
        if provider_type == ProviderType.OPENAI:
            return OpenAIProvider(api_key)
        if provider_type == ProviderType.XAI:
            return XAIProvider(api_key)
        if provider_type == ProviderType.ANTHROPIC:
            return AnthropicProvider(api_key)
        if provider_type == ProviderType.MISTRAL:
            return MistralProvider(api_key)
        if provider_type == ProviderType.GROQ:
            return GroqProvider(api_key)
        if provider_type == ProviderType.OLLAMA:
            return OllamaProvider(api_key)
        if provider_type == ProviderType.DIFFUSERS:
            return DiffusersProvider(api_key)
        if provider_type == ProviderType.STABILITY:
            return StabilityProvider(api_key)
        if provider_type == ProviderType.LEONARDO:
            return LeonardoProvider(api_key)
        if provider_type == ProviderType.RUNWAY:
            return RunwayProvider(api_key)
        if provider_type == ProviderType.PIKA:
            return PikaProvider(api_key)
        if provider_type == ProviderType.REPLICATE:
            return ReplicateProvider(api_key)
        raise ValueError(f"Unsupported provider: {provider}")


class StabilityProvider(AIProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Stability API key required")
        self.api_key = api_key
        self.session = requests.Session()

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        parts = size.split("x")
        width = int(parts[0]) if parts and parts[0].isdigit() else 1024
        height = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1024
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": height,
            "width": width,
            "samples": 1,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

        def call() -> Optional[bytes]:
            response = self.session.post(
                "https://api.stability.ai/v1/generation/sdxl-1024x1024/engine/text-to-image",
                json=payload,
                headers=headers,
                timeout=120,
            )
            if response.status_code != 200:
                return None
            data = response.json()
            artifacts = data.get("artifacts") or []
            if not artifacts:
                return None
            encoded = artifacts[0].get("base64")
            if not encoded:
                return None
            return base64.b64decode(encoded)

        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            logger.debug(f"Stability image error: {exc}")
            return None


class LeonardoProvider(AIProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Leonardo API key required")
        self.api_key = api_key
        self.session = requests.Session()

    async def generate_image(self, prompt: str, size: str = "768x768") -> Optional[bytes]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"prompt": prompt, "presetStyle": "LEONARDO", "width": size.split("x")[0], "height": size.split("x")[-1]}

        def call() -> Optional[bytes]:
            response = self.session.post("https://cloud.leonardo.ai/api/rest/v1/generations", json=payload, headers=headers, timeout=120)
            if response.status_code != 200:
                return None
            data = response.json()
            generations = data.get("generations") or []
            if not generations:
                return None
            images = generations[0].get("generated_images") or []
            if not images:
                return None
            uri = images[0].get("url")
            if not uri:
                return None
            image_response = self.session.get(uri, timeout=120)
            if image_response.status_code != 200:
                return None
            return image_response.content

        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            logger.debug(f"Leonardo image error: {exc}")
            return None


class RunwayProvider(AIProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Runway API key required")
        self.api_key = api_key
        self.session = requests.Session()

    async def generate_video(self, prompt: str, duration_seconds: int = 8, format_hint: str = "mp4") -> Optional[bytes]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"prompt": prompt, "duration": duration_seconds, "format": format_hint}

        def call() -> Optional[bytes]:
            response = self.session.post("https://api.runwayml.com/v1/videos", json=payload, headers=headers, timeout=180)
            if response.status_code != 200:
                return None
            data = response.json()
            video_url = data.get("url")
            if not video_url:
                return None
            download = self.session.get(video_url, timeout=300)
            if download.status_code != 200:
                return None
            return download.content

        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            logger.debug(f"Runway video error: {exc}")
            return None


class PikaProvider(AIProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Pika API key required")
        self.api_key = api_key
        self.session = requests.Session()

    async def generate_video(self, prompt: str, duration_seconds: int = 4, format_hint: str = "mp4") -> Optional[bytes]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"prompt": prompt, "duration": duration_seconds, "format": format_hint}

        def call() -> Optional[bytes]:
            response = self.session.post("https://api.pika.art/v1/videos", json=payload, headers=headers, timeout=180)
            if response.status_code != 200:
                return None
            data = response.json()
            video_url = data.get("video")
            if not video_url:
                return None
            download = self.session.get(video_url, timeout=300)
            if download.status_code != 200:
                return None
            return download.content

        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            logger.debug(f"Pika video error: {exc}")
            return None


class ReplicateProvider(AIProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Replicate API key required")
        self.api_key = api_key
        self.session = requests.Session()

    async def generate_text(self, prompt: str, params: GenerationParameters) -> Optional[str]:
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        payload = {"input": {"prompt": prompt, "max_length": params.max_tokens or 1024}}

        def call() -> Optional[str]:
            response = self.session.post(
                f"https://api.replicate.com/v1/models/{params.provider_model}/predictions",
                json=payload,
                headers=headers,
                timeout=180,
            )
            if response.status_code != 201:
                return None
            data = response.json()
            prediction = data.get("output")
            if isinstance(prediction, list):
                return "".join(str(part) for part in prediction)
            if isinstance(prediction, dict):
                return str(prediction)
            return str(prediction) if prediction else None

        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            logger.error(f"Replicate error: {exc}")
            return None

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        return None