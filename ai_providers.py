import asyncio
import logging
from typing import Optional, Dict

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from xai_sdk import AsyncClient as AsyncXAIClient
except ImportError:
    AsyncXAIClient = None

try:
    from diffusers import StableDiffusionPipeline
    import torch
except Exception:
    StableDiffusionPipeline = None
    torch = None

from .config import ProviderType

logger = logging.getLogger(__name__)

class AIProvider:
    async def generate_content(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        raise NotImplementedError

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        return None

    async def generate_from_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Optional[str]:
        return None

class GeminiProvider(AIProvider):
    def __init__(self, api_key: str):
        if not genai:
            raise ImportError("google-generativeai is not installed")
        genai.configure(api_key=api_key)

    async def generate_content(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        try:
            full_model = model.split(':')[1]
            gen_model = genai.GenerativeModel(
                model_name=full_model,
                generation_config=genai.GenerationConfig(temperature=temperature)
            )
            response = await gen_model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None

    async def generate_from_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Optional[str]:
        try:
            if not genai:
                raise ImportError("google-generativeai is not installed")
            model = genai.GenerativeModel(model_name="image-bison-001")
            response = await model.generate_image_caption_async(image_bytes, prompt or "Describe the image")
            return getattr(response, 'caption', None) or str(response)
        except Exception as e:
            logger.debug(f"Gemini image processing not available: {e}")
            return None

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str):
        if not AsyncOpenAI:
            raise ImportError("openai is not installed")
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_content(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        try:
            full_model = model.split(':')[1]
            response = await self.client.chat.completions.create(
                model=full_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        try:
            if not AsyncOpenAI:
                raise ImportError("openai is not installed")
            resp = await self.client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
            import base64
            b64 = resp.data[0].b64_json
            return base64.b64decode(b64)
        except Exception as e:
            logger.debug(f"OpenAI image generation not available: {e}")
            return None

    async def generate_from_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Optional[str]:
        try:
            return None
        except Exception as e:
            logger.debug(f"OpenAI image processing not available: {e}")
            return None

class XAIProvider(AIProvider):
    def __init__(self, api_key: str):
        if not AsyncXAIClient:
            raise ImportError("xai-sdk is not installed")
        self.client = AsyncXAIClient(api_key=api_key)

    async def generate_content(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        try:
            full_model = model.split(':')[1]
            chat = await self.client.chat.create(
                model=full_model,
                messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
            )
            response = await chat.asample()
            return response.content
        except Exception as e:
            logger.error(f"xAI error: {e}")
            return None

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        try:
            return None
        except Exception as e:
            logger.debug(f"xAI image generation not available: {e}")
            return None

    async def generate_from_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> Optional[str]:
        try:
            return None
        except Exception as e:
            logger.debug(f"xAI image processing not available: {e}")
            return None

def get_provider(provider_type: ProviderType, api_key: str) -> AIProvider:
    if provider_type == ProviderType.GEMINI:
        return GeminiProvider(api_key)
    elif provider_type == ProviderType.OPENAI:
        return OpenAIProvider(api_key)
    elif provider_type == ProviderType.XAI:
        return XAIProvider(api_key)
    elif provider_type == ProviderType.DIFFUSERS:
        return DiffusersProvider(api_key)
    raise ValueError(f"Unknown provider: {provider_type}")


class DiffusersProvider(AIProvider):
    def __init__(self, api_key: str = ""):
        if not StableDiffusionPipeline:
            raise ImportError("diffusers is not installed or available")
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipe.to(device)

    async def generate_image(self, prompt: str, size: str = "512x512") -> Optional[bytes]:
        try:
            width, height = [int(x) for x in size.split("x")]
            image = self.pipe(prompt, num_inference_steps=25).images[0]
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format='PNG')
            return buf.getvalue()
        except Exception as e:
            logger.debug(f"Diffusers generation failed: {e}")
            return None