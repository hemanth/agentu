"""OpenAI-compatible protocol driver (OpenAI, DeepSeek, Groq, vLLM, Ollama-v1)."""

import json
import base64
import logging
from typing import AsyncIterator, Dict, Any, List, Optional
import aiohttp

from .base import BaseDriver, DriverResponse, GeneratedImage, ImageGenerationResponse

logger = logging.getLogger(__name__)


class OpenAIDriver(BaseDriver):
    """Zero-dependency HTTP driver for OpenAI-compatible /chat/completions."""

    def __init__(
        self,
        api_base: Optional[str] = "http://localhost:11434/v1",
        api_key: Optional[str] = None,
        model: Optional[str] = "gpt-4o",
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        super().__init__(api_base=api_base or "http://localhost:11434/v1", api_key=api_key, model=model, temperature=temperature, timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def capabilities(self) -> Dict[str, bool]:
        model = (self.model or "").lower()
        vision = any(name in model for name in (
            "gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "gpt-6", "o1", "o3", "o4"
        ))
        return {
            "image": vision,
            "audio": "audio" in model or "realtime" in model,
            "video": False,
            "document": False,
        }

    async def call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> DriverResponse:
        url = f"{self.api_base}/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if output_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.get("title", "StructuredResponse"),
                    "strict": True,
                    "schema": output_schema,
                },
            }

        session = await self._get_session()
        async with session.post(
            url,
            json=body,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        choices = data.get("choices", [{}])
        choice = choices[0] if choices else {}
        msg = choice.get("message", {})
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls", [])
        usage = data.get("usage", {})

        return DriverResponse(
            text=text,
            raw=data,
            tool_calls=tool_calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", self.model or ""),
        )

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str = "gpt-image-2",
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        output_format: Optional[str] = None,
        background: Optional[str] = None,
        **kwargs,
    ) -> ImageGenerationResponse:
        """Generate images through an OpenAI-compatible Images API."""
        if not prompt.strip():
            raise ValueError("Image prompt must not be empty")
        if n < 1:
            raise ValueError("n must be at least 1")

        body: Dict[str, Any] = {"model": model, "prompt": prompt, "n": n, **kwargs}
        for key, value in {
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "background": background,
        }.items():
            if value is not None:
                body[key] = value

        session = await self._get_session()
        async with session.post(
            f"{self.api_base}/images/generations",
            json=body,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()

        image_format = output_format or "png"
        mime_type = "image/jpeg" if image_format in ("jpg", "jpeg") else f"image/{image_format}"
        images = []
        for item in payload.get("data", []):
            encoded = item.get("b64_json")
            images.append(GeneratedImage(
                data=base64.b64decode(encoded) if encoded else None,
                url=item.get("url"),
                revised_prompt=item.get("revised_prompt"),
                mime_type=mime_type,
            ))
        if not images:
            raise ValueError("Image API returned no images")
        return ImageGenerationResponse(images=images, model=model, raw=payload)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        url = f"{self.api_base}/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        session = await self._get_session()
        async with session.post(
            url,
            json=body,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                payload = decoded[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

