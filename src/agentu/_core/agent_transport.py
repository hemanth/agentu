"""LLM transport primitives shared by the Agent orchestration layer."""

import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import urljoin
from typing import Any, AsyncIterator, Dict, List, Optional, Type, Union

import aiohttp

from .multimodal import build_content_parts, detect_model_capabilities
from .structured import build_response_format
from ..middleware.observe import EventType
from ..drivers.base import GeneratedImage, ImageGenerationResponse
from ..drivers.openai_driver import OpenAIDriver

logger = logging.getLogger(__name__)


def _openai_compatible_capabilities(model: Optional[str]) -> Dict[str, bool]:
    """Limit model capabilities to parts supported by this transport."""
    model_caps = detect_model_capabilities(model)
    return {
        "image": model_caps["image"],
        "audio": model_caps["audio"],
        "video": False,
        "document": False,
    }


class TransportMixin:
    """Provider dispatch, HTTP session management, retries, and SSE parsing."""

    async def _get_llm_session(self) -> aiohttp.ClientSession:
        if self._llm_session is None or self._llm_session.closed:
            self._llm_session = aiohttp.ClientSession()
        return self._llm_session

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str = "gpt-image-2",
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> GeneratedImage:
        """Generate one image and optionally save it in the same call."""
        requested_count = kwargs.pop("n", 1)
        if requested_count != 1:
            raise ValueError("generate_image() creates one image; use generate_images() for batches")
        result = await self.generate_images(prompt, model=model, n=1, **kwargs)
        image = result.images[0]
        if output_path is not None:
            destination = Path(output_path)
            if image.data is not None:
                image.save(destination, overwrite=overwrite)
            elif image.url:
                if destination.exists() and not overwrite:
                    raise FileExistsError(f"Image already exists: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                image_url = urljoin(f"{self.api_base}/", image.url)
                session = await self._get_llm_session()
                async with session.get(image_url) as response:
                    response.raise_for_status()
                    destination.write_bytes(await response.read())
            else:
                raise ValueError("Generated image contains neither inline data nor a URL")
        return image

    async def generate_images(
        self,
        prompt: str,
        *,
        model: str = "gpt-image-2",
        n: int = 1,
        **kwargs,
    ) -> ImageGenerationResponse:
        """Generate a batch of images using the configured provider transport."""
        driver = getattr(self, "driver", None)
        if driver is None:
            driver = OpenAIDriver(
                api_base=self.api_base,
                api_key=self.api_key,
                model=model,
                timeout=120.0,
            )
        return await driver.generate_image(prompt, model=model, n=n, **kwargs)

    async def _raw_llm_call(
        self,
        prompt: str,
        output_schema: Optional[Type] = None,
        images: Optional[List[str]] = None,
        media: Optional[List[Union[str, Dict[str, Any]]]] = None,
        max_retries: int = 2,
    ) -> str:
        """Call the configured driver or OpenAI-compatible HTTP endpoint."""
        retryable_statuses = {429, 500, 502, 503, 504}

        with self.observer.trace(
            EventType.LLM_REQUEST,
            {"model": self.model, "prompt_length": len(prompt)},
        ):
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            driver = getattr(self, "driver", None)
            content = build_content_parts(
                prompt,
                images=images,
                media=media,
                model=self.model,
                custom_converter=getattr(self, "_media_converter", None),
                capabilities=(
                    driver.capabilities
                    if driver
                    else _openai_compatible_capabilities(self.model)
                ),
            )
            messages = [{"role": "user", "content": content}]

            if driver is not None:
                driver_schema = output_schema
                if output_schema is not None and not isinstance(output_schema, dict):
                    driver_schema = build_response_format(output_schema)["json_schema"]["schema"]

                last_error = None
                for attempt in range(max_retries + 1):
                    try:
                        response = await driver.call(messages, output_schema=driver_schema)
                        if not response.text:
                            raise ValueError("Empty response from API")
                        return response.text
                    except aiohttp.ClientError as exc:
                        last_error = exc
                        if attempt == max_retries:
                            raise
                        await asyncio.sleep(2 ** attempt)
                raise last_error or RuntimeError("LLM driver call failed after retries")

            body: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": False,
            }
            if output_schema is not None:
                body["response_format"] = build_response_format(output_schema)

            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    session = await self._get_llm_session()
                    async with session.post(
                        f"{self.api_base}/chat/completions",
                        json=body,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status in retryable_statuses and attempt < max_retries:
                            delay = float(2 ** attempt)
                            logger.warning(
                                "LLM API returned %s, retrying in %ss (attempt %s/%s)",
                                response.status,
                                delay,
                                attempt + 1,
                                max_retries,
                            )
                            await asyncio.sleep(delay)
                            continue

                        response.raise_for_status()
                        payload = await response.json()
                        if "error" in payload:
                            raise RuntimeError(payload["error"])
                        text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if not text:
                            raise ValueError("Empty response from API")
                        return text
                except aiohttp.ClientError as exc:
                    last_error = exc
                    if attempt == max_retries:
                        raise
                    delay = float(2 ** attempt)
                    logger.warning(
                        "LLM API connection error: %s, retrying in %ss (attempt %s/%s)",
                        exc,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)

            raise last_error or RuntimeError("LLM call failed after retries")

    async def _stream_llm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        media: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> AsyncIterator[str]:
        """Stream text chunks from a provider driver or OpenAI-compatible SSE."""
        with self.observer.trace(
            EventType.LLM_REQUEST,
            {"model": self.model, "prompt_length": len(prompt), "streaming": True},
        ):
            try:
                driver = getattr(self, "driver", None)
                capabilities = (
                    driver.capabilities
                    if driver
                    else _openai_compatible_capabilities(self.model)
                )
                content = build_content_parts(
                    prompt,
                    images=images,
                    media=media,
                    model=self.model,
                    custom_converter=getattr(self, "_media_converter", None),
                    capabilities=capabilities,
                )
                messages = [{"role": "user", "content": content}]

                if driver is not None:
                    async for chunk in driver.stream(messages):
                        yield chunk
                    return

                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                session = await self._get_llm_session()
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "stream": True,
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data: "):
                            continue
                        data = decoded[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                        if text:
                            yield text
            except aiohttp.ClientError as exc:
                logger.error("Error streaming LLM API: %s", exc)
                raise
