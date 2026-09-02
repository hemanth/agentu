"""Native Ollama API driver."""

import json
import logging
from typing import AsyncIterator, Dict, Any, List, Optional
import aiohttp

from .base import BaseDriver, DriverResponse

logger = logging.getLogger(__name__)


class OllamaDriver(BaseDriver):
    """Zero-dependency HTTP driver for native Ollama /api/chat."""

    def __init__(
        self,
        api_base: Optional[str] = "http://localhost:11434",
        model: Optional[str] = "llama3",
        temperature: float = 0.7,
        timeout: float = 120.0,
        keep_alive: str = "5m",
    ):
        base = (api_base or "http://localhost:11434").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        super().__init__(
            api_base=base,
            api_key=None,
            model=model or "llama3",
            temperature=temperature,
            timeout=timeout,
        )
        self.keep_alive = keep_alive

    async def call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> DriverResponse:
        url = f"{self.api_base}/api/chat"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            },
        }
        if output_schema:
            body["format"] = "json"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        msg = data.get("message", {})
        text = msg.get("content", "")

        return DriverResponse(
            text=text,
            raw=data,
            tool_calls=[],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            model=data.get("model", self.model or ""),
        )

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        url = f"{self.api_base}/api/chat"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if not decoded:
                        continue
                    try:
                        chunk = json.loads(decoded)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
