"""OpenAI-compatible protocol driver (OpenAI, DeepSeek, Groq, vLLM, Ollama-v1)."""

import json
import logging
from typing import AsyncIterator, Dict, Any, List, Optional
import aiohttp

from .base import BaseDriver, DriverResponse

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

        async with aiohttp.ClientSession() as session:
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

        async with aiohttp.ClientSession() as session:
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
