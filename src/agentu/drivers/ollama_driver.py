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

    @property
    def capabilities(self) -> Dict[str, bool]:
        model = (self.model or "").lower()
        vision = any(name in model for name in (
            "llava", "vision", "qwen-vl", "qwen2.5-vl", "minicpm", "moondream", "bakllava"
        ))
        return {"image": vision, "audio": False, "video": False, "document": False}

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool":
                normalized.append({
                    "role": "tool",
                    "content": str(msg.get("content", "")),
                })
                continue
            if msg.get("tool_calls"):
                calls = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
                normalized.append({
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content") or "",
                    "tool_calls": calls,
                })
                continue
            normalized.append(msg)
        return normalized

    @staticmethod
    def _parse_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        for idx, tc in enumerate(raw_calls or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args)
            parsed.append({
                "id": tc.get("id") or f"ollama_call_{idx}",
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": args,
                },
            })
        return parsed

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
            "messages": self._normalize_messages(messages),
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            },
        }
        if tools:
            body["tools"] = tools
        if output_schema:
            body["format"] = output_schema if isinstance(output_schema, dict) else "json"

        session = await self._get_session()
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
        tool_calls = self._parse_tool_calls(msg.get("tool_calls", []))

        return DriverResponse(
            text=text,
            raw=data,
            tool_calls=tool_calls,
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

        session = await self._get_session()
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

