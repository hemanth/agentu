"""Anthropic Claude Messages API driver (zero heavy anthropic SDK dependency)."""

import json
import logging
from typing import AsyncIterator, Dict, Any, List, Optional
from .._core.http_compat import aiohttp

from .base import BaseDriver, DriverResponse

logger = logging.getLogger(__name__)


class ClaudeDriver(BaseDriver):
    """Zero-dependency HTTP driver for Anthropic Claude /v1/messages API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = "claude-3-7-sonnet-20250219",
        api_base: Optional[str] = "https://api.anthropic.com/v1",
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        super().__init__(
            api_base=api_base or "https://api.anthropic.com/v1",
            api_key=api_key,
            model=model or "claude-3-7-sonnet-20250219",
            temperature=temperature,
            timeout=timeout,
        )

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {"image": True, "audio": False, "video": False, "document": True}

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert OpenAI-style messages into Anthropic system + user/assistant messages."""
        system_text = None
        claude_msgs = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_text = str(content)
                continue

            if role == "tool":
                claude_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": str(content),
                    }],
                })
                continue

            claude_role = "assistant" if role in ("assistant", "model") else "user"

            if isinstance(content, str):
                if msg.get("tool_calls"):
                    parts = [{"type": "text", "text": content}] if content else []
                    parts.extend(self._convert_tool_uses(msg["tool_calls"]))
                    claude_msgs.append({"role": claude_role, "content": parts})
                else:
                    claude_msgs.append({"role": claude_role, "content": content})
            elif isinstance(content, list):
                parts = []
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif ptype in ("image_url", "document"):
                        block_type = "image" if ptype == "image_url" else "document"
                        payload = part.get(ptype, {})
                        url = payload.get("url", "")
                        extra = {k: v for k, v in payload.items() if k not in ("url", "mime_type")}
                        if url.startswith("data:"):
                            mime_type, b64 = self._parse_data_uri(url)
                            block: Dict[str, Any] = {
                                "type": block_type,
                                "source": {"type": "base64", "media_type": mime_type, "data": b64},
                            }
                        elif url:
                            block = {
                                "type": block_type,
                                "source": {"type": "url", "url": url},
                            }
                        else:
                            continue
                        if extra:
                            block.update(extra)
                        parts.append(block)
                claude_msgs.append({"role": claude_role, "content": parts})

        return system_text, claude_msgs

    @staticmethod
    def _convert_tool_uses(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        blocks = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            blocks.append({
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": function.get("name", ""),
                "input": arguments,
            })
        return blocks

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted = []
        for tool in tools:
            function = tool.get("function", tool)
            converted.append({
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            })
        return converted

    @staticmethod
    def _parse_tool_calls(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            }
            for block in content
            if block.get("type") == "tool_use"
        ]

    async def call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> DriverResponse:
        url = f"{self.api_base}/messages"
        sys_text, claude_msgs = self._convert_messages(messages)

        body: Dict[str, Any] = {
            "model": self.model or "claude-3-7-sonnet-20250219",
            "messages": claude_msgs,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "stream": stream,
        }
        if sys_text:
            body["system"] = sys_text
        if tools:
            body["tools"] = self._convert_tools(tools)

        session = await self._get_session()
        async with session.post(
            url,
            json=body,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        text_parts = []
        for c in data.get("content", []):
            if c.get("type") == "text":
                text_parts.append(c.get("text", ""))

        text = "".join(text_parts)
        usage = data.get("usage", {})

        return DriverResponse(
            text=text,
            raw=data,
            tool_calls=self._parse_tool_calls(data.get("content", [])),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            model=data.get("model", self.model or ""),
        )

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        url = f"{self.api_base}/messages"
        sys_text, claude_msgs = self._convert_messages(messages)

        body: Dict[str, Any] = {
            "model": self.model or "claude-3-7-sonnet-20250219",
            "messages": claude_msgs,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "stream": True,
        }
        if sys_text:
            body["system"] = sys_text
        if tools:
            body["tools"] = self._convert_tools(tools)

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
                try:
                    chunk = json.loads(decoded[6:])
                    t = chunk.get("type")
                    if t == "content_block_delta":
                        delta = chunk.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text")
                            if text:
                                yield text
                except json.JSONDecodeError:
                    continue

