"""Google Gemini REST API driver (zero heavy google-genai dependency)."""

import json
import logging
from typing import AsyncIterator, Dict, Any, List, Optional
import aiohttp

from .base import BaseDriver, DriverResponse

logger = logging.getLogger(__name__)


class GeminiDriver(BaseDriver):
    """Zero-dependency HTTP driver for Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = "gemini-2.5-flash",
        api_base: Optional[str] = "https://generativelanguage.googleapis.com/v1beta",
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        super().__init__(
            api_base=api_base or "https://generativelanguage.googleapis.com/v1beta",
            api_key=api_key,
            model=model or "gemini-2.5-flash",
            temperature=temperature,
            timeout=timeout,
        )

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert OpenAI-style messages into Gemini contents & system instruction."""
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = {"parts": [{"text": str(content)}]}
                continue

            gemini_role = "model" if role in ("assistant", "model") else "user"

            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            mime, b64 = url.split(";base64,")
                            mime_type = mime.replace("data:", "")
                            parts.append({"inline_data": {"mime_type": mime_type, "data": b64}})
                        else:
                            parts.append({"file_data": {"file_uri": url}})
                    elif part.get("type") == "video_url":
                        url = part.get("video_url", {}).get("url", "")
                        parts.append({"file_data": {"file_uri": url}})
                    elif part.get("type") == "audio_url":
                        url = part.get("audio_url", {}).get("url", "")
                        parts.append({"file_data": {"file_uri": url}})

            contents.append({"role": gemini_role, "parts": parts})

        return system_instruction, contents

    async def call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> DriverResponse:
        model_name = self.model or "gemini-2.5-flash"
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        url = f"{self.api_base}/{model_name}:generateContent?key={self.api_key or ''}"
        sys_inst, contents = self._convert_messages(messages)

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
            }
        }
        if sys_inst:
            body["systemInstruction"] = sys_inst
        if output_schema:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = output_schema

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        candidates = data.get("candidates", [{}])
        cand = candidates[0] if candidates else {}
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})

        return DriverResponse(
            text=text,
            raw=data,
            tool_calls=[],
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            model=model_name,
        )

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        model_name = self.model or "gemini-2.5-flash"
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        url = f"{self.api_base}/{model_name}:streamGenerateContent?alt=sse&key={self.api_key or ''}"
        sys_inst, contents = self._convert_messages(messages)

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
            }
        }
        if sys_inst:
            body["systemInstruction"] = sys_inst

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
                    if not decoded or not decoded.startswith("data: "):
                        continue
                    try:
                        chunk = json.loads(decoded[6:])
                        for cand in chunk.get("candidates", []):
                            for p in cand.get("content", {}).get("parts", []):
                                t = p.get("text")
                                if t:
                                    yield t
                    except json.JSONDecodeError:
                        continue
