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

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {"image": True, "audio": True, "video": True, "document": True}

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        return headers

    @staticmethod
    def _build_media_part(payload: Dict[str, Any], parse_data_uri) -> Dict[str, Any]:
        """Build a Gemini inline_data or file_data part while preserving extra metadata."""
        url = payload.get("url", "")
        extra = {k: v for k, v in payload.items() if k not in ("url", "mime_type")}
        mime_type = payload.get("mime_type")

        if url.startswith("data:"):
            parsed_mime, b64 = parse_data_uri(url)
            part: Dict[str, Any] = {
                "inline_data": {
                    "mime_type": mime_type or parsed_mime,
                    "data": b64,
                }
            }
        else:
            file_data: Dict[str, Any] = {"file_uri": url}
            if mime_type:
                file_data["mime_type"] = mime_type
            part = {"file_data": file_data}

        if extra:
            part.update(extra)
        return part

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert OpenAI-style messages into Gemini contents & system instruction."""
        system_instruction = None
        contents = []
        tool_names: Dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "tool":
                call_id = msg.get("tool_call_id", "")
                name = msg.get("name") or tool_names.get(call_id, "unknown_tool")
                contents.append({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "id": call_id,
                            "name": name,
                            "response": {"result": content},
                        }
                    }],
                })
                continue

            if role == "system":
                system_instruction = {"parts": [{"text": str(content)}]}
                continue

            gemini_role = "model" if role in ("assistant", "model") else "user"

            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        parts.append({"text": part.get("text", "")})
                    elif ptype in ("image_url", "video_url", "audio_url", "document"):
                        payload = part.get(ptype, {})
                        parts.append(self._build_media_part(payload, self._parse_data_uri))

            for tool_call in msg.get("tool_calls", []):
                function = tool_call.get("function", {})
                call_id = tool_call.get("id", "")
                name = function.get("name", "")
                tool_names[call_id] = name
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                parts.append({
                    "functionCall": {"id": call_id, "name": name, "args": arguments}
                })

            contents.append({"role": gemini_role, "parts": parts})

        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        declarations = []
        for tool in tools:
            function = tool.get("function", tool)
            declarations.append({
                "name": function["name"],
                "description": function.get("description", ""),
                "parametersJsonSchema": function.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            })
        return [{"functionDeclarations": declarations}]

    @staticmethod
    def _parse_tool_calls(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        calls = []
        for index, part in enumerate(parts):
            function_call = part.get("functionCall")
            if not function_call:
                continue
            calls.append({
                "id": function_call.get("id") or f"gemini_call_{index}",
                "type": "function",
                "function": {
                    "name": function_call.get("name", ""),
                    "arguments": json.dumps(function_call.get("args", {})),
                },
            })
        return calls

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

        url = f"{self.api_base}/{model_name}:generateContent"
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

        candidates = data.get("candidates", [{}])
        cand = candidates[0] if candidates else {}
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        tool_calls = self._parse_tool_calls(parts)
        usage = data.get("usageMetadata", {})

        return DriverResponse(
            text=text,
            raw=data,
            tool_calls=tool_calls,
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

        url = f"{self.api_base}/{model_name}:streamGenerateContent?alt=sse"
        sys_inst, contents = self._convert_messages(messages)

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
            }
        }
        if sys_inst:
            body["systemInstruction"] = sys_inst
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
                    for cand in chunk.get("candidates", []):
                        for p in cand.get("content", {}).get("parts", []):
                            t = p.get("text")
                            if t:
                                yield t
                except json.JSONDecodeError:
                    continue

