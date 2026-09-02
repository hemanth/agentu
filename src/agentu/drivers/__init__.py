"""Zero-dependency LLM protocol drivers."""

import os
from typing import Optional
from .base import BaseDriver, DriverResponse
from .openai_driver import OpenAIDriver
from .gemini_driver import GeminiDriver
from .claude_driver import ClaudeDriver
from .ollama_driver import OllamaDriver


def get_driver_for_model(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> BaseDriver:
    """Auto-detect and instantiate the optimal lightweight driver for a model.

    Args:
        model: Model identifier (e.g. 'gemini-2.5-flash', 'claude-3-7-sonnet', 'gpt-4o', 'llama3')
        api_base: Custom base URL if any
        api_key: API key if any (or loaded from environment)
        temperature: Generation temperature
        timeout: Network timeout in seconds

    Returns:
        Configured BaseDriver instance
    """
    m = (model or "").lower()
    base = api_base or ""

    # Gemini
    if "gemini" in m:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return GeminiDriver(api_key=key, model=model, api_base=api_base, temperature=temperature, timeout=timeout)

    # Claude / Anthropic
    if "claude" in m:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return ClaudeDriver(api_key=key, model=model, api_base=api_base, temperature=temperature, timeout=timeout)

    # Native Ollama API endpoint
    if base and ":11434" in base and not base.rstrip("/").endswith("/v1"):
        return OllamaDriver(api_base=base, model=model, temperature=temperature, timeout=timeout)

    # Default: OpenAI-compatible endpoint (OpenAI, DeepSeek, Groq, vLLM, Ollama-v1, Azure)
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("GROQ_API_KEY")
    return OpenAIDriver(api_base=api_base or "http://localhost:11434/v1", api_key=key, model=model, temperature=temperature, timeout=timeout)


__all__ = [
    "BaseDriver",
    "DriverResponse",
    "OpenAIDriver",
    "GeminiDriver",
    "ClaudeDriver",
    "OllamaDriver",
    "get_driver_for_model",
]
