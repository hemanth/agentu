"""Zero-dependency LLM protocol drivers."""

import os
import re
from typing import Optional, Tuple
from .base import BaseDriver, DriverResponse, GeneratedImage, ImageGenerationResponse
from .openai_driver import OpenAIDriver
from .gemini_driver import GeminiDriver
from .claude_driver import ClaudeDriver
from .ollama_driver import OllamaDriver

_OPENAI_MODEL_RE = re.compile(r"^(gpt-|chatgpt-|o1(?:-|$)|o3(?:-|$)|o4(?:-|$))", re.IGNORECASE)
_PROVIDER_PREFIXES = ("openai/", "gemini/", "google/", "anthropic/", "claude/", "deepseek/", "groq/", "ollama/")


def _split_provider_prefix(model: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split an optional 'provider/model' prefix into (provider, clean_model)."""
    if not model:
        return None, model
    lower = model.lower()
    for prefix in _PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            provider = prefix.rstrip("/")
            if provider == "google":
                provider = "gemini"
            elif provider == "anthropic":
                provider = "claude"
            return provider, model[len(prefix):]
    return None, model


def is_auto_routed_model(model: Optional[str]) -> bool:
    """Return True when a model identifier unambiguously targets a native or cloud provider."""
    if not model:
        return False
    provider, clean = _split_provider_prefix(model)
    if provider is not None:
        return True
    m = (clean or "").lower()
    return (
        m.startswith("gemini")
        or m.startswith("claude")
        or m.startswith("deepseek")
        or bool(_OPENAI_MODEL_RE.match(m))
    )


def detect_env_default_model() -> Optional[str]:
    """Detect a sensible default model from environment variables when none is specified."""
    explicit = os.environ.get("AGENTU_MODEL")
    if explicit:
        return explicit
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini-2.5-flash"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-20250514"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek-chat"
    if os.environ.get("GROQ_API_KEY"):
        return "groq/llama-3.3-70b-versatile"
    return None


def get_driver_for_model(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> BaseDriver:
    """Auto-detect and instantiate the optimal lightweight driver for a model.

    Supports both bare model names ('gemini-2.5-flash', 'claude-sonnet-4-20250514',
    'gpt-4o', 'deepseek-chat') and 'provider/model' specs ('openai/gpt-4o',
    'groq/llama-3.3-70b-versatile', 'ollama/qwen3:latest').

    Args:
        model: Model identifier
        api_base: Custom base URL if any
        api_key: API key if any (or loaded from environment)
        temperature: Generation temperature
        timeout: Network timeout in seconds

    Returns:
        Configured BaseDriver instance
    """
    provider, clean_model = _split_provider_prefix(model)
    m = (clean_model or "").lower()
    base = api_base or ""

    # Gemini
    if provider == "gemini" or "gemini" in m:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        resolved_base = api_base or os.environ.get("GEMINI_BASE_URL")
        return GeminiDriver(
            api_key=key,
            model=clean_model,
            api_base=resolved_base,
            temperature=temperature,
            timeout=timeout,
        )

    # Claude / Anthropic
    if provider == "claude" or "claude" in m:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        resolved_base = api_base or os.environ.get("ANTHROPIC_BASE_URL")
        return ClaudeDriver(
            api_key=key,
            model=clean_model,
            api_base=resolved_base,
            temperature=temperature,
            timeout=timeout,
        )

    # Native Ollama API endpoint
    if provider == "ollama" or (base and ":11434" in base and not base.rstrip("/").endswith("/v1")):
        return OllamaDriver(
            api_base=api_base or "http://localhost:11434",
            model=clean_model,
            temperature=temperature,
            timeout=timeout,
        )

    # Groq (OpenAI-compatible)
    if provider == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY")
        return OpenAIDriver(
            api_base=api_base or "https://api.groq.com/openai/v1",
            api_key=key,
            model=clean_model,
            temperature=temperature,
            timeout=timeout,
        )

    # DeepSeek (OpenAI-compatible)
    if provider == "deepseek" or m.startswith("deepseek"):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return OpenAIDriver(
            api_base=api_base or "https://api.deepseek.com/v1",
            api_key=key,
            model=clean_model,
            temperature=temperature,
            timeout=timeout,
        )

    # OpenAI cloud models
    if provider == "openai" or _OPENAI_MODEL_RE.match(m):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base = api_base or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return OpenAIDriver(
            api_base=resolved_base,
            api_key=key,
            model=clean_model,
            temperature=temperature,
            timeout=timeout,
        )

    # Default: OpenAI-compatible endpoint (Ollama /v1, vLLM, LM Studio, custom gateway)
    key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    return OpenAIDriver(
        api_base=api_base or "http://localhost:11434/v1",
        api_key=key,
        model=clean_model,
        temperature=temperature,
        timeout=timeout,
    )


__all__ = [
    "BaseDriver", "DriverResponse", "GeneratedImage", "ImageGenerationResponse",
    "OpenAIDriver",
    "GeminiDriver",
    "ClaudeDriver",
    "OllamaDriver",
    "get_driver_for_model",
    "is_auto_routed_model",
    "detect_env_default_model",
]
