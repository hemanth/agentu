"""Base driver protocol and message structures for zero-dependency LLM adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any, List, Optional


@dataclass
class DriverResponse:
    """Standardized response from an LLM driver."""
    text: str
    raw: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class BaseDriver(ABC):
    """Abstract base class for zero-dependency LLM protocol drivers."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        self.api_base = api_base.rstrip('/') if api_base else None
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    @abstractmethod
    async def call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> DriverResponse:
        """Execute a completion call against the provider."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion text tokens from the provider."""
        pass
