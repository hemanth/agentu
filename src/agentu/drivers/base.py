"""Base driver protocol and response structures for LLM adapters."""

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any, List, Optional, Union


@dataclass
class DriverResponse:
    """Standardized response from an LLM driver."""
    text: str
    raw: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@dataclass
class GeneratedImage:
    """One image returned by an image-generation provider."""

    data: Optional[bytes] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None
    mime_type: str = "image/png"

    def save(self, path: Union[str, Path], *, overwrite: bool = False) -> Path:
        """Save an inline image without silently overwriting an existing file."""
        if self.data is None:
            raise ValueError("This generated image contains a URL, not inline image data")
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Image already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.data)
        return destination


@dataclass
class ImageGenerationResponse:
    """Normalized image-generation result."""

    images: List[GeneratedImage]
    model: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


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

    @property
    def capabilities(self) -> Dict[str, bool]:
        """Media inputs this concrete transport can encode safely."""
        return {"image": False, "audio": False, "video": False, "document": False}

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

    async def generate_image(self, prompt: str, **kwargs) -> ImageGenerationResponse:
        """Generate images when supported by the provider."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement image generation"
        )
