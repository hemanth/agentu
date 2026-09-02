"""InferResult[T] - Fully typed, dictionary-compatible response object."""

from typing import Generic, TypeVar, Optional, Any, Dict, List, Iterator

T = TypeVar("T")


class InferResult(Generic[T]):
    """Result from agent inference with typed structured data and backward-compatible dict access."""

    def __init__(
        self,
        text: str,
        structured: Optional[T] = None,
        turns: int = 1,
        history: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        success: bool = True,
        error: Optional[str] = None,
        raw: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.structured = structured
        self.data = structured  # Alias for structured
        self.turns = turns
        self.history = history or []
        self.tool_calls = tool_calls or []
        self.success = success
        self.error = error
        self.raw = raw or {}

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for 100% backward compatibility."""
        if key in ("text", "text_response"):
            return self.text
        if key in ("structured", "data"):
            return self.structured
        if key == "turns":
            return self.turns
        if key == "history":
            return self.history
        if key == "tool_calls":
            return self.tool_calls
        if key == "success":
            return self.success
        if key == "error":
            return self.error
        if key == "raw":
            return self.raw
        if key in self.raw:
            return self.raw[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in ("text", "text_response"):
            self.text = value
        elif key in ("structured", "data"):
            self.structured = value
            self.data = value
        elif key == "turns":
            self.turns = value
        elif key == "success":
            self.success = value
        else:
            self.raw[key] = value

    def __contains__(self, key: str) -> bool:
        if key in ("text", "text_response", "structured", "data", "turns", "history", "tool_calls", "success", "error", "raw"):
            return True
        return key in self.raw

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        return ["text_response", "structured", "turns", "history", "tool_calls", "success", "error"]

    def items(self) -> List[tuple]:
        return [(k, self.get(k)) for k in self.keys()]

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        if self.structured is not None:
            return f"InferResult(data={self.structured!r}, turns={self.turns})"
        return f"InferResult(text={self.text!r}, turns={self.turns})"
