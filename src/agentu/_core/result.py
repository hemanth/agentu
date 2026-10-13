"""InferResult[T] - Fully typed, dictionary-compatible response object."""

from typing import Generic, TypeVar, Optional, Any, Dict, List

T = TypeVar("T")


class InferResult(dict, Generic[T]):
    """Result from agent inference with typed structured data and backward-compatible dict access."""

    def __init__(
        self,
        text: str = "",
        structured: Optional[T] = None,
        turns: int = 1,
        history: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        success: bool = True,
        error: Optional[str] = None,
        raw: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ):
        base_dict: Dict[str, Any] = {
            "text_response": text,
            "result": extra.pop("result", text),
            "structured": structured,
            "turns": turns,
            "history": history if history is not None else [],
            "tool_calls": tool_calls if tool_calls is not None else [],
            "success": success,
            "error": error,
        }
        if raw:
            base_dict.update(raw)
        base_dict.update(extra)
        super().__init__(base_dict)
        self.raw = raw or {}

    @property
    def text(self) -> str:
        val = super().get("text_response")
        if val is not None:
            return val
        return str(super().get("result") or "")

    @text.setter
    def text(self, value: str) -> None:
        super().__setitem__("text_response", value)
        super().__setitem__("result", value)

    @property
    def structured(self) -> Optional[T]:
        return super().get("structured")

    @structured.setter
    def structured(self, value: Optional[T]) -> None:
        super().__setitem__("structured", value)

    @property
    def data(self) -> Optional[T]:
        return super().get("structured")

    @data.setter
    def data(self, value: Optional[T]) -> None:
        super().__setitem__("structured", value)

    @property
    def turns(self) -> int:
        return super().get("turns", 1)

    @turns.setter
    def turns(self, value: int) -> None:
        super().__setitem__("turns", value)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return super().get("history", [])

    @history.setter
    def history(self, value: List[Dict[str, Any]]) -> None:
        super().__setitem__("history", value)

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        return super().get("tool_calls", [])

    @tool_calls.setter
    def tool_calls(self, value: List[Dict[str, Any]]) -> None:
        super().__setitem__("tool_calls", value)

    @property
    def success(self) -> bool:
        return super().get("success", True)

    @success.setter
    def success(self, value: bool) -> None:
        super().__setitem__("success", value)

    @property
    def error(self) -> Optional[str]:
        return super().get("error")

    @error.setter
    def error(self, value: Optional[str]) -> None:
        super().__setitem__("error", value)

    def __getitem__(self, key: str) -> Any:
        if key == "text":
            return self.text
        if key == "data":
            return self.structured
        if key == "raw":
            return self.raw
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self.raw:
            return self.raw[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in ("text", "text_response"):
            super().__setitem__("text_response", value)
        elif key in ("structured", "data"):
            super().__setitem__("structured", value)
        else:
            super().__setitem__(key, value)

    def __contains__(self, key: object) -> bool:
        if key in ("text", "data", "raw"):
            return True
        return super().__contains__(key) or key in self.raw

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        if self.structured is not None:
            return f"InferResult(data={self.structured!r}, turns={self.turns})"
        return f"InferResult(text={self.text!r}, turns={self.turns})"

