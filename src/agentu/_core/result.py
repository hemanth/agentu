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
        *,
        include_text_key: bool = True,
        **extra: Any,
    ):
        base_dict: Dict[str, Any] = {
            "turns": turns,
            "history": history if history is not None else [],
            "tool_calls": tool_calls if tool_calls is not None else [],
            "success": False if error is not None else success,
        }
        if include_text_key:
            base_dict["text_response"] = text
            base_dict["result"] = extra.pop("result", text)
        elif "result" in extra:
            base_dict["result"] = extra.pop("result")
        if structured is not None or include_text_key:
            base_dict["structured"] = structured
        if error is not None:
            base_dict["error"] = error
        if raw:
            base_dict.update(raw)
        base_dict.update(extra)
        super().__init__(base_dict)
        self.raw = raw or {}

    @classmethod
    def from_response(
        cls,
        payload: Any,
        *,
        turns: Optional[int] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> "InferResult[Any]":
        """Normalize any legacy inference dict into an InferResult without altering existing dict keys."""
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return cls(text=str(payload), turns=turns or 1, history=history)

        has_text_response = "text_response" in payload
        text_val = payload.get("text_response")
        error_val = payload.get("error")
        hist = payload.get("history", history if history is not None else [])
        inferred_turns = payload.get("turns", turns if turns is not None else max(1, len(hist)))

        tool_calls_list = payload.get("tool_calls")
        if tool_calls_list is None:
            if hist:
                tool_calls_list = list(hist)
            elif "tool_used" in payload:
                tool_calls_list = [{
                    "turn": 1,
                    "tool_used": payload.get("tool_used"),
                    "parameters": payload.get("parameters", {}),
                    "result": payload.get("result"),
                }]
            else:
                tool_calls_list = []

        extra = {
            k: v
            for k, v in payload.items()
            if k not in ("text_response", "structured", "turns", "history", "tool_calls", "success", "error")
        }
        return cls(
            text=str(text_val) if text_val is not None else "",
            structured=payload.get("structured"),
            turns=inferred_turns,
            history=hist,
            tool_calls=tool_calls_list,
            success=error_val is None and payload.get("success", True),
            error=error_val,
            include_text_key=has_text_response,
            **extra,
        )

    @property
    def text(self) -> str:
        val = super().get("text_response")
        if val is not None:
            return str(val)
        res = super().get("result")
        if res is not None:
            return str(res)
        return ""

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
        return super().get("success", "error" not in self)

    @success.setter
    def success(self, value: bool) -> None:
        super().__setitem__("success", value)

    @property
    def error(self) -> Optional[str]:
        return super().get("error")

    @error.setter
    def error(self, value: Optional[str]) -> None:
        if value is None:
            if super().__contains__("error"):
                super().__delitem__("error")
        else:
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
        elif key == "error" and value is None:
            if super().__contains__("error"):
                super().__delitem__("error")
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
        if self.error is not None:
            return f"InferResult(error={self.error!r}, turns={self.turns})"
        return f"InferResult(text={self.text!r}, turns={self.turns})"
