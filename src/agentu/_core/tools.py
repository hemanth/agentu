"""Tool utilities, JSON Schema inference, and @tool decorator for agentu."""

import inspect
import re
from enum import Enum
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)


class ToolPermission(Enum):
    """Permission level for tool execution.

    READONLY: Safe, no side effects. Auto-approved.
    WRITE: Has side effects but allowed by default. Logged with extra scrutiny.
    DANGEROUS: Blocked unless explicitly allowed via with_permissions().
    """
    READONLY = "readonly"
    WRITE = "write"
    DANGEROUS = "dangerous"


_DOC_ARG_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.+)$")
_SPHINX_PARAM_RE = re.compile(r"^\s*:param\s+(?:\w+\s+)?([a-zA-Z_]\w*)\s*:\s*(.+)$")


def _extract_param_descriptions_from_docstring(doc: Optional[str]) -> Dict[str, str]:
    """Extract per-parameter descriptions from Google-style Args: or Sphinx :param: docstrings."""
    if not doc:
        return {}

    descriptions: Dict[str, str] = {}
    in_args = False
    args_indent: Optional[int] = None

    for raw_line in doc.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        sphinx = _SPHINX_PARAM_RE.match(raw_line)
        if sphinx:
            descriptions[sphinx.group(1)] = sphinx.group(2).strip()
            continue

        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            args_indent = None
            continue

        if in_args:
            leading = len(raw_line) - len(raw_line.lstrip())
            if stripped.endswith(":") and not _DOC_ARG_RE.match(raw_line):
                in_args = False
                continue
            if args_indent is None:
                args_indent = leading
            elif leading < args_indent:
                in_args = False
                continue

            match = _DOC_ARG_RE.match(raw_line)
            if match and leading == args_indent:
                descriptions[match.group(1)] = match.group(2).strip()

    return descriptions


def _unwrap_annotated(annotation: Any) -> tuple[Any, Optional[str]]:
    """Return (underlying_type, description) if annotation is typing.Annotated."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        desc: Optional[str] = None
        for meta in args[1:]:
            if isinstance(meta, str):
                desc = meta
                break
            meta_desc = getattr(meta, "description", None)
            if isinstance(meta_desc, str):
                desc = meta_desc
                break
        return base_type, desc
    return annotation, None


def _python_type_to_json_schema(annotation: Any) -> Dict[str, Any]:
    """Convert a Python type hint into a clean JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    annotation, annotated_desc = _unwrap_annotated(annotation)
    schema: Dict[str, Any]

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T] / Union[T, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema = _python_type_to_json_schema(non_none[0])
        else:
            schema = {"anyOf": [_python_type_to_json_schema(a) for a in non_none]}
    elif origin is Literal:
        values = list(args)
        val_types = {type(v) for v in values}
        json_type = "string"
        if val_types == {int}:
            json_type = "integer"
        elif val_types <= {int, float}:
            json_type = "number"
        elif val_types == {bool}:
            json_type = "boolean"
        schema = {"type": json_type, "enum": values}
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        values = [e.value for e in annotation]
        schema = {"type": "string", "enum": values}
    elif annotation is str:
        schema = {"type": "string"}
    elif annotation is bool:
        schema = {"type": "boolean"}
    elif annotation is int:
        schema = {"type": "integer"}
    elif annotation is float:
        schema = {"type": "number"}
    elif origin in (list, List, tuple, set, frozenset) or annotation in (list, tuple, set):
        item_schema = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        schema = {"type": "array", "items": item_schema}
    elif origin in (dict, Dict) or annotation is dict:
        schema = {"type": "object"}
    else:
        # Delegate Pydantic BaseModel, dataclasses, TypedDicts to pydantic.TypeAdapter
        try:
            from pydantic import TypeAdapter
            schema = TypeAdapter(annotation).json_schema()
            schema.pop("title", None)
        except Exception:
            schema = {"type": "string"}

    if annotated_desc and "description" not in schema:
        schema = {**schema, "description": annotated_desc}
    return schema


class Tool:
    """A tool that can be used by an agent.

    Auto-infers name, description, parameters, and full JSON Schema from the function.

    Examples:
        >>> def add(x: int, y: int) -> int:
        ...     '''Add two numbers.'''
        ...     return x + y
        >>>
        >>> t = Tool(add)  # Everything auto-inferred!
        >>>
        >>> # Or via @tool decorator:
        >>> @tool(readonly=True)
        ... def search(query: str) -> list[str]:
        ...     '''Search records.'''
        ...     return []
    """

    def __init__(
        self,
        function: Callable,
        description: Optional[str] = None,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        permission: ToolPermission = ToolPermission.WRITE,
        reads_private: bool = False,
        ingests_untrusted: bool = False,
        communicates_externally: bool = False,
        schema: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a Tool.

        Args:
            function: The function to wrap
            description: Optional description (uses docstring if not provided)
            name: Optional name (uses function name if not provided)
            parameters: Optional parameters dict (auto-inferred if not provided)
            permission: Permission level (default: WRITE)
            reads_private: Tool can access private/sensitive data (default: False)
            ingests_untrusted: Tool processes data from untrusted sources (default: False)
            communicates_externally: Tool can send data outside the system (default: False)
            schema: Optional explicit JSON Schema dict for parameters
        """
        self.function = function
        self.name = name or function.__name__
        self.description = description or self._extract_description(function)
        self.parameters = parameters if parameters is not None else self._extract_parameters(function)
        self.permission = permission
        self.reads_private = reads_private
        self.ingests_untrusted = ingests_untrusted
        self.communicates_externally = communicates_externally
        self._explicit_schema = schema

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling a Tool instance directly like the wrapped function."""
        return self.function(*args, **kwargs)

    @staticmethod
    def _extract_description(function: Callable) -> str:
        """Extract summary description from function docstring."""
        doc = function.__doc__
        if not doc:
            return f"Execute {function.__name__}"

        lines = doc.strip().split("\n")
        return lines[0].strip()

    @staticmethod
    def _extract_parameters(function: Callable) -> Dict[str, str]:
        """Extract human-readable parameter hints from function signature."""
        sig = inspect.signature(function)
        params: Dict[str, str] = {}

        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param_name in ("self", "cls"):
                continue

            ann = param.annotation
            ann, _ = _unwrap_annotated(ann)
            if ann != inspect.Parameter.empty:
                type_name = getattr(ann, "__name__", str(ann))
            else:
                type_name = "any"

            if param.default != inspect.Parameter.empty:
                params[param_name] = f"{type_name}: (default: {param.default})"
            else:
                params[param_name] = f"{type_name}"

        return params

    def to_json_schema(self) -> Dict[str, Any]:
        """Build a standard JSON Schema object describing this tool's input parameters."""
        if self._explicit_schema and isinstance(self._explicit_schema, dict):
            if "type" in self._explicit_schema or "properties" in self._explicit_schema:
                schema = dict(self._explicit_schema)
                schema.setdefault("type", "object")
                schema.setdefault("properties", {})
                return schema

        # If self.parameters was already supplied as a JSON Schema dict
        if isinstance(self.parameters, dict) and (
            self.parameters.get("type") == "object" and isinstance(self.parameters.get("properties"), dict)
        ):
            return dict(self.parameters)

        try:
            sig = inspect.signature(self.function)
        except (TypeError, ValueError):
            sig = None

        try:
            type_hints = get_type_hints(self.function, include_extras=True)
        except Exception:
            type_hints = {}

        doc_descriptions = _extract_param_descriptions_from_docstring(getattr(self.function, "__doc__", None))

        properties: Dict[str, Any] = {}
        required: List[str] = []

        has_var_kw_only = False
        if sig is not None:
            non_var_params = [
                (name, p) for name, p in sig.parameters.items()
                if name not in ("self", "cls")
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            has_var_kw_only = (
                not non_var_params
                and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            )
            for param_name, param in non_var_params:
                hint = type_hints.get(param_name, param.annotation)
                prop_schema = _python_type_to_json_schema(hint)
                if param_name in doc_descriptions and "description" not in prop_schema:
                    prop_schema["description"] = doc_descriptions[param_name]
                if param.default is not inspect.Parameter.empty:
                    if isinstance(param.default, (str, int, float, bool, list, dict)) or param.default is None:
                        prop_schema["default"] = param.default
                else:
                    # Optional[T] without explicit `= None` is still required unless default is given
                    required.append(param_name)
                properties[param_name] = prop_schema

        # Fallback for dynamically wrapped callables (e.g. **kwargs wrappers with manual self.parameters)
        if (sig is None or has_var_kw_only) and isinstance(self.parameters, dict) and self.parameters:
            _TYPE_MAP = {
                "str": "string",
                "string": "string",
                "int": "integer",
                "integer": "integer",
                "float": "number",
                "number": "number",
                "bool": "boolean",
                "boolean": "boolean",
                "list": "array",
                "array": "array",
                "dict": "object",
                "object": "object",
            }
            for param_name, spec in self.parameters.items():
                spec_str = str(spec)
                raw_type = spec_str.split(":", 1)[0].strip().lower()
                desc_part = spec_str.split(":", 1)[1].strip() if ":" in spec_str else ""
                prop: Dict[str, Any] = {"type": _TYPE_MAP.get(raw_type, "string")}
                if desc_part:
                    prop["description"] = desc_part
                properties[param_name] = prop
                if "(default:" not in spec_str and "(optional)" not in spec_str:
                    required.append(param_name)

        result: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            result["required"] = required
        return result

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI/standard function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }


@overload
def tool(func: Callable) -> Tool: ...


@overload
def tool(
    func: None = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    permission: Optional[ToolPermission] = None,
    readonly: bool = False,
    dangerous: bool = False,
    reads_private: bool = False,
    ingests_untrusted: bool = False,
    communicates_externally: bool = False,
) -> Callable[[Callable], Tool]: ...


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    permission: Optional[ToolPermission] = None,
    readonly: bool = False,
    dangerous: bool = False,
    reads_private: bool = False,
    ingests_untrusted: bool = False,
    communicates_externally: bool = False,
) -> Union[Tool, Callable[[Callable], Tool]]:
    """Decorator to turn any sync or async function into a typed agentu Tool.

    Examples:
        >>> @tool
        ... def add(x: int, y: int) -> int:
        ...     '''Add two integers.'''
        ...     return x + y

        >>> @tool(readonly=True)
        ... async def lookup_order(order_id: str) -> dict:
        ...     '''Fetch order status.'''
        ...     return {"id": order_id, "status": "shipped"}
    """
    resolved_perm = permission
    if resolved_perm is None:
        if dangerous:
            resolved_perm = ToolPermission.DANGEROUS
        elif readonly:
            resolved_perm = ToolPermission.READONLY
        else:
            resolved_perm = ToolPermission.WRITE

    def _decorator(fn: Callable) -> Tool:
        return Tool(
            function=fn,
            description=description,
            name=name,
            permission=resolved_perm,
            reads_private=reads_private,
            ingests_untrusted=ingests_untrusted,
            communicates_externally=communicates_externally,
        )

    if func is not None:
        return _decorator(func)
    return _decorator
