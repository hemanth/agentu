"""Zero-dependency structured output compiler, AST contract lifter, and PDA JSON healer.

Supports:
1. Plain Python classes with zero imports (``class Deal: pick: str; price: float``)
   and Pythonic ``def check(self): assert ...`` contracts statically lifted via ``ast``
   into JSON Schema constraints before the first LLM call.
2. Inline dict shapes (``{"pick": str, "price": float}``) hydrated into ``StructRecord``
   (supporting both ``res.data.pick`` and ``res.data["pick"]``).
3. ``@dataclass``, ``TypedDict``, ``Enum``, ``Literal``, ``Optional``, nested types, and
   raw JSON Schema dicts.
4. Progressive duck-typing for Pydantic v1/v2 ``BaseModel`` when installed.
5. Single-pass deterministic Pushdown Automaton (``heal_json``) that repairs LLM JSON
   syntax defects in O(n) time before spending retry tokens.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import enum
import inspect
import json
import logging
import re
import sys
import textwrap
import types
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
     Sequence,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

try:
    from typing import is_typeddict
except ImportError:  # pragma: no cover
    def is_typeddict(tp: Any) -> bool:
        return isinstance(tp, type) and issubclass(tp, dict) and hasattr(tp, "__annotations__")

logger = logging.getLogger(__name__)

_NONE_TYPE = type(None)
_UNION_ORIGINS: Tuple[Any, ...] = (
    (Union, types.UnionType) if hasattr(types, "UnionType") else (Union,)
)
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')
_INT_STR_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_STR_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


# ─────────────────────────────────────────────────────────────────────────────
# StructRecord: Dict + Attribute Hybrid for Inline Dict Schemas
# ─────────────────────────────────────────────────────────────────────────────


class StructRecord(dict):
    """Dictionary with ergonomic dot-attribute access for inline dict schemas."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute {name!r}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: (v.to_dict() if isinstance(v, StructRecord) else v)
            for k, v in self.items()
        }


def _wrap_struct_record(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, StructRecord):
        return StructRecord({k: _wrap_struct_record(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap_struct_record(v) for v in value]
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Single-Pass Deterministic Pushdown Automaton (PDA) JSON Healer
# ─────────────────────────────────────────────────────────────────────────────


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            body = text[first_nl + 1 :]
            end_fence = body.rfind("```")
            if end_fence != -1:
                return body[:end_fence].strip()
            return body.strip()
        return text.strip("`").strip()
    return text


def _pop_trailing_comma(out: List[str]) -> None:
    """Remove a trailing comma (ignoring whitespace) right before '}' or ']'."""
    idx = len(out) - 1
    while idx >= 0 and out[idx] in (" ", "\t", "\n", "\r"):
        idx -= 1
    if idx >= 0 and out[idx] == ",":
        del out[idx:]


def heal_json(raw: str) -> str:
    """Repair common LLM JSON syntax errors in a single O(n) pushdown pass.

    Repairs:
    - Markdown code fences and conversational preamble/postamble around JSON
    - ``//`` line comments and ``/* ... */`` block comments outside strings
    - Unescaped literal newlines, tabs, and control characters inside strings
    - Single-quoted strings (``'value'``) and single-quoted/unquoted object keys
    - Python literals ``True``, ``False``, ``None`` outside strings
    - Trailing commas before ``}`` or ``]``
    - Truncated JSON at EOF (closes unclosed strings, drops half-written keys,
      fills ``null`` after a dangling ``:``, and closes open ``{``/``[`` stacks)

    Non-JSON prose with no JSON structure is returned unchanged so ``json.loads``
    raises ``JSONDecodeError`` as expected.
    """
    if not raw:
        return raw

    stripped = raw.strip()
    # Fast C-path when output is already clean JSON
    if stripped and stripped[0] in "{[":
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    text = _strip_code_fences(stripped)
    if not text:
        return text

    # Fast retry right after fence stripping
    if text[0] in "{[":
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    # Locate earliest JSON container start ('{' or '[') if wrapped in prose
    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [pos for pos in (first_obj, first_arr) if pos != -1]
    if not starts:
        # Scalar JSON or plain prose — return stripped text without fabricating {}
        return text

    start = min(starts)
    n = len(text)
    i = start
    out: List[str] = []

    # Stack entries: [delimiter ('{' | '['), state, safe_checkpoint_len]
    # '{' states: 'EXPECT_KEY', 'AFTER_KEY', 'AFTER_COLON', 'AFTER_VAL'
    # '[' states: 'EXPECT_ELEM', 'AFTER_ELEM'
    stack: List[List[Any]] = []

    def _mark_value_start() -> None:
        if not stack:
            return
        top = stack[-1]
        if top[0] == "{" and top[1] == "EXPECT_KEY":
            top[1] = "AFTER_KEY"
        elif top[0] == "{" and top[1] == "AFTER_COLON":
            top[1] = "AFTER_VAL"
        elif top[0] == "[" and top[1] == "EXPECT_ELEM":
            top[1] = "AFTER_ELEM"

    def _mark_container_complete() -> None:
        if not stack:
            return
        top = stack[-1]
        if top[0] == "{":
            top[1] = "AFTER_VAL"
            top[2] = len(out)
        elif top[0] == "[":
            top[1] = "AFTER_ELEM"
            top[2] = len(out)

    while i < n:
        ch = text[i]

        # 1. Comments outside strings
        if ch == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                end_comment = text.find("*/", i + 2)
                i = n if end_comment == -1 else end_comment + 2
                continue

        # 2. Double-quoted string
        if ch == '"':
            _mark_value_start()
            out.append('"')
            i += 1
            closed = False
            while i < n:
                sc = text[i]
                if sc == "\\":
                    if i + 1 < n:
                        esc = text[i + 1]
                        if esc == "'":
                            # Invalid JSON escape \' -> literal '
                            out.append("'")
                            i += 2
                            continue
                        if esc == "u":
                            if i + 5 < n and all(c in _HEX_CHARS for c in text[i + 2 : i + 6]):
                                out.append(text[i : i + 6])
                                i += 6
                                continue
                            out.append("\\\\u")
                            i += 2
                            continue
                        if esc in _VALID_JSON_ESCAPES:
                            out.append("\\")
                            out.append(esc)
                            i += 2
                            continue
                        # Stray backslash before non-escape character
                        out.append("\\\\")
                        out.append(esc)
                        i += 2
                        continue
                    else:
                        # Trailing backslash right at EOF inside string
                        i += 1
                        break
                if sc == '"':
                    out.append('"')
                    i += 1
                    closed = True
                    break
                if sc == "\n":
                    out.append("\\n")
                elif sc == "\r":
                    out.append("\\r")
                elif sc == "\t":
                    out.append("\\t")
                elif ord(sc) < 0x20:
                    out.append(f"\\u{ord(sc):04x}")
                else:
                    out.append(sc)
                i += 1
            if not closed:
                out.append('"')
            if stack and stack[-1][1] in ("AFTER_VAL", "AFTER_ELEM"):
                stack[-1][2] = len(out)
            continue

        # 3. Single-quoted string -> normalized double-quoted JSON string
        if ch == "'":
            _mark_value_start()
            out.append('"')
            i += 1
            closed = False
            while i < n:
                sc = text[i]
                if sc == "\\":
                    if i + 1 < n:
                        esc = text[i + 1]
                        if esc == "'":
                            out.append("'")
                            i += 2
                            continue
                        if esc == '"':
                            out.append('\\"')
                            i += 2
                            continue
                        if esc in _VALID_JSON_ESCAPES:
                            out.append("\\")
                            out.append(esc)
                            i += 2
                            continue
                        out.append("\\\\")
                        out.append(esc)
                        i += 2
                        continue
                    else:
                        i += 1
                        break
                if sc == "'":
                    out.append('"')
                    i += 1
                    closed = True
                    break
                if sc == '"':
                    out.append('\\"')
                elif sc == "\n":
                    out.append("\\n")
                elif sc == "\r":
                    out.append("\\r")
                elif sc == "\t":
                    out.append("\\t")
                elif ord(sc) < 0x20:
                    out.append(f"\\u{ord(sc):04x}")
                else:
                    out.append(sc)
                i += 1
            if not closed:
                out.append('"')
            if stack and stack[-1][1] in ("AFTER_VAL", "AFTER_ELEM"):
                stack[-1][2] = len(out)
            continue

        # 4. Structural delimiters
        if ch == "{":
            _mark_value_start()
            out.append("{")
            stack.append(["{", "EXPECT_KEY", len(out)])
            i += 1
            continue

        if ch == "[":
            _mark_value_start()
            out.append("[")
            stack.append(["[", "EXPECT_ELEM", len(out)])
            i += 1
            continue

        if ch in "}]":
            expected_open = "{" if ch == "}" else "["
            _pop_trailing_comma(out)
            if stack:
                # Auto-close mismatched inner delimiters if present on stack
                if stack[-1][0] != expected_open and any(f[0] == expected_open for f in stack):
                    while stack and stack[-1][0] != expected_open:
                        inner = stack.pop()
                        _pop_trailing_comma(out)
                        if inner[0] == "{" and inner[1] == "AFTER_COLON":
                            out.append("null")
                        elif inner[0] == "{" and inner[1] == "AFTER_KEY":
                            del out[inner[2] :]
                            _pop_trailing_comma(out)
                        out.append("}" if inner[0] == "{" else "]")
                if stack and stack[-1][0] == expected_open:
                    stack.pop()
                    out.append(ch)
                    _mark_container_complete()
                    i += 1
                    if not stack:
                        # Root container cleanly closed; ignore trailing LLM prose
                        break
                    continue
            out.append(ch)
            i += 1
            continue

        if ch == ":":
            out.append(":")
            if stack and stack[-1][0] == "{":
                stack[-1][1] = "AFTER_COLON"
            i += 1
            continue

        if ch == ",":
            # Avoid duplicate or leading commas
            prev_idx = len(out) - 1
            while prev_idx >= 0 and out[prev_idx] in (" ", "\t", "\n", "\r"):
                prev_idx -= 1
            if prev_idx >= 0 and out[prev_idx] not in ("{", "[", ","):
                out.append(",")
                if stack:
                    if stack[-1][0] == "{":
                        stack[-1][2] = prev_idx + 1
                        stack[-1][1] = "EXPECT_KEY"
                    elif stack[-1][0] == "[":
                        stack[-1][2] = prev_idx + 1
                        stack[-1][1] = "EXPECT_ELEM"
            i += 1
            continue

        # 5. Identifiers (unquoted keys or Python booleans/None)
        if ch.isalpha() or ch in ("_", "$"):
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in ("_", "$", "-")):
                j += 1
            ident = text[j - 1 : j]  # placeholder overwritten below
            ident = text[i:j]
            k = j
            while k < n and text[k] in (" ", "\t", "\n", "\r"):
                k += 1

            if k < n and text[k] == ":" and stack and stack[-1][0] == "{" and stack[-1][1] in ("EXPECT_KEY", "AFTER_VAL"):
                # Unquoted object key
                _mark_value_start()
                out.append(f'"{ident}"')
            elif ident == "True":
                _mark_value_start()
                out.append("true")
                if stack and stack[-1][1] in ("AFTER_VAL", "AFTER_ELEM"):
                    stack[-1][2] = len(out)
            elif ident == "False":
                _mark_value_start()
                out.append("false")
                if stack and stack[-1][1] in ("AFTER_VAL", "AFTER_ELEM"):
                    stack[-1][2] = len(out)
            elif ident == "None":
                _mark_value_start()
                out.append("null")
                if stack and stack[-1][1] in ("AFTER_VAL", "AFTER_ELEM"):
                    stack[-1][2] = len(out)
            else:
                _mark_value_start()
                out.append(ident)
                if ident in ("true", "false", "null") and stack and stack[-1][1] in ("AFTER_VAL", "AFTER_ELEM"):
                    stack[-1][2] = len(out)
            i = j
            continue

        # 6. Numeric literals
        if ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            _mark_value_start()
            j = i + 1
            while j < n and (text[j].isdigit() or text[j] in ".eE+-"):
                j += 1
            num_tok = text[i:j]
            if num_tok.endswith((".", "e", "E", "+", "-")):
                num_tok = num_tok.rstrip(".eE+-")
            out.append(num_tok or "0")
            if stack and stack[-1][1] in ("AFTER_VAL", "AFTER_ELEM"):
                stack[-1][2] = len(out)
            i = j
            continue

        # 7. Whitespace & other characters
        out.append(ch)
        i += 1

    # EOF stack closure for truncated LLM generations
    while stack:
        frame = stack.pop()
        delim, state, safe_idx = frame[0], frame[1], frame[2]
        _pop_trailing_comma(out)
        if delim == "{":
            if state == "AFTER_KEY":
                # Truncated mid-key or before ':' -> roll back incomplete key
                del out[safe_idx:]
                _pop_trailing_comma(out)
            elif state == "AFTER_COLON":
                out.append("null")
            out.append("}")
        else:
            out.append("]")
        _mark_container_complete()

    return "".join(out).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Key Normalization & Bounded Damerau-Levenshtein Distance
# ─────────────────────────────────────────────────────────────────────────────


def _canonical_key(key: str) -> str:
    """Normalize snake_case, camelCase, kebab-case, and whitespace to a canonical form."""
    return key.replace("_", "").replace("-", "").replace(" ", "").lower()


def _damerau_levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Optimal string alignment (Damerau-Levenshtein) with length-bound pruning."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if a == b:
        return 0

    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j

    for i in range(1, la + 1):
        row_min = max_dist + 1
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            val = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                val = min(val, dp[i - 2][j - 2] + 1)  # adjacent transposition
            dp[i][j] = val
            if val < row_min:
                row_min = val
        if row_min > max_dist:
            return max_dist + 1

    return dp[la][lb]


# ─────────────────────────────────────────────────────────────────────────────
# Static AST Contract Lifter & Multi-Assertion Diagnostic Evaluator
# ─────────────────────────────────────────────────────────────────────────────


def _ast_const_number(node: ast.AST) -> Optional[Union[int, float]]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_const_number(node.operand)
        if inner is not None:
            return -inner
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _ast_const_number(node.operand)
    return None


def _ast_self_attr(node: ast.AST) -> Optional[str]:
    """Return 'field' if node is `self.field`."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _ast_self_len_attr(node: ast.AST) -> Optional[str]:
    """Return 'field' if node is `len(self.field)`."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
    ):
        return _ast_self_attr(node.args[0])
    return None


def _flatten_and_exprs(node: ast.AST) -> List[ast.AST]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        parts: List[ast.AST] = []
        for val in node.values:
            parts.extend(_flatten_and_exprs(val))
        return parts
    return [node]


def _apply_numeric_bound(prop: Dict[str, Any], op: ast.cmpop, num: Union[int, float], field_on_left: bool) -> None:
    # Flip operator when constant is on the left (`0 <= self.x` -> `self.x >= 0`)
    if not field_on_left:
        flip_map = {
            ast.Lt: ast.Gt,
            ast.LtE: ast.GtE,
            ast.Gt: ast.Lt,
            ast.GtE: ast.LtE,
            ast.Eq: ast.Eq,
        }
        op_cls = flip_map.get(type(op))
        if op_cls is None:
            return
        op = op_cls()

    if isinstance(op, ast.GtE):
        prop["minimum"] = max(prop["minimum"], num) if "minimum" in prop else num
    elif isinstance(op, ast.Gt):
        prop["exclusiveMinimum"] = max(prop["exclusiveMinimum"], num) if "exclusiveMinimum" in prop else num
    elif isinstance(op, ast.LtE):
        prop["maximum"] = min(prop["maximum"], num) if "maximum" in prop else num
    elif isinstance(op, ast.Lt):
        prop["exclusiveMaximum"] = min(prop["exclusiveMaximum"], num) if "exclusiveMaximum" in prop else num
    elif isinstance(op, ast.Eq):
        prop["const"] = num


def _apply_len_bound(prop: Dict[str, Any], op: ast.cmpop, num: int, len_on_left: bool) -> None:
    if not len_on_left:
        flip_map = {
            ast.Lt: ast.Gt,
            ast.LtE: ast.GtE,
            ast.Gt: ast.Lt,
            ast.GtE: ast.LtE,
            ast.Eq: ast.Eq,
        }
        op_cls = flip_map.get(type(op))
        if op_cls is None:
            return
        op = op_cls()

    is_array = prop.get("type") == "array"
    min_key = "minItems" if is_array else "minLength"
    max_key = "maxItems" if is_array else "maxLength"

    if isinstance(op, ast.GtE):
        prop[min_key] = max(prop.get(min_key, 0), max(0, int(num)))
    elif isinstance(op, ast.Gt):
        prop[min_key] = max(prop.get(min_key, 0), max(0, int(num) + 1))
    elif isinstance(op, ast.LtE):
        val = max(0, int(num))
        prop[max_key] = min(prop[max_key], val) if max_key in prop else val
    elif isinstance(op, ast.Lt):
        val = max(0, int(num) - 1)
        prop[max_key] = min(prop[max_key], val) if max_key in prop else val
    elif isinstance(op, ast.Eq):
        val = max(0, int(num))
        prop[min_key] = val
        prop[max_key] = val


def _extract_literal_sequence(node: ast.AST) -> Optional[List[Any]]:
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        vals: List[Any] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, (str, int, float, bool)):
                vals.append(elt.value)
            else:
                num = _ast_const_number(elt)
                if num is not None:
                    vals.append(num)
                else:
                    return None
        if isinstance(node, ast.Set):
            vals = sorted(vals, key=lambda x: str(x))
        return vals
    return None


def _get_contract_methods(cls: Type, include_post_init: bool = True) -> List[Any]:
    methods: List[Any] = []
    seen = set()
    names = ("check", "validate", "__post_init__") if include_post_init else ("check", "validate")
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        for mname in names:
            if mname in base.__dict__:
                fn = base.__dict__[mname]
                if callable(fn) and fn not in seen:
                    seen.add(fn)
                    methods.append(fn)
    return methods


def _parse_contract_ast(fn: Any) -> Optional[ast.Module]:
    """Parse the AST of a contract method from disk, linecache, or `python -c` (`sys.orig_argv`)."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        return ast.parse(src)
    except (OSError, TypeError, IndentationError, SyntaxError):
        pass

    # Fallback for Python < 3.13 `python -c "..."` where `<string>` is not in `linecache`
    code_obj = getattr(fn, "__code__", None)
    if code_obj is not None and getattr(code_obj, "co_filename", "") == "<string>":
        orig_argv = getattr(sys, "orig_argv", None) or ()
        if "-c" in orig_argv:
            c_idx = orig_argv.index("-c")
            if c_idx + 1 < len(orig_argv):
                try:
                    full_tree = ast.parse(orig_argv[c_idx + 1])
                    target_lineno = code_obj.co_firstlineno
                    target_name = getattr(fn, "__name__", "")
                    for node in ast.walk(full_tree):
                        if (
                            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == target_name
                            and node.lineno == target_lineno
                        ):
                            return ast.Module(body=[node], type_ignores=[])
                except SyntaxError:
                    pass
    return None


def _lift_ast_constraints(cls: Type, properties: Dict[str, Dict[str, Any]]) -> None:
    """Statically inspect `check`/`validate`/`__post_init__` via `ast` and enrich JSON Schema."""
    for fn in _get_contract_methods(cls, include_post_init=True):
        tree = _parse_contract_ast(fn)
        if tree is None:
            continue

        for stmt in ast.walk(tree):
            if not isinstance(stmt, ast.Assert):
                continue
            for expr in _flatten_and_exprs(stmt.test):
                # 1. Comparisons (bounds, lengths, membership)
                if isinstance(expr, ast.Compare):
                    operands = [expr.left] + list(expr.comparators)
                    for left, op, right in zip(operands, expr.ops, operands[1:]):
                        # Membership: `self.tier in ("standard", "vip")`
                        if isinstance(op, ast.In):
                            fname = _ast_self_attr(left)
                            seq = _extract_literal_sequence(right)
                            if fname and fname in properties and seq:
                                properties[fname]["enum"] = seq
                            continue

                        # Numeric bounds: `0 <= self.refund_usd <= 100`
                        f_left = _ast_self_attr(left)
                        n_right = _ast_const_number(right)
                        if f_left and f_left in properties and n_right is not None:
                            _apply_numeric_bound(properties[f_left], op, n_right, field_on_left=True)
                            continue

                        n_left = _ast_const_number(left)
                        f_right = _ast_self_attr(right)
                        if f_right and f_right in properties and n_left is not None:
                            _apply_numeric_bound(properties[f_right], op, n_left, field_on_left=False)
                            continue

                        # Length bounds: `len(self.summary) >= 12` or `1 <= len(self.tags) <= 5`
                        l_left = _ast_self_len_attr(left)
                        if l_left and l_left in properties and isinstance(n_right, int):
                            _apply_len_bound(properties[l_left], op, n_right, len_on_left=True)
                            continue

                        l_right = _ast_self_len_attr(right)
                        if l_right and l_right in properties and isinstance(n_left, int):
                            _apply_len_bound(properties[l_right], op, n_left, len_on_left=False)
                            continue

                # 2. `assert self.summary.strip()` -> minLength >= 1
                elif (
                    isinstance(expr, ast.Call)
                    and isinstance(expr.func, ast.Attribute)
                    and expr.func.attr == "strip"
                    and not expr.args
                ):
                    fname = _ast_self_attr(expr.func.value)
                    if fname and fname in properties and properties[fname].get("type") == "string":
                        properties[fname]["minLength"] = max(properties[fname].get("minLength", 0), 1)

                # 3. `assert re.match(r"...", self.code)` / `re.fullmatch` / `re.search`
                elif (
                    isinstance(expr, ast.Call)
                    and len(expr.args) >= 2
                    and isinstance(expr.args[0], ast.Constant)
                    and isinstance(expr.args[0].value, str)
                ):
                    func_name = ""
                    if isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name) and expr.func.value.id == "re":
                        func_name = expr.func.attr
                    elif isinstance(expr.func, ast.Name):
                        func_name = expr.func.id
                    if func_name in ("match", "fullmatch", "search"):
                        fname = _ast_self_attr(expr.args[1])
                        if fname and fname in properties and properties[fname].get("type") == "string":
                            pat = expr.args[0].value
                            if func_name == "fullmatch" and not pat.startswith("^"):
                                pat = f"^(?:{pat})$"
                            properties[fname]["pattern"] = pat


def _referenced_self_fields(node: ast.AST) -> List[str]:
    fields: List[str] = []
    for sub in ast.walk(node):
        fname = _ast_self_attr(sub)
        if fname and fname not in fields:
            fields.append(fname)
    return fields


def _diagnose_failed_contract_fn(fn: Any, instance: Any, fallback_exc: Exception, prefix: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    """Evaluate all `assert` clauses in `fn` so every violated invariant is reported at once."""
    collected: List[Dict[str, Any]] = []
    tree = _parse_contract_ast(fn)
    if tree is None:
        msg = str(fallback_exc) or f"{type(fallback_exc).__name__} in {getattr(fn, '__name__', 'check')}()"
        return [{"loc": prefix or ("__root__",), "msg": msg}]

    fn_globals = getattr(fn, "__globals__", {})
    local_ns = {"self": instance}

    # Find top-level function body statements
    func_defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    body_stmts = func_defs[0].body if func_defs else tree.body

    for stmt in body_stmts:
        if not isinstance(stmt, ast.Assert):
            continue
        custom_msg: Optional[str] = None
        if stmt.msg is not None:
            try:
                msg_code = compile(ast.Expression(stmt.msg), "<contract_msg>", "eval")
                custom_msg = str(eval(msg_code, fn_globals, local_ns))
            except Exception:
                custom_msg = ast.unparse(stmt.msg)

        for clause in _flatten_and_exprs(stmt.test):
            try:
                code = compile(ast.Expression(clause), "<contract>", "eval")
                ok = bool(eval(code, fn_globals, local_ns))
            except Exception as eval_exc:
                ok = False
                custom_msg = custom_msg or str(eval_exc)

            if not ok:
                refs = _referenced_self_fields(clause)
                loc = prefix + ((refs[0],) if refs else ("__contract__",))
                expr_str = ast.unparse(clause)
                live_vals = ", ".join(
                    f"{r}={getattr(instance, r, '<unset>')!r}" for r in refs
                )
                detail = f"Contract failed: `{expr_str}`"
                if custom_msg:
                    detail = f"{custom_msg} (`{expr_str}`)"
                if live_vals:
                    detail = f"{detail} [got {live_vals}]"
                collected.append({"loc": loc, "msg": detail})

    if not collected:
        msg = str(fallback_exc) or f"Contract check failed in {getattr(fn, '__name__', 'check')}()"
        collected.append({"loc": prefix or ("__contract__",), "msg": msg})

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# Zero-Dependency Schema Introspection & Type Compiler
# ─────────────────────────────────────────────────────────────────────────────


class _SchemaValidationError(ValueError):
    """Structured validation error exposing `.errors()` compatible with `format_validation_error`."""

    def __init__(self, errors: List[Dict[str, Any]]):
        self._errors = errors
        summary = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}"
            for e in errors
        )
        super().__init__(summary or "Schema validation failed")

    def errors(self) -> List[Dict[str, Any]]:
        return self._errors


def _is_pydantic_model(tp: Any) -> bool:
    return isinstance(tp, type) and (
        (hasattr(tp, "model_json_schema") and callable(tp.model_json_schema))
        or (hasattr(tp, "schema") and callable(tp.schema) and hasattr(tp, "__fields__"))
    )


def _is_type_spec_value(val: Any) -> bool:
    """True when `val` is a Python type or typing annotation rather than JSON Schema data."""
    if val is Any or val is None or isinstance(val, type):
        return True
    if get_origin(val) is not None:
        return True
    if isinstance(val, dict) and val and not _is_json_schema_dict(val):
        return True
    return False


def _is_json_schema_dict(d: Dict[str, Any]) -> bool:
    """Distinguish a raw JSON Schema dict from an inline Python shape dict like `{'x': int}`."""
    if not d:
        return True
    if any(isinstance(v, type) or get_origin(v) is not None or v is Any for v in d.values()):
        return False
    if "$schema" in d or "$defs" in d or "definitions" in d:
        return True
    if "type" in d and isinstance(d["type"], (str, list)):
        return True
    if "properties" in d and isinstance(d["properties"], dict):
        # Verify properties look like JSON schema objects, not Python types
        return not any(_is_type_spec_value(v) for v in d["properties"].values())
    if any(k in d for k in ("anyOf", "oneOf", "allOf", "enum", "items", "additionalProperties")):
        return True
    return True


def _is_structured_class(tp: Any) -> bool:
    """Return True for user classes, @dataclass, and TypedDict (excluding primitives/enums/containers)."""
    if is_typeddict(tp):
        return True
    if not isinstance(tp, type):
        return False
    if tp in (str, int, float, bool, bytes, dict, list, tuple, set, frozenset, object, _NONE_TYPE):
        return False
    if issubclass(tp, enum.Enum):
        return False
    return True


_MISSING = object()


def _resolve_class_fields(cls: Type) -> List[Tuple[str, Any, Any]]:
    """Return ordered `(name, type_hint, default_or_MISSING)` for plain classes, dataclasses, TypedDicts."""
    # Collect annotations in MRO order (base -> subclass)
    raw_ann: Dict[str, Any] = {}
    globalns: Dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        mod = sys.modules.get(base.__module__)
        if mod is not None:
            globalns.update(vars(mod))
        ann = base.__dict__.get("__annotations__", {})
        if ann:
            raw_ann.update(ann)

    try:
        hints = get_type_hints(cls, globalns=globalns or None)
    except Exception:
        hints = dict(raw_ann)

    if dataclasses.is_dataclass(cls):
        out: List[Tuple[str, Any, Any]] = []
        for f in dataclasses.fields(cls):
            if not f.init or f.name.startswith("_"):
                continue
            tp = hints.get(f.name, f.type)
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = f.default_factory
            else:
                default = _MISSING
            out.append((f.name, tp, default))
        return out

    if is_typeddict(cls):
        opt_keys = getattr(cls, "__optional_keys__", frozenset())
        total = getattr(cls, "__total__", True)
        out = []
        for fname, tp in hints.items():
            if fname.startswith("_"):
                continue
            is_opt = (fname in opt_keys) or (not total and fname not in getattr(cls, "__required_keys__", frozenset()))
            out.append((fname, tp, None if is_opt else _MISSING))
        return out

    # Plain Python class! Also include unannotated class attributes with concrete values if no annotations exist
    out = []
    seen: Set[str] = set()
    for fname, tp in hints.items():
        if fname.startswith("_"):
            continue
        if get_origin(tp) is ClassVar:
            continue
        seen.add(fname)
        default = _MISSING
        for base in cls.__mro__:
            if base is object:
                continue
            if fname in base.__dict__:
                attr = base.__dict__[fname]
                if not inspect.isroutine(attr) and not isinstance(attr, (property, staticmethod, classmethod)):
                    default = attr
                break
        out.append((fname, tp, default))

    if not out:
        # Fallback if a plain class defined `x = ""` defaults without annotations
        for base in reversed(cls.__mro__):
            if base is object:
                continue
            for fname, attr in base.__dict__.items():
                if fname.startswith("_") or fname in seen:
                    continue
                if inspect.isroutine(attr) or isinstance(attr, (property, staticmethod, classmethod)):
                    continue
                seen.add(fname)
                out.append((fname, type(attr) if attr is not None else Any, attr))

    return out


def _type_to_json_schema(tp: Any, seen_classes: Optional[Set[Type]] = None) -> Dict[str, Any]:
    seen_classes = seen_classes or set()

    if tp is Any or tp is object:
        return {}
    if tp is None or tp is _NONE_TYPE:
        return {"type": "null"}
    if tp is str:
        return {"type": "string"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}

    if isinstance(tp, dict):
        if _is_json_schema_dict(tp):
            return copy.deepcopy(tp)
        props: Dict[str, Any] = {}
        req: List[str] = []
        for k, spec in tp.items():
            # Optional inline dict syntax: `{"tier?": str}` or `Optional[str]`
            clean_key = k[:-1] if isinstance(k, str) and k.endswith("?") else k
            is_opt_key = isinstance(k, str) and k.endswith("?")
            sub_schema = _type_to_json_schema(spec, seen_classes)
            props[clean_key] = sub_schema
            if not is_opt_key and not _is_optional_type(spec):
                req.append(clean_key)
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": props,
        }
        if req:
            schema["required"] = req
        return schema

    if _is_pydantic_model(tp):
        if hasattr(tp, "model_json_schema"):
            return tp.model_json_schema()
        return tp.schema()

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        vals = [m.value for m in tp]
        prim_types = {type(v) for v in vals}
        js: Dict[str, Any] = {"enum": vals}
        if prim_types == {str}:
            js["type"] = "string"
        elif prim_types == {int}:
            js["type"] = "integer"
        return js

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Literal:
        vals = list(args)
        js = {"enum": vals}
        prim_types = {type(v) for v in vals if not isinstance(v, bool)}
        if all(isinstance(v, str) for v in vals):
            js["type"] = "string"
        elif all(isinstance(v, bool) for v in vals):
            js["type"] = "boolean"
        elif prim_types == {int}:
            js["type"] = "integer"
        return js

    if origin in _UNION_ORIGINS:
        sub_schemas = [_type_to_json_schema(a, seen_classes) for a in args]
        return {"anyOf": sub_schemas}

    if origin in (list, List, Sequence, set, Set, frozenset):
        item_tp = args[0] if args else Any
        js = {"type": "array", "items": _type_to_json_schema(item_tp, seen_classes)}
        if origin in (set, Set, frozenset):
            js["uniqueItems"] = True
        return js

    if origin in (tuple, Tuple):
        if not args:
            return {"type": "array"}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _type_to_json_schema(args[0], seen_classes)}
        return {
            "type": "array",
            "prefixItems": [_type_to_json_schema(a, seen_classes) for a in args],
            "minItems": len(args),
            "maxItems": len(args),
        }

    if origin in (dict, Dict):
        val_tp = args[1] if len(args) == 2 else Any
        js = {"type": "object"}
        if val_tp is not Any:
            js["additionalProperties"] = _type_to_json_schema(val_tp, seen_classes)
        return js

    if tp in (list, set, tuple):
        return {"type": "array"}
    if tp is dict:
        return {"type": "object"}

    if _is_structured_class(tp):
        if tp in seen_classes:
            return {"type": "object", "title": tp.__name__}
        next_seen = set(seen_classes) | {tp}
        fields = _resolve_class_fields(tp)
        props = {}
        req = []
        for fname, ftype, fdefault in fields:
            prop_schema = _type_to_json_schema(ftype, next_seen)
            if fdefault is not _MISSING and not callable(fdefault):
                try:
                    json.dumps(fdefault)
                    prop_schema["default"] = copy.deepcopy(fdefault)
                except TypeError:
                    pass
            elif fdefault is _MISSING and not _is_optional_type(ftype):
                req.append(fname)
            props[fname] = prop_schema

        # Lift AST invariants (`def check(self): assert ...`) into JSON Schema!
        _lift_ast_constraints(tp, props)

        schema = {
            "title": tp.__name__,
            "type": "object",
            "properties": props,
        }
        doc = inspect.getdoc(tp)
        # Ignore auto-generated dataclass signature docstrings `ClassName(a: int, ...)`
        if doc and not doc.startswith(f"{tp.__name__}("):
            schema["description"] = doc
        if req:
            schema["required"] = req
        return schema

    raise TypeError(f"Unsupported structured output schema type: {tp!r}")


def _is_optional_type(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin in _UNION_ORIGINS:
        return _NONE_TYPE in get_args(tp)
    return False


def schema_for_type(model: Union[Type, Dict[str, Any]]) -> Dict[str, Any]:
    """Compile any plain Python class, inline dict shape, dataclass, TypedDict, or Pydantic model to JSON Schema."""
    if _is_pydantic_model(model):
        if hasattr(model, "model_json_schema"):
            return model.model_json_schema()
        return model.schema()  # type: ignore[union-attr]
    if isinstance(model, dict) and _is_json_schema_dict(model):
        return model
    if isinstance(model, dict) or _is_structured_class(model):
        return _type_to_json_schema(model)
    raise TypeError(
        f"Expected a Python class, inline dict shape, JSON schema dict, or BaseModel; got {type(model)}"
    )


def pydantic_to_json_schema(model: Union[Type, Dict[str, Any]]) -> Dict[str, Any]:
    """Backwards-compatible alias for :func:`schema_for_type`."""
    return schema_for_type(model)


def build_response_format(
    schema: Union[Type, Dict[str, Any]],
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build OpenAI-compatible ``response_format`` payload for any supported schema."""
    if isinstance(schema, dict):
        json_schema = schema if _is_json_schema_dict(schema) else schema_for_type(schema)
        schema_name = name or json_schema.get("title") or "response"
    else:
        json_schema = schema_for_type(schema)
        schema_name = name or getattr(schema, "__name__", "response")

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": json_schema,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plain-Class Ergonomics Injection & Recursive Hydration
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_plain_class_ergonomics(cls: Type, field_names: List[str]) -> None:
    """Attach clean `__repr__`, `__eq__`, and `to_dict()` to plain classes if not user-defined."""
    if "__repr__" not in cls.__dict__ and cls.__repr__ is object.__repr__:
        def _auto_repr(self: Any) -> str:
            parts = ", ".join(f"{k}={getattr(self, k, None)!r}" for k in field_names)
            return f"{type(self).__name__}({parts})"
        cls.__repr__ = _auto_repr  # type: ignore[method-assign]

    if "__eq__" not in cls.__dict__ and cls.__eq__ is object.__eq__:
        def _auto_eq(self: Any, other: Any) -> bool:
            if type(self) is not type(other):
                return NotImplemented
            return all(getattr(self, k, _MISSING) == getattr(other, k, _MISSING) for k in field_names)
        cls.__eq__ = _auto_eq  # type: ignore[method-assign]

    if not hasattr(cls, "to_dict"):
        def _to_dict(self: Any) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k in field_names:
                v = getattr(self, k, None)
                if hasattr(v, "to_dict") and callable(v.to_dict):
                    v = v.to_dict()
                elif isinstance(v, list):
                    v = [x.to_dict() if hasattr(x, "to_dict") and callable(x.to_dict) else x for x in v]
                out[k] = v
            return out
        cls.to_dict = _to_dict  # type: ignore[attr-defined]


def _validate_raw_json_schema(data: Any, schema: Dict[str, Any], path: Tuple[Any, ...]) -> Any:
    """Lightweight validator for raw JSON Schema dicts (`output_schema={...}`)."""
    errs: List[Dict[str, Any]] = []
    expected_type = schema.get("type")

    if expected_type == "object":
        if not isinstance(data, dict):
            raise _SchemaValidationError([{"loc": path or ("__root__",), "msg": f"Expected object, got {type(data).__name__}"}])
        for req_key in schema.get("required", ()):
            if req_key not in data:
                errs.append({"loc": path + (req_key,), "msg": "Field required"})
        props = schema.get("properties", {})
        result: Dict[str, Any] = {}
        for k, v in data.items():
            if k in props:
                try:
                    result[k] = _validate_raw_json_schema(v, props[k], path + (k,))
                except _SchemaValidationError as sub_exc:
                    errs.extend(sub_exc.errors())
            else:
                result[k] = _wrap_struct_record(v)
        if errs:
            raise _SchemaValidationError(errs)
        return StructRecord(result)

    if expected_type == "array":
        if not isinstance(data, list):
            raise _SchemaValidationError([{"loc": path or ("__root__",), "msg": f"Expected array, got {type(data).__name__}"}])
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            out_list = []
            for idx, elem in enumerate(data):
                try:
                    out_list.append(_validate_raw_json_schema(elem, item_schema, path + (idx,)))
                except _SchemaValidationError as sub_exc:
                    errs.extend(sub_exc.errors())
            if errs:
                raise _SchemaValidationError(errs)
            return out_list
        return _wrap_struct_record(data)

    if expected_type == "string" and not isinstance(data, str):
        errs.append({"loc": path, "msg": f"Expected string, got {type(data).__name__}"})
    elif expected_type == "integer" and (isinstance(data, bool) or not isinstance(data, int)):
        errs.append({"loc": path, "msg": f"Expected integer, got {type(data).__name__}"})
    elif expected_type == "number" and (isinstance(data, bool) or not isinstance(data, (int, float))):
        errs.append({"loc": path, "msg": f"Expected number, got {type(data).__name__}"})
    elif expected_type == "boolean" and not isinstance(data, bool):
        errs.append({"loc": path, "msg": f"Expected boolean, got {type(data).__name__}"})

    if not errs:
        if "enum" in schema and data not in schema["enum"]:
            errs.append({"loc": path, "msg": f"Value {data!r} not in allowed enum {schema['enum']!r}"})
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            if "minimum" in schema and data < schema["minimum"]:
                errs.append({"loc": path, "msg": f"Input should be >= {schema['minimum']} (got {data})"})
            if "maximum" in schema and data > schema["maximum"]:
                errs.append({"loc": path, "msg": f"Input should be <= {schema['maximum']} (got {data})"})
        if isinstance(data, str):
            if "minLength" in schema and len(data) < schema["minLength"]:
                errs.append({"loc": path, "msg": f"String length should be >= {schema['minLength']} (got {len(data)})"})
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                errs.append({"loc": path, "msg": f"String length should be <= {schema['maxLength']} (got {len(data)})"})

    if errs:
        raise _SchemaValidationError(errs)
    return data


def _coerce_and_validate(data: Any, tp: Any, path: Tuple[Any, ...] = ()) -> Any:
    if tp is Any or tp is object:
        return _wrap_struct_record(data)

    if tp is None or tp is _NONE_TYPE:
        if data is None:
            return None
        raise _SchemaValidationError([{"loc": path, "msg": f"Expected null, got {type(data).__name__}"}])

    if tp is str:
        if isinstance(data, str):
            return data
        raise _SchemaValidationError([{"loc": path, "msg": f"Expected str, got {type(data).__name__} ({data!r})"}])

    if tp is bool:
        if isinstance(data, bool):
            return data
        if isinstance(data, str) and data.lower() in ("true", "false"):
            return data.lower() == "true"
        raise _SchemaValidationError([{"loc": path, "msg": f"Expected bool, got {type(data).__name__} ({data!r})"}])

    if tp is int:
        if isinstance(data, int) and not isinstance(data, bool):
            return data
        if isinstance(data, float) and data.is_integer():
            return int(data)
        if isinstance(data, str) and _INT_STR_RE.match(data.strip()):
            return int(data.strip())
        raise _SchemaValidationError([{"loc": path, "msg": f"Expected int, got {type(data).__name__} ({data!r})"}])

    if tp is float:
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            return float(data)
        if isinstance(data, str) and _FLOAT_STR_RE.match(data.strip()):
            return float(data.strip())
        raise _SchemaValidationError([{"loc": path, "msg": f"Expected float, got {type(data).__name__} ({data!r})"}])

    # Inline Python dict shape vs Raw JSON Schema dict
    if isinstance(tp, dict):
        if _is_json_schema_dict(tp):
            return _validate_raw_json_schema(data, tp, path)
        if not isinstance(data, dict):
            raise _SchemaValidationError([{"loc": path or ("__root__",), "msg": f"Expected JSON object, got {type(data).__name__}"}])
        spec_fields: List[Tuple[str, Any, Any]] = []
        for raw_k, sub_tp in tp.items():
            is_opt = isinstance(raw_k, str) and raw_k.endswith("?")
            clean_k = raw_k[:-1] if is_opt else raw_k
            default = None if (is_opt or _is_optional_type(sub_tp)) else _MISSING
            spec_fields.append((clean_k, sub_tp, default))
        hydrated_map = _hydrate_mapping(data, spec_fields, path)
        return StructRecord(hydrated_map)

    if _is_pydantic_model(tp):
        try:
            if hasattr(tp, "model_validate"):
                return tp.model_validate(data)
            return tp(**data)
        except Exception as exc:
            if hasattr(exc, "errors") and callable(exc.errors):
                raise _SchemaValidationError(exc.errors()) from exc
            raise _SchemaValidationError([{"loc": path, "msg": str(exc)}]) from exc

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        for member in tp:
            if data == member.value or data == member:
                return member
        if isinstance(data, str):
            folded = data.strip().lower()
            for member in tp:
                if (isinstance(member.value, str) and member.value.lower() == folded) or member.name.lower() == folded:
                    return member
        allowed = [m.value for m in tp]
        raise _SchemaValidationError([{"loc": path, "msg": f"Expected one of {allowed!r}, got {data!r}"}])

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Literal:
        for lit in args:
            if data == lit and type(data) is type(lit):
                return lit
        if isinstance(data, str):
            folded = data.strip().lower()
            str_matches = [lit for lit in args if isinstance(lit, str) and lit.lower() == folded]
            if len(str_matches) == 1:
                return str_matches[0]
        raise _SchemaValidationError([{"loc": path, "msg": f"Expected literal in {list(args)!r}, got {data!r}"}])

    if origin in _UNION_ORIGINS:
        if data is None and _NONE_TYPE in args:
            return None
        non_none = [a for a in args if a is not _NONE_TYPE]
        sub_errors: List[Dict[str, Any]] = []
        for candidate in non_none:
            try:
                return _coerce_and_validate(data, candidate, path)
            except _SchemaValidationError as sub_exc:
                sub_errors.extend(sub_exc.errors())
        raise _SchemaValidationError(sub_errors or [{"loc": path, "msg": f"Value {data!r} did not match union {tp}"}])

    if origin in (list, List, Sequence, set, Set, frozenset):
        if not isinstance(data, (list, tuple)):
            raise _SchemaValidationError([{"loc": path, "msg": f"Expected array, got {type(data).__name__}"}])
        item_tp = args[0] if args else Any
        out_items: List[Any] = []
        item_errs: List[Dict[str, Any]] = []
        for idx, elem in enumerate(data):
            try:
                out_items.append(_coerce_and_validate(elem, item_tp, path + (idx,)))
            except _SchemaValidationError as sub_exc:
                item_errs.extend(sub_exc.errors())
        if item_errs:
            raise _SchemaValidationError(item_errs)
        if origin in (set, Set):
            return set(out_items)
        if origin is frozenset:
            return frozenset(out_items)
        return out_items

    if origin in (tuple, Tuple):
        if not isinstance(data, (list, tuple)):
            raise _SchemaValidationError([{"loc": path, "msg": f"Expected array/tuple, got {type(data).__name__}"}])
        if not args:
            return tuple(data)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_and_validate(elem, args[0], path + (idx,)) for idx, elem in enumerate(data))
        if len(data) != len(args):
            raise _SchemaValidationError([{"loc": path, "msg": f"Expected tuple of length {len(args)}, got {len(data)}"}])
        out_tup: List[Any] = []
        tup_errs: List[Dict[str, Any]] = []
        for idx, (elem, sub_tp) in enumerate(zip(data, args)):
            try:
                out_tup.append(_coerce_and_validate(elem, sub_tp, path + (idx,)))
            except _SchemaValidationError as sub_exc:
                tup_errs.extend(sub_exc.errors())
        if tup_errs:
            raise _SchemaValidationError(tup_errs)
        return tuple(out_tup)

    if origin in (dict, Dict):
        if not isinstance(data, dict):
            raise _SchemaValidationError([{"loc": path, "msg": f"Expected object, got {type(data).__name__}"}])
        key_tp = args[0] if len(args) >= 1 else str
        val_tp = args[1] if len(args) >= 2 else Any
        out_dict: Dict[Any, Any] = {}
        dict_errs: List[Dict[str, Any]] = []
        for k, v in data.items():
            try:
                ck = _coerce_and_validate(k, key_tp, path + (k,))
                cv = _coerce_and_validate(v, val_tp, path + (k,))
                out_dict[ck] = cv
            except _SchemaValidationError as sub_exc:
                dict_errs.extend(sub_exc.errors())
        if dict_errs:
            raise _SchemaValidationError(dict_errs)
        return out_dict

    if tp in (list, set, tuple):
        if not isinstance(data, (list, tuple)):
            raise _SchemaValidationError([{"loc": path, "msg": f"Expected array, got {type(data).__name__}"}])
        return tp(data)
    if tp is dict:
        if not isinstance(data, dict):
            raise _SchemaValidationError([{"loc": path, "msg": f"Expected object, got {type(data).__name__}"}])
        return _wrap_struct_record(data)

    if _is_structured_class(tp):
        if not is_typeddict(tp) and isinstance(data, tp):
            return data
        if not isinstance(data, dict):
            raise _SchemaValidationError([{"loc": path or ("__root__",), "msg": f"Expected JSON object for {tp.__name__}, got {type(data).__name__}"}])
        fields = _resolve_class_fields(tp)
        hydrated = _hydrate_mapping(data, fields, path)

        if is_typeddict(tp):
            return StructRecord(hydrated)

        if dataclasses.is_dataclass(tp):
            try:
                instance = tp(**hydrated)
            except (AssertionError, ValueError, TypeError) as post_exc:
                post_fn = getattr(tp, "__post_init__", None)
                if post_fn is not None and isinstance(post_exc, AssertionError):
                    # Construct bare instance to diagnose all __post_init__ asserts
                    bare = object.__new__(tp)
                    for k, v in hydrated.items():
                        object.__setattr__(bare, k, v)
                    raise _SchemaValidationError(_diagnose_failed_contract_fn(post_fn, bare, post_exc, path)) from post_exc
                raise _SchemaValidationError([{"loc": path or ("__init__",), "msg": str(post_exc)}]) from post_exc

            contract_errs = _run_contracts(instance, tp, path, include_post_init=False)
            if contract_errs:
                raise _SchemaValidationError(contract_errs)
            return instance

        # Plain Python class!
        field_names = [f[0] for f in fields]
        _ensure_plain_class_ergonomics(tp, field_names)

        has_custom_init = "__init__" in tp.__dict__ or tp.__init__ is not object.__init__
        if has_custom_init:
            try:
                instance = tp(**hydrated)
            except TypeError:
                # Fallback if user wrote a positional/no-arg __init__ or inherited a base
                instance = tp.__new__(tp)
                for k, v in hydrated.items():
                    setattr(instance, k, v)
        else:
            instance = tp.__new__(tp)
            for k, v in hydrated.items():
                setattr(instance, k, v)

        contract_errs = _run_contracts(instance, tp, path, include_post_init=True)
        if contract_errs:
            raise _SchemaValidationError(contract_errs)
        return instance

    raise TypeError(f"Unsupported validation target: {tp!r}")


def _hydrate_mapping(
    data: Dict[str, Any],
    fields: List[Tuple[str, Any, Any]],
    path: Tuple[Any, ...],
) -> Dict[str, Any]:
    """Match input keys (exact + canonical camelCase/snake_case), coerce types, and diagnose near-misses."""
    errs: List[Dict[str, Any]] = []
    consumed_input_keys: Set[str] = set()

    # Build canonical lookup for unambiguous camelCase <-> snake_case alignment
    canon_to_input: Dict[str, List[str]] = {}
    for in_key in data:
        canon_to_input.setdefault(_canonical_key(str(in_key)), []).append(in_key)

    hydrated: Dict[str, Any] = {}
    missing_required: List[str] = []

    for fname, ftype, fdefault in fields:
        matched_in_key: Optional[str] = None
        if fname in data:
            matched_in_key = fname
        else:
            canon_candidates = [
                k for k in canon_to_input.get(_canonical_key(fname), ())
                if k not in consumed_input_keys
            ]
            if len(canon_candidates) == 1:
                matched_in_key = canon_candidates[0]

        if matched_in_key is not None:
            consumed_input_keys.add(matched_in_key)
            raw_val = data[matched_in_key]
            try:
                hydrated[fname] = _coerce_and_validate(raw_val, ftype, path + (fname,))
            except _SchemaValidationError as sub_exc:
                errs.extend(sub_exc.errors())
        else:
            if fdefault is not _MISSING:
                hydrated[fname] = fdefault() if callable(fdefault) else copy.deepcopy(fdefault)
            elif _is_optional_type(ftype):
                hydrated[fname] = None
            else:
                missing_required.append(fname)

    # Check unused input keys for Damerau-Levenshtein <= 2 near-misses against missing required fields
    leftover_keys = [k for k in data if k not in consumed_input_keys]
    for fname in missing_required:
        hint = ""
        best_key = None
        best_dist = 3
        for uk in leftover_keys:
            d = _damerau_levenshtein(fname.lower(), str(uk).lower(), max_dist=2)
            if d < best_dist:
                best_dist = d
                best_key = uk
        if best_key is not None:
            hint = f" (received near-miss key {best_key!r} at edit distance {best_dist}; use {fname!r})"
        errs.append({"loc": path + (fname,), "msg": f"Field required{hint}"})

    if errs:
        raise _SchemaValidationError(errs)
    return hydrated


def _run_contracts(instance: Any, cls: Type, path: Tuple[Any, ...], include_post_init: bool) -> List[Dict[str, Any]]:
    errs: List[Dict[str, Any]] = []
    for fn in _get_contract_methods(cls, include_post_init=include_post_init):
        try:
            fn(instance)
        except Exception as exc:
            errs.extend(_diagnose_failed_contract_fn(fn, instance, exc, path))
    return errs


# ─────────────────────────────────────────────────────────────────────────────
# Public Error & Validation Entrypoints
# ─────────────────────────────────────────────────────────────────────────────


class StructuredOutputError(Exception):
    """Raised when structured output validation fails after all retry attempts.

    Attributes:
        raw_output: The raw LLM output that failed validation.
        model: The schema class or dict the output was validated against.
        attempts: Number of attempts made (including the original).
        last_error: The last validation error message.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: str = "",
        model: Optional[Any] = None,
        attempts: int = 1,
        last_error: str = "",
    ):
        super().__init__(message)
        self.raw_output = raw_output
        self.model = model
        self.attempts = attempts
        self.last_error = last_error


def format_validation_error(error: Exception, model: Union[Type, Dict[str, Any]]) -> str:
    """Format a validation error into a concise, field-localized correction prompt for the LLM."""
    schema = schema_for_type(model)
    schema_str = json.dumps(schema, indent=2)

    details: Optional[str] = None
    cause = error.__cause__ if error.__cause__ else error
    if hasattr(cause, "errors") and callable(cause.errors):
        try:
            errs = cause.errors()
            parts = []
            for e in errs:
                loc = " -> ".join(str(l) for l in e.get("loc", [])) or "__root__"
                msg = e.get("msg", str(e))
                parts.append(f"  - {loc}: {msg}")
            if parts:
                details = "\n".join(parts)
        except Exception:
            pass

    if details:
        return (
            f"Your output did not match the required schema.\n"
            f"Field errors:\n{details}\n\n"
            f"Required JSON schema:\n{schema_str}\n\n"
            f"Produce valid JSON matching this schema exactly."
        )

    return (
        f"Your output did not match the required schema.\n"
        f"Error: {error}\n\n"
        f"Required JSON schema:\n{schema_str}\n\n"
        f"Produce valid JSON matching this schema exactly."
    )


def parse_and_validate(raw_json: str, model: Union[Type, Dict[str, Any]]) -> Any:
    """Repair JSON syntax if needed, parse, and validate against any supported schema.

    Args:
        raw_json: Raw JSON string (or fenced/prose-wrapped LLM response)
        model: Plain Python class, inline dict shape, dataclass, TypedDict,
            JSON Schema dict, or Pydantic BaseModel

    Returns:
        Validated instance (or ``StructRecord`` for dict schemas)

    Raises:
        ValueError: If JSON cannot be parsed or fails schema/contract validation
    """
    healed = heal_json(raw_json)

    try:
        data = json.loads(healed)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}") from e

    try:
        if _is_pydantic_model(model):
            if hasattr(model, "model_validate"):
                return model.model_validate(data)
            return model(**data)  # type: ignore[misc]
        return _coerce_and_validate(data, model)
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}") from e
