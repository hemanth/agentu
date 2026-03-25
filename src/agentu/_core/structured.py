"""Structured output utilities for validated LLM responses.

Converts Pydantic models → OpenAI response_format payloads,
parses raw JSON → validated model instances.
"""

import json
import logging
from typing import Any, Dict, Optional, Type, Union

logger = logging.getLogger(__name__)


def pydantic_to_json_schema(model: Type) -> Dict[str, Any]:
    """Extract JSON schema from a Pydantic BaseModel class.

    Works with both Pydantic v1 and v2.

    Args:
        model: A Pydantic BaseModel subclass

    Returns:
        JSON schema dict
    """
    if hasattr(model, "model_json_schema"):
        # Pydantic v2
        return model.model_json_schema()
    elif hasattr(model, "schema"):
        # Pydantic v1
        return model.schema()
    else:
        raise TypeError(f"Expected Pydantic BaseModel, got {type(model)}")


def build_response_format(
    schema: Union[Type, Dict[str, Any]],
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build OpenAI-compatible response_format payload.

    Args:
        schema: Pydantic BaseModel class or raw JSON schema dict
        name: Schema name (auto-derived from class name if not provided)

    Returns:
        Dict for the `response_format` field in chat completion requests
    """
    if isinstance(schema, dict):
        json_schema = schema
        schema_name = name or "response"
    else:
        json_schema = pydantic_to_json_schema(schema)
        schema_name = name or getattr(schema, "__name__", "response")

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": json_schema,
        },
    }


def parse_and_validate(raw_json: str, model: Type) -> Any:
    """Parse JSON string and validate against a Pydantic model.

    Args:
        raw_json: Raw JSON string from LLM response
        model: Pydantic BaseModel class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If JSON is invalid or fails validation
    """
    # Strip markdown code fences if present
    text = raw_json.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}") from e

    try:
        if hasattr(model, "model_validate"):
            # Pydantic v2
            return model.model_validate(data)
        else:
            # Pydantic v1
            return model(**data)
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}") from e
