"""Guardrails for input/output validation on agent calls.

Composable validators that can block, filter, or transform text
before it reaches the LLM or before the response reaches the user.
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    guardrail: str
    reason: Optional[str] = None
    matches: List[str] = field(default_factory=list)


class Guardrail:
    """Base guardrail — subclass and override `check`."""

    name: str = "base"

    def check(self, text: str) -> GuardrailResult:
        """Check text against this guardrail.

        Args:
            text: The text to validate

        Returns:
            GuardrailResult indicating pass/fail
        """
        return GuardrailResult(passed=True, guardrail=self.name)


class PII(Guardrail):
    """Detect personally identifiable information via regex patterns.

    Catches emails, phone numbers, SSNs, and credit card numbers.
    """

    name = "pii"

    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    }

    def __init__(self, detect: Optional[List[str]] = None):
        """Initialize PII guardrail.

        Args:
            detect: List of PII types to detect. Default: all types.
                   Options: "email", "phone", "ssn", "credit_card"
        """
        self.detect = detect or list(self.PATTERNS.keys())

    def check(self, text: str) -> GuardrailResult:
        matches = []
        for pii_type in self.detect:
            pattern = self.PATTERNS.get(pii_type)
            if pattern:
                found = re.findall(pattern, text)
                if found:
                    matches.extend(f"{pii_type}:{m}" for m in found)

        if matches:
            return GuardrailResult(
                passed=False,
                guardrail=self.name,
                reason=f"PII detected: {', '.join(set(t.split(':')[0] for t in matches))}",
                matches=matches,
            )
        return GuardrailResult(passed=True, guardrail=self.name)


class ContentFilter(Guardrail):
    """Block messages containing specific keywords or topics."""

    name = "content_filter"

    def __init__(self, block: Optional[List[str]] = None, case_sensitive: bool = False):
        """Initialize content filter.

        Args:
            block: List of blocked keywords/phrases
            case_sensitive: Whether matching is case-sensitive
        """
        self.blocked = block or []
        self.case_sensitive = case_sensitive

    def check(self, text: str) -> GuardrailResult:
        check_text = text if self.case_sensitive else text.lower()
        matches = []

        for keyword in self.blocked:
            check_keyword = keyword if self.case_sensitive else keyword.lower()
            if check_keyword in check_text:
                matches.append(keyword)

        if matches:
            return GuardrailResult(
                passed=False,
                guardrail=self.name,
                reason=f"Blocked content detected: {', '.join(matches)}",
                matches=matches,
            )
        return GuardrailResult(passed=True, guardrail=self.name)


class MaxLength(Guardrail):
    """Enforce maximum text length."""

    name = "max_length"

    def __init__(self, max_chars: int = 10000):
        """Initialize max length guardrail.

        Args:
            max_chars: Maximum allowed character count
        """
        self.max_chars = max_chars

    def check(self, text: str) -> GuardrailResult:
        if len(text) > self.max_chars:
            return GuardrailResult(
                passed=False,
                guardrail=self.name,
                reason=f"Text length {len(text)} exceeds limit of {self.max_chars}",
            )
        return GuardrailResult(passed=True, guardrail=self.name)


class JSONSchema(Guardrail):
    """Validate that output matches a JSON schema or Pydantic model."""

    name = "json_schema"

    def __init__(self, schema: Optional[Dict[str, Any]] = None, model: Optional[Any] = None):
        """Initialize JSON schema guardrail.

        Args:
            schema: JSON schema dict with "required" and "properties" keys
            model: Pydantic model class for validation
        """
        self.schema = schema
        self.model = model

    def check(self, text: str) -> GuardrailResult:
        # Try to parse as JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return GuardrailResult(
                passed=False,
                guardrail=self.name,
                reason=f"Invalid JSON: {str(e)}",
            )

        # Validate against Pydantic model
        if self.model is not None:
            try:
                self.model.model_validate(data)
                return GuardrailResult(passed=True, guardrail=self.name)
            except Exception as e:
                return GuardrailResult(
                    passed=False,
                    guardrail=self.name,
                    reason=f"Schema validation failed: {str(e)}",
                )

        # Validate against JSON schema dict
        if self.schema is not None:
            missing = []
            for field_name in self.schema.get("required", []):
                if field_name not in data:
                    missing.append(field_name)

            if missing:
                return GuardrailResult(
                    passed=False,
                    guardrail=self.name,
                    reason=f"Missing required fields: {', '.join(missing)}",
                    matches=missing,
                )

            # Type checking for properties
            properties = self.schema.get("properties", {})
            type_errors = []
            type_map = {
                "string": str, "number": (int, float),
                "integer": int, "boolean": bool,
                "array": list, "object": dict,
            }

            for prop_name, prop_def in properties.items():
                if prop_name in data:
                    expected_type = prop_def.get("type")
                    if expected_type and expected_type in type_map:
                        if not isinstance(data[prop_name], type_map[expected_type]):
                            type_errors.append(
                                f"{prop_name}: expected {expected_type}, got {type(data[prop_name]).__name__}"
                            )

            if type_errors:
                return GuardrailResult(
                    passed=False,
                    guardrail=self.name,
                    reason=f"Type errors: {'; '.join(type_errors)}",
                    matches=type_errors,
                )

        return GuardrailResult(passed=True, guardrail=self.name)


class GuardrailSet:
    """Run multiple guardrails and collect all violations."""

    def __init__(self, guardrails: List[Guardrail]):
        self.guardrails = guardrails

    def check(self, text: str) -> List[GuardrailResult]:
        """Run all guardrails against text.

        Args:
            text: Text to validate

        Returns:
            List of GuardrailResult (only failures, empty = all passed)
        """
        failures = []
        for guardrail in self.guardrails:
            result = guardrail.check(text)
            if not result.passed:
                failures.append(result)
        return failures

    def check_or_raise(self, text: str, direction: str = "input") -> None:
        """Check text and raise GuardrailError if any guardrail fails.

        Args:
            text: Text to validate
            direction: "input" or "output" (for error message context)

        Raises:
            GuardrailError: If any guardrail fails
        """
        failures = self.check(text)
        if failures:
            reasons = [f.reason for f in failures]
            raise GuardrailError(
                f"Guardrail violation on {direction}: {'; '.join(reasons)}",
                failures=failures,
            )


class GuardrailError(Exception):
    """Raised when a guardrail check fails."""

    def __init__(self, message: str, failures: Optional[List[GuardrailResult]] = None):
        super().__init__(message)
        self.failures = failures or []
