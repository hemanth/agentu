"""Tests for structured output hardening (output_type with validation-retry).

Covers:
- StructuredOutputError attributes
- format_validation_error with Pydantic v2 field errors and generic errors
- parse_and_validate unchanged behaviour
- Agent.infer(output_type=...) happy path (first-attempt success)
- Agent.infer(output_type=...) retry path (fails then succeeds)
- Agent.infer(output_type=...) exhausted retries → StructuredOutputError
- output_type and output_schema mutual exclusion
- Backwards compatibility: output_schema still works exactly as before
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import BaseModel, Field
from typing import List, Optional

from agentu import Agent
from agentu._core.structured import (
    StructuredOutputError,
    format_validation_error,
    parse_and_validate,
    pydantic_to_json_schema,
    build_response_format,
)


# ─── Test Models ───


class MovieReview(BaseModel):
    title: str
    rating: float = Field(ge=0.0, le=10.0)
    summary: str
    tags: List[str] = []


class UserProfile(BaseModel):
    name: str
    age: int = Field(ge=0)
    email: Optional[str] = None


# ─── StructuredOutputError ───


class TestStructuredOutputError:
    def test_basic_attributes(self):
        err = StructuredOutputError(
            "validation failed",
            raw_output='{"bad": true}',
            model=MovieReview,
            attempts=3,
            last_error="rating must be <= 10",
        )
        assert str(err) == "validation failed"
        assert err.raw_output == '{"bad": true}'
        assert err.model is MovieReview
        assert err.attempts == 3
        assert err.last_error == "rating must be <= 10"

    def test_default_attributes(self):
        err = StructuredOutputError("oops")
        assert err.raw_output == ""
        assert err.model is None
        assert err.attempts == 1
        assert err.last_error == ""

    def test_is_exception(self):
        err = StructuredOutputError("fail")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(StructuredOutputError) as exc_info:
            raise StructuredOutputError(
                "bad schema",
                model=UserProfile,
                attempts=2,
            )
        assert exc_info.value.model is UserProfile
        assert exc_info.value.attempts == 2


# ─── format_validation_error ───


class TestFormatValidationError:
    def test_generic_error(self):
        err = ValueError("Invalid JSON from LLM: blah")
        msg = format_validation_error(err, MovieReview)
        assert "did not match the required schema" in msg
        assert "Invalid JSON from LLM" in msg
        assert "MovieReview" in msg or "title" in msg  # schema should appear

    def test_includes_json_schema(self):
        err = ValueError("missing field")
        msg = format_validation_error(err, MovieReview)
        # The schema for MovieReview should include the field names
        assert "title" in msg
        assert "rating" in msg
        assert "summary" in msg

    def test_pydantic_v2_field_errors(self):
        """When the cause has .errors() method, field-level details appear."""
        # Simulate a Pydantic validation error
        try:
            MovieReview(title="x", rating=15.0, summary="ok")  # rating > 10
        except Exception as pydantic_err:
            # Wrap it like parse_and_validate does
            wrapper = ValueError(f"Schema validation failed: {pydantic_err}")
            wrapper.__cause__ = pydantic_err

            msg = format_validation_error(wrapper, MovieReview)
            assert "Field errors" in msg
            assert "rating" in msg

    def test_instructs_llm_to_produce_valid_json(self):
        err = ValueError("bad")
        msg = format_validation_error(err, MovieReview)
        assert "Produce valid JSON matching this schema exactly" in msg


# ─── parse_and_validate (existing behaviour preserved) ───


class TestParseAndValidate:
    def test_valid_json(self):
        raw = json.dumps({"title": "Inception", "rating": 9.0, "summary": "Great film"})
        result = parse_and_validate(raw, MovieReview)
        assert isinstance(result, MovieReview)
        assert result.title == "Inception"
        assert result.rating == 9.0

    def test_valid_with_optional_fields(self):
        raw = json.dumps({
            "title": "Dune",
            "rating": 8.5,
            "summary": "Epic",
            "tags": ["sci-fi", "action"],
        })
        result = parse_and_validate(raw, MovieReview)
        assert result.tags == ["sci-fi", "action"]

    def test_strips_markdown_fences(self):
        raw = '```json\n{"title": "Test", "rating": 5.0, "summary": "ok"}\n```'
        result = parse_and_validate(raw, MovieReview)
        assert result.title == "Test"

    def test_invalid_json_raises_valueerror(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_and_validate("not json at all", MovieReview)

    def test_schema_mismatch_raises_valueerror(self):
        # Missing required field 'summary'
        raw = json.dumps({"title": "X", "rating": 5.0})
        with pytest.raises(ValueError, match="validation failed"):
            parse_and_validate(raw, MovieReview)

    def test_constraint_violation_raises_valueerror(self):
        # rating > 10 violates ge=0, le=10
        raw = json.dumps({"title": "X", "rating": 15.0, "summary": "bad"})
        with pytest.raises(ValueError, match="validation failed"):
            parse_and_validate(raw, MovieReview)


# ─── Agent.infer(output_type=...) ───


def _make_agent(**kwargs):
    """Create a test agent with mocked Ollama detection."""
    with patch("agentu._core.agent._get_ollama_models_sync", return_value=["test:latest"]):
        return Agent("test_agent", model="test:latest", enable_memory=False, **kwargs)


class TestInferOutputType:
    """Integration tests for output_type in Agent.infer()."""

    @pytest.mark.asyncio
    async def test_happy_path_first_attempt(self):
        """LLM returns valid JSON on first try → validated instance returned."""
        agent = _make_agent()
        agent._max_corrections = 2  # has budget, but shouldn't need it

        valid_json = json.dumps({"title": "Inception", "rating": 9.0, "summary": "Amazing"})
        agent._call_llm = AsyncMock(return_value=valid_json)

        result = await agent.infer("Review Inception", output_type=MovieReview)

        assert "structured" in result
        assert isinstance(result["structured"], MovieReview)
        assert result["structured"].title == "Inception"
        assert result["structured"].rating == 9.0
        assert result["attempts"] == 1
        # _call_llm should be called exactly once
        assert agent._call_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        """LLM returns invalid JSON first, then valid JSON → retries and succeeds."""
        agent = _make_agent()
        agent._max_corrections = 2

        invalid_json = json.dumps({"title": "X", "rating": 15.0, "summary": "bad"})
        valid_json = json.dumps({"title": "X", "rating": 8.0, "summary": "good"})
        agent._call_llm = AsyncMock(side_effect=[invalid_json, valid_json])

        result = await agent.infer("Review X", output_type=MovieReview)

        assert isinstance(result["structured"], MovieReview)
        assert result["structured"].rating == 8.0
        assert result["attempts"] == 2
        assert agent._call_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_third_attempt(self):
        """LLM fails twice then succeeds on third try."""
        agent = _make_agent()
        agent._max_corrections = 2  # allows up to 3 total attempts

        bad1 = json.dumps({"title": "X"})  # missing fields
        bad2 = json.dumps({"title": "X", "rating": -1.0, "summary": "s"})  # rating < 0
        good = json.dumps({"title": "X", "rating": 7.0, "summary": "ok"})
        agent._call_llm = AsyncMock(side_effect=[bad1, bad2, good])

        result = await agent.infer("Review X", output_type=MovieReview)

        assert isinstance(result["structured"], MovieReview)
        assert result["attempts"] == 3
        assert agent._call_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises_error(self):
        """All attempts fail → StructuredOutputError raised."""
        agent = _make_agent()
        agent._max_corrections = 1  # 2 total attempts

        bad = json.dumps({"title": "X"})  # always invalid (missing fields)
        agent._call_llm = AsyncMock(return_value=bad)

        with pytest.raises(StructuredOutputError) as exc_info:
            await agent.infer("Review X", output_type=MovieReview)

        err = exc_info.value
        assert err.model is MovieReview
        assert err.attempts == 2
        assert err.raw_output == bad
        assert "validation failed" in err.last_error.lower() or "validation" in str(err).lower()

    @pytest.mark.asyncio
    async def test_no_retries_raises_immediately(self):
        """max_corrections=0 → fails immediately on first invalid output."""
        agent = _make_agent()
        agent._max_corrections = 0

        bad = json.dumps({"title": "X"})
        agent._call_llm = AsyncMock(return_value=bad)

        with pytest.raises(StructuredOutputError) as exc_info:
            await agent.infer("Review X", output_type=MovieReview)

        assert exc_info.value.attempts == 1
        assert agent._call_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_json_triggers_retry(self):
        """LLM returns non-JSON text first, then valid JSON."""
        agent = _make_agent()
        agent._max_corrections = 1

        not_json = "Here's my review: Inception is great!"
        valid_json = json.dumps({"title": "Inception", "rating": 9.0, "summary": "Great"})
        agent._call_llm = AsyncMock(side_effect=[not_json, valid_json])

        result = await agent.infer("Review Inception", output_type=MovieReview)

        assert isinstance(result["structured"], MovieReview)
        assert result["attempts"] == 2

    @pytest.mark.asyncio
    async def test_correction_prompt_includes_error(self):
        """Verify the retry call includes the validation error in the prompt."""
        agent = _make_agent()
        agent._max_corrections = 1

        bad = json.dumps({"title": "X"})
        good = json.dumps({"title": "X", "rating": 5.0, "summary": "ok"})
        agent._call_llm = AsyncMock(side_effect=[bad, good])

        await agent.infer("Review X", output_type=MovieReview)

        # Second call should have a correction prompt
        second_call_args = agent._call_llm.call_args_list[1]
        prompt = second_call_args[0][0] if second_call_args[0] else second_call_args[1].get("prompt", "")
        # The correction prompt should mention the error and schema
        assert "invalid" in prompt.lower() or "schema" in prompt.lower()

    @pytest.mark.asyncio
    async def test_conversation_history_updated(self):
        """Successful output_type call updates conversation_history."""
        agent = _make_agent()
        agent._max_corrections = 0

        valid_json = json.dumps({"title": "T", "rating": 5.0, "summary": "s"})
        agent._call_llm = AsyncMock(return_value=valid_json)

        await agent.infer("Review T", output_type=MovieReview)

        assert len(agent.conversation_history) == 1
        entry = agent.conversation_history[0]
        assert entry["user_input"] == "Review T"
        assert "structured" in entry["response"]

    @pytest.mark.asyncio
    async def test_result_includes_raw_json(self):
        """Result dict includes the raw JSON string under 'result'."""
        agent = _make_agent()
        valid_json = json.dumps({"title": "T", "rating": 5.0, "summary": "s"})
        agent._call_llm = AsyncMock(return_value=valid_json)

        result = await agent.infer("test", output_type=MovieReview)

        assert result["result"] == valid_json
        parsed_back = json.loads(result["result"])
        assert parsed_back["title"] == "T"


# ─── Mutual Exclusion ───


class TestMutualExclusion:
    @pytest.mark.asyncio
    async def test_both_raises_valueerror(self):
        """Passing both output_schema and output_type raises ValueError."""
        agent = _make_agent()

        with pytest.raises(ValueError, match="mutually exclusive"):
            await agent.infer(
                "test",
                output_schema=MovieReview,
                output_type=MovieReview,
            )

    @pytest.mark.asyncio
    async def test_output_schema_alone_works(self):
        """output_schema without output_type still works (backwards compat)."""
        agent = _make_agent()
        valid_json = json.dumps({"title": "T", "rating": 5.0, "summary": "s"})
        agent._call_llm = AsyncMock(return_value=valid_json)

        result = await agent.infer("test", output_schema=MovieReview)

        assert "structured" in result
        assert isinstance(result["structured"], MovieReview)

    @pytest.mark.asyncio
    async def test_output_type_alone_works(self):
        """output_type without output_schema works."""
        agent = _make_agent()
        valid_json = json.dumps({"title": "T", "rating": 5.0, "summary": "s"})
        agent._call_llm = AsyncMock(return_value=valid_json)

        result = await agent.infer("test", output_type=MovieReview)

        assert "structured" in result
        assert isinstance(result["structured"], MovieReview)

    @pytest.mark.asyncio
    async def test_neither_works(self):
        """No output_schema or output_type still works (no structured output)."""
        agent = _make_agent()
        # Without tools and without schema, the agent takes the tool-calling path
        # which requires evaluate_tool_use. Just verify no error from the
        # mutual-exclusion check.
        # We'll mock evaluate_tool_use to return a text response.
        agent.evaluate_tool_use = AsyncMock(return_value={
            "text_response": "Hello!",
        })

        result = await agent.infer("hello")
        assert "text_response" in result


# ─── with_guardrails integration ───


class TestGuardrailsIntegration:
    @pytest.mark.asyncio
    async def test_max_corrections_from_guardrails(self):
        """with_guardrails(max_corrections=N) controls output_type retries."""
        agent = _make_agent()
        agent.with_guardrails(max_corrections=3)

        bad = json.dumps({"title": "X"})
        good = json.dumps({"title": "X", "rating": 5.0, "summary": "ok"})

        # Fail 3 times, succeed on 4th (max_corrections=3 → 4 total attempts)
        agent._call_llm = AsyncMock(side_effect=[bad, bad, bad, good])

        result = await agent.infer("test", output_type=MovieReview)
        assert result["attempts"] == 4
        assert isinstance(result["structured"], MovieReview)

    @pytest.mark.asyncio
    async def test_default_max_corrections_zero(self):
        """Default max_corrections=0 means no retries."""
        agent = _make_agent()
        assert agent._max_corrections == 0

        bad = json.dumps({"title": "X"})
        agent._call_llm = AsyncMock(return_value=bad)

        with pytest.raises(StructuredOutputError) as exc_info:
            await agent.infer("test", output_type=MovieReview)
        assert exc_info.value.attempts == 1


# ─── Edge Cases ───


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_markdown_fenced_json(self):
        """LLM wraps JSON in markdown code fences."""
        agent = _make_agent()
        fenced = '```json\n{"title": "T", "rating": 5.0, "summary": "s"}\n```'
        agent._call_llm = AsyncMock(return_value=fenced)

        result = await agent.infer("test", output_type=MovieReview)
        assert isinstance(result["structured"], MovieReview)

    @pytest.mark.asyncio
    async def test_optional_fields_default(self):
        """Optional fields use defaults when not provided."""
        agent = _make_agent()
        raw = json.dumps({"name": "Alice", "age": 30})
        agent._call_llm = AsyncMock(return_value=raw)

        result = await agent.infer("get user", output_type=UserProfile)
        assert result["structured"].name == "Alice"
        assert result["structured"].email is None

    @pytest.mark.asyncio
    async def test_extra_fields_in_response(self):
        """Extra fields in LLM response are ignored by Pydantic."""
        agent = _make_agent()
        raw = json.dumps({
            "title": "T",
            "rating": 5.0,
            "summary": "s",
            "director": "Unknown",  # extra field
        })
        agent._call_llm = AsyncMock(return_value=raw)

        result = await agent.infer("test", output_type=MovieReview)
        assert isinstance(result["structured"], MovieReview)
        assert not hasattr(result["structured"], "director")


# ─── pydantic_to_json_schema and build_response_format (unchanged) ───


class TestSchemaHelpers:
    def test_pydantic_to_json_schema(self):
        schema = pydantic_to_json_schema(MovieReview)
        assert "properties" in schema
        assert "title" in schema["properties"]
        assert "rating" in schema["properties"]

    def test_build_response_format_from_model(self):
        fmt = build_response_format(MovieReview)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "MovieReview"
        assert fmt["json_schema"]["strict"] is True

    def test_build_response_format_from_dict(self):
        raw_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        fmt = build_response_format(raw_schema, name="custom")
        assert fmt["json_schema"]["name"] == "custom"
        assert fmt["json_schema"]["schema"] == raw_schema
