import pytest
from agentu.middleware.guardrails import (
    Guardrail, GuardrailResult, GuardrailSet, GuardrailError,
    PII, ContentFilter, MaxLength, JSONSchema,
)
from agentu import Agent


# ─── PII Guardrail ───


class TestPII:
    def test_detects_email(self):
        g = PII()
        result = g.check("Contact me at john@example.com")
        assert not result.passed
        assert "email" in result.reason
        assert any("john@example.com" in m for m in result.matches)

    def test_detects_phone(self):
        g = PII()
        result = g.check("Call me at 555-123-4567")
        assert not result.passed
        assert "phone" in result.reason

    def test_detects_ssn(self):
        g = PII()
        result = g.check("My SSN is 123-45-6789")
        assert not result.passed
        assert "ssn" in result.reason

    def test_detects_credit_card(self):
        g = PII()
        result = g.check("Card: 4111-1111-1111-1111")
        assert not result.passed
        assert "credit_card" in result.reason

    def test_clean_text_passes(self):
        g = PII()
        result = g.check("Hello, how are you today?")
        assert result.passed

    def test_selective_detection(self):
        """Only detect specified PII types."""
        g = PII(detect=["email"])
        # Should catch email
        result = g.check("Email: a@b.com, SSN: 123-45-6789")
        assert not result.passed
        assert "email" in result.reason
        # SSN match should not appear in reason since we only detect email
        assert "ssn" not in result.reason

    def test_multiple_pii_types(self):
        g = PII()
        result = g.check("Email: a@b.com and SSN: 123-45-6789")
        assert not result.passed
        assert len(result.matches) >= 2


# ─── ContentFilter Guardrail ───


class TestContentFilter:
    def test_blocks_keyword(self):
        g = ContentFilter(block=["violence", "weapons"])
        result = g.check("This text mentions violence")
        assert not result.passed
        assert "violence" in result.matches

    def test_clean_text_passes(self):
        g = ContentFilter(block=["violence"])
        result = g.check("This is a nice, friendly message")
        assert result.passed

    def test_case_insensitive_default(self):
        g = ContentFilter(block=["BADWORD"])
        result = g.check("this has badword in it")
        assert not result.passed

    def test_case_sensitive(self):
        g = ContentFilter(block=["BADWORD"], case_sensitive=True)
        # Lowercase should pass
        result = g.check("this has badword in it")
        assert result.passed
        # Exact case should fail
        result = g.check("this has BADWORD in it")
        assert not result.passed

    def test_multiple_blocked_keywords(self):
        g = ContentFilter(block=["spam", "scam"])
        result = g.check("This is a spam scam message")
        assert not result.passed
        assert len(result.matches) == 2

    def test_empty_block_list(self):
        g = ContentFilter(block=[])
        result = g.check("Anything goes")
        assert result.passed


# ─── MaxLength Guardrail ───


class TestMaxLength:
    def test_within_limit_passes(self):
        g = MaxLength(max_chars=100)
        result = g.check("Short text")
        assert result.passed

    def test_exceeds_limit_fails(self):
        g = MaxLength(max_chars=10)
        result = g.check("This is more than ten characters")
        assert not result.passed
        assert "exceeds limit" in result.reason

    def test_exact_limit_passes(self):
        g = MaxLength(max_chars=5)
        result = g.check("Hello")
        assert result.passed

    def test_one_over_limit_fails(self):
        g = MaxLength(max_chars=5)
        result = g.check("Hello!")
        assert not result.passed


# ─── JSONSchema Guardrail ───


class TestJSONSchema:
    def test_invalid_json_fails(self):
        g = JSONSchema(schema={"required": ["name"]})
        result = g.check("not valid json")
        assert not result.passed
        assert "Invalid JSON" in result.reason

    def test_missing_required_field(self):
        g = JSONSchema(schema={
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}
        })
        result = g.check('{"name": "Alice"}')
        assert not result.passed
        assert "age" in result.reason

    def test_valid_schema(self):
        g = JSONSchema(schema={
            "required": ["name"],
            "properties": {"name": {"type": "string"}}
        })
        result = g.check('{"name": "Alice"}')
        assert result.passed

    def test_type_error(self):
        g = JSONSchema(schema={
            "required": ["count"],
            "properties": {"count": {"type": "integer"}}
        })
        result = g.check('{"count": "not a number"}')
        assert not result.passed
        assert "Type errors" in result.reason

    def test_pydantic_model_valid(self):
        try:
            from pydantic import BaseModel

            class UserModel(BaseModel):
                name: str
                age: int

            g = JSONSchema(model=UserModel)
            result = g.check('{"name": "Alice", "age": 30}')
            assert result.passed
        except ImportError:
            pytest.skip("pydantic not available")

    def test_pydantic_model_invalid(self):
        try:
            from pydantic import BaseModel

            class UserModel(BaseModel):
                name: str
                age: int

            g = JSONSchema(model=UserModel)
            result = g.check('{"name": "Alice"}')
            assert not result.passed
        except ImportError:
            pytest.skip("pydantic not available")

    def test_no_schema_no_model_passes(self):
        """With no schema or model, any valid JSON passes."""
        g = JSONSchema()
        result = g.check('{"anything": true}')
        assert result.passed


# ─── GuardrailSet ───


class TestGuardrailSet:
    def test_all_pass(self):
        gs = GuardrailSet([
            MaxLength(max_chars=1000),
            ContentFilter(block=["spam"]),
        ])
        failures = gs.check("Hello world")
        assert failures == []

    def test_one_fails(self):
        gs = GuardrailSet([
            MaxLength(max_chars=1000),
            ContentFilter(block=["spam"]),
        ])
        failures = gs.check("This is spam content")
        assert len(failures) == 1
        assert failures[0].guardrail == "content_filter"

    def test_multiple_failures(self):
        gs = GuardrailSet([
            MaxLength(max_chars=5),
            ContentFilter(block=["bad"]),
        ])
        failures = gs.check("This is bad and long text")
        assert len(failures) == 2

    def test_check_or_raise_passes(self):
        gs = GuardrailSet([MaxLength(max_chars=1000)])
        gs.check_or_raise("short text")  # Should not raise

    def test_check_or_raise_fails(self):
        gs = GuardrailSet([MaxLength(max_chars=5)])
        with pytest.raises(GuardrailError) as exc_info:
            gs.check_or_raise("this is too long", direction="input")
        assert "input" in str(exc_info.value)
        assert len(exc_info.value.failures) == 1


# ─── Base Guardrail ───


class TestBaseGuardrail:
    def test_base_always_passes(self):
        g = Guardrail()
        result = g.check("anything")
        assert result.passed

    def test_result_dataclass(self):
        r = GuardrailResult(passed=False, guardrail="test", reason="bad", matches=["x"])
        assert not r.passed
        assert r.guardrail == "test"
        assert r.reason == "bad"
        assert r.matches == ["x"]


# ─── Agent Integration ───


class TestAgentIntegration:
    def test_with_guardrails_returns_self(self):
        agent = Agent("test", model="test-model")
        result = agent.with_guardrails(
            input_guardrails=[PII()],
            output_guardrails=[ContentFilter(block=["violence"])]
        )
        assert result is agent
        assert agent._input_guardrails is not None
        assert agent._output_guardrails is not None

    def test_chaining_with_guardrails(self):
        agent = (
            Agent("test", model="test-model")
            .with_guardrails(input_guardrails=[MaxLength(max_chars=5000)])
        )
        assert agent._input_guardrails is not None

    def test_no_guardrails_by_default(self):
        agent = Agent("test", model="test-model")
        assert agent._input_guardrails is None
        assert agent._output_guardrails is None
