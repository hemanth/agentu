"""Tests for harness engineering features: self-correction, rules, permissions."""

import os
import asyncio
import tempfile
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agentu import Agent, Tool, ToolPermission
from agentu.middleware.guardrails import (
    ContentFilter, PII, MaxLength, GuardrailSet, GuardrailError
)
from agentu.middleware.observe import EventType


# ──────────────────────────────────────────────
# 1. Tool Permissions
# ──────────────────────────────────────────────

class TestToolPermissions:
    """Test tool permission scoping."""

    def test_tool_default_permission_is_write(self):
        """Default permission should be WRITE."""
        def my_func():
            pass
        tool = Tool(my_func)
        assert tool.permission == ToolPermission.WRITE

    def test_tool_permission_readonly(self):
        """Can set READONLY permission."""
        def read_data():
            pass
        tool = Tool(read_data, permission=ToolPermission.READONLY)
        assert tool.permission == ToolPermission.READONLY

    def test_tool_permission_dangerous(self):
        """Can set DANGEROUS permission."""
        def delete_all():
            pass
        tool = Tool(delete_all, permission=ToolPermission.DANGEROUS)
        assert tool.permission == ToolPermission.DANGEROUS

    @pytest.mark.asyncio
    async def test_dangerous_tool_blocked_by_default(self):
        """DANGEROUS tools should raise PermissionError by default."""
        def destroy():
            return "destroyed"

        agent = Agent("test")
        agent.with_tools([Tool(destroy, permission=ToolPermission.DANGEROUS)])

        with pytest.raises(PermissionError, match="DANGEROUS"):
            await agent.call("destroy", {})

    @pytest.mark.asyncio
    async def test_dangerous_tool_allowed_with_permissions(self):
        """DANGEROUS tools should work when explicitly allowed."""
        def destroy():
            return "destroyed"

        agent = Agent("test")
        agent.with_tools([Tool(destroy, permission=ToolPermission.DANGEROUS)])
        agent.with_permissions(allow_dangerous=True)

        result = await agent.call("destroy", {})
        assert result == "destroyed"

    @pytest.mark.asyncio
    async def test_readonly_tool_always_works(self):
        """READONLY tools should always work regardless of permissions."""
        def read_data():
            return "data"

        agent = Agent("test")
        agent.with_tools([Tool(read_data, permission=ToolPermission.READONLY)])

        result = await agent.call("read_data", {})
        assert result == "data"

    @pytest.mark.asyncio
    async def test_write_tool_always_works(self):
        """WRITE (default) tools should always work."""
        def save_data():
            return "saved"

        agent = Agent("test")
        agent.with_tools([Tool(save_data)])

        result = await agent.call("save_data", {})
        assert result == "saved"

    @pytest.mark.asyncio
    async def test_tool_blocked_event_recorded(self):
        """Blocked tool should record TOOL_BLOCKED event."""
        def destroy():
            return "destroyed"

        agent = Agent("test")
        agent.with_tools([Tool(destroy, permission=ToolPermission.DANGEROUS)])

        try:
            await agent.call("destroy", {})
        except PermissionError:
            pass

        events = agent.observer.get_events()
        blocked_events = [e for e in events if e.get("event") == "tool_blocked"]
        assert len(blocked_events) == 1
        assert blocked_events[0]["tool_name"] == "destroy"

    def test_with_permissions_chaining(self):
        """with_permissions should return self for chaining."""
        agent = Agent("test")
        result = agent.with_permissions(allow_dangerous=True)
        assert result is agent


# ──────────────────────────────────────────────
# 2. Rule File Loader
# ──────────────────────────────────────────────

class TestRuleFileLoader:
    """Test with_rules() feedforward guide loading."""

    def test_with_rules_loads_file(self):
        """with_rules should load file content into context."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Rules\n- Never reveal passwords\n- Always be polite")
            f.flush()
            
            agent = Agent("test")
            agent.with_rules(f.name)
            
            assert "Never reveal passwords" in agent.context
            assert "=== Project Rules ===" in agent.context
            assert "=== End Rules ===" in agent.context
            
            os.unlink(f.name)

    def test_with_rules_prepends_to_existing_context(self):
        """Rules should be prepended to existing context."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("Be safe")
            f.flush()
            
            agent = Agent("test")
            agent.set_context("You are a helpful assistant")
            agent.with_rules(f.name)
            
            # Rules should come before the existing context
            rules_pos = agent.context.index("Be safe")
            context_pos = agent.context.index("You are a helpful assistant")
            assert rules_pos < context_pos
            
            os.unlink(f.name)

    def test_with_rules_missing_file_warns(self):
        """Missing rules file should warn and return self."""
        agent = Agent("test")
        agent.set_context("original context")
        result = agent.with_rules("/nonexistent/AGENTS.md")
        
        assert result is agent
        assert agent.context == "original context"

    def test_with_rules_chaining(self):
        """with_rules should return self for chaining."""
        agent = Agent("test")
        result = agent.with_rules("/nonexistent/file.md")
        assert result is agent

    def test_with_rules_default_path(self):
        """Default path should be AGENTS.md."""
        agent = Agent("test")
        # Should not raise, just warn about missing file
        agent.with_rules()


# ──────────────────────────────────────────────
# 3. Self-Correction Loop
# ──────────────────────────────────────────────

class TestSelfCorrection:
    """Test output guardrail self-correction loop."""

    def test_with_guardrails_max_corrections_default(self):
        """Default max_corrections should be 2."""
        agent = Agent("test").with_guardrails(
            output_guardrails=[ContentFilter(block=["bad"])]
        )
        assert agent._max_corrections == 2

    def test_with_guardrails_max_corrections_custom(self):
        """Can set custom max_corrections."""
        agent = Agent("test").with_guardrails(
            output_guardrails=[ContentFilter(block=["bad"])],
            max_corrections=5,
        )
        assert agent._max_corrections == 5

    def test_with_guardrails_max_corrections_zero(self):
        """Setting max_corrections=0 disables self-correction."""
        agent = Agent("test").with_guardrails(
            output_guardrails=[ContentFilter(block=["bad"])],
            max_corrections=0,
        )
        assert agent._max_corrections == 0

    @pytest.mark.asyncio
    async def test_self_correction_succeeds_on_retry(self):
        """Agent should self-correct when first response fails guardrails."""
        agent = Agent("test")
        agent.with_guardrails(
            output_guardrails=[ContentFilter(block=["badword"])],
            max_corrections=2,
        )

        # First call returns bad content, second returns clean content
        call_count = 0
        async def mock_raw_llm_call(prompt, output_schema=None, images=None, max_retries=2):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "This contains badword in it"
            return "This is a clean response"

        agent._raw_llm_call = mock_raw_llm_call

        result = await agent._call_llm("Tell me something")
        assert result == "This is a clean response"
        assert call_count == 2  # Called twice: original + 1 correction

    @pytest.mark.asyncio
    async def test_self_correction_records_events(self):
        """Self-correction should record SELF_CORRECTION events."""
        agent = Agent("test")
        agent.with_guardrails(
            output_guardrails=[ContentFilter(block=["badword"])],
            max_corrections=2,
        )

        call_count = 0
        async def mock_raw_llm_call(prompt, output_schema=None, images=None, max_retries=2):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "This contains badword"
            return "Clean response"

        agent._raw_llm_call = mock_raw_llm_call

        await agent._call_llm("Test")

        events = agent.observer.get_events()
        correction_events = [e for e in events if e.get("event") == "self_correction"]
        assert len(correction_events) == 1
        assert correction_events[0]["attempt"] == 1

    @pytest.mark.asyncio
    async def test_self_correction_exhausted_raises(self):
        """Should raise GuardrailError after exhausting correction attempts."""
        agent = Agent("test")
        agent.with_guardrails(
            output_guardrails=[ContentFilter(block=["badword"])],
            max_corrections=1,
        )

        # Always returns bad content
        async def mock_raw_llm_call(prompt, output_schema=None, images=None, max_retries=2):
            return "This always contains badword"

        agent._raw_llm_call = mock_raw_llm_call

        with pytest.raises(GuardrailError, match="Blocked content detected"):
            await agent._call_llm("Test")

    @pytest.mark.asyncio
    async def test_no_correction_when_disabled(self):
        """max_corrections=0 should raise immediately on guardrail failure."""
        agent = Agent("test")
        agent.with_guardrails(
            output_guardrails=[ContentFilter(block=["badword"])],
            max_corrections=0,
        )

        call_count = 0
        async def mock_raw_llm_call(prompt, output_schema=None, images=None, max_retries=2):
            nonlocal call_count
            call_count += 1
            return "Contains badword"

        agent._raw_llm_call = mock_raw_llm_call

        with pytest.raises(GuardrailError):
            await agent._call_llm("Test")
        
        assert call_count == 1  # Only called once, no retry

    @pytest.mark.asyncio
    async def test_no_correction_when_response_is_clean(self):
        """Clean responses should not trigger any correction."""
        agent = Agent("test")
        agent.with_guardrails(
            output_guardrails=[ContentFilter(block=["badword"])],
            max_corrections=2,
        )

        call_count = 0
        async def mock_raw_llm_call(prompt, output_schema=None, images=None, max_retries=2):
            nonlocal call_count
            call_count += 1
            return "Perfectly clean response"

        agent._raw_llm_call = mock_raw_llm_call

        result = await agent._call_llm("Test")
        assert result == "Perfectly clean response"
        assert call_count == 1  # Only called once


# ──────────────────────────────────────────────
# 4. Integration: Config with rules
# ──────────────────────────────────────────────

class TestConfigRules:
    """Test that AgentConfig supports rules field."""

    def test_config_accepts_rules(self):
        """AgentConfig should accept a rules field."""
        import json
        
        config = {
            "name": "test-agent",
            "model": "openai/gpt-4o",
            "rules": "AGENTS.md"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            f.flush()
            
            from agentu._core.config import AgentConfig
            cfg = AgentConfig.load(f.name)
            
            assert cfg.rules == "AGENTS.md"
            
            os.unlink(f.name)
