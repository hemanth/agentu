"""Tests for lethal-trifecta accounting and spotlighting (§5.1)."""

import logging
import pytest

from agentu._core.tools import Tool, ToolPermission
from agentu._core.safety import (
    check_lethal_trifecta,
    spotlight_untrusted,
    TrifectaReport,
    UNTRUSTED_OPEN,
    UNTRUSTED_CLOSE,
    UNTRUSTED_SUFFIX,
)
from agentu import Agent


# ── Helpers ──────────────────────────────────────────────────────────────

def _noop():
    """No-op tool function."""
    return "ok"


def _make_tool(name: str, **capability_flags) -> Tool:
    """Create a named Tool with specified capability flags."""
    return Tool(_noop, name=name, **capability_flags)


# ── Tool capability flags ────────────────────────────────────────────────


class TestToolCapabilityFlags:
    """Verify that Tool stores the three capability flags correctly."""

    def test_defaults_are_false(self):
        t = Tool(_noop)
        assert t.reads_private is False
        assert t.ingests_untrusted is False
        assert t.communicates_externally is False

    def test_reads_private_flag(self):
        t = Tool(_noop, reads_private=True)
        assert t.reads_private is True
        assert t.ingests_untrusted is False
        assert t.communicates_externally is False

    def test_ingests_untrusted_flag(self):
        t = Tool(_noop, ingests_untrusted=True)
        assert t.ingests_untrusted is True

    def test_communicates_externally_flag(self):
        t = Tool(_noop, communicates_externally=True)
        assert t.communicates_externally is True

    def test_all_flags(self):
        t = Tool(
            _noop,
            reads_private=True,
            ingests_untrusted=True,
            communicates_externally=True,
        )
        assert t.reads_private is True
        assert t.ingests_untrusted is True
        assert t.communicates_externally is True


# ── Spotlighting ─────────────────────────────────────────────────────────


class TestSpotlightUntrusted:
    """spotlight_untrusted() wraps text in XML-style delimiters."""

    def test_wraps_result(self):
        raw = "Hello from the internet!"
        wrapped = spotlight_untrusted(raw)
        assert wrapped.startswith(UNTRUSTED_OPEN)
        assert raw in wrapped
        assert UNTRUSTED_CLOSE in wrapped
        assert UNTRUSTED_SUFFIX in wrapped

    def test_preserves_original_content(self):
        raw = "line1\nline2\nline3"
        wrapped = spotlight_untrusted(raw)
        assert raw in wrapped

    def test_empty_string(self):
        wrapped = spotlight_untrusted("")
        assert UNTRUSTED_OPEN in wrapped
        assert UNTRUSTED_CLOSE in wrapped

    def test_structure_order(self):
        """Open tag → content → close tag → suffix."""
        wrapped = spotlight_untrusted("data")
        open_pos = wrapped.index(UNTRUSTED_OPEN)
        data_pos = wrapped.index("data")
        close_pos = wrapped.index(UNTRUSTED_CLOSE)
        suffix_pos = wrapped.index(UNTRUSTED_SUFFIX)
        assert open_pos < data_pos < close_pos < suffix_pos


# ── Lethal-trifecta accounting ───────────────────────────────────────────


class TestCheckLethalTrifecta:
    """check_lethal_trifecta() correctly identifies the dangerous combo."""

    def test_no_tools(self):
        report = check_lethal_trifecta([])
        assert report.has_trifecta is False
        assert report.message == ""

    def test_no_flags_set(self):
        tools = [_make_tool("a"), _make_tool("b")]
        report = check_lethal_trifecta(tools)
        assert report.has_trifecta is False

    def test_single_flag_only(self):
        tools = [_make_tool("reader", reads_private=True)]
        report = check_lethal_trifecta(tools)
        assert report.has_trifecta is False

    def test_two_of_three_flags(self):
        tools = [
            _make_tool("reader", reads_private=True),
            _make_tool("ingester", ingests_untrusted=True),
        ]
        report = check_lethal_trifecta(tools)
        assert report.has_trifecta is False

    def test_all_three_flags_across_tools(self):
        tools = [
            _make_tool("reader", reads_private=True),
            _make_tool("ingester", ingests_untrusted=True),
            _make_tool("sender", communicates_externally=True),
        ]
        report = check_lethal_trifecta(tools)
        assert report.has_trifecta is True
        assert "reader" in report.reads_private_tools
        assert "ingester" in report.ingests_untrusted_tools
        assert "sender" in report.communicates_externally_tools
        assert "LETHAL TRIFECTA" in report.message

    def test_all_three_flags_on_single_tool(self):
        tools = [
            _make_tool(
                "swiss_army",
                reads_private=True,
                ingests_untrusted=True,
                communicates_externally=True,
            ),
        ]
        report = check_lethal_trifecta(tools)
        assert report.has_trifecta is True
        assert report.reads_private_tools == ["swiss_army"]
        assert report.ingests_untrusted_tools == ["swiss_army"]
        assert report.communicates_externally_tools == ["swiss_army"]

    def test_logs_warning(self, caplog):
        tools = [
            _make_tool("r", reads_private=True),
            _make_tool("i", ingests_untrusted=True),
            _make_tool("c", communicates_externally=True),
        ]
        with caplog.at_level(logging.WARNING, logger="agentu._core.safety"):
            check_lethal_trifecta(tools)
        assert any("LETHAL TRIFECTA" in m for m in caplog.messages)

    def test_no_warning_when_safe(self, caplog):
        tools = [_make_tool("safe")]
        with caplog.at_level(logging.WARNING, logger="agentu._core.safety"):
            check_lethal_trifecta(tools)
        assert not any("LETHAL TRIFECTA" in m for m in caplog.messages)

    def test_report_message_contains_tool_names(self):
        tools = [
            _make_tool("db_reader", reads_private=True),
            _make_tool("web_fetch", ingests_untrusted=True),
            _make_tool("email_send", communicates_externally=True),
        ]
        report = check_lethal_trifecta(tools)
        assert "db_reader" in report.message
        assert "web_fetch" in report.message
        assert "email_send" in report.message

    def test_multiple_tools_per_capability(self):
        tools = [
            _make_tool("r1", reads_private=True),
            _make_tool("r2", reads_private=True),
            _make_tool("i1", ingests_untrusted=True),
            _make_tool("c1", communicates_externally=True),
        ]
        report = check_lethal_trifecta(tools)
        assert report.has_trifecta is True
        assert len(report.reads_private_tools) == 2


# ── TrifectaReport dataclass ────────────────────────────────────────────


class TestTrifectaReport:
    def test_default_values(self):
        r = TrifectaReport()
        assert r.has_trifecta is False
        assert r.reads_private_tools == []
        assert r.ingests_untrusted_tools == []
        assert r.communicates_externally_tools == []
        assert r.message == ""

    def test_custom_values(self):
        r = TrifectaReport(
            has_trifecta=True,
            reads_private_tools=["a"],
            ingests_untrusted_tools=["b"],
            communicates_externally_tools=["c"],
            message="danger",
        )
        assert r.has_trifecta is True
        assert r.message == "danger"


# ── Agent integration ────────────────────────────────────────────────────


class TestAgentTrifectaIntegration:
    """with_tools() runs the trifecta check and stores the report."""

    def test_no_trifecta_no_report(self):
        agent = Agent("test", model="test-model")
        agent.with_tools([_make_tool("safe")])
        assert not hasattr(agent, "_trifecta_report") or not getattr(
            agent, "_trifecta_report", None
        )

    def test_trifecta_detected_stores_report(self):
        agent = Agent("test", model="test-model")
        agent.with_tools([
            _make_tool("r", reads_private=True),
            _make_tool("i", ingests_untrusted=True),
            _make_tool("c", communicates_externally=True),
        ])
        report = getattr(agent, "_trifecta_report", None)
        assert report is not None
        assert report.has_trifecta is True

    def test_trifecta_across_active_and_deferred(self):
        agent = Agent("test", model="test-model")
        agent.with_tools(
            tools=[_make_tool("r", reads_private=True)],
            defer=[
                _make_tool("i", ingests_untrusted=True),
                _make_tool("c", communicates_externally=True),
            ],
        )
        report = getattr(agent, "_trifecta_report", None)
        assert report is not None
        assert report.has_trifecta is True

    def test_with_tools_returns_self(self):
        agent = Agent("test", model="test-model")
        result = agent.with_tools([
            _make_tool("r", reads_private=True),
            _make_tool("i", ingests_untrusted=True),
            _make_tool("c", communicates_externally=True),
        ])
        assert result is agent


class TestAgentSpotlighting:
    """call() wraps results from ingests_untrusted tools."""

    @pytest.mark.asyncio
    async def test_untrusted_tool_result_is_spotlighted(self):
        def fetch_web(url: str) -> str:
            """Fetch a web page."""
            return "Hello from the web"

        agent = Agent("test", model="test-model")
        agent.with_tools([
            Tool(fetch_web, ingests_untrusted=True),
        ])
        result = await agent.call("fetch_web", {"url": "https://example.com"})
        assert UNTRUSTED_OPEN in result
        assert UNTRUSTED_CLOSE in result
        assert UNTRUSTED_SUFFIX in result
        assert "Hello from the web" in result

    @pytest.mark.asyncio
    async def test_trusted_tool_result_is_not_spotlighted(self):
        def safe_calc(x: int) -> str:
            """Calculate something."""
            return "42"

        agent = Agent("test", model="test-model")
        agent.with_tools([Tool(safe_calc)])
        result = await agent.call("safe_calc", {"x": 1})
        assert UNTRUSTED_OPEN not in str(result)

    @pytest.mark.asyncio
    async def test_non_string_result_is_not_spotlighted(self):
        def fetch_data() -> dict:
            """Fetch structured data."""
            return {"key": "value"}

        agent = Agent("test", model="test-model")
        agent.with_tools([
            Tool(fetch_data, ingests_untrusted=True),
        ])
        result = await agent.call("fetch_data", {})
        # Non-string results should pass through unchanged
        assert result == {"key": "value"}
