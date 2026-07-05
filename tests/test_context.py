"""Tests for context management and compaction."""

import pytest
from agentu._core.context import (
    ContextConfig,
    CompactionMode,
    ContextStats,
    estimate_tokens,
    estimate_history_tokens,
    truncate_tool_results,
    drop_old_turns,
    summarize_turns,
    compact_context,
)


class TestTokenEstimation:
    """Test approximate token counting."""

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        # 20 chars / 4 = 5 tokens
        assert estimate_tokens("Hello World! How are") == 5

    def test_estimate_history_tokens(self):
        history = [
            {"user_input": "Hello", "response": {"result": "Hi"}},
            {"user_input": "Bye", "response": {"result": "Goodbye"}},
        ]
        tokens = estimate_history_tokens(history)
        assert tokens > 0


class TestTruncateToolResults:
    """Test Tier 1 compaction: truncating old tool results."""

    def test_no_truncation_when_under_keep_recent(self):
        history = [
            {"user_input": "q1", "response": {"result": "x" * 5000}},
            {"user_input": "q2", "response": {"result": "y" * 5000}},
        ]
        result = truncate_tool_results(history, max_result_chars=100, keep_recent=5)
        # Both are within keep_recent, so no truncation
        assert len(result[0]["response"]["result"]) == 5000

    def test_truncates_old_results(self):
        history = [
            {"user_input": "old", "response": {"result": "x" * 5000}},
            {"user_input": "recent1", "response": {"result": "y" * 100}},
            {"user_input": "recent2", "response": {"result": "z" * 100}},
        ]
        result = truncate_tool_results(history, max_result_chars=200, keep_recent=2)
        # Old entry should be truncated
        assert len(result[0]["response"]["result"]) < 5000
        assert "truncated" in result[0]["response"]["result"]
        # Recent entries should be intact
        assert len(result[1]["response"]["result"]) == 100

    def test_preserves_short_results(self):
        history = [
            {"user_input": "old", "response": {"result": "short"}},
            {"user_input": "recent", "response": {"result": "also short"}},
        ]
        result = truncate_tool_results(history, max_result_chars=200, keep_recent=1)
        assert result[0]["response"]["result"] == "short"

    def test_truncates_nested_history(self):
        history = [
            {
                "user_input": "old",
                "response": {
                    "result": "ok",
                    "history": [
                        {"result": "x" * 5000},
                    ]
                },
            },
            {"user_input": "recent", "response": {"result": "y"}},
        ]
        result = truncate_tool_results(history, max_result_chars=200, keep_recent=1)
        nested = result[0]["response"]["history"][0]["result"]
        assert "truncated" in nested


class TestDropOldTurns:
    """Test emergency compaction: dropping oldest turns."""

    def test_drops_oldest_to_fit_budget(self):
        history = [
            {"user_input": f"q{i}", "response": {"result": "x" * 1000}}
            for i in range(10)
        ]
        original_len = len(history)
        result = drop_old_turns(history, max_tokens=500, keep_recent=2)
        assert len(result) < original_len
        assert len(result) >= 2  # Keep at least keep_recent

    def test_no_drop_when_under_budget(self):
        history = [
            {"user_input": "q1", "response": {"result": "short"}},
        ]
        result = drop_old_turns(history, max_tokens=10000, keep_recent=1)
        assert len(result) == 1


class TestSummarizeTurns:
    """Test Tier 2 compaction: LLM summarization."""

    @pytest.mark.asyncio
    async def test_summarizes_old_turns(self):
        history = [
            {"user_input": f"q{i}", "response": {"result": f"r{i}", "tool_used": "tool"}}
            for i in range(8)
        ]

        async def mock_llm(prompt):
            return "Summary: 8 queries were made using tool."

        result = await summarize_turns(history, mock_llm, keep_recent=3)
        # Should have 1 summary + 3 recent
        assert len(result) == 4
        assert result[0]["user_input"] == "[Context Summary]"
        assert "Summary" in result[0]["response"]["text_response"]

    @pytest.mark.asyncio
    async def test_no_summarization_when_under_keep_recent(self):
        history = [
            {"user_input": "q1", "response": {"result": "r1"}},
        ]

        async def mock_llm(prompt):
            return "Should not be called"

        result = await summarize_turns(history, mock_llm, keep_recent=5)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        history = [
            {"user_input": f"q{i}", "response": {"result": f"r{i}"}}
            for i in range(8)
        ]

        async def failing_llm(prompt):
            raise Exception("LLM unavailable")

        result = await summarize_turns(history, failing_llm, keep_recent=3)
        # Should fall back to just keeping recent
        assert len(result) == 3


class TestCompactContext:
    """Test the full compaction pipeline."""

    @pytest.mark.asyncio
    async def test_no_compaction_when_mode_none(self):
        history = [
            {"user_input": f"q{i}", "response": {"result": "x" * 5000}}
            for i in range(20)
        ]
        config = ContextConfig(compaction=CompactionMode.NONE)
        result, stats = await compact_context(history, config)
        assert len(result) == 20
        assert stats.compactions_performed == 0

    @pytest.mark.asyncio
    async def test_truncate_mode(self):
        history = [
            {"user_input": f"q{i}", "response": {"result": "x" * 5000}}
            for i in range(10)
        ]
        config = ContextConfig(
            compaction=CompactionMode.TRUNCATE,
            max_tokens=100_000,
            max_result_chars=200,
            keep_recent=2,
        )
        result, stats = await compact_context(history, config)
        # Old results should be truncated
        for entry in result[:-2]:
            assert len(entry["response"]["result"]) <= 250  # 200 + truncation message

    @pytest.mark.asyncio
    async def test_auto_mode_with_llm(self):
        # Create history that exceeds budget after truncation
        history = [
            {"user_input": f"q{i}", "response": {"result": f"r{i}", "tool_used": "tool"}}
            for i in range(20)
        ]
        config = ContextConfig(
            compaction=CompactionMode.AUTO,
            max_tokens=50,  # Very low budget to trigger summarization
            max_result_chars=200,
            keep_recent=3,
        )

        async def mock_llm(prompt):
            return "Summary of conversation"

        result, stats = await compact_context(history, config, llm_call=mock_llm)
        # Should have been compacted
        assert len(result) <= 4  # 1 summary + 3 recent

    @pytest.mark.asyncio
    async def test_auto_mode_without_llm(self):
        history = [
            {"user_input": f"q{i}", "response": {"result": f"r{i}"}}
            for i in range(20)
        ]
        config = ContextConfig(
            compaction=CompactionMode.AUTO,
            max_tokens=50,
            keep_recent=3,
        )
        result, stats = await compact_context(history, config, llm_call=None)
        # Should fall back to dropping
        assert len(result) >= 3


class TestContextConfig:
    """Test context configuration."""

    def test_default_config(self):
        config = ContextConfig()
        assert config.max_tokens == 128_000
        assert config.compaction == CompactionMode.NONE
        assert config.keep_recent == 5

    def test_custom_config(self):
        config = ContextConfig(
            max_tokens=50_000,
            compaction=CompactionMode.AUTO,
            keep_recent=10,
        )
        assert config.max_tokens == 50_000
        assert config.compaction == CompactionMode.AUTO

    def test_compaction_mode_values(self):
        assert CompactionMode("none") == CompactionMode.NONE
        assert CompactionMode("auto") == CompactionMode.AUTO
        assert CompactionMode("truncate") == CompactionMode.TRUNCATE
        assert CompactionMode("summarize") == CompactionMode.SUMMARIZE
