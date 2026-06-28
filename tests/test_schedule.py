"""Tests for scheduled automations (loop engineering)."""

import pytest
import asyncio
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agentu.workflow.schedule import (
    CronParser, ScheduleConfig, ScheduleStore, Scheduler, Finding, FindingStatus
)


class TestCronParser:
    """Tests for lightweight cron expression parser."""

    def test_wildcard_matches_any(self):
        """Test that * matches any value."""
        assert CronParser.matches("* * * * *") is True

    def test_exact_minute_match(self):
        dt = datetime(2024, 6, 15, 9, 30, 0)
        assert CronParser.matches("30 * * * *", dt) is True
        assert CronParser.matches("31 * * * *", dt) is False

    def test_exact_hour_match(self):
        dt = datetime(2024, 6, 15, 9, 0, 0)
        assert CronParser.matches("0 9 * * *", dt) is True
        assert CronParser.matches("0 10 * * *", dt) is False

    def test_step_expression(self):
        """Test */N step expressions."""
        dt = datetime(2024, 6, 15, 9, 0, 0)
        assert CronParser.matches("*/5 * * * *", dt) is True  # 0 % 5 == 0

        dt = datetime(2024, 6, 15, 9, 15, 0)
        assert CronParser.matches("*/5 * * * *", dt) is True  # 15 % 5 == 0

        dt = datetime(2024, 6, 15, 9, 7, 0)
        assert CronParser.matches("*/5 * * * *", dt) is False  # 7 % 5 != 0

    def test_range_expression(self):
        """Test N-M range expressions."""
        dt = datetime(2024, 6, 15, 9, 0, 0)
        assert CronParser.matches("* 8-17 * * *", dt) is True  # 9 in 8-17
        assert CronParser.matches("* 10-17 * * *", dt) is False  # 9 not in 10-17

    def test_list_expression(self):
        """Test N,M list expressions."""
        dt = datetime(2024, 6, 15, 9, 0, 0)
        assert CronParser.matches("0 9,17 * * *", dt) is True
        assert CronParser.matches("0 8,10 * * *", dt) is False

    def test_day_of_week(self):
        """Test day of week field (0=Sunday)."""
        # 2024-06-15 is a Saturday
        dt = datetime(2024, 6, 15, 9, 0, 0)
        assert CronParser.matches("0 9 * * 6", dt) is True  # Saturday
        assert CronParser.matches("0 9 * * 1", dt) is False  # Monday

    def test_invalid_expression_raises(self):
        """Test that invalid expression raises ValueError."""
        with pytest.raises(ValueError, match="5 fields"):
            CronParser.matches("* * *")

    def test_complex_expression(self):
        """Test complex multi-field expression."""
        # 9:30 AM on the 15th of June
        dt = datetime(2024, 6, 15, 9, 30, 0)
        assert CronParser.matches("30 9 15 6 *", dt) is True
        assert CronParser.matches("30 9 15 7 *", dt) is False

    def test_next_minute_boundary(self):
        """Test next minute boundary calculation."""
        seconds = CronParser.next_minute_boundary()
        assert 0 < seconds <= 60

    def test_range_with_step(self):
        """Test N-M/S range with step."""
        dt = datetime(2024, 6, 15, 9, 10, 0)
        assert CronParser.matches("0-30/10 * * * *", dt) is True  # 10 in 0-30, step 10
        dt = datetime(2024, 6, 15, 9, 15, 0)
        assert CronParser.matches("0-30/10 * * * *", dt) is False  # 15 not on step 10


class TestScheduleConfig:
    """Tests for ScheduleConfig."""

    def test_defaults(self):
        config = ScheduleConfig()
        assert config.every is None
        assert config.cron is None
        assert config.max_runs is None
        assert config.id is not None  # auto-generated

    def test_with_interval(self):
        config = ScheduleConfig(every=30, prompt="Check issues")
        assert config.every == 30
        assert config.prompt == "Check issues"


class TestScheduleStore:
    """Tests for SQLite-backed schedule storage."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temp store."""
        return ScheduleStore(db_path=str(tmp_path / "test_schedules.db"))

    def test_save_and_get_findings(self, store):
        """Test saving and retrieving findings."""
        finding = Finding(
            id="f1",
            schedule_id="s1",
            result="Found 3 issues",
            status="pending",
            duration_ms=150.0,
            iteration=1,
        )
        store.save_finding(finding)

        results = store.get_findings(status="pending")
        assert len(results) == 1
        assert results[0].id == "f1"
        assert results[0].result == "Found 3 issues"

    def test_update_finding_status(self, store):
        """Test status updates."""
        finding = Finding(id="f2", schedule_id="s1", result="test")
        store.save_finding(finding)

        store.update_finding_status("f2", "archived")
        pending = store.get_findings(status="pending")
        archived = store.get_findings(status="archived")

        assert len(pending) == 0
        assert len(archived) == 1
        assert archived[0].id == "f2"

    def test_empty_findings(self, store):
        """Test empty results."""
        results = store.get_findings()
        assert results == []

    def test_save_schedule(self, store):
        """Test saving a schedule config."""
        config = ScheduleConfig(every=30, prompt="test")
        store.save_schedule("test-agent", config)
        # No error = success. We verify via internal DB state.
        with sqlite3.connect(store.db_path) as conn:
            rows = conn.execute("SELECT * FROM schedules").fetchall()
        assert len(rows) == 1

    def test_filter_by_schedule_id(self, store):
        """Test filtering findings by schedule_id."""
        store.save_finding(Finding(id="f1", schedule_id="s1", result="r1"))
        store.save_finding(Finding(id="f2", schedule_id="s2", result="r2"))

        results = store.get_findings(schedule_id="s1")
        assert len(results) == 1
        assert results[0].schedule_id == "s1"


class TestScheduler:
    """Tests for the Scheduler class."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test-agent"
        agent.infer = AsyncMock(return_value="Found 2 issues")
        agent.observer = MagicMock()
        return agent

    @pytest.fixture
    def store(self, tmp_path):
        return ScheduleStore(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_interval_with_max_runs(self, mock_agent, store):
        """Test interval mode stops after max_runs."""
        config = ScheduleConfig(every=1, prompt="check", max_runs=2)
        scheduler = Scheduler(mock_agent, config, store=store)

        # Patch asyncio.wait_for to not actually wait
        original_wait_for = asyncio.wait_for

        async def fast_wait_for(coro, timeout):
            raise asyncio.TimeoutError()

        with patch('asyncio.wait_for', side_effect=fast_wait_for):
            await scheduler.start()

        assert mock_agent.infer.call_count == 2
        findings = store.get_findings(status="pending")
        assert len(findings) == 2

    @pytest.mark.asyncio
    async def test_stop_request(self, mock_agent, store):
        """Test graceful stop."""
        config = ScheduleConfig(every=1, prompt="check")
        scheduler = Scheduler(mock_agent, config, store=store)

        # Stop before starting
        scheduler.stop()

        await scheduler.start()
        # Should exit immediately, agent.infer never called
        assert mock_agent.infer.call_count == 0

    @pytest.mark.asyncio
    async def test_run_with_error(self, mock_agent, store):
        """Test handling of agent errors during scheduled run."""
        mock_agent.infer = AsyncMock(side_effect=Exception("LLM failed"))
        config = ScheduleConfig(every=1, prompt="check", max_runs=1)
        scheduler = Scheduler(mock_agent, config, store=store)

        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            await scheduler.start()

        findings = store.get_findings(status="pending")
        assert len(findings) == 1
        assert findings[0].error == "LLM failed"

    @pytest.mark.asyncio
    async def test_run_with_prompt_file(self, mock_agent, store, tmp_path):
        """Test scheduled run reads prompt from file."""
        prompt_file = tmp_path / "TRIAGE.md"
        prompt_file.write_text("Scan for issues")

        config = ScheduleConfig(every=1, prompt_file=str(prompt_file), max_runs=1)
        scheduler = Scheduler(mock_agent, config, store=store)

        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            await scheduler.start()

        mock_agent.infer.assert_called_once_with("Scan for issues")

    @pytest.mark.asyncio
    async def test_empty_result_auto_archives(self, mock_agent, store):
        """Test that empty results are auto-archived."""
        mock_agent.infer = AsyncMock(return_value="")
        config = ScheduleConfig(every=1, prompt="check", max_runs=1)
        scheduler = Scheduler(mock_agent, config, store=store)

        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            await scheduler.start()

        pending = store.get_findings(status="pending")
        archived = store.get_findings(status="archived")
        assert len(pending) == 0
        assert len(archived) == 1


class TestFinding:
    """Tests for Finding dataclass."""

    def test_to_dict(self):
        f = Finding(id="test", schedule_id="s1", result="found stuff")
        d = f.to_dict()
        assert d["id"] == "test"
        assert d["result"] == "found stuff"
        assert d["status"] == "pending"
        assert "created_at" in d


class TestAgentScheduleIntegration:
    """Test Agent.with_schedule() integration."""

    def test_with_schedule_returns_self(self):
        from agentu import Agent
        agent = Agent("test")
        result = agent.with_schedule(every=30, prompt="check")
        assert result is agent

    def test_with_schedule_requires_every_or_cron(self):
        from agentu import Agent
        agent = Agent("test")
        with pytest.raises(ValueError, match="Must specify"):
            agent.with_schedule(prompt="check")

    def test_findings_empty_without_schedule(self):
        from agentu import Agent
        agent = Agent("test")
        assert agent.findings() == []

    @pytest.mark.asyncio
    async def test_start_without_schedule_raises(self):
        from agentu import Agent
        agent = Agent("test")
        with pytest.raises(RuntimeError, match="No schedules"):
            await agent.start()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
