"""Tests for Ralph autonomous loop module."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agentu.ralph import RalphRunner, RalphConfig, RalphState, ralph


class TestRalphState:
    """Tests for RalphState dataclass."""
    
    def test_default_state(self):
        state = RalphState()
        assert state.iteration == 0
        assert state.checkpoints_completed == []
        assert state.last_result is None
        assert state.errors == []
        
    def test_to_dict(self):
        state = RalphState(iteration=5, last_result="test result")
        d = state.to_dict()
        assert d["iteration"] == 5
        assert d["last_result"] == "test result"
        assert "started_at" in d


class TestRalphConfig:
    """Tests for RalphConfig dataclass."""
    
    def test_defaults(self):
        config = RalphConfig(prompt_file="PROMPT.md")
        assert config.max_iterations == 50
        assert config.timeout_minutes == 30
        assert config.checkpoint_every == 5


class TestRalphRunner:
    """Tests for RalphRunner class."""
    
    def test_stop_request(self):
        mock_agent = MagicMock()
        config = RalphConfig(prompt_file="PROMPT.md")
        runner = RalphRunner(mock_agent, config)
        
        assert runner._stop_requested is False
        runner.stop()
        assert runner._stop_requested is True
    
    def test_check_completion_all_done(self):
        mock_agent = MagicMock()
        config = RalphConfig(prompt_file="PROMPT.md")
        runner = RalphRunner(mock_agent, config)
        
        prompt = """
# Goal
- [x] Task 1
- [x] Task 2
"""
        assert runner._check_completion(prompt) is True
    
    def test_check_completion_incomplete(self):
        mock_agent = MagicMock()
        config = RalphConfig(prompt_file="PROMPT.md")
        runner = RalphRunner(mock_agent, config)
        
        prompt = """
# Goal
- [x] Task 1
- [ ] Task 2
"""
        assert runner._check_completion(prompt) is False
    
    def test_check_completion_no_checkboxes(self):
        mock_agent = MagicMock()
        config = RalphConfig(prompt_file="PROMPT.md")
        runner = RalphRunner(mock_agent, config)
        
        prompt = "# Goal\nJust some text"
        assert runner._check_completion(prompt) is False

    @pytest.mark.asyncio
    async def test_run_stops_on_completion(self):
        """Test that Ralph stops when all checkpoints complete."""
        mock_agent = MagicMock()
        mock_agent.infer = AsyncMock(return_value={"result": "done"})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Goal
- [x] Task 1
- [x] Task 2
""")
            prompt_path = f.name
        
        try:
            config = RalphConfig(prompt_file=prompt_path, max_iterations=10)
            runner = RalphRunner(mock_agent, config)
            
            result = await runner.run()
            
            assert result["stopped_by"] == "completion"
            assert result["iterations"] == 1  # Checks completion at start of first iteration
        finally:
            Path(prompt_path).unlink()

    @pytest.mark.asyncio  
    async def test_run_respects_max_iterations(self):
        """Test that Ralph respects max_iterations limit."""
        mock_agent = MagicMock()
        mock_agent.infer = AsyncMock(return_value={"result": "working..."})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Goal
- [ ] Never complete task
""")
            prompt_path = f.name
        
        try:
            config = RalphConfig(prompt_file=prompt_path, max_iterations=3)
            runner = RalphRunner(mock_agent, config)
            
            result = await runner.run()
            
            assert result["iterations"] == 3
            assert mock_agent.infer.call_count == 3
        finally:
            Path(prompt_path).unlink()


class TestRalphFunction:
    """Tests for ralph() convenience function."""
    
    @pytest.mark.asyncio
    async def test_ralph_function(self):
        mock_agent = MagicMock()
        mock_agent.infer = AsyncMock(return_value={"result": "ok"})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Goal\n- [x] Done")
            prompt_path = f.name
        
        try:
            result = await ralph(mock_agent, prompt_path, max_iterations=5)
            assert "iterations" in result
            assert "stopped_by" in result
        finally:
            Path(prompt_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
