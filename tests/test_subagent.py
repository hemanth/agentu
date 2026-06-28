"""Tests for sub-agent system (loop engineering)."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from agentu.workflow.subagent import (
    SubAgentConfig, load_subagent_configs, _build_subagent, run_maker_checker
)


class TestSubAgentConfig:
    """Tests for SubAgentConfig dataclass."""

    def test_from_dict_minimal(self):
        config = SubAgentConfig.from_dict({"name": "coder"})
        assert config.name == "coder"
        assert config.role == "maker"
        assert config.model is None
        assert config.instructions == ""

    def test_from_dict_full(self):
        config = SubAgentConfig.from_dict({
            "name": "reviewer",
            "instructions": "Review code for bugs.",
            "description": "Code reviewer",
            "role": "checker",
            "model": "gpt-4o",
            "temperature": 0.3,
        })
        assert config.name == "reviewer"
        assert config.role == "checker"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.3

    def test_from_json_file(self, tmp_path):
        """Test loading from JSON file."""
        config_file = tmp_path / "agent.json"
        config_file.write_text(json.dumps({
            "name": "tester",
            "instructions": "Write tests.",
            "role": "maker",
        }))

        config = SubAgentConfig.from_yaml(str(config_file))
        assert config.name == "tester"
        assert config.role == "maker"

    def test_from_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            SubAgentConfig.from_yaml("/nonexistent/path.json")


class TestLoadSubagentConfigs:
    """Tests for loading configs from dicts or directories."""

    def test_from_list_of_dicts(self):
        configs = load_subagent_configs([
            {"name": "coder", "role": "maker"},
            {"name": "reviewer", "role": "checker"},
        ])
        assert len(configs) == 2
        assert configs[0].name == "coder"
        assert configs[1].role == "checker"

    def test_from_directory(self, tmp_path):
        """Test loading from a directory of JSON files."""
        (tmp_path / "maker.json").write_text(json.dumps({
            "name": "coder",
            "instructions": "Write code.",
            "role": "maker",
        }))
        (tmp_path / "checker.json").write_text(json.dumps({
            "name": "reviewer",
            "instructions": "Review code.",
            "role": "checker",
        }))
        # Hidden files should be skipped
        (tmp_path / ".hidden.json").write_text(json.dumps({"name": "hidden"}))

        configs = load_subagent_configs(str(tmp_path))
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert "coder" in names
        assert "reviewer" in names

    def test_empty_directory(self, tmp_path):
        """Test empty directory returns empty list."""
        configs = load_subagent_configs(str(tmp_path))
        assert configs == []

    def test_nonexistent_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_subagent_configs("/nonexistent/dir")


class TestBuildSubagent:
    """Tests for _build_subagent."""

    def test_inherits_parent_model(self):
        """Test that sub-agent inherits parent model when not specified."""
        parent = MagicMock()
        parent.model = "qwen3"
        parent.temperature = 0.7
        parent.api_base = "http://localhost:11434/v1"
        parent.api_key = None
        parent.tools = []

        config = SubAgentConfig(name="child", instructions="Do stuff.")

        with patch('agentu._core.agent.Agent') as MockAgent:
            mock_instance = MagicMock()
            mock_instance.tools = []
            mock_instance.context = ""
            MockAgent.return_value = mock_instance

            agent = _build_subagent(config, parent)

            MockAgent.assert_called_once_with(
                name="child",
                model="qwen3",
                temperature=0.7,
                api_base="http://localhost:11434/v1",
                api_key=None,
                enable_memory=False,
            )

    def test_overrides_parent_model(self):
        """Test that explicit model overrides parent."""
        parent = MagicMock()
        parent.model = "qwen3"
        parent.temperature = 0.7
        parent.api_base = "http://localhost:11434/v1"
        parent.api_key = None
        parent.tools = []

        config = SubAgentConfig(name="child", model="gpt-4o", temperature=0.3)

        with patch('agentu._core.agent.Agent') as MockAgent:
            mock_instance = MagicMock()
            mock_instance.tools = []
            mock_instance.context = ""
            MockAgent.return_value = mock_instance

            _build_subagent(config, parent)

            MockAgent.assert_called_once_with(
                name="child",
                model="gpt-4o",
                temperature=0.3,
                api_base="http://localhost:11434/v1",
                api_key=None,
                enable_memory=False,
            )

    def test_sets_instructions_as_context(self):
        """Test that instructions are set as system context."""
        parent = MagicMock()
        parent.model = "qwen3"
        parent.temperature = 0.7
        parent.api_base = "http://localhost:11434/v1"
        parent.api_key = None
        parent.tools = []

        config = SubAgentConfig(name="child", instructions="Be helpful.")

        with patch('agentu._core.agent.Agent') as MockAgent:
            mock_instance = MagicMock()
            mock_instance.tools = []
            mock_instance.context = ""
            MockAgent.return_value = mock_instance

            agent = _build_subagent(config, parent)
            assert mock_instance.context == "Be helpful."


class TestRunMakerChecker:
    """Tests for maker-checker delegation pattern."""

    @pytest.mark.asyncio
    async def test_maker_only(self):
        """Test delegation with maker only (no checker)."""
        maker = MagicMock()
        maker.name = "coder"
        maker.infer = AsyncMock(return_value="def hello(): pass")

        result = await run_maker_checker(
            task="Write a hello function",
            makers=[maker],
            checkers=[],
        )

        assert result["approved"] is True
        assert result["result"] == "def hello(): pass"
        assert result["checker"] is None
        assert result["corrections"] == 0

    @pytest.mark.asyncio
    async def test_maker_checker_approved(self):
        """Test maker-checker where checker approves."""
        maker = MagicMock()
        maker.name = "coder"
        maker.infer = AsyncMock(return_value="def hello(): pass")

        checker = MagicMock()
        checker.name = "reviewer"
        checker.infer = AsyncMock(return_value="APPROVED: Looks good!")

        result = await run_maker_checker(
            task="Write a hello function",
            makers=[maker],
            checkers=[checker],
        )

        assert result["approved"] is True
        assert result["maker"] == "coder"
        assert result["checker"] == "reviewer"
        assert result["corrections"] == 0

    @pytest.mark.asyncio
    async def test_maker_checker_rejected_then_corrected(self):
        """Test maker-checker with rejection and correction."""
        maker = MagicMock()
        maker.name = "coder"
        maker.infer = AsyncMock(
            side_effect=["def hello(): pass", "def hello():\n    print('hello')"]
        )

        checker = MagicMock()
        checker.name = "reviewer"
        checker.infer = AsyncMock(
            side_effect=["NEEDS REVISION: Function body is empty", "APPROVED: Good fix!"]
        )

        result = await run_maker_checker(
            task="Write a hello function",
            makers=[maker],
            checkers=[checker],
            max_corrections=1,
        )

        assert result["approved"] is True
        assert result["corrections"] == 1
        assert maker.infer.call_count == 2
        assert checker.infer.call_count == 2

    @pytest.mark.asyncio
    async def test_maker_checker_max_corrections_exceeded(self):
        """Test maker-checker when max corrections is exceeded."""
        maker = MagicMock()
        maker.name = "coder"
        maker.infer = AsyncMock(return_value="bad code")

        checker = MagicMock()
        checker.name = "reviewer"
        checker.infer = AsyncMock(return_value="NEEDS REVISION: Still bad")

        result = await run_maker_checker(
            task="Write code",
            makers=[maker],
            checkers=[checker],
            max_corrections=1,
        )

        assert result["approved"] is False
        assert result["corrections"] == 1

    @pytest.mark.asyncio
    async def test_no_makers_raises(self):
        """Test that no makers raises ValueError."""
        with pytest.raises(ValueError, match="maker"):
            await run_maker_checker(task="test", makers=[], checkers=[])

    @pytest.mark.asyncio
    async def test_observer_events(self):
        """Test that observer events are emitted."""
        maker = MagicMock()
        maker.name = "coder"
        maker.infer = AsyncMock(return_value="code")

        checker = MagicMock()
        checker.name = "reviewer"
        checker.infer = AsyncMock(return_value="APPROVED: OK")

        observer = MagicMock()

        result = await run_maker_checker(
            task="test",
            makers=[maker],
            checkers=[checker],
            parent_observer=observer,
        )

        # Should have recorded delegate_start and delegate_review events
        assert observer.record.call_count >= 2

    @pytest.mark.asyncio
    async def test_configurable_max_corrections(self):
        """Test max_corrections=2 allows two correction attempts."""
        maker = MagicMock()
        maker.name = "coder"
        maker.infer = AsyncMock(
            side_effect=["v1", "v2", "v3"]
        )

        checker = MagicMock()
        checker.name = "reviewer"
        checker.infer = AsyncMock(
            side_effect=["NEEDS REVISION: v1 bad", "NEEDS REVISION: v2 bad", "APPROVED: v3 ok"]
        )

        result = await run_maker_checker(
            task="test", makers=[maker], checkers=[checker],
            max_corrections=2,
        )

        assert result["approved"] is True
        assert result["corrections"] == 2


class TestAgentSubagentIntegration:
    """Test Agent.with_subagents() integration."""

    def test_with_subagents_from_dicts(self):
        from agentu import Agent
        agent = Agent("lead").with_subagents([
            {"name": "coder", "role": "maker"},
            {"name": "reviewer", "role": "checker"},
        ])
        assert hasattr(agent, '_subagent_configs')
        assert len(agent._subagent_configs) == 2

    def test_with_subagents_from_directory(self, tmp_path):
        from agentu import Agent
        (tmp_path / "maker.json").write_text(json.dumps({
            "name": "coder", "role": "maker"
        }))

        agent = Agent("lead").with_subagents(str(tmp_path))
        assert len(agent._subagent_configs) == 1

    @pytest.mark.asyncio
    async def test_delegate_without_subagents_raises(self):
        from agentu import Agent
        agent = Agent("lead")
        with pytest.raises(RuntimeError, match="No sub-agents"):
            await agent.delegate("test task")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
