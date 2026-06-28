"""Tests for git worktree isolation (loop engineering)."""

import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

from agentu.workflow.worktree import WorktreeManager


class TestWorktreeManager:
    """Tests for WorktreeManager."""

    def test_init_defaults(self):
        wt = WorktreeManager()
        assert wt.base_path is None
        assert wt.branch is None
        assert wt.cleanup is True
        assert wt.worktree_path is None

    def test_init_custom(self):
        wt = WorktreeManager(base_path="/repo", branch="feature", cleanup=False)
        assert wt.base_path == "/repo"
        assert wt.branch == "feature"
        assert wt.cleanup is False

    def test_is_git_repo_true(self):
        """Test git repo detection in a real git repo."""
        wt = WorktreeManager()
        # This test runs inside the agentu repo, so it should be True
        assert wt._is_git_repo() is True

    def test_detect_git_root(self):
        """Test git root detection."""
        wt = WorktreeManager()
        root = wt._detect_git_root()
        assert root is not None
        assert Path(root).is_dir()

    def test_detect_git_root_non_repo(self, tmp_path):
        """Test git root detection outside a repo."""
        wt = WorktreeManager(base_path=str(tmp_path))
        root = wt._detect_git_root()
        assert root is None

    def test_is_git_repo_false(self, tmp_path):
        """Test git repo detection outside a repo."""
        wt = WorktreeManager(base_path=str(tmp_path))
        assert wt._is_git_repo() is False

    @pytest.mark.asyncio
    async def test_create_in_non_git_repo(self, tmp_path):
        """Test create returns None outside a git repo."""
        wt = WorktreeManager(base_path=str(tmp_path))
        result = await wt.create(agent_name="test")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_success(self):
        """Test worktree creation with mocked git commands."""
        wt = WorktreeManager()

        mock_run_result = MagicMock()
        mock_run_result.returncode = 0

        with patch.object(wt, '_detect_git_root', return_value="/fake/repo"):
            with patch('subprocess.run', return_value=mock_run_result):
                path = await wt.create(agent_name="builder")

                assert path is not None
                assert "builder" in path
                assert wt.worktree_path == path

    @pytest.mark.asyncio
    async def test_create_failure(self):
        """Test worktree creation handles git failure."""
        wt = WorktreeManager()

        mock_run_result = MagicMock()
        mock_run_result.returncode = 1
        mock_run_result.stderr = "error: branch already exists"

        with patch.object(wt, '_detect_git_root', return_value="/fake/repo"):
            with patch('subprocess.run', return_value=mock_run_result):
                path = await wt.create(agent_name="builder")
                assert path is None

    @pytest.mark.asyncio
    async def test_remove_success(self):
        """Test worktree removal."""
        wt = WorktreeManager()
        wt.worktree_path = "/fake/worktree"

        mock_run_result = MagicMock()
        mock_run_result.returncode = 0

        with patch.object(wt, '_detect_git_root', return_value="/fake/repo"):
            with patch('subprocess.run', return_value=mock_run_result):
                await wt.remove()
                assert wt.worktree_path is None

    @pytest.mark.asyncio
    async def test_remove_without_worktree(self):
        """Test remove is a no-op without a worktree."""
        wt = WorktreeManager()
        await wt.remove()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test context manager creates and cleans up."""
        wt = WorktreeManager()

        with patch.object(wt, 'create', new_callable=AsyncMock) as mock_create:
            with patch.object(wt, 'remove', new_callable=AsyncMock) as mock_remove:
                mock_create.return_value = "/fake/worktree"

                async with wt:
                    pass

                mock_create.assert_called_once()
                mock_remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_no_cleanup(self):
        """Test context manager skips cleanup when cleanup=False."""
        wt = WorktreeManager(cleanup=False)

        with patch.object(wt, 'create', new_callable=AsyncMock) as mock_create:
            with patch.object(wt, 'remove', new_callable=AsyncMock) as mock_remove:
                mock_create.return_value = "/fake/worktree"

                async with wt:
                    pass

                mock_create.assert_called_once()
                mock_remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_error(self):
        """Test context manager still cleans up on exception."""
        wt = WorktreeManager()

        with patch.object(wt, 'create', new_callable=AsyncMock) as mock_create:
            with patch.object(wt, 'remove', new_callable=AsyncMock) as mock_remove:
                mock_create.return_value = "/fake/worktree"

                with pytest.raises(ValueError):
                    async with wt:
                        raise ValueError("test error")

                mock_remove.assert_called_once()


class TestAgentWorktreeIntegration:
    """Test Agent.with_worktree() integration."""

    def test_with_worktree_returns_self(self):
        from agentu import Agent
        agent = Agent("test")
        result = agent.with_worktree()
        assert result is agent
        assert hasattr(agent, '_worktree_config')

    def test_with_worktree_config(self):
        from agentu import Agent
        agent = Agent("test").with_worktree(branch="feature", cleanup=False)
        assert agent._worktree_config["branch"] == "feature"
        assert agent._worktree_config["cleanup"] is False

    def test_with_worktree_defaults(self):
        from agentu import Agent
        agent = Agent("test").with_worktree()
        assert agent._worktree_config["branch"] is None
        assert agent._worktree_config["cleanup"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
