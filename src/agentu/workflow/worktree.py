"""
Git worktree isolation for agentu loop engineering.

Create isolated git worktrees so parallel agents don't collide.

Usage:
    agent = Agent("builder").with_worktree()
    result = await agent.infer("Refactor auth module")
    # runs in an isolated git worktree, auto-cleaned after
"""

import asyncio
import subprocess
import uuid
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WorktreeManager:
    """Manages git worktree lifecycle for isolated agent execution.

    Creates temporary worktrees for agents to work in, ensuring
    parallel agents don't step on each other's files.
    """

    def __init__(self, base_path: Optional[str] = None, branch: Optional[str] = None,
                 cleanup: bool = True):
        """Initialize worktree manager.

        Args:
            base_path: Git repository root. Auto-detected if None.
            branch: Branch name for the worktree. Auto-generated if None.
            cleanup: Whether to auto-remove worktree after use (default: True).
        """
        self.base_path = base_path
        self.branch = branch
        self.cleanup = cleanup
        self.worktree_path: Optional[str] = None
        self._branch_created = False
        self._auto_branch_name: Optional[str] = None

    def _detect_git_root(self) -> Optional[str]:
        """Detect the git repository root."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
                cwd=self.base_path or "."
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _is_git_repo(self) -> bool:
        """Check if the current directory is inside a git repo."""
        return self._detect_git_root() is not None

    async def create(self, agent_name: str = "agent") -> Optional[str]:
        """Create an isolated git worktree.

        Args:
            agent_name: Agent name for branch naming.

        Returns:
            Path to the worktree, or None if not in a git repo.
        """
        git_root = self._detect_git_root()
        if git_root is None:
            logger.warning("Not in a git repo — skipping worktree isolation")
            return None

        short_id = str(uuid.uuid4())[:8]
        branch_name = self.branch or f"agent/{agent_name}-{short_id}"
        worktree_dir = str(Path(git_root) / ".worktrees" / f"{agent_name}-{short_id}")

        # Create worktree with new branch
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "worktree", "add", "-b", branch_name, worktree_dir],
                capture_output=True, text=True, timeout=30,
                cwd=git_root
            )
            if result.returncode != 0:
                # Branch might already exist, try without -b
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["git", "worktree", "add", worktree_dir, branch_name],
                    capture_output=True, text=True, timeout=30,
                    cwd=git_root
                )
                if result.returncode != 0:
                    logger.error(f"Failed to create worktree: {result.stderr}")
                    return None
            else:
                self._branch_created = True
                if self.branch is None:
                    self._auto_branch_name = branch_name

            self.worktree_path = worktree_dir
            logger.info(f"Created worktree: {worktree_dir} (branch: {branch_name})")

            # Emit observer event
            return worktree_dir

        except subprocess.TimeoutExpired:
            logger.error("Worktree creation timed out")
            return None
        except FileNotFoundError:
            logger.warning("git not found in PATH — skipping worktree isolation")
            return None

    async def remove(self):
        """Remove the worktree and optionally the branch."""
        if not self.worktree_path:
            return

        git_root = self._detect_git_root()
        if git_root is None:
            return

        try:
            # Remove the worktree
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "worktree", "remove", self.worktree_path, "--force"],
                capture_output=True, text=True, timeout=30,
                cwd=git_root
            )
            if result.returncode == 0:
                logger.info(f"Removed worktree: {self.worktree_path}")
            else:
                logger.warning(f"Failed to remove worktree: {result.stderr}")

            # Clean up the branch if we created it
            if self._branch_created and self.branch is None and self._auto_branch_name:
                # We don't delete user-specified branches
                branch_result = await asyncio.to_thread(
                    subprocess.run,
                    ["git", "branch", "-D", self._auto_branch_name],
                    capture_output=True, text=True, timeout=30,
                    cwd=git_root
                )
                if branch_result.returncode == 0:
                    logger.info(f"Deleted auto-created branch: {self._auto_branch_name}")
                else:
                    logger.warning(f"Failed to delete branch {self._auto_branch_name}: {branch_result.stderr}")
                self._auto_branch_name = None

            self.worktree_path = None

        except subprocess.TimeoutExpired:
            logger.error("Worktree removal timed out")
        except FileNotFoundError:
            pass

    async def __aenter__(self):
        """Enter context manager — create worktree."""
        await self.create()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager — remove worktree if cleanup is enabled."""
        if self.cleanup:
            await self.remove()
