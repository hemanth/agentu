"""Durable agent support — file locking and crash-safe state management."""

try:
    import fcntl
except ImportError:
    # Windows fallback/stub for tests
    class fcntl:
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8
        @staticmethod
        def flock(fd, op):
            pass

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_DIR = os.path.expanduser("~/.agentu/agents")


class AgentLock:
    """Single-writer file lock for durable agents.
    
    Uses OS-level advisory locking (fcntl.flock) to ensure only one
    instance of an agent runs at a time. The lock is automatically
    released if the process crashes.
    """

    def __init__(self, agent_name: str, base_dir: Optional[str] = None):
        self.agent_name = agent_name
        self.base_dir = base_dir or _DEFAULT_AGENT_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        self.lock_path = os.path.join(self.base_dir, f"{agent_name}.lock")
        self._fd = None

    def acquire(self) -> bool:
        """Acquire an exclusive lock. Raises RuntimeError if already held."""
        self._fd = open(self.lock_path, 'w')
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd.write(str(os.getpid()))
            self._fd.flush()
            logger.info("Acquired lock for agent '%s' (pid=%d)", self.agent_name, os.getpid())
            return True
        except (IOError, OSError):
            self._fd.close()
            self._fd = None
            raise RuntimeError(
                f"Agent '{self.agent_name}' is already running "
                f"(lock held at {self.lock_path})"
            )

    def release(self) -> None:
        """Release the lock."""
        if self._fd:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
            except Exception:
                pass
            self._fd = None
            logger.info("Released lock for agent '%s'", self.agent_name)

    @property
    def is_locked(self) -> bool:
        """Check if the lock is currently held by this instance."""
        return self._fd is not None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        self.release()
