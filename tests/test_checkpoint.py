"""Tests for checkpoint/resume and sandbox egress control."""

import asyncio
import json
import os
import platform
import tempfile
import shutil
import time

import pytest

from agentu import Agent, Tool
from agentu.runtime.sandbox import SubprocessSandbox, SandboxLimits
from agentu.runtime.checkpoint import CheckpointData, CheckpointStore
from agentu.runtime.session import Session, SessionManager


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for checkpoint DBs."""
    return str(tmp_path)


@pytest.fixture
def store(tmp_dir):
    """Create a CheckpointStore backed by a temp DB."""
    db = os.path.join(tmp_dir, "cp_test.db")
    s = CheckpointStore(db_path=db)
    yield s
    s.close()


@pytest.fixture
def test_agent():
    """Create a minimal agent for session tests."""
    def echo(message: str) -> str:
        """Echo the message."""
        return f"Echo: {message}"

    return Agent(
        name="TestAgent",
        model="qwen3:latest",
        enable_memory=True,
    ).with_tools([Tool(echo)])


# ══════════════════════════════════════════════
# §4.3 — Sandbox egress control
# ══════════════════════════════════════════════

class TestSandboxLimitsDefaults:
    """SandboxLimits new fields have safe defaults."""

    def test_allow_network_defaults_true(self):
        limits = SandboxLimits()
        assert limits.allow_network is True

    def test_network_allowlist_defaults_none(self):
        limits = SandboxLimits()
        assert limits.network_allowlist is None


class TestBuildEnv:
    """SubprocessSandbox._build_env unit tests."""

    def test_returns_none_when_network_allowed(self):
        env = SubprocessSandbox._build_env(SandboxLimits(allow_network=True))
        assert env is None

    def test_sets_proxy_when_network_blocked(self):
        env = SubprocessSandbox._build_env(SandboxLimits(allow_network=False))
        assert env is not None
        assert env["HTTP_PROXY"] == "http://blocked"
        assert env["HTTPS_PROXY"] == "http://blocked"
        assert env["http_proxy"] == "http://blocked"
        assert env["https_proxy"] == "http://blocked"
        assert env["NO_PROXY"] == ""
        assert env["no_proxy"] == ""

    def test_allowlist_sets_no_proxy(self):
        limits = SandboxLimits(
            allow_network=False,
            network_allowlist=["api.example.com", "localhost"],
        )
        env = SubprocessSandbox._build_env(limits)
        assert env["NO_PROXY"] == "api.example.com,localhost"
        assert env["no_proxy"] == "api.example.com,localhost"
        # proxy should still be set
        assert env["HTTP_PROXY"] == "http://blocked"

    def test_inherits_parent_env_when_blocked(self):
        """Blocked env should contain parent PATH so Python can run."""
        env = SubprocessSandbox._build_env(SandboxLimits(allow_network=False))
        assert "PATH" in env


class TestSandboxEgressIntegration:
    """Integration tests: subprocess actually sees the env vars."""

    @pytest.mark.asyncio
    async def test_subprocess_sees_blocked_proxy(self):
        sandbox = SubprocessSandbox()
        code = "import os; print(os.environ.get('HTTP_PROXY', ''))"
        result = await sandbox.execute(
            code,
            SandboxLimits(allow_network=False),
        )
        assert result.success
        assert result.output.strip() == "http://blocked"

    @pytest.mark.asyncio
    async def test_subprocess_sees_no_proxy_allowlist(self):
        sandbox = SubprocessSandbox()
        code = "import os; print(os.environ.get('NO_PROXY', ''))"
        limits = SandboxLimits(
            allow_network=False,
            network_allowlist=["example.com"],
        )
        result = await sandbox.execute(code, limits)
        assert result.success
        assert result.output.strip() == "example.com"

    @pytest.mark.asyncio
    async def test_subprocess_has_no_proxy_when_allowed(self):
        """When allow_network=True the proxy vars should NOT be injected."""
        sandbox = SubprocessSandbox()
        code = "import os; print(os.environ.get('HTTP_PROXY', 'UNSET'))"
        # Temporarily make sure our own env has no HTTP_PROXY
        old = os.environ.pop("HTTP_PROXY", None)
        try:
            result = await sandbox.execute(code, SandboxLimits(allow_network=True))
            assert result.success
            assert result.output.strip() == "UNSET"
        finally:
            if old is not None:
                os.environ["HTTP_PROXY"] = old


# ══════════════════════════════════════════════
# §4.4 — Checkpoint store
# ══════════════════════════════════════════════

class TestCheckpointData:
    """CheckpointData serialisation round-trip."""

    def test_to_dict_and_back(self):
        data = CheckpointData(
            session_id="s1",
            agent_name="TestAgent",
            conversation_history=[{"content": "hello"}],
            metadata={"user": "alice"},
            turn_count=3,
            created_at=1000.0,
            checkpointed_at=2000.0,
        )
        d = data.to_dict()
        restored = CheckpointData.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.turn_count == 3
        assert restored.conversation_history == [{"content": "hello"}]
        assert restored.parent_session_id is None


class TestCheckpointStore:
    """CRUD operations on CheckpointStore."""

    def test_save_and_load(self, store):
        data = CheckpointData(
            session_id="s1",
            agent_name="Agent",
            conversation_history=[{"role": "user", "content": "hi"}],
            metadata={"key": "val"},
            turn_count=1,
            created_at=time.time(),
            checkpointed_at=time.time(),
        )
        row_id = store.save(data)
        assert row_id >= 1

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.agent_name == "Agent"
        assert loaded.conversation_history == data.conversation_history
        assert loaded.metadata == data.metadata
        assert loaded.turn_count == 1

    def test_load_returns_latest(self, store):
        """When multiple checkpoints exist, load returns the newest."""
        for i in range(3):
            store.save(CheckpointData(
                session_id="s-multi",
                agent_name="A",
                conversation_history=[{"turn": i}],
                metadata={},
                turn_count=i,
                created_at=1000.0,
                checkpointed_at=1000.0 + i,
            ))

        latest = store.load("s-multi")
        assert latest is not None
        assert latest.turn_count == 2
        assert latest.conversation_history == [{"turn": 2}]

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("does-not-exist") is None

    def test_list_checkpoints_all(self, store):
        for sid in ("a", "b", "c"):
            store.save(CheckpointData(
                session_id=sid,
                agent_name="A",
                conversation_history=[],
                metadata={},
                turn_count=0,
                created_at=time.time(),
                checkpointed_at=time.time(),
            ))

        result = store.list_checkpoints()
        assert len(result) == 3

    def test_list_checkpoints_filtered(self, store):
        for sid in ("a", "a", "b"):
            store.save(CheckpointData(
                session_id=sid,
                agent_name="A",
                conversation_history=[],
                metadata={},
                turn_count=0,
                created_at=time.time(),
                checkpointed_at=time.time(),
            ))

        result = store.list_checkpoints(session_id="a")
        assert len(result) == 2
        assert all(r["session_id"] == "a" for r in result)

    def test_delete(self, store):
        store.save(CheckpointData(
            session_id="del-me",
            agent_name="A",
            conversation_history=[],
            metadata={},
            turn_count=0,
            created_at=time.time(),
            checkpointed_at=time.time(),
        ))
        deleted = store.delete("del-me")
        assert deleted == 1
        assert store.load("del-me") is None

    def test_delete_nonexistent_returns_zero(self, store):
        assert store.delete("nope") == 0

    def test_fork(self, store):
        store.save(CheckpointData(
            session_id="origin",
            agent_name="A",
            conversation_history=[{"msg": "hello"}],
            metadata={"key": "val"},
            turn_count=5,
            created_at=1000.0,
            checkpointed_at=2000.0,
        ))

        forked = store.fork("origin")
        assert forked is not None
        assert forked.session_id != "origin"
        assert forked.parent_session_id == "origin"
        assert forked.conversation_history == [{"msg": "hello"}]
        assert forked.turn_count == 5

        # Forked checkpoint should be loadable
        reloaded = store.load(forked.session_id)
        assert reloaded is not None
        assert reloaded.parent_session_id == "origin"

    def test_fork_nonexistent_returns_none(self, store):
        assert store.fork("ghost") is None


# ══════════════════════════════════════════════
# §4.4 — Session.checkpoint() and SessionManager.resume()
# ══════════════════════════════════════════════

class TestSessionCheckpoint:
    """Session.checkpoint() serialises state correctly."""

    def test_checkpoint_saves_state(self, test_agent, tmp_dir):
        db = os.path.join(tmp_dir, "cp.db")
        cp_store = CheckpointStore(db_path=db)

        session = Session(
            session_id="cp-test",
            agent=test_agent,
            metadata={"user": "alice"},
        )
        # Add some conversation entries
        session.agent.remember("User: hello", memory_type="conversation")
        session.agent.remember("Agent: hi!", memory_type="conversation")
        session.turn_count = 3

        data = session.checkpoint(store=cp_store)

        assert data.session_id == "cp-test"
        assert data.agent_name == "TestAgent"
        assert data.turn_count == 3
        assert data.metadata == {"user": "alice"}
        assert len(data.conversation_history) >= 2
        assert data.parent_session_id is None
        assert data.checkpointed_at > 0

        # Verify it's persisted
        loaded = cp_store.load("cp-test")
        assert loaded is not None
        assert loaded.session_id == "cp-test"

        cp_store.close()

    def test_checkpoint_fork(self, test_agent, tmp_dir):
        db = os.path.join(tmp_dir, "cp_fork.db")
        cp_store = CheckpointStore(db_path=db)

        session = Session(
            session_id="original",
            agent=test_agent,
        )
        session.agent.remember("User: test", memory_type="conversation")
        session.turn_count = 2

        forked_data = session.checkpoint(store=cp_store, fork=True)

        assert forked_data.session_id != "original"
        assert forked_data.parent_session_id == "original"
        assert forked_data.turn_count == 2
        assert len(forked_data.conversation_history) >= 1

        # Original should NOT have a checkpoint (we only saved the fork)
        assert cp_store.load("original") is None
        # Fork should be loadable
        assert cp_store.load(forked_data.session_id) is not None

        cp_store.close()


class TestSessionManagerResume:
    """SessionManager.resume() restores sessions from checkpoints."""

    def test_resume_restores_session(self, test_agent, tmp_dir):
        db = os.path.join(tmp_dir, "resume.db")
        cp_store = CheckpointStore(db_path=db)

        # Create and checkpoint a session
        manager = SessionManager()
        session = manager.create_session(
            agent=test_agent,
            session_id="resume-me",
            metadata={"user": "bob"},
        )
        session.agent.remember("User: important context", memory_type="conversation")
        session.turn_count = 5
        session.checkpoint(store=cp_store)

        # Simulate restart: new manager, new agent
        manager2 = SessionManager()
        new_agent = Agent(name="TestAgent", model="qwen3:latest", enable_memory=True)

        resumed = manager2.resume("resume-me", agent=new_agent, store=cp_store)
        assert resumed is not None
        assert resumed.session_id == "resume-me"
        assert resumed.turn_count == 5
        assert resumed.metadata == {"user": "bob"}

        # Conversation history should be replayed into memory
        history = resumed.get_history(limit=100)
        assert len(history) >= 1
        assert any("important context" in e.content for e in history)

        cp_store.close()

    def test_resume_nonexistent_returns_none(self, test_agent, tmp_dir):
        db = os.path.join(tmp_dir, "empty.db")
        cp_store = CheckpointStore(db_path=db)

        manager = SessionManager()
        result = manager.resume("ghost", agent=test_agent, store=cp_store)
        assert result is None

        cp_store.close()

    def test_resume_returns_existing_if_already_active(self, test_agent, tmp_dir):
        db = os.path.join(tmp_dir, "active.db")
        cp_store = CheckpointStore(db_path=db)

        manager = SessionManager()
        session = manager.create_session(
            agent=test_agent, session_id="active-one"
        )
        session.checkpoint(store=cp_store)

        # Resume while already active — should return existing session
        resumed = manager.resume("active-one", agent=test_agent, store=cp_store)
        assert resumed is session

        cp_store.close()


# ══════════════════════════════════════════════
# Checkpoint store: edge cases
# ══════════════════════════════════════════════

class TestCheckpointStoreEdgeCases:

    def test_creates_parent_directories(self, tmp_dir):
        deep_path = os.path.join(tmp_dir, "a", "b", "c", "test.db")
        store = CheckpointStore(db_path=deep_path)
        store.save(CheckpointData(
            session_id="deep",
            agent_name="A",
            conversation_history=[],
            metadata={},
            turn_count=0,
            created_at=time.time(),
            checkpointed_at=time.time(),
        ))
        assert store.load("deep") is not None
        store.close()

    def test_empty_conversation_history(self, store):
        store.save(CheckpointData(
            session_id="empty-hist",
            agent_name="A",
            conversation_history=[],
            metadata={},
            turn_count=0,
            created_at=time.time(),
            checkpointed_at=time.time(),
        ))
        loaded = store.load("empty-hist")
        assert loaded is not None
        assert loaded.conversation_history == []

    def test_large_metadata(self, store):
        big_meta = {f"key_{i}": f"value_{i}" for i in range(100)}
        store.save(CheckpointData(
            session_id="big-meta",
            agent_name="A",
            conversation_history=[],
            metadata=big_meta,
            turn_count=0,
            created_at=time.time(),
            checkpointed_at=time.time(),
        ))
        loaded = store.load("big-meta")
        assert loaded is not None
        assert len(loaded.metadata) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
