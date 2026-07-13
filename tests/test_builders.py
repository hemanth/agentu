"""Tests for builder methods and mixin features of Agent.

Covers: with_backend, with_vectors, with_context, with_otel,
mixin inheritance/MRO, and checkpoint-backend integration.
"""

import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from agentu import Agent
from agentu.storage import InMemoryBackend, InMemoryVectorBackend


# ── Helpers ──────────────────────────────────────────────────────────

def _make_agent(name="test_agent", **kw):
    """Create an Agent with Ollama auto-detect mocked out."""
    with patch("agentu._core.agent._get_ollama_models_sync", return_value=[]):
        return Agent(name, **kw)


# ══════════════════════════════════════════════════════════════════════
# with_backend() tests
# ══════════════════════════════════════════════════════════════════════


class TestWithBackend:

    def test_string_url_sets_backend_url(self):
        agent = _make_agent()
        result = agent.with_backend("redis://localhost:6379/0")
        assert agent._backend_url == "redis://localhost:6379/0"
        assert agent._storage_backend is None
        assert result is agent  # returns self

    def test_custom_backend_object_sets_storage_backend(self):
        agent = _make_agent()
        backend = InMemoryBackend()
        agent.with_backend(backend)
        assert agent._storage_backend is backend
        assert agent._backend_url is None

    def test_chaining_with_tools(self):
        agent = _make_agent()

        def dummy(x: int) -> int:
            """Dummy."""
            return x

        result = agent.with_backend("redis://localhost:6379").with_tools([dummy])
        assert isinstance(result, Agent)
        assert agent._backend_url == "redis://localhost:6379"
        assert len(agent.tools) == 1

    @pytest.mark.asyncio
    async def test_get_storage_backend_returns_custom_backend(self):
        agent = _make_agent()
        backend = InMemoryBackend()
        agent.with_backend(backend)
        result = await agent.get_storage_backend()
        assert result is backend

    @pytest.mark.asyncio
    async def test_get_storage_backend_lazy_creates_from_url(self):
        agent = _make_agent()
        agent.with_backend("redis://localhost:6379")

        mock_backend = MagicMock()
        mock_create = AsyncMock(return_value=mock_backend)

        with patch("agentu.storage.RedisStorageBackend.create", mock_create):
            result = await agent.get_storage_backend()
            mock_create.assert_awaited_once_with("redis://localhost:6379")
            assert result is mock_backend

    @pytest.mark.asyncio
    async def test_get_storage_backend_returns_none_when_unconfigured(self):
        agent = _make_agent()
        result = await agent.get_storage_backend()
        assert result is None

    def test_string_url_clears_existing_backend(self):
        agent = _make_agent()
        backend = InMemoryBackend()
        agent.with_backend(backend)
        assert agent._storage_backend is backend
        # Now switch to URL
        agent.with_backend("redis://new-host:6379")
        assert agent._storage_backend is None
        assert agent._backend_url == "redis://new-host:6379"


# ══════════════════════════════════════════════════════════════════════
# with_vectors() tests
# ══════════════════════════════════════════════════════════════════════


class TestWithVectors:

    def test_string_path_sets_vector_dsn(self):
        agent = _make_agent()
        result = agent.with_vectors("./vectors")
        assert agent._vector_dsn == "./vectors"
        assert agent._vector_backend is None
        assert result is agent

    def test_custom_backend_sets_vector_backend(self):
        agent = _make_agent()
        vb = InMemoryVectorBackend()
        agent.with_vectors(vb)
        assert agent._vector_backend is vb
        assert agent._vector_dsn is None

    def test_dimension_parameter_passes_through(self):
        agent = _make_agent()
        agent.with_vectors("./vectors", dimension=768)
        assert agent._vector_dimension == 768

    def test_default_dimension(self):
        agent = _make_agent()
        assert agent._vector_dimension == 384  # default from __init__

    @pytest.mark.asyncio
    async def test_get_vector_backend_returns_custom_backend(self):
        agent = _make_agent()
        vb = InMemoryVectorBackend()
        agent.with_vectors(vb)
        result = await agent.get_vector_backend()
        assert result is vb

    @pytest.mark.asyncio
    async def test_get_vector_backend_lazy_creates_from_dsn(self):
        agent = _make_agent()
        agent.with_vectors("./my_vectors")

        mock_backend = MagicMock()
        mock_create = MagicMock(return_value=mock_backend)

        with patch("agentu.storage.LanceDBBackend.create", mock_create):
            result = await agent.get_vector_backend()
            mock_create.assert_called_once_with("./my_vectors")
            assert result is mock_backend

    @pytest.mark.asyncio
    async def test_get_vector_backend_returns_none_when_unconfigured(self):
        agent = _make_agent()
        result = await agent.get_vector_backend()
        assert result is None

    def test_chaining_works(self):
        agent = _make_agent()
        result = agent.with_vectors("./vectors", dimension=512).with_backend("redis://localhost")
        assert isinstance(result, Agent)
        assert agent._vector_dsn == "./vectors"
        assert agent._backend_url == "redis://localhost"


# ══════════════════════════════════════════════════════════════════════
# with_context() tests
# ══════════════════════════════════════════════════════════════════════


class TestWithContext:

    def test_sets_context_config(self):
        agent = _make_agent()
        assert agent._context_config is None
        agent.with_context()
        assert agent._context_config is not None

    def test_default_values(self):
        agent = _make_agent()
        agent.with_context()
        cfg = agent._context_config
        assert cfg.max_tokens == 128_000
        assert cfg.compaction.value == "auto"
        assert cfg.keep_recent == 5
        assert cfg.max_result_chars == 2000

    def test_custom_values(self):
        agent = _make_agent()
        agent.with_context(
            max_tokens=50_000,
            compaction="summarize",
            keep_recent=10,
            max_result_chars=500,
        )
        cfg = agent._context_config
        assert cfg.max_tokens == 50_000
        assert cfg.compaction.value == "summarize"
        assert cfg.keep_recent == 10
        assert cfg.max_result_chars == 500

    def test_compaction_none(self):
        agent = _make_agent()
        agent.with_context(compaction="none")
        assert agent._context_config.compaction.value == "none"

    def test_compaction_truncate(self):
        agent = _make_agent()
        agent.with_context(compaction="truncate")
        assert agent._context_config.compaction.value == "truncate"

    def test_chaining_works(self):
        agent = _make_agent()
        result = agent.with_context(max_tokens=64_000).with_backend("redis://host")
        assert isinstance(result, Agent)
        assert agent._context_config.max_tokens == 64_000
        assert agent._backend_url == "redis://host"


# ══════════════════════════════════════════════════════════════════════
# with_otel() tests
# ══════════════════════════════════════════════════════════════════════


class TestWithOtel:

    def test_sets_otel_exporter(self):
        agent = _make_agent()
        with patch("agentu.middleware.otel.OTelExporter") as MockExporter:
            mock_instance = MagicMock()
            MockExporter.return_value = mock_instance

            result = agent.with_otel(
                endpoint="http://localhost:4318",
                service_name="my-svc",
            )
            MockExporter.assert_called_once_with(
                service_name="my-svc",
                endpoint="http://localhost:4318",
                model=agent.model,
                observer=agent.observer,
            )
            mock_instance.attach.assert_called_once_with(agent.observer)
            assert agent._otel_exporter is mock_instance
            assert result is agent

    def test_default_service_name(self):
        agent = _make_agent()
        with patch("agentu.middleware.otel.OTelExporter") as MockExporter:
            mock_instance = MagicMock()
            MockExporter.return_value = mock_instance

            agent.with_otel()
            call_kwargs = MockExporter.call_args[1]
            assert call_kwargs["service_name"] == "agentu"

    def test_chaining_works(self):
        agent = _make_agent()
        with patch("agentu.middleware.otel.OTelExporter") as MockExporter:
            MockExporter.return_value = MagicMock()
            result = agent.with_otel().with_backend("redis://host")
        assert isinstance(result, Agent)


# ══════════════════════════════════════════════════════════════════════
# Mixin inheritance tests
# ══════════════════════════════════════════════════════════════════════


class TestMixinInheritance:

    def test_agent_has_memory_mixin_methods(self):
        """Agent should have MemoryMixin methods."""
        assert hasattr(Agent, "remember")
        assert hasattr(Agent, "recall")

    def test_agent_has_sandbox_mixin_methods(self):
        """Agent should have SandboxMixin methods."""
        assert hasattr(Agent, "with_sandbox")

    def test_agent_has_hooks_mixin_methods(self):
        """Agent should have HooksMixin methods."""
        assert hasattr(Agent, "with_hooks")
        assert hasattr(Agent, "with_permissions")

    def test_agent_has_context_mixin_methods(self):
        """Agent should have ContextMixin methods."""
        assert hasattr(Agent, "with_context")
        assert hasattr(Agent, "set_context")

    def test_agent_has_workflow_mixin_methods(self):
        """Agent should have WorkflowMixin methods."""
        assert hasattr(Agent, "with_schedule")
        assert hasattr(Agent, "with_subagents")
        assert hasattr(Agent, "with_worktree")
        assert hasattr(Agent, "delegate")
        assert hasattr(Agent, "best_of")
        assert hasattr(Agent, "start")
        assert hasattr(Agent, "stop")
        assert hasattr(Agent, "findings")

    def test_mro_includes_all_mixins(self):
        """Agent MRO must include all five mixins."""
        from agentu._core.agent_memory import MemoryMixin
        from agentu._core.agent_sandbox import SandboxMixin
        from agentu._core.agent_hooks import HooksMixin
        from agentu._core.agent_context import ContextMixin
        from agentu._core.agent_workflow import WorkflowMixin

        mro = Agent.__mro__
        for mixin in [MemoryMixin, SandboxMixin, HooksMixin, ContextMixin, WorkflowMixin]:
            assert mixin in mro, f"{mixin.__name__} not in Agent MRO"

    def test_chaining_multiple_mixins(self):
        """Calling builder methods from different mixins returns Agent."""
        agent = _make_agent()

        def hook(name, params, ctx):
            from agentu._core.hooks import HookResult
            return HookResult()

        result = (
            agent
            .with_context(max_tokens=64_000)
            .with_hooks(pre_tool=hook)
            .with_backend("redis://host")
        )
        assert isinstance(result, Agent)
        assert result is agent

    def test_set_context_sets_context_string(self):
        agent = _make_agent()
        agent.set_context("You are a helpful assistant.")
        assert agent.context == "You are a helpful assistant."


# ══════════════════════════════════════════════════════════════════════
# WorkflowMixin builder tests
# ══════════════════════════════════════════════════════════════════════


class TestWorkflowBuilders:

    def test_with_worktree_sets_config(self):
        agent = _make_agent()
        result = agent.with_worktree(branch="feature-x", cleanup=False)
        assert agent._worktree_config == {"branch": "feature-x", "cleanup": False}
        assert result is agent

    def test_with_worktree_defaults(self):
        agent = _make_agent()
        agent.with_worktree()
        assert agent._worktree_config["branch"] is None
        assert agent._worktree_config["cleanup"] is True

    def test_with_schedule_requires_every_or_cron(self):
        agent = _make_agent()
        with pytest.raises(ValueError, match="Must specify"):
            agent.with_schedule(prompt="check")

    def test_with_schedule_every(self):
        agent = _make_agent()
        result = agent.with_schedule(every=30, prompt="Check status")
        assert result is agent
        assert hasattr(agent, "_schedulers")
        assert len(agent._schedulers) == 1

    def test_with_schedule_cron(self):
        agent = _make_agent()
        agent.with_schedule(cron="0 9 * * *", prompt="Daily check")
        assert len(agent._schedulers) == 1

    def test_findings_empty_when_no_store(self):
        agent = _make_agent()
        assert agent.findings() == []

    def test_findings_returns_list(self):
        agent = _make_agent()
        agent.with_schedule(every=60, prompt="test")
        # No runs yet, so findings should be empty
        assert agent.findings() == []

    @pytest.mark.asyncio
    async def test_start_raises_without_schedules(self):
        agent = _make_agent()
        with pytest.raises(RuntimeError, match="No schedules configured"):
            await agent.start()


# ══════════════════════════════════════════════════════════════════════
# Checkpoint-backend integration tests
# ══════════════════════════════════════════════════════════════════════


class TestCheckpointBackendIntegration:

    def test_session_checkpoint_uses_storage_backend(self):
        """Session.checkpoint() should persist to the agent's storage backend."""
        from agentu.runtime.session import Session

        agent = _make_agent(enable_memory=True)
        backend = InMemoryBackend()
        agent.with_backend(backend)

        session = Session(session_id="sess-1", agent=agent)
        session.turn_count = 3

        # checkpoint() internally calls backend.set() via asyncio.run
        data = session.checkpoint()

        assert data.session_id == "sess-1"
        assert data.agent_name == "test_agent"
        assert data.turn_count == 3

    def test_session_checkpoint_fork_creates_new_id(self):
        """Forked checkpoint should have a new session_id and parent ref."""
        from agentu.runtime.session import Session

        agent = _make_agent(enable_memory=True)
        backend = InMemoryBackend()
        agent.with_backend(backend)

        session = Session(session_id="orig-sess", agent=agent)
        data = session.checkpoint(fork=True)

        assert data.session_id != "orig-sess"
        assert data.parent_session_id == "orig-sess"

    def test_session_manager_resume_from_store(self):
        """SessionManager.resume() reads from checkpoint store."""
        from agentu.runtime.session import Session, SessionManager
        from agentu.runtime.checkpoint import CheckpointStore

        agent = _make_agent(enable_memory=True)
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cp.db")
            store = CheckpointStore(db_path=db_path)

            # Create and checkpoint a session
            session = Session(session_id="resume-test", agent=agent)
            session.turn_count = 7
            session.checkpoint(store=store)

            # Create a fresh manager and resume
            manager = SessionManager()
            agent2 = _make_agent(enable_memory=True)
            resumed = manager.resume("resume-test", agent2, store=store)

            assert resumed is not None
            assert resumed.session_id == "resume-test"
            assert resumed.turn_count == 7

    def test_session_manager_resume_returns_none_for_unknown(self):
        """resume() returns None when no checkpoint exists."""
        from agentu.runtime.session import SessionManager
        from agentu.runtime.checkpoint import CheckpointStore

        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "empty.db")
            store = CheckpointStore(db_path=db_path)
            manager = SessionManager()
            agent = _make_agent(enable_memory=True)
            result = manager.resume("nonexistent", agent, store=store)
            assert result is None


# ══════════════════════════════════════════════════════════════════════
# Vector backend ↔ MemoryMixin integration tests
# ══════════════════════════════════════════════════════════════════════


class TestVectorBackendIntegration:

    def test_get_vector_backend_sync_returns_none_unconfigured(self):
        """Returns None when no vector backend is configured."""
        agent = _make_agent(enable_memory=True)
        assert agent._get_vector_backend_sync() is None

    def test_get_vector_backend_sync_returns_direct_backend(self):
        """Returns the backend when set via with_vectors(backend_obj)."""
        backend = InMemoryVectorBackend()
        agent = _make_agent(enable_memory=True).with_vectors(backend)
        assert agent._get_vector_backend_sync() is backend

    def test_get_vector_backend_sync_caches_created_backend(self):
        """Created backend is cached on self._vector_backend."""
        backend = InMemoryVectorBackend()
        agent = _make_agent(enable_memory=True)
        agent._vector_backend = backend
        result = agent._get_vector_backend_sync()
        assert result is backend

    def test_remember_stores_to_vector_backend(self):
        """remember() with high importance stores to vector backend."""
        backend = InMemoryVectorBackend()
        agent = _make_agent(enable_memory=True).with_vectors(backend)
        # Without an embedding provider, _store_to_vector_backend exits early
        # Verify it doesn't crash
        agent.remember("important fact", importance=0.9, store_long_term=True)

    def test_remember_skips_vector_when_low_importance(self):
        """remember() with low importance doesn't try vector backend."""
        backend = InMemoryVectorBackend()
        agent = _make_agent(enable_memory=True).with_vectors(backend)
        agent.remember("trivial note", importance=0.3, store_long_term=False)
        # Should not crash, vector backend not invoked

    def test_recall_without_vector_backend_uses_memory(self):
        """recall(semantic=True) falls through to Memory when no vector backend."""
        agent = _make_agent(enable_memory=True)
        agent.remember("test content", importance=0.5)
        # Should not crash, falls through to substring matching
        results = agent.recall(query="test", semantic=True)
        assert isinstance(results, list)

    def test_recall_from_vector_backend_returns_empty_without_provider(self):
        """_recall_from_vector_backend returns [] without embedding provider."""
        backend = InMemoryVectorBackend()
        agent = _make_agent(enable_memory=True).with_vectors(backend)
        result = agent._recall_from_vector_backend("query", limit=5, threshold=0.0)
        assert result == []


# ── Write-ahead checkpoint tests ─────────────────────────

class TestWriteAheadCheckpoint:
    """Tests for mid-tool-call checkpoint/resume."""

    def test_checkpoint_data_was_interrupted_false_by_default(self):
        """Clean checkpoints have was_interrupted == False."""
        from agentu.runtime.checkpoint import CheckpointData
        cp = CheckpointData(
            session_id="s1", agent_name="bot",
            conversation_history=[], metadata={},
            turn_count=0, created_at=0.0, checkpointed_at=1.0,
        )
        assert cp.was_interrupted is False
        assert cp.pending_tool_calls is None

    def test_checkpoint_data_was_interrupted_true(self):
        """Checkpoints with pending_tool_calls have was_interrupted == True."""
        from agentu.runtime.checkpoint import CheckpointData
        pending = {
            "user_input": "search for cats",
            "turn": 1,
            "completed_turns": [],
            "pending_tool": "web_search",
            "pending_parameters": {"query": "cats"},
        }
        cp = CheckpointData(
            session_id="s1", agent_name="bot",
            conversation_history=[], metadata={},
            turn_count=1, created_at=0.0, checkpointed_at=1.0,
            pending_tool_calls=pending,
        )
        assert cp.was_interrupted is True
        assert cp.pending_tool_calls["pending_tool"] == "web_search"

    def test_checkpoint_data_from_dict_without_pending(self):
        """from_dict handles old checkpoints without pending_tool_calls."""
        from agentu.runtime.checkpoint import CheckpointData
        old_data = {
            "session_id": "s1", "agent_name": "bot",
            "conversation_history": [], "metadata": {},
            "turn_count": 0, "created_at": 0.0,
            "checkpointed_at": 1.0, "parent_session_id": None,
        }
        cp = CheckpointData.from_dict(old_data)
        assert cp.pending_tool_calls is None
        assert cp.was_interrupted is False

    def test_checkpoint_data_roundtrip(self):
        """to_dict → from_dict preserves pending_tool_calls."""
        from agentu.runtime.checkpoint import CheckpointData
        pending = {"pending_tool": "calc", "turn": 2, "completed_turns": []}
        cp = CheckpointData(
            session_id="s1", agent_name="bot",
            conversation_history=[], metadata={},
            turn_count=1, created_at=0.0, checkpointed_at=1.0,
            pending_tool_calls=pending,
        )
        restored = CheckpointData.from_dict(cp.to_dict())
        assert restored.was_interrupted is True
        assert restored.pending_tool_calls["pending_tool"] == "calc"

    def test_checkpoint_store_save_load_with_pending(self, tmp_path):
        """SQLite store persists and restores pending_tool_calls."""
        from agentu.runtime.checkpoint import CheckpointData, CheckpointStore
        db_path = str(tmp_path / "test.db")
        store = CheckpointStore(db_path=db_path)

        pending = {
            "user_input": "search",
            "turn": 1,
            "completed_turns": [],
            "pending_tool": "web_search",
            "pending_parameters": {"q": "test"},
        }
        cp = CheckpointData(
            session_id="s-pending", agent_name="bot",
            conversation_history=[], metadata={"key": "val"},
            turn_count=1, created_at=0.0, checkpointed_at=1.0,
            pending_tool_calls=pending,
        )
        store.save(cp)

        loaded = store.load("s-pending")
        assert loaded is not None
        assert loaded.was_interrupted is True
        assert loaded.pending_tool_calls["pending_tool"] == "web_search"
        assert loaded.pending_tool_calls["pending_parameters"] == {"q": "test"}
        store.close()

    def test_checkpoint_store_save_load_without_pending(self, tmp_path):
        """SQLite store handles clean checkpoints (no pending)."""
        from agentu.runtime.checkpoint import CheckpointData, CheckpointStore
        db_path = str(tmp_path / "test.db")
        store = CheckpointStore(db_path=db_path)

        cp = CheckpointData(
            session_id="s-clean", agent_name="bot",
            conversation_history=[], metadata={},
            turn_count=2, created_at=0.0, checkpointed_at=1.0,
        )
        store.save(cp)

        loaded = store.load("s-clean")
        assert loaded is not None
        assert loaded.was_interrupted is False
        assert loaded.pending_tool_calls is None
        store.close()

    def test_session_auto_checkpoint_default_off(self):
        """Sessions have auto_checkpoint disabled by default."""
        from agentu.runtime.session import Session
        agent = _make_agent()
        session = Session(session_id="s1", agent=agent)
        assert session.auto_checkpoint is False

    def test_session_enable_auto_checkpoint(self):
        """enable_auto_checkpoint sets the flag and returns self."""
        from agentu.runtime.session import Session
        agent = _make_agent()
        session = Session(session_id="s1", agent=agent)
        result = session.enable_auto_checkpoint()
        assert result is session
        assert session.auto_checkpoint is True

    def test_session_links_to_agent(self):
        """Session.__post_init__ sets agent._active_session."""
        from agentu.runtime.session import Session
        agent = _make_agent()
        session = Session(session_id="s1", agent=agent)
        assert agent._active_session is session

    def test_agent_has_active_session_none_by_default(self):
        """Agent._active_session is None before any session is created."""
        agent = _make_agent()
        assert agent._active_session is None

    def test_checkpoint_with_pending_tool_calls(self):
        """Session.checkpoint() passes pending_tool_calls to CheckpointData."""
        from agentu.runtime.session import Session
        from agentu.runtime.checkpoint import CheckpointStore
        import tempfile, os
        agent = _make_agent(enable_memory=True)
        session = Session(session_id="s-test", agent=agent)

        pending = {"pending_tool": "calculator", "turn": 1}
        with tempfile.TemporaryDirectory() as d:
            store = CheckpointStore(db_path=os.path.join(d, "cp.db"))
            cp = session.checkpoint(store=store, pending_tool_calls=pending)
            assert cp.was_interrupted is True
            assert cp.pending_tool_calls["pending_tool"] == "calculator"

            # Load back
            loaded = store.load("s-test")
            assert loaded.was_interrupted is True
            store.close()

    def test_resume_detects_interrupted_session(self):
        """SessionManager.resume() sets metadata on interrupted sessions."""
        from agentu.runtime.session import Session, SessionManager
        from agentu.runtime.checkpoint import CheckpointData, CheckpointStore
        import tempfile, os

        with tempfile.TemporaryDirectory() as d:
            store = CheckpointStore(db_path=os.path.join(d, "cp.db"))

            # Save an interrupted checkpoint
            pending = {
                "user_input": "what is 2+2",
                "turn": 1,
                "completed_turns": [],
                "pending_tool": "calculator",
                "pending_parameters": {"expr": "2+2"},
            }
            cp = CheckpointData(
                session_id="s-crash", agent_name="bot",
                conversation_history=[], metadata={},
                turn_count=1, created_at=0.0, checkpointed_at=1.0,
                pending_tool_calls=pending,
            )
            store.save(cp)

            # Resume it
            manager = SessionManager()
            agent = _make_agent(enable_memory=True)
            session = manager.resume("s-crash", agent, store=store)
            assert session is not None
            assert session.metadata.get('_interrupted') is True
            assert session.metadata.get('_interrupted_tool') == "calculator"
            assert session.metadata.get('_interrupted_turn') == 1
            store.close()

