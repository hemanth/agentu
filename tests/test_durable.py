import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock

from agentu._core.agent import Agent
from agentu.runtime.durable import AgentLock
from agentu.runtime.checkpoint import CheckpointData, CheckpointStore


def _make_agent(**kwargs):
    return Agent('test-agent', model='test-model', auto_discover_rules=False, **kwargs)


class TestAgentLock:
    def test_acquire_release(self, tmp_path):
        lock = AgentLock("test-agent", base_dir=str(tmp_path))
        assert lock.acquire() is True
        assert lock.is_locked is True
        lock.release()
        assert lock.is_locked is False

    def test_double_acquire_raises(self, tmp_path):
        lock1 = AgentLock("test-agent", base_dir=str(tmp_path))
        lock2 = AgentLock("test-agent", base_dir=str(tmp_path))
        
        assert lock1.acquire() is True
        with pytest.raises(RuntimeError, match="is already running"):
            lock2.acquire()
            
        lock1.release()

    def test_context_manager(self, tmp_path):
        lock = AgentLock("test-agent", base_dir=str(tmp_path))
        with lock:
            assert lock.is_locked is True
        assert lock.is_locked is False


class TestWithBackendDurable:
    def test_durable_true_defaults(self):
        agent = _make_agent()
        agent.with_backend(durable=True)
        assert agent._durable is True
        assert agent._durable_dir is not None
        assert agent._storage_backend is not None or agent._backend_url is not None

    def test_durable_true_custom_backend(self):
        agent = _make_agent()
        agent.with_backend(backend="redis://localhost:6379", durable=True)
        assert agent._durable is True
        assert agent._backend_url == "redis://localhost:6379"


class TestCheckpointDataExtended:
    def test_fields_serialize_deserialize(self):
        data = CheckpointData(
            session_id="123",
            agent_name="test",
            conversation_history=[],
            metadata={},
            turn_count=1,
            created_at=100.0,
            checkpointed_at=101.0,
            schedule_state={"step": 1},
            inbox_cursor=["file1"]
        )
        as_dict = data.to_dict()
        assert as_dict["schedule_state"] == {"step": 1}
        assert as_dict["inbox_cursor"] == ["file1"]

        reconstructed = CheckpointData.from_dict(as_dict)
        assert reconstructed.schedule_state == {"step": 1}
        assert reconstructed.inbox_cursor == ["file1"]


@pytest.mark.asyncio
class TestDurableStart:
    @patch("agentu.runtime.session.SessionManager")
    @patch("agentu.runtime.durable.AgentLock")
    async def test_durable_start_acquires_lock(self, mock_lock_cls, mock_sm_cls):
        mock_lock = MagicMock()
        mock_lock_cls.return_value = mock_lock
        
        mock_sm = MagicMock()
        mock_session = MagicMock()
        mock_sm.resume.return_value = mock_session
        mock_sm_cls.return_value = mock_sm
        
        agent = _make_agent()
        agent.with_backend(durable=True)
        
        mock_scheduler = MagicMock()
        mock_scheduler.start = MagicMock(return_value=asyncio.sleep(0.01))
        agent._schedulers = [mock_scheduler]
        
        await agent.start()
        
        mock_lock_cls.assert_called_once_with(agent.name, base_dir=agent._durable_dir)
        mock_lock.acquire.assert_called_once()
        mock_sm_cls.assert_called_once()
        mock_sm.resume.assert_called_once_with(agent.name, agent)
        mock_session.enable_auto_checkpoint.assert_called_once()
        mock_lock.release.assert_called_once()
