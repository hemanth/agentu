"""Tests for session-based stateful intelligence."""

import pytest
import asyncio
import time
from pathlib import Path
import tempfile
import shutil

from agentu import Agent, Tool, Session, SessionManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for test sessions."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def test_agent():
    """Create a test agent with simple tools."""
    def echo(message: str) -> str:
        """Echo the message back."""
        return f"Echo: {message}"
    
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    agent = Agent(
        name="TestAgent",
        model="qwen3:latest",
        enable_memory=True
    ).with_tools([
        Tool(echo),
        Tool(add)
    ])
    
    return agent


class TestSession:
    """Test Session class."""
    
    def test_session_creation(self, test_agent, temp_dir):
        """Test session can be created."""
        session = Session(
            session_id="test-123",
            agent=test_agent
        )
        
        assert session.session_id == "test-123"
        assert session.turn_count == 0
        assert session.agent == test_agent
        assert session.agent.memory_enabled
    
    @pytest.mark.asyncio
    async def test_session_send_increments_turn(self, test_agent):
        """Test that sending messages increments turn counter."""
        session = Session(session_id="test-turn", agent=test_agent)
        
        initial_turn = session.turn_count
        await session.send("test message")
        
        assert session.turn_count == initial_turn + 1
    
    @pytest.mark.asyncio
    async def test_session_send_returns_response(self, test_agent):
        """Test that send returns a response with session info."""
        session = Session(session_id="test-response", agent=test_agent)
        
        response = await session.send("echo hello")
        
        assert 'session_info' in response
        assert response['session_info']['session_id'] == "test-response"
        assert response['session_info']['turn'] >= 1
        assert 'memory_stats' in response['session_info']
    
    def test_session_get_history(self, test_agent):
        """Test getting conversation history."""
        session = Session(session_id="test-history", agent=test_agent)
        
        # Add some memories
        session.agent.remember("User: hello", memory_type='conversation')
        session.agent.remember("Agent: hi there", memory_type='conversation')
        
        history = session.get_history(limit=10)
        
        assert len(history) >= 2
        assert any('hello' in entry.content for entry in history)
    
    def test_session_clear_history(self, test_agent):
        """Test clearing conversation history."""
        session = Session(session_id="test-clear", agent=test_agent)
        
        # Add conversation and fact
        session.agent.remember("User: test", memory_type='conversation')
        session.agent.remember("Important fact", memory_type='fact')
        
        session.clear_history()
        
        # Conversation memories should be cleared
        history = session.get_history(limit=10)
        conv_entries = [e for e in history if e.memory_type == 'conversation']
        assert len(conv_entries) == 0
        
        # Facts should remain
        facts = session.agent.recall(memory_type='fact', limit=10)
        assert len(facts) > 0
    
    def test_session_to_dict(self, test_agent):
        """Test session serialization."""
        session = Session(
            session_id="test-dict",
            agent=test_agent,
            metadata={'user_id': 'user123'}
        )
        
        data = session.to_dict()
        
        assert data['session_id'] == "test-dict"
        assert 'created_at' in data
        assert 'last_accessed' in data
        assert data['metadata']['user_id'] == 'user123'
        assert 'memory_stats' in data


class TestSessionManager:
    """Test SessionManager class."""
    
    def test_manager_creation(self):
        """Test manager can be created."""
        manager = SessionManager(max_sessions=100, session_timeout=3600)
        
        assert manager.max_sessions == 100
        assert manager.session_timeout == 3600
        assert len(manager.sessions) == 0
    
    def test_create_session(self, test_agent):
        """Test creating a session."""
        manager = SessionManager()
        
        session = manager.create_session(
            agent=test_agent,
            metadata={'user_id': 'alice'}
        )
        
        assert session.session_id in manager.sessions
        assert session.metadata['user_id'] == 'alice'
    
    def test_create_session_with_custom_id(self, test_agent):
        """Test creating session with custom ID."""
        manager = SessionManager()
        
        session = manager.create_session(
            agent=test_agent,
            session_id="custom-id-123"
        )
        
        assert session.session_id == "custom-id-123"
        assert "custom-id-123" in manager.sessions
    
    def test_create_duplicate_session_returns_existing(self, test_agent):
        """Test creating duplicate session returns existing one."""
        manager = SessionManager()
        
        session1 = manager.create_session(agent=test_agent, session_id="dup")
        session2 = manager.create_session(agent=test_agent, session_id="dup")
        
        assert session1 == session2
    
    def test_get_session(self, test_agent):
        """Test retrieving a session."""
        manager = SessionManager()
        
        session = manager.create_session(agent=test_agent, session_id="retrieve")
        retrieved = manager.get_session("retrieve")
        
        assert retrieved == session
        assert retrieved.session_id == "retrieve"
    
    def test_get_nonexistent_session(self):
        """Test getting non-existent session returns None."""
        manager = SessionManager()
        
        result = manager.get_session("does-not-exist")
        
        assert result is None
    
    def test_delete_session(self, test_agent):
        """Test deleting a session."""
        manager = SessionManager()
        
        session = manager.create_session(agent=test_agent, session_id="delete-me")
        assert "delete-me" in manager.sessions
        
        deleted = manager.delete_session("delete-me")
        
        assert deleted is True
        assert "delete-me" not in manager.sessions
    
    def test_delete_nonexistent_session(self):
        """Test deleting non-existent session returns False."""
        manager = SessionManager()
        
        result = manager.delete_session("does-not-exist")
        
        assert result is False
    
    def test_list_sessions(self, test_agent):
        """Test listing all sessions."""
        manager = SessionManager()
        
        manager.create_session(agent=test_agent, metadata={'user_id': 'alice'})
        manager.create_session(agent=test_agent, metadata={'user_id': 'bob'})
        
        sessions = manager.list_sessions()
        
        assert len(sessions) == 2
    
    def test_list_sessions_filtered_by_user(self, test_agent):
        """Test listing sessions filtered by user_id."""
        manager = SessionManager()
        
        manager.create_session(agent=test_agent, metadata={'user_id': 'alice'})
        manager.create_session(agent=test_agent, metadata={'user_id': 'bob'})
        manager.create_session(agent=test_agent, metadata={'user_id': 'alice'})
        
        alice_sessions = manager.list_sessions(user_id='alice')
        
        assert len(alice_sessions) == 2
        assert all(s['metadata']['user_id'] == 'alice' for s in alice_sessions)
    
    def test_session_timeout(self, test_agent):
        """Test that expired sessions are removed on access."""
        manager = SessionManager(session_timeout=1)  # 1 second timeout
        
        session = manager.create_session(agent=test_agent, session_id="expire")
        
        # Wait for timeout
        time.sleep(1.5)
        
        # Try to get expired session
        retrieved = manager.get_session("expire")
        
        assert retrieved is None
        assert "expire" not in manager.sessions
    
    def test_max_sessions_cleanup(self, test_agent):
        """Test that old sessions are cleaned up when limit reached."""
        manager = SessionManager(max_sessions=3)
        
        # Create 3 sessions
        for i in range(3):
            manager.create_session(agent=test_agent, session_id=f"session-{i}")
        
        assert len(manager.sessions) == 3
        
        # Create one more - should trigger cleanup
        manager.create_session(agent=test_agent, session_id="session-new")
        
        # Should still be at limit (cleaned up oldest)
        assert len(manager.sessions) <= 3
    
    def test_stats(self, test_agent):
        """Test getting manager statistics."""
        manager = SessionManager(max_sessions=100, session_timeout=3600)
        
        manager.create_session(agent=test_agent)
        manager.create_session(agent=test_agent)
        
        stats = manager.stats()
        
        assert stats['total_sessions'] == 2
        assert stats['max_sessions'] == 100
        assert stats['session_timeout'] == 3600
        assert len(stats['sessions']) == 2


class TestSessionIntegration:
    """Integration tests for sessions with agent inference."""
    
    @pytest.mark.asyncio
    async def test_session_maintains_context(self, test_agent):
        """Test that session maintains context across multiple turns."""
        manager = SessionManager()
        session = manager.create_session(agent=test_agent)
        
        # First message
        response1 = await session.send("echo hello")
        
        # Check turn count increased
        assert session.turn_count == 1
        
        # Second message
        response2 = await session.send("add 5 and 3")
        
        # Check turn count increased again
        assert session.turn_count == 2
        
        # Verify both messages are in history
        history = session.get_history(limit=10)
        assert len(history) >= 2
    
    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self, test_agent):
        """Test that multiple sessions maintain separate state."""
        manager = SessionManager()
        
        session1 = manager.create_session(agent=test_agent, session_id="user1")
        session2 = manager.create_session(agent=test_agent, session_id="user2")
        
        # Each session sends different message
        await session1.send("echo user1 message")
        await session2.send("echo user2 message")
        
        # Get histories
        history1 = session1.get_history(limit=10)
        history2 = session2.get_history(limit=10)
        
        # Verify separation
        user1_contents = [e.content for e in history1]
        user2_contents = [e.content for e in history2]
        
        assert any('user1' in c for c in user1_contents)
        assert any('user2' in c for c in user2_contents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
