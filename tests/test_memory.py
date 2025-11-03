"""Tests for memory module."""

import os
import time
import tempfile
import pytest
from agentu import Memory, MemoryEntry, ShortTermMemory, LongTermMemory, Agent


class TestMemoryEntry:
    """Test MemoryEntry class."""

    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            content="Test content",
            timestamp=time.time(),
            metadata={"key": "value"},
            memory_type="fact",
            importance=0.8
        )

        assert entry.content == "Test content"
        assert entry.memory_type == "fact"
        assert entry.importance == 0.8
        assert entry.metadata["key"] == "value"
        assert entry.access_count == 0

    def test_memory_entry_to_dict(self):
        """Test converting memory entry to dictionary."""
        entry = MemoryEntry(
            content="Test",
            timestamp=123.456,
            metadata={},
            memory_type="conversation"
        )

        data = entry.to_dict()
        assert data["content"] == "Test"
        assert data["timestamp"] == 123.456
        assert data["memory_type"] == "conversation"

    def test_memory_entry_from_dict(self):
        """Test creating memory entry from dictionary."""
        data = {
            "content": "Test",
            "timestamp": 123.456,
            "metadata": {"foo": "bar"},
            "memory_type": "task",
            "importance": 0.7,
            "access_count": 5,
            "last_accessed": 124.0
        }

        entry = MemoryEntry.from_dict(data)
        assert entry.content == "Test"
        assert entry.importance == 0.7
        assert entry.access_count == 5


class TestShortTermMemory:
    """Test ShortTermMemory class."""

    def test_add_entry(self):
        """Test adding entries to short-term memory."""
        stm = ShortTermMemory(max_size=5)

        entry = stm.add("Test memory", memory_type="conversation")
        assert entry.content == "Test memory"
        assert len(stm.entries) == 1

    def test_max_size_limit(self):
        """Test that short-term memory respects max size."""
        stm = ShortTermMemory(max_size=3)

        for i in range(5):
            stm.add(f"Memory {i}", importance=0.5)

        assert len(stm.entries) <= 3

    def test_get_recent(self):
        """Test getting recent memories."""
        stm = ShortTermMemory(max_size=10)

        for i in range(5):
            stm.add(f"Memory {i}")
            time.sleep(0.01)  # Ensure different timestamps

        recent = stm.get_recent(n=3)
        assert len(recent) == 3
        assert recent[0].content == "Memory 4"  # Most recent

    def test_clear(self):
        """Test clearing short-term memory."""
        stm = ShortTermMemory()
        stm.add("Test 1")
        stm.add("Test 2")

        assert len(stm.entries) == 2

        stm.clear()
        assert len(stm.entries) == 0


class TestLongTermMemory:
    """Test LongTermMemory class."""

    def test_add_entry(self):
        """Test adding entries to long-term memory."""
        ltm = LongTermMemory()

        entry = ltm.add("Important fact", memory_type="fact", importance=0.9)
        assert entry.content == "Important fact"
        assert len(ltm.entries) == 1

    def test_search(self):
        """Test searching long-term memory."""
        ltm = LongTermMemory()

        ltm.add("Python is a programming language", memory_type="fact")
        ltm.add("JavaScript is also a programming language", memory_type="fact")
        ltm.add("Apples are fruits", memory_type="fact")

        results = ltm.search("programming", limit=5)
        assert len(results) == 2
        assert all("programming" in r.content.lower() for r in results)

    def test_get_by_type(self):
        """Test getting memories by type."""
        ltm = LongTermMemory()

        ltm.add("Conversation 1", memory_type="conversation")
        ltm.add("Task 1", memory_type="task")
        ltm.add("Conversation 2", memory_type="conversation")

        conversations = ltm.get_by_type("conversation")
        assert len(conversations) == 2
        assert all(m.memory_type == "conversation" for m in conversations)

    def test_consolidate(self):
        """Test memory consolidation."""
        ltm = LongTermMemory()

        # Add memories with different importance
        ltm.add("Important", importance=0.8)
        ltm.add("Not important", importance=0.2)
        ltm.add("Medium", importance=0.5)

        # Wait a bit to make last_accessed old enough
        import time
        time.sleep(0.1)

        # Modify last_accessed to be old enough for consolidation
        for entry in ltm.entries:
            entry.last_accessed = time.time() - 90000  # 25 hours ago

        # Consolidate with threshold 0.4
        ltm.consolidate(importance_threshold=0.4)

        # Should keep important and medium, remove not important
        assert len(ltm.entries) == 2

    def test_persistence(self):
        """Test saving and loading from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Create and save
            ltm1 = LongTermMemory(storage_path=temp_path)
            ltm1.add("Persistent memory", memory_type="fact", importance=0.9)
            ltm1.save()

            # Load in new instance
            ltm2 = LongTermMemory(storage_path=temp_path)
            ltm2.load()

            assert len(ltm2.entries) == 1
            assert ltm2.entries[0].content == "Persistent memory"
            assert ltm2.entries[0].importance == 0.9

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestMemory:
    """Test unified Memory system."""

    def test_remember_and_recall(self):
        """Test remembering and recalling memories."""
        memory = Memory(short_term_size=5)

        memory.remember("Test memory", memory_type="conversation")

        results = memory.recall(limit=5)
        assert len(results) > 0
        assert any("Test memory" in r.content for r in results)

    def test_recall_with_query(self):
        """Test recalling with search query."""
        memory = Memory()

        memory.remember("Python programming", memory_type="fact", store_long_term=True)
        memory.remember("Coffee brewing", memory_type="fact", store_long_term=True)

        results = memory.recall(query="python", limit=5)
        assert len(results) > 0
        assert any("Python" in r.content for r in results)

    def test_recall_by_type(self):
        """Test recalling by memory type."""
        memory = Memory()

        memory.remember("Task 1", memory_type="task", store_long_term=True)
        memory.remember("Fact 1", memory_type="fact", store_long_term=True)
        memory.remember("Task 2", memory_type="task", store_long_term=True)

        tasks = memory.recall(memory_type="task", limit=10, include_short_term=False)
        # Should have 2 tasks (both stored in long-term)
        # Filter to only tasks from the results
        task_memories = [m for m in tasks if m.memory_type == "task"]
        assert len(task_memories) == 2

    def test_important_memories_go_to_long_term(self):
        """Test that important memories are stored in long-term."""
        memory = Memory()

        memory.remember("Very important", importance=0.8)  # Should auto-store

        assert len(memory.long_term.entries) > 0
        assert any("Very important" in e.content for e in memory.long_term.entries)

    def test_consolidate_to_long_term(self):
        """Test consolidating short-term to long-term."""
        memory = Memory()

        memory.remember("Important short-term", importance=0.7)
        memory.remember("Not important", importance=0.3)

        memory.consolidate_to_long_term(importance_threshold=0.6)

        # Important one should be in long-term
        assert any("Important short-term" in e.content for e in memory.long_term.entries)

    def test_get_context(self):
        """Test getting formatted context."""
        memory = Memory()

        memory.remember("First memory")
        memory.remember("Second memory")

        context = memory.get_context(max_entries=5)
        assert "First memory" in context
        assert "Second memory" in context
        assert "Recent memories:" in context

    def test_stats(self):
        """Test getting memory statistics."""
        memory = Memory(short_term_size=5)

        memory.remember("Memory 1", memory_type="conversation")
        memory.remember("Memory 2", memory_type="fact", store_long_term=True)

        stats = memory.stats()
        assert stats['short_term_size'] > 0
        assert stats['total_memories'] > 0


class TestAgentMemory:
    """Test memory integration with Agent."""

    def test_agent_with_memory_enabled(self):
        """Test agent with memory enabled."""
        agent = Agent(name="test_agent", enable_memory=True)

        assert agent.memory_enabled is True
        assert agent.memory is not None

    def test_agent_with_memory_disabled(self):
        """Test agent with memory disabled."""
        agent = Agent(name="test_agent", enable_memory=False)

        assert agent.memory_enabled is False
        assert agent.memory is None

    def test_agent_remember(self):
        """Test agent remembering information."""
        agent = Agent(name="test_agent", enable_memory=True)

        agent.remember("User likes Python", memory_type="fact", importance=0.8)

        memories = agent.recall(query="Python")
        assert len(memories) > 0

    def test_agent_recall(self):
        """Test agent recalling memories."""
        agent = Agent(name="test_agent", enable_memory=True)

        agent.remember("Project deadline is Friday", memory_type="task")
        agent.remember("Team meeting at 2pm", memory_type="task")

        tasks = agent.recall(memory_type="task")
        assert len(tasks) == 2

    def test_agent_memory_stats(self):
        """Test getting memory stats from agent."""
        agent = Agent(name="test_agent", enable_memory=True)

        agent.remember("Test memory")

        stats = agent.get_memory_stats()
        assert stats['memory_enabled'] is True
        assert stats['total_memories'] > 0

    def test_agent_memory_context(self):
        """Test getting memory context from agent."""
        agent = Agent(name="test_agent", enable_memory=True)

        agent.remember("Context memory 1")
        agent.remember("Context memory 2")

        context = agent.get_memory_context(max_entries=5)
        assert "Context memory" in context

    def test_agent_save_memory(self):
        """Test agent saving memory to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            agent = Agent(name="test_agent", enable_memory=True, memory_path=temp_path)

            agent.remember("Persistent data", store_long_term=True)
            agent.save_memory()

            assert os.path.exists(temp_path)

            # Load in new agent
            agent2 = Agent(name="test_agent2", enable_memory=True, memory_path=temp_path)
            memories = agent2.recall(query="Persistent")
            assert len(memories) > 0

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_agent_consolidate_memory(self):
        """Test agent consolidating memory."""
        agent = Agent(name="test_agent", enable_memory=True)

        agent.remember("Important", importance=0.9)
        agent.remember("Not important", importance=0.2)

        agent.consolidate_memory(importance_threshold=0.5)

        # Should have consolidated important memories to long-term
        assert len(agent.memory.long_term.entries) > 0

    def test_agent_clear_short_term(self):
        """Test clearing agent's short-term memory."""
        agent = Agent(name="test_agent", enable_memory=True)

        agent.remember("Short-term data")
        assert len(agent.memory.short_term.entries) > 0

        agent.clear_short_term_memory()
        assert len(agent.memory.short_term.entries) == 0
