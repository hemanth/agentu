"""Tests for always-on memory patterns: structured extraction, consolidation, inbox."""
import os
import time
import pytest
import asyncio
from agentu import Agent
from agentu.memory.memory import MemoryEntry, Memory


def _make_agent(**kwargs):
    return Agent('test-agent', model='test-model', auto_discover_rules=False, **kwargs)


class TestStructuredMemory:
    def test_memory_entry_new_fields_default(self):
        entry = MemoryEntry(content='test', timestamp=0, metadata={}, memory_type='fact')
        assert entry.summary is None
        assert entry.entities == []
        assert entry.topics == []
        assert entry.source is None
        assert entry.consolidated is False

    def test_memory_entry_with_new_fields(self):
        entry = MemoryEntry(
            content='AI news', timestamp=0, metadata={}, memory_type='fact',
            summary='AI is growing', entities=['OpenAI'], topics=['AI'],
            source='news.md', consolidated=True,
        )
        assert entry.summary == 'AI is growing'
        assert entry.entities == ['OpenAI']
        assert entry.topics == ['AI']
        assert entry.source == 'news.md'
        assert entry.consolidated is True

    def test_memory_entry_roundtrip(self):
        entry = MemoryEntry(
            content='test', timestamp=1.0, metadata={'k': 'v'}, memory_type='fact',
            entities=['X'], topics=['Y'], source='file.md',
        )
        restored = MemoryEntry.from_dict(entry.to_dict())
        assert restored.entities == ['X']
        assert restored.topics == ['Y']
        assert restored.source == 'file.md'

    def test_from_dict_handles_old_data(self):
        old_data = {
            'content': 'old', 'timestamp': 1.0, 'metadata': {},
            'memory_type': 'fact', 'importance': 0.5,
            'access_count': 0, 'last_accessed': 0.0,
        }
        entry = MemoryEntry.from_dict(old_data)
        assert entry.summary is None
        assert entry.entities == []
        assert entry.consolidated is False

    def test_remember_with_structured_fields(self):
        agent = _make_agent(enable_memory=True)
        agent.remember(
            'AI agents are growing', memory_type='fact',
            entities=['AI'], topics=['technology'], source='report.md',
        )
        entries = agent.recall(memory_type='fact', include_short_term=True)
        assert len(entries) >= 1
        entry = entries[0]
        assert entry.entities == ['AI']
        assert entry.topics == ['technology']
        assert entry.source == 'report.md'

    def test_remember_backward_compatible(self):
        agent = _make_agent(enable_memory=True)
        agent.remember('old style', memory_type='fact')
        entries = agent.recall(memory_type='fact', include_short_term=True)
        assert len(entries) >= 1
        assert entries[0].entities == []


class TestConsolidation:
    def test_with_consolidation_returns_self(self):
        agent = _make_agent(enable_memory=True)
        result = agent.with_consolidation(every=60)
        assert result is agent

    def test_with_consolidation_adds_tool(self):
        agent = _make_agent(enable_memory=True)
        agent.with_consolidation(every=60)
        tool_names = {t.name for t in agent.tools}
        assert 'consolidate_memories' in tool_names

    def test_consolidation_tool_stores_memory(self):
        agent = _make_agent(enable_memory=True)
        agent.with_consolidation(every=60)
        tool = next(t for t in agent.tools if t.name == 'consolidate_memories')
        result = tool.function(
            insight='AI and cost reduction are connected themes',
            related_topics=['AI', 'cost'],
            source_summaries=[],
        )
        assert result['status'] == 'consolidated'
        entries = agent.recall(memory_type='consolidation', include_short_term=True)
        assert len(entries) >= 1


class TestInbox:
    def test_with_inbox_returns_self(self):
        agent = _make_agent(enable_memory=True)
        result = agent.with_inbox('/tmp/test-inbox')
        assert result is agent
        assert agent._inbox_path == '/tmp/test-inbox'

    def test_with_inbox_default_path(self):
        agent = _make_agent(enable_memory=True)
        agent.with_inbox()
        assert agent._inbox_path.endswith('inbox')

    @pytest.mark.asyncio
    async def test_poll_inbox_empty(self, tmp_path):
        agent = _make_agent(enable_memory=True)
        inbox = str(tmp_path / 'inbox')
        os.makedirs(inbox)
        agent.with_inbox(inbox)
        await agent._poll_inbox()  # Should not crash

    @pytest.mark.asyncio
    async def test_poll_inbox_skips_hidden(self, tmp_path):
        agent = _make_agent(enable_memory=True)
        inbox = str(tmp_path / 'inbox')
        os.makedirs(inbox)
        (tmp_path / 'inbox' / '.hidden').write_text('secret')
        agent.with_inbox(inbox)
        await agent._poll_inbox()
        # Hidden file should still be there
        assert (tmp_path / 'inbox' / '.hidden').exists()

    def test_workspace_inbox_config(self, tmp_path):
        from agentu.workspace import parse_agent_yaml
        ws = tmp_path / '.agentu'
        ws.mkdir()
        (ws / 'agent.yaml').write_text('name: bot\ninbox:\n  watch: ./inbox\n  poll_interval: 10\n')
        config = parse_agent_yaml(str(ws / 'agent.yaml'))
        assert config.inbox_path is not None
        assert config.inbox_poll_interval == 10
