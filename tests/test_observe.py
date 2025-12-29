"""Tests for observability system."""

import pytest
import asyncio
from agentu import Agent, Tool, observe
from agentu.observe import Observer, EventType, OutputFormat, Event, Metrics


@pytest.fixture
def test_observer():
    """Create test observer."""
    return Observer(agent_name="test", output=OutputFormat.SILENT)


@pytest.fixture
def test_agent():
    """Create test agent with simple tools."""
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    # Configure observe to be silent for tests
    observe.configure(output="silent", enabled=True)
    
    agent = Agent("test_agent").with_tools([Tool(add)])
    return agent


class TestEvent:
    """Test Event dataclass."""
    
    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            event_type=EventType.TOOL_CALL,
            agent_name="test",
            metadata={"tool": "add"}
        )
        
        assert event.event_type == EventType.TOOL_CALL
        assert event.agent_name == "test"
        assert event.metadata["tool"] == "add"
        assert event.timestamp is not None
    
    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = Event(
            event_type=EventType.TOOL_CALL,
            agent_name="test",
            duration_ms=123.45,
            metadata={"tool": "search"}
        )
        
        data = event.to_dict()
        
        assert data["event"] == "tool_call"
        assert data["agent"] == "test"
        assert data["duration_ms"] == 123.45
        assert data["tool"] == "search"
    
    def test_event_to_json(self):
        """Test JSON serialization."""
        import json
        
        event = Event(
            event_type=EventType.TOOL_CALL,
            agent_name="test"
        )
        
        json_str = event.to_json()
        data = json.loads(json_str)
        
        assert data["event"] == "tool_call"
        assert data["agent"] == "test"


class TestObserver:
    """Test Observer class."""
    
    def test_observer_creation(self, test_observer):
        """Test observer initialization."""
        assert test_observer.agent_name == "test"
        assert test_observer.output == OutputFormat.SILENT
        assert test_observer.enabled
        assert len(test_observer.events) == 0
    
    def test_record_event(self, test_observer):
        """Test recording an event."""
        test_observer.record(
            EventType.TOOL_CALL,
            metadata={"tool": "search"}
        )
        
        assert len(test_observer.events) == 1
        assert test_observer.events[0].event_type == EventType.TOOL_CALL
    
    def test_metrics_update(self, test_observer):
        """Test that metrics are updated."""
        test_observer.record(EventType.TOOL_CALL)
        test_observer.record(EventType.LLM_REQUEST)
        test_observer.record(EventType.ERROR)
        
        metrics = test_observer.get_metrics()
        
        assert metrics["tool_calls"] == 1
        assert metrics["llm_requests"] == 1
        assert metrics["errors"] == 1
    
    def test_trace_context_manager(self, test_observer):
        """Test trace context manager."""
        with test_observer.trace(EventType.TOOL_CALL, {"tool": "add"}):
            pass  # Simulate work
        
        assert len(test_observer.events) == 1
        event = test_observer.events[0]
        assert event.duration_ms is not None
        assert event.duration_ms >= 0
    
    def test_get_events(self, test_observer):
        """Test getting events."""
        test_observer.record(EventType.TOOL_CALL)
        test_observer.record(EventType.LLM_REQUEST)
        
        events = test_observer.get_events()
        
        assert len(events) == 2
        assert all(isinstance(e, dict) for e in events)
    
    def test_clear(self, test_observer):
        """Test clearing events and metrics."""
        test_observer.record(EventType.TOOL_CALL)
        test_observer.record(EventType.LLM_REQUEST)
        
        test_observer.clear()
        
        assert len(test_observer.events) == 0
        assert test_observer.get_metrics()["tool_calls"] == 0


class TestAgentInstrumentation:
    """Test agent instrumentation."""
    
    @pytest.mark.asyncio
    async def test_tool_call_tracked(self, test_agent):
        """Test that tool calls are tracked."""
        initial_events = len(test_agent.observer.events)
        
        await test_agent.call("add", {"x": 5, "y": 3})
        
        # Should have recorded tool call event
        assert len(test_agent.observer.events) > initial_events
        
        # Check for tool call event
        tool_events = [
            e for e in test_agent.observer.events
            if e.event_type == EventType.TOOL_CALL
        ]
        assert len(tool_events) > 0
    
    @pytest.mark.asyncio
    async def test_inference_tracked(self, test_agent):
        """Test that inference is tracked."""
        initial_events = len(test_agent.observer.events)
        
        await test_agent.infer("Add 5 and 3")
        
        # Should have start and end events
        start_events = [
            e for e in test_agent.observer.events
            if e.event_type == Event Type.INFERENCE_START
        ]
        end_events = [
            e for e in test_agent.observer.events
            if e.event_type == EventType.INFERENCE_END
        ]
        
        assert len(start_events) > 0
        assert len(end_events) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_collected(self, test_agent):
        """Test that metrics are collected."""
        await test_agent.call("add", {"x": 5, "y": 3})
        
        metrics = test_agent.observer.get_metrics()
        
        assert metrics["tool_calls"] >= 1


class TestConfiguration:
    """Test global configuration."""
    
    def test_configure(self):
        """Test global configuration."""
        observe.configure(output="json", enabled=False)
        
        output, enabled = observe.get_config()
        
        assert output == OutputFormat.JSON
        assert enabled is False
        
        # Reset
        observe.configure(output="console", enabled=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
