"""Observability and monitoring for agentu.

Provides simple event tracking, metrics, and logging for debugging
and production monitoring.
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of observable events."""
    TOOL_CALL = "tool_call"
    LLM_REQUEST = "llm_request"
    INFERENCE_START = "inference_start"
    INFERENCE_END = "inference_end"
    ERROR = "error"
    SESSION_CREATE = "session_create"
    SESSION_END = "session_end"


class OutputFormat(Enum):
    """Output format for events."""
    JSON = "json"
    CONSOLE = "console"
    SILENT = "silent"


@dataclass
class Event:
    """Represents an observable event."""
    event_type: EventType
    agent_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Handle both EventType enum and string
        event_value = self.event_type.value if hasattr(self.event_type, 'value') else str(self.event_type)
        
        return {
            "event": event_value,
            "agent": self.agent_name,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            **self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Metrics:
    """Performance metrics."""
    tool_calls: int = 0
    llm_requests: int = 0
    total_duration_ms: float = 0.0
    errors: int = 0
    sessions_created: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class Observer:
    """Central observability manager.
    
    Tracks events, metrics, and provides simple monitoring interface.
    """
    
    def __init__(
        self,
        agent_name: str = "agent",
        output: OutputFormat = OutputFormat.CONSOLE,
        enabled: bool = True
    ):
        """Initialize observer.
        
        Args:
            agent_name: Name of the agent being observed
            output: Output format (json, console, silent)
            enabled: Whether observation is enabled
        """
        self.agent_name = agent_name
        self.output = output
        self.enabled = enabled
        self.events: List[Event] = []
        self.metrics = Metrics()
    
    def record(
        self,
        event_type: EventType,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None
    ):
        """Record an event.
        
        Args:
            event_type: Type of event
            metadata: Additional event data
            duration_ms: Optional duration
        """
        if not self.enabled:
            return
        
        event = Event(
            event_type=event_type,
            agent_name=self.agent_name,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        self._update_metrics(event)
        self._output_event(event)
    
    def _update_metrics(self, event: Event):
        """Update metrics based on event."""
        # Normalize event type to string for comparison
        event_str = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
        
        if event_str == "tool_call" or event.event_type == EventType.TOOL_CALL:
            self.metrics.tool_calls += 1
        elif event_str == "llm_request" or event.event_type == EventType.LLM_REQUEST:
            self.metrics.llm_requests += 1
        elif event_str == "error" or event.event_type == EventType.ERROR:
            self.metrics.errors += 1
        elif event_str == "session_create" or event.event_type == EventType.SESSION_CREATE:
            self.metrics.sessions_created += 1
        
        if event.duration_ms:
            self.metrics.total_duration_ms += event.duration_ms
    
    def _output_event(self, event: Event):
        """Output event based on configured format."""
        if self.output == OutputFormat.SILENT:
            return
        
        if self.output == OutputFormat.JSON:
            print(event.to_json())
        elif self.output == OutputFormat.CONSOLE:
            self._console_output(event)
    
    def _console_output(self, event: Event):
        """Human-readable console output."""
        try:
            BLUE = '\033[94m'
            GREEN = '\033[92m'
            RED = '\033[91m'
            RESET = '\033[0m'
        except:
            BLUE = GREEN = RED = RESET = ''
        
        symbol = {
            EventType.TOOL_CALL: f"{BLUE}🔧{RESET}",
            EventType.LLM_REQUEST: f"{BLUE}🤖{RESET}",
            EventType.INFERENCE_START: f"{GREEN}▶{RESET}",
            EventType.INFERENCE_END: f"{GREEN}✓{RESET}",
            EventType.ERROR: f"{RED}✗{RESET}",
            EventType.SESSION_CREATE: f"{GREEN}📝{RESET}",
            EventType.SESSION_END: f"{BLUE}📋{RESET}",
        }.get(event.event_type, "•")
        
        parts = [f"{symbol} {event.event_type.value}"]
        
        if event.duration_ms:
            parts.append(f"({event.duration_ms:.1f}ms)")
        
        # Add relevant metadata
        if event.event_type == EventType.TOOL_CALL:
            tool = event.metadata.get('tool_name', 'unknown')
            parts.append(f"- {tool}")
        elif event.event_type == EventType.LLM_REQUEST:
            tokens = event.metadata.get('tokens', 0)
            if tokens:
                parts.append(f"- {tokens} tokens")
        
        logger.info(" ".join(parts))
    
    @contextmanager
    def trace(self, event_type: EventType, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracing execution time.
        
        Example:
            with observer.trace(EventType.TOOL_CALL, {"tool": "search"}):
                # execute tool
                pass
        """
        if not self.enabled:
            yield
            return
        
        start = time.time()
        error = None
        
        try:
            yield
        except Exception as e:
            error = e
            raise
        finally:
            duration_ms = (time.time() - start) * 1000
            
            if error:
                self.record(
                    EventType.ERROR,
                    metadata={**(metadata or {}), "error": str(error)},
                    duration_ms=duration_ms
                )
            else:
                self.record(event_type, metadata, duration_ms)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary with all metrics
        """
        return self.metrics.to_dict()
    
    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events.
        
        Args:
            limit: Maximum number of events to return
        
        Returns:
            List of event dictionaries
        """
        return [e.to_dict() for e in self.events[-limit:]]
    
    def clear(self):
        """Clear all events and reset metrics."""
        self.events.clear()
        self.metrics = Metrics()


# Global configuration
_global_output = OutputFormat.CONSOLE
_global_enabled = True


def configure(
    output: str = "console",
    enabled: bool = True
):
    """Configure global observability settings.
    
    Args:
        output: Output format ("json" | "console" | "silent")
        enabled: Whether to enable observation
    
    Example:
        >>> from agentu import observe
        >>> observe.configure(output="json", enabled=True)
    """
    global _global_output, _global_enabled
    
    _global_output = OutputFormat[output.upper()]
    _global_enabled = enabled


def get_config() -> tuple[OutputFormat, bool]:
    """Get current global configuration."""
    return _global_output, _global_enabled
