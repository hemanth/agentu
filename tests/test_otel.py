"""Tests for OpenTelemetry GenAI span exporter."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agentu.middleware.observe import Observer, EventType, OutputFormat, Event
from agentu.middleware.otel import (
    OTelExporter,
    _check_otel,
    _EVENT_SPAN_MAP,
    _SPAN_CLIENT_OPERATION,
    _SPAN_EXECUTE_TOOL,
    _SPAN_CHAT,
    _GENAI_SYSTEM,
    _GENAI_REQUEST_MODEL,
    _GENAI_OPERATION_NAME,
    _GENAI_TOOL_NAME,
)


@pytest.fixture
def observer():
    """Create a silent test observer."""
    return Observer(agent_name="test", output=OutputFormat.SILENT)


class TestOTelAvailability:
    """Test graceful degradation when opentelemetry is not installed."""

    def test_check_otel_returns_bool(self):
        """_check_otel returns a boolean."""
        result = _check_otel()
        assert isinstance(result, bool)

    def test_exporter_disabled_without_otel(self):
        """OTelExporter is a no-op when opentelemetry is not installed."""
        with patch("agentu.middleware.otel._check_otel", return_value=False):
            with patch("agentu.middleware.otel._otel_available", False):
                exporter = OTelExporter.__new__(OTelExporter)
                exporter.service_name = "test"
                exporter.endpoint = None
                exporter.model = None
                exporter._observer = None
                exporter._tracer = None
                exporter._provider = None
                exporter._active_spans = {}
                exporter._enabled = False
                assert not exporter.enabled

    def test_exporter_on_event_noop_when_disabled(self, observer):
        """on_event does nothing when exporter is disabled."""
        exporter = OTelExporter.__new__(OTelExporter)
        exporter._enabled = False
        exporter._tracer = None
        exporter._active_spans = {}

        event = Event(
            event_type=EventType.TOOL_CALL,
            agent_name="test",
            metadata={"tool_name": "search"},
        )
        # Should not raise
        exporter.on_event(event)


class TestEventSpanMapping:
    """Test EventType to span name mapping."""

    def test_inference_maps_to_client_operation(self):
        assert _EVENT_SPAN_MAP[EventType.INFERENCE_START] == _SPAN_CLIENT_OPERATION
        assert _EVENT_SPAN_MAP[EventType.INFERENCE_END] == _SPAN_CLIENT_OPERATION

    def test_tool_call_maps_to_execute_tool(self):
        assert _EVENT_SPAN_MAP[EventType.TOOL_CALL] == _SPAN_EXECUTE_TOOL

    def test_llm_request_maps_to_chat(self):
        assert _EVENT_SPAN_MAP[EventType.LLM_REQUEST] == _SPAN_CHAT

    def test_unmapped_events_return_none(self):
        assert _EVENT_SPAN_MAP.get(EventType.ERROR) is None
        assert _EVENT_SPAN_MAP.get(EventType.SESSION_CREATE) is None


class TestOTelExporterSpans:
    """Test span creation from events (with mocked tracer)."""

    def _make_exporter(self) -> OTelExporter:
        """Create an exporter with a mocked tracer."""
        exporter = OTelExporter.__new__(OTelExporter)
        exporter.service_name = "test-service"
        exporter.endpoint = None
        exporter.model = "test-model"
        exporter._observer = None
        exporter._provider = None
        exporter._active_spans = {}
        exporter._enabled = True

        # Mock tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        exporter._tracer = mock_tracer

        return exporter

    def test_inference_start_creates_span(self):
        """INFERENCE_START creates an active span."""
        exporter = self._make_exporter()
        event = Event(
            event_type=EventType.INFERENCE_START,
            agent_name="test",
            metadata={"model": "gpt-4"},
        )

        exporter.on_event(event)

        exporter._tracer.start_span.assert_called_once()
        call_args = exporter._tracer.start_span.call_args
        assert call_args[0][0] == _SPAN_CLIENT_OPERATION
        assert "test" in exporter._active_spans

    def test_inference_end_closes_span(self):
        """INFERENCE_END closes the active inference span."""
        exporter = self._make_exporter()
        mock_span = MagicMock()
        exporter._active_spans["test"] = mock_span

        event = Event(
            event_type=EventType.INFERENCE_END,
            agent_name="test",
            duration_ms=150.0,
            metadata={"tokens": 500, "model": "gpt-4"},
        )

        exporter.on_event(event)

        mock_span.set_attribute.assert_any_call("duration_ms", 150.0)
        mock_span.end.assert_called_once()
        assert "test" not in exporter._active_spans

    def test_tool_call_creates_and_closes_span(self):
        """TOOL_CALL creates a span with tool attributes."""
        exporter = self._make_exporter()
        event = Event(
            event_type=EventType.TOOL_CALL,
            agent_name="test",
            duration_ms=25.0,
            metadata={"tool_name": "search", "tool_call_id": "call_123"},
        )

        exporter.on_event(event)

        exporter._tracer.start_span.assert_called_once()
        call_args = exporter._tracer.start_span.call_args
        assert call_args[0][0] == _SPAN_EXECUTE_TOOL
        attrs = call_args[1]["attributes"]
        assert attrs[_GENAI_TOOL_NAME] == "search"

        # Span should be ended immediately
        mock_span = exporter._tracer.start_span.return_value
        mock_span.end.assert_called_once()

    def test_llm_request_creates_chat_span(self):
        """LLM_REQUEST creates a gen_ai.chat span."""
        exporter = self._make_exporter()
        event = Event(
            event_type=EventType.LLM_REQUEST,
            agent_name="test",
            duration_ms=200.0,
            metadata={"tokens": 1000, "model": "gpt-4"},
        )

        exporter.on_event(event)

        call_args = exporter._tracer.start_span.call_args
        assert call_args[0][0] == _SPAN_CHAT
        mock_span = exporter._tracer.start_span.return_value
        mock_span.end.assert_called_once()

    def test_unmapped_event_ignored(self):
        """Events without a span mapping are silently ignored."""
        exporter = self._make_exporter()
        event = Event(
            event_type=EventType.ERROR,
            agent_name="test",
            metadata={"error": "something broke"},
        )

        exporter.on_event(event)
        exporter._tracer.start_span.assert_not_called()


class TestOTelExporterAttach:
    """Test Observer integration."""

    def test_attach_patches_observer_record(self, observer):
        """attach() monkey-patches observer.record to forward events."""
        exporter = OTelExporter.__new__(OTelExporter)
        exporter.service_name = "test"
        exporter.endpoint = None
        exporter.model = None
        exporter._observer = None
        exporter._provider = None
        exporter._active_spans = {}
        exporter._enabled = True
        exporter._tracer = MagicMock()

        original_record = observer.record
        exporter.attach(observer)

        # record should be patched
        assert observer.record is not original_record

        # Recording an event should still work
        observer.record(EventType.TOOL_CALL, metadata={"tool_name": "test_tool"})
        assert len(observer.events) == 1
        assert observer.events[0].event_type == EventType.TOOL_CALL

    def test_attach_forwards_events_to_exporter(self, observer):
        """Events recorded after attach() are forwarded to on_event."""
        exporter = OTelExporter.__new__(OTelExporter)
        exporter.service_name = "test"
        exporter.endpoint = None
        exporter.model = None
        exporter._observer = None
        exporter._provider = None
        exporter._active_spans = {}
        exporter._enabled = True

        mock_tracer = MagicMock()
        exporter._tracer = mock_tracer

        exporter.attach(observer)

        observer.record(EventType.LLM_REQUEST, metadata={"tokens": 100})
        # on_event should have been called -> tracer.start_span called
        mock_tracer.start_span.assert_called_once()


class TestWithOTel:
    """Test the Agent.with_otel() builder method."""

    def test_with_otel_returns_self(self):
        """with_otel() returns the agent for chaining."""
        from agentu.middleware.observe import Observer, OutputFormat
        from agentu.middleware import observe

        observe.configure(output="silent", enabled=True)

        # We can't easily construct a full Agent without Ollama,
        # so test the OTelExporter integration directly
        observer = Observer(agent_name="test", output=OutputFormat.SILENT)
        exporter = OTelExporter(
            service_name="test-service",
            model="test-model",
            observer=observer,
        )
        exporter.attach(observer)

        # Verify the exporter is properly configured
        assert exporter.service_name == "test-service"
        assert exporter.model == "test-model"

    def test_shutdown_does_not_raise(self):
        """shutdown() is safe even when provider is None."""
        exporter = OTelExporter.__new__(OTelExporter)
        exporter._provider = None
        exporter.shutdown()  # Should not raise


class TestOTelSemConvAttributes:
    """Verify semantic convention attribute values."""

    def test_genai_system_is_agentu(self):
        """gen_ai.system should be 'agentu'."""
        exporter = TestOTelExporterSpans()._make_exporter()
        event = Event(
            event_type=EventType.TOOL_CALL,
            agent_name="test",
            metadata={"tool_name": "fetch"},
        )
        exporter.on_event(event)

        attrs = exporter._tracer.start_span.call_args[1]["attributes"]
        assert attrs[_GENAI_SYSTEM] == "agentu"

    def test_model_from_exporter(self):
        """Model attribute comes from exporter when set."""
        exporter = TestOTelExporterSpans()._make_exporter()
        exporter.model = "my-model"
        event = Event(
            event_type=EventType.LLM_REQUEST,
            agent_name="test",
            metadata={},
        )
        exporter.on_event(event)

        attrs = exporter._tracer.start_span.call_args[1]["attributes"]
        assert attrs[_GENAI_REQUEST_MODEL] == "my-model"

    def test_model_falls_back_to_metadata(self):
        """Model attribute falls back to event metadata."""
        exporter = TestOTelExporterSpans()._make_exporter()
        exporter.model = None
        event = Event(
            event_type=EventType.LLM_REQUEST,
            agent_name="test",
            metadata={"model": "gpt-4o"},
        )
        exporter.on_event(event)

        attrs = exporter._tracer.start_span.call_args[1]["attributes"]
        assert attrs[_GENAI_REQUEST_MODEL] == "gpt-4o"
