"""OpenTelemetry GenAI span exporter for agentu.

Maps Observer events to OpenTelemetry GenAI semantic convention spans.
The opentelemetry SDK is an optional dependency — this module gracefully
degrades when it is not installed.

Install with::

    pip install agentu[otel]

Usage::

    agent = Agent("assistant").with_otel(
        endpoint="http://localhost:4318",
        service_name="my-agent",
    )
"""

import logging
import time
from typing import Any, Dict, Optional, List
from contextlib import contextmanager

from .observe import Observer, Event, EventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import helpers – opentelemetry is optional
# ---------------------------------------------------------------------------

_otel_available: Optional[bool] = None
_trace_mod = None
_resource_mod = None
_otlp_exporter_mod = None


def _check_otel() -> bool:
    """Check whether opentelemetry SDK is importable."""
    global _otel_available, _trace_mod, _resource_mod, _otlp_exporter_mod
    if _otel_available is not None:
        return _otel_available

    try:
        import opentelemetry.trace as _t  # type: ignore[import-untyped]
        import opentelemetry.sdk.trace as _st  # type: ignore[import-untyped]
        import opentelemetry.sdk.resources as _r  # type: ignore[import-untyped]
        import opentelemetry.sdk.trace.export as _e  # type: ignore[import-untyped]

        _trace_mod = _t
        _resource_mod = _r
        _otel_available = True
    except ImportError:
        _otel_available = False
        logger.debug("opentelemetry SDK not installed — OTel exporter disabled")

    return _otel_available


# ---------------------------------------------------------------------------
# Semantic-convention attribute names (OpenTelemetry GenAI SemConv)
# ---------------------------------------------------------------------------

_GENAI_SYSTEM = "gen_ai.system"
_GENAI_REQUEST_MODEL = "gen_ai.request.model"
_GENAI_OPERATION_NAME = "gen_ai.operation.name"
_GENAI_TOOL_NAME = "gen_ai.tool.name"
_GENAI_TOOL_CALL_ID = "gen_ai.tool.call.id"
_GENAI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GENAI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_GENAI_RESPONSE_MODEL = "gen_ai.response.model"

# Span names following the GenAI semantic conventions
_SPAN_CLIENT_OPERATION = "gen_ai.client.operation"
_SPAN_EXECUTE_TOOL = "gen_ai.execute_tool"
_SPAN_CHAT = "gen_ai.chat"

# ---------------------------------------------------------------------------
# EventType -> OTel span mapping
# ---------------------------------------------------------------------------

_EVENT_SPAN_MAP: Dict[EventType, str] = {
    EventType.INFERENCE_START: _SPAN_CLIENT_OPERATION,
    EventType.INFERENCE_END: _SPAN_CLIENT_OPERATION,
    EventType.TOOL_CALL: _SPAN_EXECUTE_TOOL,
    EventType.LLM_REQUEST: _SPAN_CHAT,
}


class OTelExporter:
    """Export Observer events as OpenTelemetry GenAI spans.

    This class hooks into the :class:`Observer` recording pipeline and
    translates each event into a properly-attributed OTel span.

    When the ``opentelemetry`` SDK is not installed the exporter is
    a silent no-op — no errors are raised.

    Args:
        service_name: OTel resource service name (default: ``"agentu"``).
        endpoint: OTLP HTTP exporter endpoint.  ``None`` uses the SDK
            default (env var ``OTEL_EXPORTER_OTLP_ENDPOINT`` or
            ``http://localhost:4318``).
        model: Model name to attach to spans.
        observer: An :class:`Observer` instance to listen to.
    """

    def __init__(
        self,
        service_name: str = "agentu",
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        observer: Optional[Observer] = None,
    ):
        self.service_name = service_name
        self.endpoint = endpoint
        self.model = model
        self._observer = observer
        self._tracer = None
        self._provider = None
        self._active_spans: Dict[str, Any] = {}
        self._enabled = False

        if _check_otel():
            self._setup_tracer()

    # -----------------------------------------------------------------
    # Tracer initialisation
    # -----------------------------------------------------------------

    def _setup_tracer(self) -> None:
        """Initialise the OTel TracerProvider and tracer."""
        try:
            import opentelemetry.trace as trace_api  # type: ignore[import-untyped]
            import opentelemetry.sdk.trace as sdk_trace  # type: ignore[import-untyped]
            import opentelemetry.sdk.resources as resources  # type: ignore[import-untyped]
            import opentelemetry.sdk.trace.export as export  # type: ignore[import-untyped]

            resource = resources.Resource.create({
                "service.name": self.service_name,
            })

            self._provider = sdk_trace.TracerProvider(resource=resource)

            # Try OTLP HTTP exporter
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-untyped]
                    OTLPSpanExporter,
                )

                exporter_kwargs: Dict[str, Any] = {}
                if self.endpoint:
                    exporter_kwargs["endpoint"] = self.endpoint

                span_exporter = OTLPSpanExporter(**exporter_kwargs)
                self._provider.add_span_processor(
                    export.BatchSpanProcessor(span_exporter)
                )
            except ImportError:
                # Fall back to console exporter for debugging
                self._provider.add_span_processor(
                    export.SimpleSpanProcessor(export.ConsoleSpanExporter())
                )
                logger.debug(
                    "OTLP exporter not installed, using ConsoleSpanExporter"
                )

            trace_api.set_tracer_provider(self._provider)
            self._tracer = trace_api.get_tracer(
                "agentu",
                schema_url="https://opentelemetry.io/schemas/1.28.0",
            )
            self._enabled = True
            logger.info(
                "OTel GenAI exporter enabled (service=%s)", self.service_name
            )
        except Exception as exc:
            logger.warning("Failed to initialise OTel tracer: %s", exc)
            self._enabled = False

    # -----------------------------------------------------------------
    # Event -> Span translation
    # -----------------------------------------------------------------

    def on_event(self, event: Event) -> None:
        """Process an Observer event and create/finish OTel spans.

        This method is designed to be called from the Observer record
        pipeline (see :meth:`attach`).

        Args:
            event: The Observer event to translate.
        """
        if not self._enabled or self._tracer is None:
            return

        span_name = _EVENT_SPAN_MAP.get(event.event_type)
        if span_name is None:
            return

        if event.event_type == EventType.INFERENCE_START:
            self._start_inference_span(event)
        elif event.event_type == EventType.INFERENCE_END:
            self._end_inference_span(event)
        elif event.event_type == EventType.TOOL_CALL:
            self._record_tool_span(event)
        elif event.event_type == EventType.LLM_REQUEST:
            self._record_chat_span(event)

    def _start_inference_span(self, event: Event) -> None:
        """Start a span wrapping the whole inference operation."""
        span = self._tracer.start_span(
            _SPAN_CLIENT_OPERATION,
            attributes={
                _GENAI_SYSTEM: "agentu",
                _GENAI_OPERATION_NAME: "inference",
                _GENAI_REQUEST_MODEL: self.model or event.metadata.get("model", "unknown"),
            },
        )
        # Key by agent name so nested agents don't collide
        self._active_spans[event.agent_name] = span

    def _end_inference_span(self, event: Event) -> None:
        """End the inference span, recording duration + tokens."""
        span = self._active_spans.pop(event.agent_name, None)
        if span is None:
            return

        if event.duration_ms is not None:
            span.set_attribute("duration_ms", event.duration_ms)

        tokens = event.metadata.get("tokens")
        if tokens:
            span.set_attribute(_GENAI_USAGE_INPUT_TOKENS, tokens)

        model = event.metadata.get("model")
        if model:
            span.set_attribute(_GENAI_RESPONSE_MODEL, model)

        span.end()

    def _record_tool_span(self, event: Event) -> None:
        """Create a child span for a tool call."""
        tool_name = event.metadata.get("tool_name", "unknown")
        attrs = {
            _GENAI_SYSTEM: "agentu",
            _GENAI_TOOL_NAME: tool_name,
            _GENAI_OPERATION_NAME: "execute_tool",
        }

        tool_call_id = event.metadata.get("tool_call_id")
        if tool_call_id:
            attrs[_GENAI_TOOL_CALL_ID] = tool_call_id

        span = self._tracer.start_span(_SPAN_EXECUTE_TOOL, attributes=attrs)
        if event.duration_ms is not None:
            span.set_attribute("duration_ms", event.duration_ms)
        span.end()

    def _record_chat_span(self, event: Event) -> None:
        """Create a span for an LLM chat completion request."""
        attrs = {
            _GENAI_SYSTEM: "agentu",
            _GENAI_OPERATION_NAME: "chat",
            _GENAI_REQUEST_MODEL: self.model or event.metadata.get("model", "unknown"),
        }

        tokens = event.metadata.get("tokens")
        if tokens:
            attrs[_GENAI_USAGE_INPUT_TOKENS] = tokens

        span = self._tracer.start_span(_SPAN_CHAT, attributes=attrs)
        if event.duration_ms is not None:
            span.set_attribute("duration_ms", event.duration_ms)
        span.end()

    # -----------------------------------------------------------------
    # Observer integration
    # -----------------------------------------------------------------

    def attach(self, observer: Observer) -> None:
        """Hook into an Observer so events are automatically exported.

        This monkey-patches the observer's ``record`` method to also
        call :meth:`on_event`.

        Args:
            observer: The Observer to listen to.
        """
        self._observer = observer
        original_record = observer.record

        def _patched_record(event_type, metadata=None, duration_ms=None):
            original_record(event_type, metadata=metadata, duration_ms=duration_ms)
            # Grab the last recorded event
            if observer.events:
                self.on_event(observer.events[-1])

        observer.record = _patched_record  # type: ignore[assignment]

    def shutdown(self) -> None:
        """Flush and shut down the tracer provider."""
        if self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception as exc:
                logger.warning("Error shutting down OTel provider: %s", exc)

    @property
    def enabled(self) -> bool:
        """Whether the exporter is actively exporting spans."""
        return self._enabled
