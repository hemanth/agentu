"""Middleware pipeline for agent LLM calls.

Express-style before/after hooks that wrap every LLM call.
Middleware runs in order for `before`, and reverse order for `after`.
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class CallContext:
    """Context passed through the middleware pipeline."""
    prompt: str
    namespace: str
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


@runtime_checkable
class Middleware(Protocol):
    """Protocol for middleware implementations."""

    name: str

    async def before(self, context: CallContext) -> CallContext:
        """Called before the LLM request. Return modified context."""
        ...

    async def after(self, context: CallContext, response: str) -> str:
        """Called after the LLM response. Return modified response."""
        ...


class BaseMiddleware:
    """Base class with default pass-through implementations."""

    name: str = "base"

    async def before(self, context: CallContext) -> CallContext:
        return context

    async def after(self, context: CallContext, response: str) -> str:
        return response


class CostTracker(BaseMiddleware):
    """Track estimated token usage and cost per namespace.

    Uses a simple heuristic: ~4 chars per token (works for English).
    """

    name = "cost_tracker"

    # Cost per 1M tokens (input, output) — defaults for GPT-4o
    DEFAULT_PRICING = {
        "input": 2.50,   # $ per 1M input tokens
        "output": 10.00,  # $ per 1M output tokens
    }

    def __init__(self, pricing: Optional[Dict[str, float]] = None):
        self.pricing = pricing or self.DEFAULT_PRICING
        self._usage: Dict[str, Dict[str, Any]] = {}  # namespace → stats

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    async def before(self, context: CallContext) -> CallContext:
        context.metadata["cost_tracker_input_tokens"] = self._estimate_tokens(context.prompt)
        return context

    async def after(self, context: CallContext, response: str) -> str:
        ns = context.namespace
        input_tokens = context.metadata.get("cost_tracker_input_tokens", 0)
        output_tokens = self._estimate_tokens(response)

        if ns not in self._usage:
            self._usage[ns] = {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "total_cost": 0.0,
            }

        stats = self._usage[ns]
        stats["calls"] += 1
        stats["input_tokens"] += input_tokens
        stats["output_tokens"] += output_tokens
        stats["total_cost"] += (
            (input_tokens / 1_000_000) * self.pricing["input"]
            + (output_tokens / 1_000_000) * self.pricing["output"]
        )

        return response

    def get_usage(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get usage stats for a namespace or all namespaces."""
        if namespace:
            return self._usage.get(namespace, {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "total_cost": 0.0,
            })
        return dict(self._usage)

    def reset(self, namespace: Optional[str] = None) -> None:
        """Reset usage stats."""
        if namespace:
            self._usage.pop(namespace, None)
        else:
            self._usage.clear()


class LoggerMiddleware(BaseMiddleware):
    """Structured logging of all LLM calls."""

    name = "logger"

    def __init__(self, log_level: int = logging.INFO, log_prompts: bool = False):
        """Initialize logger middleware.

        Args:
            log_level: Python logging level
            log_prompts: If True, log full prompt text (careful with PII!)
        """
        self.log_level = log_level
        self.log_prompts = log_prompts
        self._logger = logging.getLogger("agentu.middleware.logger")

    async def before(self, context: CallContext) -> CallContext:
        msg = f"[{context.namespace}] LLM call starting (prompt_len={len(context.prompt)}, temp={context.temperature})"
        if self.log_prompts:
            msg += f"\n  prompt: {context.prompt[:200]}..."
        self._logger.log(self.log_level, msg)
        return context

    async def after(self, context: CallContext, response: str) -> str:
        self._logger.log(
            self.log_level,
            f"[{context.namespace}] LLM call completed "
            f"(response_len={len(response)}, elapsed={context.elapsed_ms:.0f}ms)"
        )
        return response


class RetryMiddleware(BaseMiddleware):
    """Retry failed LLM calls with exponential backoff.

    This middleware wraps the call context with retry metadata.
    The actual retry logic is triggered by the MiddlewareChain when
    the LLM call raises an exception.
    """

    name = "retry"

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 30.0, backoff_factor: float = 2.0):
        """Initialize retry middleware.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay cap in seconds
            backoff_factor: Multiplier for each retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    async def before(self, context: CallContext) -> CallContext:
        context.metadata["retry_max"] = self.max_retries
        context.metadata["retry_base_delay"] = self.base_delay
        context.metadata["retry_max_delay"] = self.max_delay
        context.metadata["retry_backoff_factor"] = self.backoff_factor
        return context

    async def after(self, context: CallContext, response: str) -> str:
        return response

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt (0-indexed)."""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


class MiddlewareChain:
    """Execute middleware in order (before) and reverse order (after)."""

    def __init__(self, middlewares: Optional[List[BaseMiddleware]] = None):
        self.middlewares: List[BaseMiddleware] = middlewares or []

    def add(self, middleware: BaseMiddleware) -> 'MiddlewareChain':
        """Add middleware to the chain."""
        self.middlewares.append(middleware)
        return self

    async def run_before(self, context: CallContext) -> CallContext:
        """Run all before hooks in order."""
        for mw in self.middlewares:
            context = await mw.before(context)
        return context

    async def run_after(self, context: CallContext, response: str) -> str:
        """Run all after hooks in reverse order."""
        for mw in reversed(self.middlewares):
            response = await mw.after(context, response)
        return response

    async def execute(self, context: CallContext, call_fn) -> str:
        """Execute the full pipeline: before → call → after.

        Handles retries if RetryMiddleware is in the chain.

        Args:
            context: The call context
            call_fn: Async callable that makes the actual LLM call

        Returns:
            The (possibly modified) response string
        """
        context = await self.run_before(context)

        max_retries = context.metadata.get("retry_max", 0)
        base_delay = context.metadata.get("retry_base_delay", 1.0)
        max_delay = context.metadata.get("retry_max_delay", 30.0)
        backoff_factor = context.metadata.get("retry_backoff_factor", 2.0)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await call_fn(context.prompt)
                return await self.run_after(context, response)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        raise last_error
