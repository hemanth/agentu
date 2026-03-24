import pytest
import asyncio
import logging
from unittest.mock import AsyncMock, patch, MagicMock
from agentu.middleware import (
    BaseMiddleware, MiddlewareChain, CallContext,
    CostTracker, LoggerMiddleware, RetryMiddleware,
)
from agentu import Agent


# ─── CallContext ───


class TestCallContext:
    def test_defaults(self):
        ctx = CallContext(prompt="hello", namespace="test")
        assert ctx.prompt == "hello"
        assert ctx.namespace == "test"
        assert ctx.temperature == 0.7
        assert ctx.metadata == {}

    def test_elapsed_ms(self):
        import time
        ctx = CallContext(prompt="x", namespace="n", start_time=time.time() - 1.0)
        assert ctx.elapsed_ms >= 900  # At least ~1 second


# ─── CostTracker ───


class TestCostTracker:
    @pytest.mark.asyncio
    async def test_tracks_usage(self):
        tracker = CostTracker()
        ctx = CallContext(prompt="Hello world, this is a test prompt", namespace="gpt-4o")

        ctx = await tracker.before(ctx)
        assert "cost_tracker_input_tokens" in ctx.metadata

        response = "This is a response from the model"
        await tracker.after(ctx, response)

        usage = tracker.get_usage("gpt-4o")
        assert usage["calls"] == 1
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
        assert usage["total_cost"] > 0

    @pytest.mark.asyncio
    async def test_tracks_multiple_calls(self):
        tracker = CostTracker()

        for i in range(3):
            ctx = CallContext(prompt=f"Prompt {i}", namespace="gpt-4o")
            ctx = await tracker.before(ctx)
            await tracker.after(ctx, f"Response {i}")

        usage = tracker.get_usage("gpt-4o")
        assert usage["calls"] == 3

    @pytest.mark.asyncio
    async def test_separate_namespaces(self):
        tracker = CostTracker()

        ctx1 = CallContext(prompt="P1", namespace="gpt-4o")
        ctx1 = await tracker.before(ctx1)
        await tracker.after(ctx1, "R1")

        ctx2 = CallContext(prompt="P2", namespace="gpt-4o-mini")
        ctx2 = await tracker.before(ctx2)
        await tracker.after(ctx2, "R2")

        all_usage = tracker.get_usage()
        assert "gpt-4o" in all_usage
        assert "gpt-4o-mini" in all_usage

    def test_get_usage_empty_namespace(self):
        tracker = CostTracker()
        usage = tracker.get_usage("nonexistent")
        assert usage["calls"] == 0
        assert usage["total_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_reset(self):
        tracker = CostTracker()

        ctx = CallContext(prompt="test", namespace="gpt-4o")
        ctx = await tracker.before(ctx)
        await tracker.after(ctx, "response")
        assert tracker.get_usage("gpt-4o")["calls"] == 1

        tracker.reset("gpt-4o")
        assert tracker.get_usage("gpt-4o")["calls"] == 0

    @pytest.mark.asyncio
    async def test_reset_all(self):
        tracker = CostTracker()

        ctx = CallContext(prompt="test", namespace="ns1")
        ctx = await tracker.before(ctx)
        await tracker.after(ctx, "r")

        ctx = CallContext(prompt="test", namespace="ns2")
        ctx = await tracker.before(ctx)
        await tracker.after(ctx, "r")

        tracker.reset()
        assert tracker.get_usage() == {}

    def test_custom_pricing(self):
        tracker = CostTracker(pricing={"input": 5.0, "output": 15.0})
        assert tracker.pricing["input"] == 5.0
        assert tracker.pricing["output"] == 15.0

    def test_estimate_tokens(self):
        tracker = CostTracker()
        assert tracker._estimate_tokens("") == 1  # minimum 1
        assert tracker._estimate_tokens("Hello world!") == 3  # 12 chars / 4


# ─── LoggerMiddleware ───


class TestLoggerMiddleware:
    @pytest.mark.asyncio
    async def test_logs_before_and_after(self, caplog):
        mw = LoggerMiddleware(log_level=logging.INFO)
        ctx = CallContext(prompt="test prompt", namespace="test")

        with caplog.at_level(logging.INFO, logger="agentu.middleware.logger"):
            await mw.before(ctx)
            await mw.after(ctx, "test response")

        assert any("LLM call starting" in r.message for r in caplog.records)
        assert any("LLM call completed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_log_prompts(self, caplog):
        mw = LoggerMiddleware(log_prompts=True)
        ctx = CallContext(prompt="secret test prompt", namespace="test")

        with caplog.at_level(logging.INFO, logger="agentu.middleware.logger"):
            await mw.before(ctx)

        assert any("secret test prompt" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_log_prompts_by_default(self, caplog):
        mw = LoggerMiddleware()
        ctx = CallContext(prompt="secret prompt", namespace="test")

        with caplog.at_level(logging.INFO, logger="agentu.middleware.logger"):
            await mw.before(ctx)

        # The prompt text should not appear in logs
        assert not any("secret prompt" in r.message for r in caplog.records)


# ─── RetryMiddleware ───


class TestRetryMiddleware:
    def test_get_delay(self):
        mw = RetryMiddleware(base_delay=1.0, backoff_factor=2.0, max_delay=10.0)
        assert mw.get_delay(0) == 1.0
        assert mw.get_delay(1) == 2.0
        assert mw.get_delay(2) == 4.0
        assert mw.get_delay(3) == 8.0
        assert mw.get_delay(4) == 10.0  # capped at max_delay

    @pytest.mark.asyncio
    async def test_sets_retry_metadata(self):
        mw = RetryMiddleware(max_retries=5, base_delay=0.5)
        ctx = CallContext(prompt="test", namespace="test")
        ctx = await mw.before(ctx)
        assert ctx.metadata["retry_max"] == 5
        assert ctx.metadata["retry_base_delay"] == 0.5


# ─── BaseMiddleware ───


class TestBaseMiddleware:
    @pytest.mark.asyncio
    async def test_passthrough(self):
        mw = BaseMiddleware()
        ctx = CallContext(prompt="test", namespace="ns")
        assert await mw.before(ctx) is ctx
        assert await mw.after(ctx, "response") == "response"


# ─── MiddlewareChain ───


class TestMiddlewareChain:
    @pytest.mark.asyncio
    async def test_before_order(self):
        """Before hooks run in order (first added → first run)."""
        order = []

        class MW1(BaseMiddleware):
            name = "mw1"
            async def before(self, ctx):
                order.append("mw1")
                return ctx

        class MW2(BaseMiddleware):
            name = "mw2"
            async def before(self, ctx):
                order.append("mw2")
                return ctx

        chain = MiddlewareChain([MW1(), MW2()])
        ctx = CallContext(prompt="test", namespace="ns")
        await chain.run_before(ctx)
        assert order == ["mw1", "mw2"]

    @pytest.mark.asyncio
    async def test_after_reverse_order(self):
        """After hooks run in reverse order (last added → first run)."""
        order = []

        class MW1(BaseMiddleware):
            name = "mw1"
            async def after(self, ctx, response):
                order.append("mw1")
                return response

        class MW2(BaseMiddleware):
            name = "mw2"
            async def after(self, ctx, response):
                order.append("mw2")
                return response

        chain = MiddlewareChain([MW1(), MW2()])
        ctx = CallContext(prompt="test", namespace="ns")
        await chain.run_after(ctx, "resp")
        assert order == ["mw2", "mw1"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        chain = MiddlewareChain()
        ctx = CallContext(prompt="test", namespace="ns")

        async def mock_call(prompt):
            return f"response to: {prompt}"

        result = await chain.execute(ctx, mock_call)
        assert result == "response to: test"

    @pytest.mark.asyncio
    async def test_execute_with_retry(self):
        """RetryMiddleware in chain retries on failure."""
        chain = MiddlewareChain([RetryMiddleware(max_retries=2, base_delay=0.01)])
        ctx = CallContext(prompt="test", namespace="ns")

        call_count = 0

        async def flaky_call(prompt):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = await chain.execute(ctx, flaky_call)
        assert result == "success"
        assert call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_execute_retry_exhausted(self):
        """Raises last error after all retries exhausted."""
        chain = MiddlewareChain([RetryMiddleware(max_retries=1, base_delay=0.01)])
        ctx = CallContext(prompt="test", namespace="ns")

        async def always_fail(prompt):
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError, match="Always fails"):
            await chain.execute(ctx, always_fail)

    @pytest.mark.asyncio
    async def test_middleware_modifies_response(self):
        class UppercaseMiddleware(BaseMiddleware):
            name = "uppercase"
            async def after(self, ctx, response):
                return response.upper()

        chain = MiddlewareChain([UppercaseMiddleware()])
        ctx = CallContext(prompt="test", namespace="ns")

        async def mock_call(prompt):
            return "hello world"

        result = await chain.execute(ctx, mock_call)
        assert result == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_middleware_modifies_prompt(self):
        class PrefixMiddleware(BaseMiddleware):
            name = "prefix"
            async def before(self, ctx):
                ctx.prompt = f"[SYSTEM] {ctx.prompt}"
                return ctx

        chain = MiddlewareChain([PrefixMiddleware()])
        ctx = CallContext(prompt="hello", namespace="ns")

        captured_prompt = None

        async def capturing_call(prompt):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "ok"

        await chain.execute(ctx, capturing_call)
        assert captured_prompt == "[SYSTEM] hello"

    def test_add_returns_self(self):
        chain = MiddlewareChain()
        result = chain.add(BaseMiddleware())
        assert result is chain
        assert len(chain.middlewares) == 1


# ─── Agent Integration ───


class TestAgentMiddlewareIntegration:
    def test_use_returns_self(self):
        agent = Agent("test", model="test-model")
        result = agent.use(CostTracker())
        assert result is agent
        assert agent._middleware_chain is not None

    def test_use_multiple(self):
        agent = Agent("test", model="test-model")
        agent.use(CostTracker(), LoggerMiddleware(), RetryMiddleware())
        assert len(agent._middleware_chain.middlewares) == 3

    def test_use_chainable(self):
        agent = (
            Agent("test", model="test-model")
            .use(CostTracker())
            .use(LoggerMiddleware())
        )
        assert len(agent._middleware_chain.middlewares) == 2

    def test_no_middleware_by_default(self):
        agent = Agent("test", model="test-model")
        assert agent._middleware_chain is None

    def test_full_chaining(self):
        """Test that with_guardrails and use can be chained together."""
        from agentu.guardrails import PII, MaxLength

        agent = (
            Agent("test", model="test-model")
            .with_guardrails(input_guardrails=[PII(), MaxLength(max_chars=5000)])
            .use(CostTracker(), LoggerMiddleware())
        )
        assert agent._input_guardrails is not None
        assert agent._middleware_chain is not None
        assert len(agent._middleware_chain.middlewares) == 2
