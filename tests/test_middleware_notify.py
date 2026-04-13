import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from agentu.middleware.middleware import CallContext

def test_notify_middleware_import_error():
    """Test that an ImportError is raised if apprise is missing."""
    with patch.dict(sys.modules, {'apprise': None}):
        # Force reload to pick up the mocked None
        import importlib
        import agentu.middleware.notify
        importlib.reload(agentu.middleware.notify)
        
        with pytest.raises(ImportError, match="The 'apprise' library is required"):
            agentu.middleware.notify.NotifyMiddleware(targets=["slack://test"])

@pytest.mark.asyncio
async def test_notify_middleware_success():
    """Test that notifications are created and sent properly."""
    mock_apprise_obj = MagicMock()
    mock_apprise_obj.notify.return_value = True
    
    mock_apprise_module = MagicMock()
    mock_apprise_module.Apprise.return_value = mock_apprise_obj
    
    with patch.dict(sys.modules, {'apprise': mock_apprise_module}):
        import importlib
        import agentu.middleware.notify
        importlib.reload(agentu.middleware.notify)
        
        middleware = agentu.middleware.notify.NotifyMiddleware(
            targets=["slack://tok/tok/chan", "mailto://a@b.com"],
            title="Task Complete"
        )
        
        assert mock_apprise_obj.add.call_count == 2
        
        context = CallContext(
            prompt="Tell me a joke",
            namespace="bot",
        )
        # Mock token count from CostTracker
        context.metadata["cost_tracker_input_tokens"] = 42
        
        # Test before hook -> passthrough
        before_ctx = await middleware.before(context)
        assert before_ctx == context
        
        # Test after hook -> triggers run_in_executor
        response = "Why did the chicken cross the road? To get to the other side."
        out_response = await middleware.after(context, response)
        
        assert out_response == response
        
        # Yield to event loop slightly so run_in_executor can finish in test
        await asyncio.sleep(0.01)
        
        assert mock_apprise_obj.notify.called
        kwargs = mock_apprise_obj.notify.call_args.kwargs
        assert kwargs["title"] == "Task Complete"
        
        body = kwargs["body"]
        assert "Agent (Namespace): bot" in body
        assert "Status: SUCCESS" in body
        assert "Input Tokens (Estimated): 42" in body
        assert "Why did the chicken cross the road?" in body

@pytest.mark.asyncio
async def test_notify_middleware_formatter_and_error():
    mock_apprise_obj = MagicMock()
    mock_apprise_obj.notify.return_value = True
    
    mock_apprise_module = MagicMock()
    mock_apprise_module.Apprise.return_value = mock_apprise_obj
    
    with patch.dict(sys.modules, {'apprise': mock_apprise_module}):
        import importlib
        import agentu.middleware.notify
        importlib.reload(agentu.middleware.notify)
        
        def custom_fmt(ctx, resp, err):
            if err:
                return f"CRASH: {err}"
            return f"SWEET: {resp}"
            
        middleware = agentu.middleware.notify.NotifyMiddleware(
            targets=["slack://test"],
            formatter=custom_fmt
        )
        
        context = CallContext(prompt="test", namespace="test_bot")
        
        # Test custom success formatter
        await middleware.after(context, "yay")
        await asyncio.sleep(0.01) # Yield to executor
        assert mock_apprise_obj.notify.call_args.kwargs["body"] == "SWEET: yay"
        
        # Test on_error parsing
        await middleware.on_error(context, ValueError("Bad prompt"))
        await asyncio.sleep(0.01)
        assert mock_apprise_obj.notify.call_args.kwargs["body"] == "CRASH: Bad prompt"
        assert mock_apprise_obj.notify.call_args.kwargs["title"] == "Agentu Execution Failed"
