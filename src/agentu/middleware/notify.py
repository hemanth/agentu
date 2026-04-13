import logging
from typing import List, Optional, Callable

try:
    import apprise
except ImportError:
    apprise = None

from .middleware import BaseMiddleware, CallContext

logger = logging.getLogger(__name__)

def default_formatter(context: CallContext, response: Optional[str], error: Optional[Exception]) -> str:
    """Default text formatter for notifications."""
    namespace = context.namespace
    elapsed = context.elapsed_ms
    input_tokens = context.metadata.get("cost_tracker_input_tokens")
    
    status = "SUCCESS" if error is None else f"FAILED: {error.__class__.__name__}"
    
    lines = [
        f"Agent (Namespace): {namespace}",
        f"Status: {status}",
        f"Elapsed Time: {elapsed:.0f}ms",
    ]
    
    if input_tokens is not None:
        lines.append(f"Input Tokens (Estimated): {input_tokens}")
        
    if error:
        lines.append(f"Error Details: {str(error)}")
    elif response:
        truncated_resp = response if len(response) < 500 else response[:497] + "..."
        lines.append("")
        lines.append("Final Response Snippet:")
        lines.append(truncated_resp)
        
    return "\n".join(lines)


class NotifyMiddleware(BaseMiddleware):
    """Sends a notification after an LLM call finishes using apprise.
    
    Requires the `[notify]` extra to be installed: `pip install agentu[notify]`
    """
    
    name = "notify"
    
    def __init__(self, targets: List[str], title: Optional[str] = None,
                 formatter: Optional[Callable[[CallContext, Optional[str], Optional[Exception]], str]] = None):
        """Initialize notify middleware.
        
        Args:
            targets: List of Apprise notification URLs (e.g. 'slack://...', 'mailto://...')
            title: Optional title for the notification
            formatter: Optional function to override the message formatting (`def fmt(ctx, resp, err) -> str:`)
        """
        if apprise is None:
            raise ImportError(
                "The 'apprise' library is required for NotifyMiddleware. "
                "Please install it with `pip install agentu[notify]`."
            )
            
        self.targets = targets
        self.title = title
        self.formatter = formatter or default_formatter
        
        self.apobj = apprise.Apprise()
        for target in targets:
            self.apobj.add(target)
            
    async def before(self, context: CallContext) -> CallContext:
        """Called before the LLM request. Just passes context through."""
        return context
        
    async def _send_notification(self, context: CallContext, response: Optional[str], error: Optional[Exception]):
        """Internal helper to format and send the notification."""
        body = self.formatter(context, response, error)
        title = self.title or ("Agentu Execution Complete" if not error else "Agentu Execution Failed")
        
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            
            def _send():
                return self.apobj.notify(body=body, title=title)

            success = await loop.run_in_executor(None, _send)
            
            if success:
                logger.info(f"[{context.namespace}] Notification sent successfully to {len(self.targets)} targets.")
            else:
                logger.warning(f"[{context.namespace}] Apprise returned False when sending notifications.")
            
        except Exception as e:
            logger.error(f"[{context.namespace}] Failed to send notification: {e}")

    async def after(self, context: CallContext, response: str) -> str:
        """Called after the LLM response. Sends the success notification non-blocking."""
        await self._send_notification(context, response, None)
        return response

    async def on_error(self, context: CallContext, error: Exception) -> None:
        """Called when an LLM call fails completely. Sends the failure notification non-blocking."""
        await self._send_notification(context, None, error)
