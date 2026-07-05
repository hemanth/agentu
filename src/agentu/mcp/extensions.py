"""MCP protocol extensions: Streamable HTTP, elicitation, OAuth 2.1, tasks.

These extensions align agentu with the MCP v2 spec (2026-07-28 RC):
- Streamable HTTP: stateless transport without session handshake
- Elicitation: servers can ask users for input mid-call
- OAuth 2.1 + CIMD: client auth via HTTPS metadata URL
- Tasks: call-now/fetch-later for long-running operations
"""

import aiohttp
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ── Streamable HTTP Transport ─────────────────────────────────────

class StreamableHTTPTransport:
    """MCP Streamable HTTP transport (v2 spec compatible).

    Unlike the legacy HTTP transport, this implementation:
    - Does NOT require an initialize handshake or session ID
    - Sends JSON-RPC requests as HTTP POST to the server URL
    - Supports streaming responses via SSE on the same connection
    - Falls back to JSON response for non-streaming servers
    - Compatible with the 2026-07-28 stateless core RC

    Usage:
        transport = StreamableHTTPTransport(url="http://localhost:3000/mcp")
        tools = await transport.list_tools()
        result = await transport.call_tool("search", {"query": "test"})
        await transport.close()
    """

    def __init__(
        self,
        url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ):
        self.url = url.rstrip("/")
        self.auth_headers = auth_headers or {}
        self.timeout = timeout
        self.request_id = 0
        self._session: Optional[aiohttp.ClientSession] = None

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request via HTTP POST.

        Supports both JSON and SSE responses (checks Content-Type).
        """
        session = await self._get_session()
        request_id = self._next_id()

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.auth_headers,
        }

        try:
            async with session.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                content_type = response.headers.get("Content-Type", "")

                if "text/event-stream" in content_type:
                    # SSE streaming response — collect events
                    return await self._read_sse_response(response, request_id)
                else:
                    # Standard JSON response
                    data = await response.json()
                    if "error" in data:
                        raise Exception(
                            f"MCP error: {data['error'].get('message', 'Unknown')}"
                        )
                    return data.get("result", {})

        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP request '{method}' timed out after {self.timeout}s")

    async def _read_sse_response(
        self,
        response: aiohttp.ClientResponse,
        request_id: int,
    ) -> Dict[str, Any]:
        """Read SSE events from a streaming response until we get our result."""
        result = None
        async for line_bytes in response.content:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Check if this is our response
            if event.get("id") == request_id:
                if "error" in event:
                    raise Exception(
                        f"MCP error: {event['error'].get('message', 'Unknown')}"
                    )
                result = event.get("result", {})
                break

        return result or {}

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        result = await self.send_request("tools/list")
        return result.get("tools", [])

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Call a tool on the server."""
        result = await self.send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

        # Extract content from MCP response format
        content = result.get("content", [])
        if content and isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(text_parts) if text_parts else result

        return result

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ── Elicitation Support ───────────────────────────────────────────

@dataclass
class ElicitationRequest:
    """A request from the server for user input.

    When an MCP server needs user input during a tool call, it sends
    an elicitation request. The client must collect the response and
    send it back.
    """
    request_id: str
    message: str
    schema: Optional[Dict[str, Any]] = None  # JSON Schema for expected input
    url: Optional[str] = None  # URL mode: redirect user to this URL


@dataclass
class ElicitationResponse:
    """User's response to an elicitation request."""
    request_id: str
    action: str = "submit"  # "submit", "cancel", "dismiss"
    data: Optional[Dict[str, Any]] = None


# Type for elicitation handler callback
ElicitationHandler = Callable[[ElicitationRequest], Awaitable[ElicitationResponse]]


class ElicitationMixin:
    """Mixin that adds elicitation support to a transport.

    Elicitation allows MCP servers to ask the user for input mid-call.
    Supports both form mode (structured input) and URL mode (redirect).

    Usage:
        async def handle_elicitation(req: ElicitationRequest) -> ElicitationResponse:
            user_input = await get_user_input(req.message)
            return ElicitationResponse(request_id=req.request_id, data=user_input)

        transport.set_elicitation_handler(handle_elicitation)
    """

    _elicitation_handler: Optional[ElicitationHandler] = None

    def set_elicitation_handler(self, handler: ElicitationHandler):
        """Set the callback for handling elicitation requests."""
        self._elicitation_handler = handler

    async def _handle_elicitation(
        self,
        elicitation_data: Dict[str, Any],
    ) -> Optional[ElicitationResponse]:
        """Process an elicitation request from the server."""
        if not self._elicitation_handler:
            logger.warning("Elicitation requested but no handler configured")
            return ElicitationResponse(
                request_id=elicitation_data.get("requestId", ""),
                action="dismiss",
            )

        request = ElicitationRequest(
            request_id=elicitation_data.get("requestId", ""),
            message=elicitation_data.get("message", ""),
            schema=elicitation_data.get("schema"),
            url=elicitation_data.get("url"),
        )

        return await self._elicitation_handler(request)


# ── OAuth 2.1 + CIMD Auth ────────────────────────────────────────

@dataclass
class OAuthConfig:
    """OAuth 2.1 + CIMD authentication configuration.

    CIMD (Client ID Metadata Document) is the recommended MCP auth
    pattern: the client_id is an HTTPS URL pointing to a metadata document.

    Usage:
        auth = OAuthConfig(
            client_id="https://myapp.example.com/.well-known/oauth-client",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
    """
    client_id: str
    authorization_endpoint: str
    token_endpoint: str
    client_secret: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    redirect_uri: str = "http://localhost:8080/callback"

    # CIMD: client_id as HTTPS metadata URL
    @property
    def is_cimd(self) -> bool:
        """Whether this uses CIMD (client_id is an HTTPS URL)."""
        return self.client_id.startswith("https://")


class OAuthTokenManager:
    """Manages OAuth 2.1 access tokens with automatic refresh.

    Handles the authorization code flow and token refresh for
    MCP server authentication.
    """

    def __init__(self, config: OAuthConfig):
        self.config = config
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: float = 0

    @property
    def has_valid_token(self) -> bool:
        return self._access_token is not None and time.time() < self._expires_at

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers with the current token."""
        if not self._access_token:
            return {}
        return {"Authorization": f"Bearer {self._access_token}"}

    async def exchange_code(self, code: str) -> Dict[str, Any]:
        """Exchange an authorization code for tokens."""
        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.config.redirect_uri,
                "client_id": self.config.client_id,
            }
            if self.config.client_secret:
                data["client_secret"] = self.config.client_secret

            async with session.post(
                self.config.token_endpoint,
                data=data,
            ) as resp:
                result = await resp.json()
                self._access_token = result.get("access_token")
                self._refresh_token = result.get("refresh_token")
                expires_in = result.get("expires_in", 3600)
                self._expires_at = time.time() + expires_in
                return result

    async def refresh(self) -> Dict[str, Any]:
        """Refresh the access token."""
        if not self._refresh_token:
            raise ValueError("No refresh token available")

        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self.config.client_id,
            }
            if self.config.client_secret:
                data["client_secret"] = self.config.client_secret

            async with session.post(
                self.config.token_endpoint,
                data=data,
            ) as resp:
                result = await resp.json()
                self._access_token = result.get("access_token")
                if "refresh_token" in result:
                    self._refresh_token = result["refresh_token"]
                expires_in = result.get("expires_in", 3600)
                self._expires_at = time.time() + expires_in
                return result

    def get_authorization_url(self, state: str = "") -> str:
        """Generate the authorization URL for the user to visit."""
        import urllib.parse
        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "state": state,
        }
        if self.config.scopes:
            params["scope"] = " ".join(self.config.scopes)
        return f"{self.config.authorization_endpoint}?{urllib.parse.urlencode(params)}"


# ── MCP Tasks (Call-Now/Fetch-Later) ──────────────────────────────

class TaskState(Enum):
    """State of an MCP task."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MCPTask:
    """Represents a long-running MCP task.

    Tasks follow the call-now/fetch-later pattern:
    1. Client initiates a tool call that returns a task ID
    2. Client polls for status until complete
    3. Client fetches the result

    This matches the A2A task lifecycle pattern.
    """
    task_id: str
    state: TaskState = TaskState.SUBMITTED
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskManager:
    """Manages MCP task lifecycle for long-running operations.

    Usage:
        manager = TaskManager(transport)
        task = await manager.submit("long_running_tool", {"input": "data"})
        result = await manager.wait_for_completion(task.task_id, poll_interval=2.0)
    """

    def __init__(self, transport):
        """Initialize with an MCP transport that supports send_request."""
        self.transport = transport
        self._tasks: Dict[str, MCPTask] = {}

    async def submit(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MCPTask:
        """Submit a task for async execution."""
        result = await self.transport.send_request(
            "tasks/create",
            {"tool": tool_name, "arguments": arguments},
        )
        task = MCPTask(
            task_id=result.get("taskId", ""),
            state=TaskState(result.get("state", "submitted")),
        )
        self._tasks[task.task_id] = task
        return task

    async def get_status(self, task_id: str) -> MCPTask:
        """Get the current status of a task."""
        result = await self.transport.send_request(
            "tasks/get",
            {"taskId": task_id},
        )
        task = self._tasks.get(task_id, MCPTask(task_id=task_id))
        task.state = TaskState(result.get("state", "submitted"))
        task.result = result.get("result")
        task.error = result.get("error")
        task.progress = result.get("progress")
        self._tasks[task_id] = task
        return task

    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> MCPTask:
        """Poll a task until it completes or times out."""
        start = time.time()
        while time.time() - start < timeout:
            task = await self.get_status(task_id)
            if task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
                return task
            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

    async def cancel(self, task_id: str) -> MCPTask:
        """Cancel a running task."""
        result = await self.transport.send_request(
            "tasks/cancel",
            {"taskId": task_id},
        )
        task = self._tasks.get(task_id, MCPTask(task_id=task_id))
        task.state = TaskState.CANCELLED
        return task
