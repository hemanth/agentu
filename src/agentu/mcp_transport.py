"""MCP transport module for connecting to remote MCP servers with auth support."""
import requests
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransportType(Enum):
    """Supported transport types for MCP servers."""
    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"


@dataclass
class AuthConfig:
    """Authentication configuration for MCP servers."""
    type: str  # 'bearer', 'apikey', 'custom'
    headers: Dict[str, str]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AuthConfig':
        """Create AuthConfig from dictionary."""
        return AuthConfig(
            type=data.get('type', 'bearer'),
            headers=data.get('headers', {})
        )

    @staticmethod
    def bearer_token(token: str) -> 'AuthConfig':
        """Create bearer token auth config."""
        return AuthConfig(
            type='bearer',
            headers={'Authorization': f'Bearer {token}'}
        )

    @staticmethod
    def api_key(api_key: str, header_name: str = 'X-API-Key') -> 'AuthConfig':
        """Create API key auth config."""
        return AuthConfig(
            type='apikey',
            headers={header_name: api_key}
        )


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport_type: TransportType
    url: Optional[str] = None
    command: Optional[str] = None
    auth: Optional[AuthConfig] = None
    timeout: int = 30

    @staticmethod
    def from_dict(name: str, data: Dict[str, Any]) -> 'MCPServerConfig':
        """Create MCPServerConfig from dictionary."""
        transport_str = data.get('type', 'http')
        transport_type = TransportType(transport_str)

        auth = None
        if 'auth' in data:
            auth = AuthConfig.from_dict(data['auth'])

        return MCPServerConfig(
            name=name,
            transport_type=transport_type,
            url=data.get('url'),
            command=data.get('command'),
            auth=auth,
            timeout=data.get('timeout', 30)
        )


class MCPTransport:
    """Base transport class for MCP protocol."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session_id: Optional[str] = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers including auth if configured."""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        if self.config.auth:
            headers.update(self.config.auth.headers)

        return headers

    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send MCP JSON-RPC request. To be implemented by subclasses."""
        raise NotImplementedError

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        raise NotImplementedError

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        raise NotImplementedError

    def close(self):
        """Close the transport connection."""
        pass


class MCPHTTPTransport(MCPTransport):
    """HTTP-based transport for MCP servers."""

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self.request_id = 0

    def _next_request_id(self) -> int:
        """Get next request ID for JSON-RPC."""
        self.request_id += 1
        return self.request_id

    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send MCP JSON-RPC request over HTTP."""
        if not self.config.url:
            raise ValueError("URL is required for HTTP transport")

        request_payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_request_id()
        }

        if params:
            request_payload["params"] = params

        try:
            logger.info(f"Sending MCP request to {self.config.url}: {method}")
            response = requests.post(
                self.config.url,
                json=request_payload,
                headers=self._get_headers(),
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()

            if "error" in result:
                logger.error(f"MCP server error: {result['error']}")
                raise Exception(f"MCP server error: {result['error']}")

            return result.get("result", {})

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling MCP server: {str(e)}")
            raise

    def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP session."""
        return self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "agentu",
                "version": "0.1.0"
            }
        })

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        try:
            # Initialize session if not already done
            if not self.session_id:
                init_result = self.initialize()
                logger.info(f"Initialized MCP session: {init_result}")

            result = self.send_request("tools/list", {})
            return result.get("tools", [])

        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            raise

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        try:
            result = self.send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })

            # MCP tool responses can contain content array
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Return the text from the first content item
                    return content[0].get("text", result)

            return result

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            raise


class MCPSSETransport(MCPTransport):
    """Server-Sent Events transport for MCP servers.

    SSE transport uses a persistent connection to receive events from the server.
    Messages are sent via POST requests to a session-specific endpoint, and
    responses are received through the SSE stream.
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self.request_id = 0
        self.session_endpoint: Optional[str] = None
        self.sse_stream = None
        self.event_queue: Queue = Queue()
        self.listener_thread: Optional[threading.Thread] = None
        self.running = False
        self._initialized = False

    def _next_request_id(self) -> int:
        """Get next request ID for JSON-RPC."""
        self.request_id += 1
        return self.request_id

    def _listen_to_sse_events(self, response):
        """Background thread to listen to SSE events and queue them."""
        logger.info("[SSE] Listener thread started")
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not self.running:
                    logger.info("[SSE] Stopping listener (running=False)")
                    break

                if line:
                    self.event_queue.put(line)

        except Exception as e:
            logger.error(f"[SSE] Listener error: {e}")
            self.event_queue.put(f"ERROR: {e}")
        finally:
            logger.info("[SSE] Listener thread stopped")

    def _connect(self):
        """Establish SSE connection and get session endpoint."""
        if not self.config.url:
            raise ValueError("URL is required for SSE transport")

        logger.info(f"[SSE] Connecting to {self.config.url}")

        # Get headers without Content-Type for GET request
        headers = {}
        if self.config.auth:
            headers.update(self.config.auth.headers)

        # Connect to SSE endpoint
        self.sse_stream = requests.get(
            self.config.url,
            headers=headers,
            stream=True,
            timeout=300  # Long timeout for persistent connection
        )

        if self.sse_stream.status_code != 200:
            raise Exception(
                f"SSE connection failed: {self.sse_stream.status_code}"
            )

        logger.info(f"[SSE] Connected (Status: {self.sse_stream.status_code})")

        # Start listener thread
        self.running = True
        self.listener_thread = threading.Thread(
            target=self._listen_to_sse_events,
            args=(self.sse_stream,),
            daemon=True
        )
        self.listener_thread.start()

        # Wait for session endpoint from SSE stream
        logger.info("[SSE] Waiting for session endpoint...")
        timeout = time.time() + 10

        while time.time() < timeout:
            try:
                line = self.event_queue.get(timeout=1)
                if 'data: /sse/message' in line or 'data: /' in line:
                    self.session_endpoint = line.split('data: ')[1].strip()
                    logger.info(f"[SSE] Got session endpoint: ...{self.session_endpoint[-40:]}")
                    break
            except Empty:
                continue

        if not self.session_endpoint:
            raise Exception("Failed to get SSE session endpoint")

    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send MCP JSON-RPC request over SSE.

        Sends a POST request to the session endpoint and waits for the
        response to arrive via the SSE stream.
        """
        # Connect if not already connected
        if not self.session_endpoint:
            self._connect()

        req_id = self._next_request_id()

        # Build JSON-RPC message
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id
        }

        if params is not None:
            message["params"] = params

        # Send message to session endpoint
        message_url = f"{self.config.url.rstrip('/sse')}{self.session_endpoint}"

        headers = self._get_headers()

        logger.info(f"[SSE] Sending {method} (id:{req_id})")

        try:
            response = requests.post(
                message_url,
                json=message,
                headers=headers,
                timeout=self.config.timeout
            )

            if response.status_code != 202:
                logger.error(
                    f"[SSE] Unexpected response: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"SSE message not accepted: {response.status_code}"
                )

            logger.debug(f"[SSE] Message accepted, waiting for response (id:{req_id})")

        except requests.exceptions.RequestException as e:
            logger.error(f"[SSE] Error sending message: {e}")
            raise

        # Wait for response from SSE stream
        return self._wait_for_response(req_id)

    def _wait_for_response(self, expected_id: int, timeout_sec: int = 30) -> Dict[str, Any]:
        """Wait for a JSON-RPC response with the expected ID from the SSE stream."""
        logger.debug(f"[SSE] Waiting for response (id:{expected_id})")

        timeout = time.time() + timeout_sec
        response_data = None

        while time.time() < timeout:
            try:
                line = self.event_queue.get(timeout=1)

                # Skip non-data lines
                if not line.startswith('data: {'):
                    continue

                # Parse JSON response
                try:
                    data_str = line.split('data: ', 1)[1]
                    data = json.loads(data_str)

                    # Check if this is our response
                    if data.get('id') == expected_id:
                        logger.debug(f"[SSE] Received response (id:{expected_id})")

                        # Check for errors
                        if 'error' in data:
                            logger.error(f"[SSE] Server error: {data['error']}")
                            raise Exception(f"MCP server error: {data['error']}")

                        response_data = data.get('result', {})
                        break

                except json.JSONDecodeError as e:
                    logger.warning(f"[SSE] Failed to parse JSON: {e}")
                    continue

            except Empty:
                continue

        if response_data is None:
            raise TimeoutError(
                f"Timeout waiting for response (id:{expected_id}) after {timeout_sec}s"
            )

        return response_data

    def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP session."""
        if self._initialized:
            return {}

        result = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "agentu",
                "version": "0.1.0"
            }
        })

        self._initialized = True
        logger.info(f"[SSE] Initialized MCP session")
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        try:
            # Initialize if not already done
            if not self._initialized:
                self.initialize()

            result = self.send_request("tools/list")
            tools = result.get("tools", [])

            logger.info(f"[SSE] Listed {len(tools)} tools")
            return tools

        except Exception as e:
            logger.error(f"[SSE] Error listing tools: {e}")
            raise

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        try:
            result = self.send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })

            # MCP tool responses can contain content array
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Return the text from the first content item
                    first_content = content[0]
                    if isinstance(first_content, dict):
                        return first_content.get("text", result)
                    return first_content

            return result

        except Exception as e:
            logger.error(f"[SSE] Error calling tool {tool_name}: {e}")
            raise

    def close(self):
        """Close the SSE connection."""
        logger.info("[SSE] Closing connection")
        self.running = False

        if self.sse_stream:
            try:
                self.sse_stream.close()
            except:
                pass

        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2)

        logger.info("[SSE] Connection closed")


def create_transport(config: MCPServerConfig) -> MCPTransport:
    """Factory function to create appropriate transport based on config."""
    if config.transport_type == TransportType.HTTP:
        return MCPHTTPTransport(config)
    elif config.transport_type == TransportType.SSE:
        return MCPSSETransport(config)
    elif config.transport_type == TransportType.STDIO:
        raise NotImplementedError("STDIO transport not yet implemented")
    else:
        raise ValueError(f"Unsupported transport type: {config.transport_type}")
