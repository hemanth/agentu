"""Tests for agent serving functionality."""

import pytest
from fastapi.testclient import TestClient
from agentu import Agent, Tool, AgentServer, serve


def calculator(x: float, y: float, operation: str) -> float:
    """Simple calculator function."""
    operations = {
        "add": x + y,
        "subtract": x - y,
        "multiply": x * y,
        "divide": x / y if y != 0 else 0
    }
    return operations.get(operation, 0)


@pytest.fixture
def agent():
    """Create a test agent with calculator tool."""
    agent = Agent("test_agent", model="qwen3:latest", enable_memory=True)
    calc_tool = Tool(
        name="calculator",
        description="Perform basic arithmetic operations",
        function=calculator,
        parameters={
            "x": "float: First number",
            "y": "float: Second number",
            "operation": "str: Operation (add, subtract, multiply, divide)"
        }
    )
    agent.with_tools([calc_tool])
    return agent


@pytest.fixture
def server(agent):
    """Create agent server."""
    return AgentServer(agent, title="Test API")


@pytest.fixture
def client(server):
    """Create test client."""
    return TestClient(server.app)


def test_server_creation(server, agent):
    """Test server is created successfully."""
    assert server.agent == agent
    assert server.app is not None


def test_root_endpoint(client):
    """Test root endpoint returns agent info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_agent"
    assert data["model"] == "qwen3:latest"
    assert data["tools"] == 1
    assert data["memory_enabled"] is True


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["agent"] == "test_agent"


def test_list_tools(client):
    """Test listing tools endpoint."""
    response = client.get("/tools")
    assert response.status_code == 200
    tools = response.json()
    assert len(tools) == 1
    assert tools[0]["name"] == "calculator"
    assert tools[0]["description"] == "Perform basic arithmetic operations"
    assert "parameters" in tools[0]


@pytest.mark.asyncio
async def test_execute_tool(client):
    """Test direct tool execution endpoint."""
    response = client.post(
        "/execute",
        json={
            "tool_name": "calculator",
            "parameters": {"x": 10, "y": 5, "operation": "add"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["tool"] == "calculator"
    assert data["result"] == 15


@pytest.mark.asyncio
async def test_execute_tool_multiply(client):
    """Test tool execution with multiplication."""
    response = client.post(
        "/execute",
        json={
            "tool_name": "calculator",
            "parameters": {"x": 7, "y": 6, "operation": "multiply"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"] == 42


@pytest.mark.asyncio
async def test_execute_nonexistent_tool(client):
    """Test executing a non-existent tool returns 404."""
    response = client.post(
        "/execute",
        json={
            "tool_name": "nonexistent",
            "parameters": {}
        }
    )
    assert response.status_code == 404


def test_memory_stats(client):
    """Test memory statistics endpoint."""
    response = client.get("/memory/stats")
    assert response.status_code == 200
    stats = response.json()
    assert stats["memory_enabled"] is True
    assert "short_term_size" in stats
    assert "long_term_size" in stats


def test_remember(client):
    """Test storing memory endpoint."""
    response = client.post(
        "/memory/remember",
        params={
            "content": "Test memory content",
            "memory_type": "fact",
            "importance": 0.8
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Memory stored"


def test_recall(client):
    """Test recalling memories endpoint."""
    # First store a memory
    client.post(
        "/memory/remember",
        params={
            "content": "Python is a programming language",
            "memory_type": "fact",
            "importance": 0.9
        }
    )

    # Then recall it
    response = client.get("/memory/recall", params={"limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "memories" in data
    assert data["count"] >= 0


def test_recall_with_query(client):
    """Test recalling memories with query."""
    # Store a memory
    client.post(
        "/memory/remember",
        params={
            "content": "FastAPI is a web framework",
            "memory_type": "fact",
            "importance": 0.8
        }
    )

    # Recall with query
    response = client.get(
        "/memory/recall",
        params={"query": "FastAPI", "limit": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.skip(reason="Requires OpenAI-compatible inference endpoint")
@pytest.mark.asyncio
async def test_process_input(client):
    """Test natural language processing endpoint."""
    response = client.post(
        "/process",
        json={"input": "Calculate 10 plus 5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "tool_used" in data
    assert "result" in data


def test_cors_disabled_by_default(agent):
    """Test that CORS is disabled by default."""
    server = AgentServer(agent)
    client = TestClient(server.app)
    
    response = client.options("/tools")
    # Should not have CORS headers
    assert "access-control-allow-origin" not in response.headers


def test_cors_enabled(agent):
    """Test CORS with enable_cors=True."""
    server = AgentServer(agent, enable_cors=True)
    client = TestClient(server.app)
    
    # Make OPTIONS request (preflight)
    response = client.options("/tools", headers={"Origin": "http://example.com"})
    
    # Should have CORS headers
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"


def test_cors_specific_origins(agent):
    """Test CORS with specific origins."""
    allowed_origins = ["http://localhost:3000", "https://app.example.com"]
    server = AgentServer(
        agent,
        enable_cors=True,
        cors_origins=allowed_origins
    )
    client = TestClient(server.app)
    
    # Test allowed origin
    response = client.options(
        "/tools",
        headers={"Origin": "http://localhost:3000"}
    )
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_credentials(agent):
    """Test CORS with credentials enabled."""
    server = AgentServer(
        agent,
        enable_cors=True,
        cors_origins=["http://localhost:3000"],
        cors_credentials=True
    )
    client = TestClient(server.app)
    
    response = client.options(
        "/tools",
        headers={"Origin": "http://localhost:3000"}
    )
    
    assert "access-control-allow-credentials" in response.headers
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_on_actual_request(agent):
    """Test CORS headers are present on actual API requests."""
    server = AgentServer(agent, enable_cors=True)
    client = TestClient(server.app)
    
    # Make actual GET request with Origin header
    response = client.get("/tools", headers={"Origin": "http://example.com"})
    
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
