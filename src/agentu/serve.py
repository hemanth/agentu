"""Serve agents as REST API services using FastAPI."""

import asyncio
from typing import Dict, Any, Optional, List, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .agent import Agent


class ExecuteRequest(BaseModel):
    """Request model for tool execution."""
    tool_name: str
    parameters: Dict[str, Any]


class ProcessRequest(BaseModel):
    """Request model for natural language processing."""
    input: str


class ToolInfo(BaseModel):
    """Tool information model."""
    name: str
    description: str
    parameters: Dict[str, Any]


class AgentServer:
    """Server wrapper for Agent."""

    def __init__(
        self,
        agent: Agent,
        title: str = "agentu API",
        version: str = "0.1.0",
        enable_cors: bool = False,
        cors_origins: Optional[List[str]] = None,
        cors_methods: Optional[List[str]] = None,
        cors_headers: Optional[List[str]] = None,
        cors_credentials: bool = False
    ):
        """Initialize agent server.

        Args:
            agent: Agent instance to serve
            title: API title
            version: API version
            enable_cors: Enable CORS middleware (default: False)
            cors_origins: Allowed origins for CORS (default: ["*"])
            cors_methods: Allowed HTTP methods (default: ["*"])
            cors_headers: Allowed headers (default: ["*"])
            cors_credentials: Allow credentials (default: False)
        """
        self.agent = agent
        self.app = FastAPI(title=title, version=version)

        # Setup CORS if enabled
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins or ["*"],
                allow_credentials=cors_credentials,
                allow_methods=cors_methods or ["*"],
                allow_headers=cors_headers or ["*"],
            )

        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", tags=["info"])
        async def root():
            """Root endpoint with API information."""
            return {
                "name": self.agent.name,
                "model": self.agent.model,
                "tools": len(self.agent.tools),
                "memory_enabled": self.agent.memory_enabled
            }

        @self.app.get("/health", tags=["info"])
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "agent": self.agent.name}

        @self.app.get("/tools", response_model=List[ToolInfo], tags=["tools"])
        async def list_tools():
            """List all available tools."""
            return [
                ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters
                )
                for tool in self.agent.tools
            ]

        @self.app.post("/execute", tags=["execution"])
        async def execute_tool(request: ExecuteRequest):
            """Execute a tool directly with given parameters.

            Args:
                request: ExecuteRequest with tool_name and parameters

            Returns:
                Tool execution result
            """
            try:
                result = await self.agent.call(
                    request.tool_name,
                    request.parameters
                )
                return {
                    "success": True,
                    "tool": request.tool_name,
                    "result": result
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/process", tags=["execution"])
        async def process_input(request: ProcessRequest):
            """Process natural language input and execute appropriate tool.

            Args:
                request: ProcessRequest with natural language input

            Returns:
                Processing result including tool used and output
            """
            try:
                result = await self.agent.infer(request.input)
                return {
                    "success": True,
                    "input": request.input,
                    **result
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/memory/stats", tags=["memory"])
        async def memory_stats():
            """Get memory statistics."""
            if not self.agent.memory_enabled:
                raise HTTPException(status_code=400, detail="Memory not enabled")
            return self.agent.get_memory_stats()

        @self.app.post("/memory/remember", tags=["memory"])
        async def remember(
            content: str,
            memory_type: str = "fact",
            importance: float = 0.5,
            store_long_term: bool = False
        ):
            """Store information in agent memory."""
            if not self.agent.memory_enabled:
                raise HTTPException(status_code=400, detail="Memory not enabled")

            self.agent.remember(
                content=content,
                memory_type=memory_type,
                importance=importance,
                store_long_term=store_long_term
            )
            return {"success": True, "message": "Memory stored"}

        @self.app.get("/memory/recall", tags=["memory"])
        async def recall(
            query: Optional[str] = None,
            memory_type: Optional[str] = None,
            limit: int = 5
        ):
            """Recall memories from agent."""
            if not self.agent.memory_enabled:
                raise HTTPException(status_code=400, detail="Memory not enabled")

            memories = self.agent.recall(query, memory_type, limit)
            return {
                "success": True,
                "count": len(memories),
                "memories": [
                    {
                        "content": m.content,
                        "type": m.memory_type,
                        "importance": m.importance,
                        "timestamp": m.timestamp.isoformat() if hasattr(m, 'timestamp') and hasattr(m.timestamp, 'isoformat') else None
                    }
                    for m in memories
                ]
            }
        
        @self.app.get("/dashboard", include_in_schema=False, tags=["dashboard"])
        async def dashboard():
            """Serve observability dashboard."""
            from fastapi.responses import HTMLResponse
            from pathlib import Path
            
            dashboard_path = Path(__file__).parent / "static" / "dashboard.html"
            if not dashboard_path.exists():
                raise HTTPException(status_code=404, detail="Dashboard not found")
            
            with open(dashboard_path, 'r') as f:
                html_content = f.read()
            
            return HTMLResponse(content=html_content)
        
        @self.app.get("/api/metrics", tags=["dashboard"])
        async def get_metrics():
            """Get agent metrics and recent events for dashboard."""
            return {
                "metrics": self.agent.observer.get_metrics(),
                "events": self.agent.observer.get_events(limit=20)
            }
        
        @self.app.get("/playground", include_in_schema=False, tags=["playground"])
        async def playground():
            """Serve interactive playground."""
            from fastapi.responses import HTMLResponse
            from pathlib import Path
            
            playground_path = Path(__file__).parent / "static" / "playground.html"
            if not playground_path.exists():
                raise HTTPException(status_code=404, detail="Playground not found")
            
            with open(playground_path, 'r') as f:
                html_content = f.read()
            
            return HTMLResponse(content=html_content)
        
        @self.app.get("/api/examples", tags=["playground"])
        async def list_examples():
            """List all available examples."""
            from .sandbox import ExampleRunner
            
            runner = ExampleRunner()
            examples = runner.discover_examples()
            
            return {
                "examples": [
                    {
                        "name": ex.name,
                        "title": ex.title,
                        "description": ex.description,
                        "category": ex.category,
                        "lesson_number": ex.lesson_number,
                        "learning_objectives": ex.learning_objectives
                    }
                    for ex in examples
                ]
            }
        
        @self.app.get("/api/examples/{name}", tags=["playground"])
        async def get_example(name: str):
            """Get example details including code."""
            from .sandbox import ExampleRunner
            
            runner = ExampleRunner()
            examples = runner.discover_examples()
            
            example = next((ex for ex in examples if ex.name == name), None)
            if not example:
                raise HTTPException(status_code=404, detail=f"Example not found: {name}")
            
            return {
                "name": example.name,
                "title": example.title,
                "description": example.description,
                "category": example.category,
                "code": example.code
            }
        
        @self.app.post("/api/examples/{name}/run", tags=["playground"])
        async def run_example(name: str):
            """Execute example and return output."""
            from .sandbox import ExampleRunner
            
            runner = ExampleRunner()
            result = runner.run(name, timeout=30)
            
            return result
        
        @self.app.get("/api/examples/{name}/steps", tags=["playground"])
        async def get_example_steps(name: str):
            """Get step-by-step guide for example."""
            from .sandbox import ExampleRunner
            
            runner = ExampleRunner()
            examples = runner.discover_examples()
            
            example = next((ex for ex in examples if ex.name == name), None)
            if not example:
                raise HTTPException(status_code=404, detail=f"Example not found: {name}")
            
            steps = runner.get_steps(example.code)
            
            return {"steps": steps}

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the server.

        Args:
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional uvicorn.run arguments
        """
        uvicorn.run(self.app, host=host, port=port, **kwargs)


def serve(
    agent: Agent,
    host: str = "0.0.0.0",
    port: int = 8000,
    title: str = "agentu API",
    version: str = "0.1.0",
    enable_cors: bool = False,
    cors_origins: Optional[List[str]] = None,
    cors_methods: Optional[List[str]] = None,
    cors_headers: Optional[List[str]] = None,
    cors_credentials: bool = False,
    **kwargs
):
    """Serve an agent as a REST API service.

    Args:
        agent: Agent instance to serve
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        title: API title
        version: API version
        enable_cors: Enable CORS middleware (default: False)
        cors_origins: Allowed origins for CORS (default: ["*"])
        cors_methods: Allowed HTTP methods (default: ["*"])
        cors_headers: Allowed headers (default: ["*"])
        cors_credentials: Allow credentials (default: False)
        **kwargs: Additional uvicorn.run arguments

    Examples:
        Basic usage:
        >>> from agentu import Agent, serve
        >>> agent = Agent("assistant", model="llama3")
        >>> serve(agent, port=8000)

        With CORS enabled:
        >>> serve(agent, port=8000, enable_cors=True)

        With specific CORS origins:
        >>> serve(
        ...     agent,
        ...     port=8000,
        ...     enable_cors=True,
        ...     cors_origins=["http://localhost:3000", "https://app.example.com"],
        ...     cors_credentials=True
        ... )
    """
    server = AgentServer(
        agent,
        title=title,
        version=version,
        enable_cors=enable_cors,
        cors_origins=cors_origins,
        cors_methods=cors_methods,
        cors_headers=cors_headers,
        cors_credentials=cors_credentials
    )
    server.run(host=host, port=port, **kwargs)
