import requests
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

from .tools import Tool
from .mcp_config import load_mcp_servers
from .mcp_tool import MCPToolManager
from .memory import Memory
from .workflow import Step

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ollama_models(api_base: str = "http://localhost:11434") -> List[str]:
    """Get list of available Ollama models.

    Args:
        api_base: Base URL for Ollama API

    Returns:
        List of model names, or empty list if unable to fetch
    """
    try:
        response = requests.get(f"{api_base.rstrip('/')}/api/tags", timeout=2)
        response.raise_for_status()
        models_data = response.json()
        models = [model["name"] for model in models_data.get("models", [])]
        return models
    except Exception as e:
        logger.warning(f"Unable to fetch Ollama models: {e}")
        return []


def get_default_model(api_base: str = "http://localhost:11434") -> str:
    """Get the default model to use (first available from Ollama).

    Args:
        api_base: Base URL for Ollama API

    Returns:
        Model name (first available model, or "qwen3:latest" as fallback)
    """
    models = get_ollama_models(api_base)
    if models:
        logger.info(f"Available Ollama models: {models}")
        logger.info(f"Using default model: {models[0]}")
        return models[0]
    logger.warning("No Ollama models found, using 'qwen3:latest' as fallback")
    return "qwen3:latest"


class Agent:
    def __init__(self, name: str, model: Optional[str] = None, temperature: float = 0.7,
                 mcp_config_path: Optional[str] = None, load_mcp_tools: bool = False,
                 enable_memory: bool = True, memory_path: Optional[str] = None,
                 short_term_size: int = 10, use_sqlite: bool = True,
                 priority: int = 5, api_base: str = "http://localhost:11434/v1",
                 api_key: Optional[str] = None):
        """Initialize an Agent.

        Args:
            name: Name of the agent
            model: Model name to use (default: auto-detect from Ollama, fallback to qwen3:latest)
            temperature: Temperature for model generation (default: 0.7)
            mcp_config_path: Optional path to MCP configuration file
            load_mcp_tools: Whether to automatically load tools from MCP servers (default: False)
            enable_memory: Whether to enable memory system (default: True)
            memory_path: Path for persistent memory storage (default: None)
            short_term_size: Size of short-term memory buffer (default: 10)
            use_sqlite: If True, use SQLite database for memory; otherwise use JSON (default: True)
            priority: Agent priority for task assignment (default: 5)
            api_base: Base URL for OpenAI-compatible API (default: http://localhost:11434/v1 for Ollama)
            api_key: Optional API key for authentication
        """
        self.name = name
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key

        # Auto-detect model if not specified
        if model is None:
            # Extract base URL without /v1 suffix for Ollama API
            ollama_base = self.api_base.replace('/v1', '')
            self.model = get_default_model(ollama_base)
        else:
            self.model = model

        self.temperature = temperature
        self.tools: List[Tool] = []
        self.context = ""
        self.conversation_history = []
        self.mcp_manager = MCPToolManager()

        # Initialize memory system
        self.memory_enabled = enable_memory
        self.memory = Memory(
            short_term_size=short_term_size,
            storage_path=memory_path,
            use_sqlite=use_sqlite
        ) if enable_memory else None

        # Orchestration attributes
        self.priority = priority

        # Load MCP tools if requested
        if load_mcp_tools and mcp_config_path:
            self.with_mcp([mcp_config_path])
        
    def _add_tool_internal(self, tool: Union[Tool, Callable]) -> None:
        """Internal method to add a single tool."""
        if isinstance(tool, Tool):
            self.tools.append(tool)
            logger.info(f"Added tool: {tool.name} to agent {self.name}")
        elif callable(tool):
            tool_obj = Tool(tool)
            self.tools.append(tool_obj)
            logger.info(f"Added tool: {tool_obj.name} to agent {self.name}")
        else:
            raise TypeError(f"Expected Tool or callable, got {type(tool)}")

    def with_tools(self, tools: List[Union[Tool, Callable]]) -> 'Agent':
        """Add tools and return self for chaining.

        Args:
            tools: List of Tool objects or callable functions (auto-wrapped)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("MyAgent").with_tools([my_func])  # Single tool
            >>> agent = Agent("MyAgent").with_tools([func1, func2])  # Multiple tools
        """
        for tool in tools:
            self._add_tool_internal(tool)
        return self

    def with_mcp(self, servers: List[Union[str, Dict[str, Any]]]) -> 'Agent':
        """Connect to MCP servers and load their tools (chainable).

        Args:
            servers: List of MCP server configurations. Each item can be:
                - String URL: "http://localhost:3000"
                - Dict with url and headers: {"url": "...", "headers": {...}}
                - Config file path: "~/.agentu/mcp_config.json"

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_mcp([
            ...     "http://localhost:3000",
            ...     {"url": "https://api.com/mcp", "headers": {"Auth": "Bearer xyz"}}
            ... ])
        """
        from .mcp_config import load_mcp_servers
        from .mcp_transport import MCPServerConfig

        for server in servers:
            try:
                # Handle config file path
                if isinstance(server, str) and server.endswith('.json'):
                    server_configs = load_mcp_servers(server)
                    for server_name, server_config in server_configs.items():
                        adapter = self.mcp_manager.add_server(server_config)
                        tools = adapter.load_tools()
                        for tool in tools:
                            self._add_tool_internal(tool)
                        logger.info(f"Loaded {len(tools)} tools from MCP server: {server_name}")

                # Handle URL string
                elif isinstance(server, str):
                    from .mcp_transport import TransportType
                    config = MCPServerConfig(
                        name=f"mcp_{len(self.mcp_manager.adapters)}",
                        transport_type=TransportType.HTTP,
                        url=server
                    )
                    adapter = self.mcp_manager.add_server(config)
                    tools = adapter.load_tools()
                    for tool in tools:
                        self._add_tool_internal(tool)
                    logger.info(f"Loaded {len(tools)} tools from MCP server: {server}")

                # Handle dict with url and headers
                elif isinstance(server, dict):
                    from .mcp_transport import TransportType, AuthConfig
                    url = server.get('url')
                    if not url:
                        raise ValueError("MCP server dict must contain 'url' key")

                    auth = None
                    if 'headers' in server:
                        auth = AuthConfig(
                            type="custom",
                            headers=server.get('headers', {})
                        )

                    config = MCPServerConfig(
                        name=server.get('name', f"mcp_{len(self.mcp_manager.adapters)}"),
                        transport_type=TransportType.HTTP,
                        url=url,
                        auth=auth
                    )
                    adapter = self.mcp_manager.add_server(config)
                    tools = adapter.load_tools()
                    for tool in tools:
                        self._add_tool_internal(tool)
                    logger.info(f"Loaded {len(tools)} tools from MCP server: {url}")

                else:
                    raise TypeError(f"Invalid MCP server type: {type(server)}")

            except Exception as e:
                logger.error(f"Error connecting to MCP server {server}: {str(e)}")
                raise

        return self

    def close_mcp_connections(self):
        """Close all MCP server connections."""
        self.mcp_manager.close_all()

    def __call__(self, task: Union[str, Callable]) -> Step:
        """Make agent callable to create workflow steps.

        Args:
            task: Task string or lambda function

        Returns:
            Step instance for workflow composition

        Example:
            >>> workflow = researcher("Find trends") >> analyst("Analyze")
            >>> result = await workflow.run()
        """
        return Step(self, task)

    def remember(self, content: str, memory_type: str = 'conversation',
                metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5,
                store_long_term: bool = False):
        """Store information in memory.

        Args:
            content: The content to remember
            memory_type: Type of memory ('conversation', 'fact', 'task', 'observation')
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            store_long_term: If True, store directly in long-term memory
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.remember(content, memory_type, metadata, importance, store_long_term)

    def recall(self, query: Optional[str] = None, memory_type: Optional[str] = None,
              limit: int = 5):
        """Recall memories.

        Args:
            query: Search query (if None, returns recent memories)
            memory_type: Filter by memory type
            limit: Maximum number of results

        Returns:
            List of MemoryEntry objects
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return []

        return self.memory.recall(query, memory_type, limit)

    def get_memory_context(self, max_entries: int = 5) -> str:
        """Get formatted context from memories.

        Args:
            max_entries: Maximum number of memory entries to include

        Returns:
            Formatted string with memory context
        """
        if not self.memory_enabled:
            return ""

        return self.memory.get_context(max_entries)

    def consolidate_memory(self, importance_threshold: float = 0.6):
        """Consolidate short-term memories to long-term storage.

        Args:
            importance_threshold: Minimum importance to consolidate
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.consolidate_to_long_term(importance_threshold)

    def clear_short_term_memory(self):
        """Clear short-term memory."""
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.clear_short_term()

    def save_memory(self):
        """Save memory to persistent storage."""
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.save()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        if not self.memory_enabled:
            return {'memory_enabled': False}

        stats = self.memory.stats()
        stats['memory_enabled'] = True
        return stats
        
    def set_context(self, context: str) -> None:
        """Set the context for the agent."""
        self.context = context
        
    def _format_tools_for_prompt(self) -> str:
        """Format tools into a string for the prompt."""
        tools_str = "Available tools:\n\n"
        for tool in self.tools:
            tools_str += f"Tool: {tool.name}\n"
            tools_str += f"Description: {tool.description}\n"
            tools_str += f"Parameters: {json.dumps(tool.parameters, indent=2)}\n\n"
        return tools_str

    async def _call_llm(self, prompt: str) -> str:
        """Make an async API call to OpenAI-compatible endpoint."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "stream": False
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()

                    if "error" in response_json:
                        logger.error(f"API error: {response_json['error']}")
                        raise Exception(response_json['error'])

                    full_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

                    if not full_response:
                        logger.error("Empty response from API")
                        raise Exception("Empty response from API")

                    return full_response

        except aiohttp.ClientError as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            raise

    async def evaluate_tool_use(self, user_input: str) -> Dict[str, Any]:
        """Evaluate which tool to use based on user input (async)."""
        prompt = f"""Context: {self.context}

{self._format_tools_for_prompt()}

User Input: {user_input}

You are an AI assistant that helps determine which tool to use and how to use it.
Analyze the user input and available tools to determine the appropriate action.

Your response must be valid JSON in this exact format:
{{
    "selected_tool": "name_of_tool",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "Your explanation here"
}}

For the calculator tool, ensure numeric parameters are numbers, not strings.
Remember to match the parameter names exactly as specified in the tool description.

Example response for calculator:
{{
    "selected_tool": "calculator",
    "parameters": {{
        "x": 5,
        "y": 3,
        "operation": "multiply"
    }},
    "reasoning": "User wants to multiply 5 and 3"
}}"""

        try:
            response = await self._call_llm(prompt)
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "selected_tool": None,
                "parameters": {},
                "reasoning": "Error parsing response"
            }

    async def call(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a specific tool with given parameters.

        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool

        Returns:
            Tool execution result
        """
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.function(**parameters)
                    # Check if result is a coroutine (async function)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                except Exception as e:
                    logger.error(f"Error calling tool {tool_name}: {str(e)}")
                    raise
        raise ValueError(f"Tool {tool_name} not found")

    async def infer(self, user_input: str) -> Dict[str, Any]:
        """Infer tool and parameters from natural language input.

        Args:
            user_input: Natural language query

        Returns:
            Dict with tool_used, parameters, reasoning, and result
        """
        # Store user input in memory
        if self.memory_enabled:
            self.memory.remember(
                content=f"User: {user_input}",
                memory_type='conversation',
                metadata={'role': 'user'},
                importance=0.5
            )

        evaluation = await self.evaluate_tool_use(user_input)

        if not evaluation["selected_tool"]:
            return {"error": "No appropriate tool found"}

        result = await self.call(
            evaluation["selected_tool"],
            evaluation["parameters"]
        )

        response = {
            "tool_used": evaluation["selected_tool"],
            "parameters": evaluation["parameters"],
            "reasoning": evaluation["reasoning"],
            "result": result
        }

        # Store agent response in memory
        if self.memory_enabled:
            self.memory.remember(
                content=f"Agent: Used {evaluation['selected_tool']} - {evaluation['reasoning']}",
                memory_type='conversation',
                metadata={
                    'role': 'agent',
                    'tool': evaluation['selected_tool'],
                    'parameters': evaluation['parameters']
                },
                importance=0.6
            )

        self.conversation_history.append({
            "user_input": user_input,
            "response": response
        })

        return response