import requests
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

from .tools import Tool
from .mcp_config import load_mcp_servers
from .mcp_tool import MCPToolManager
from .memory import Memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, name: str, model: str = "llama2", temperature: float = 0.7,
                 mcp_config_path: Optional[str] = None, load_mcp_tools: bool = False,
                 enable_memory: bool = True, memory_path: Optional[str] = None,
                 short_term_size: int = 10, use_sqlite: bool = True):
        """Initialize an Agent.

        Args:
            name: Name of the agent
            model: Ollama model to use (default: llama2)
            temperature: Temperature for model generation (default: 0.7)
            mcp_config_path: Optional path to MCP configuration file
            load_mcp_tools: Whether to automatically load tools from MCP servers (default: False)
            enable_memory: Whether to enable memory system (default: True)
            memory_path: Path for persistent memory storage (default: None)
            short_term_size: Size of short-term memory buffer (default: 10)
            use_sqlite: If True, use SQLite database for memory; otherwise use JSON (default: True)
        """
        self.name = name
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

        # Load MCP tools if requested
        if load_mcp_tools:
            self.load_mcp_tools(mcp_config_path)
        
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)
        logger.info(f"Added tool: {tool.name} to agent {self.name}")

    def load_mcp_tools(self, config_path: Optional[str] = None) -> List[Tool]:
        """Load tools from MCP servers defined in configuration.

        Args:
            config_path: Optional path to MCP config file. If None, uses default location.

        Returns:
            List of loaded Tool objects from MCP servers
        """
        try:
            # Load MCP server configurations
            server_configs = load_mcp_servers(config_path)

            if not server_configs:
                logger.warning("No MCP servers found in configuration")
                return []

            # Add each server to the manager
            for server_name, server_config in server_configs.items():
                self.mcp_manager.add_server(server_config)

            # Load all tools from all servers
            mcp_tools = self.mcp_manager.load_all_tools()

            # Add to agent's tools
            for tool in mcp_tools:
                self.tools.append(tool)

            logger.info(f"Loaded {len(mcp_tools)} tools from {len(server_configs)} MCP server(s)")
            return mcp_tools

        except Exception as e:
            logger.error(f"Error loading MCP tools: {str(e)}")
            raise

    def add_mcp_server(self, server_config) -> List[Tool]:
        """Add a single MCP server and load its tools.

        Args:
            server_config: MCPServerConfig object

        Returns:
            List of loaded Tool objects from the MCP server
        """
        try:
            adapter = self.mcp_manager.add_server(server_config)
            tools = adapter.load_tools()

            for tool in tools:
                self.tools.append(tool)

            logger.info(f"Added {len(tools)} tools from MCP server: {server_config.name}")
            return tools

        except Exception as e:
            logger.error(f"Error adding MCP server {server_config.name}: {str(e)}")
            raise

    def close_mcp_connections(self):
        """Close all MCP server connections."""
        self.mcp_manager.close_all()

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

    async def _call_ollama(self, prompt: str) -> str:
        """Make an async API call to Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "stream": False
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()

                    if "error" in response_json:
                        logger.error(f"Ollama API error: {response_json['error']}")
                        raise Exception(response_json['error'])

                    full_response = response_json.get("response", "")

                    if not full_response:
                        logger.error("Empty response from Ollama")
                        raise Exception("Empty response from Ollama")

                    return full_response

        except aiohttp.ClientError as e:
            logger.error(f"Error calling Ollama: {str(e)}")
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
            response = await self._call_ollama(prompt)
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Ollama response: {str(e)}")
            return {
                "selected_tool": None,
                "parameters": {},
                "reasoning": "Error parsing response"
            }

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a specific tool with given parameters (async)."""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.function(**parameters)
                    # Check if result is a coroutine (async function)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    raise
        raise ValueError(f"Tool {tool_name} not found")

    async def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and execute appropriate tool (async)."""
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

        result = await self.execute_tool(
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