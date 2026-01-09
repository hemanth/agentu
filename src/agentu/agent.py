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
from .skill import Skill
from .observe import Observer, EventType, get_config

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
                 api_key: Optional[str] = None, max_turns: int = 10):
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
            max_turns: Maximum turns for multi-turn inference (default: 10)
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
        self.deferred_tools: List[Tool] = []
        self.skills: List[Skill] = []  # Progressive loading skills
        self.max_turns = max_turns
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
        
        # Initialize observer
        output_format, enabled = get_config()
        self.observer = Observer(
            agent_name=self.name,
            output=output_format,
            enabled=enabled
        )

        # Load MCP tools if requested
        if load_mcp_tools and mcp_config_path:
            self.with_mcp([mcp_config_path])
        
    def _add_tool_internal(self, tool: Union[Tool, Callable], deferred: bool = False) -> Tool:
        """Internal method to add a single tool.

        Returns:
            The Tool object (created or passed in)
        """
        if isinstance(tool, Tool):
            tool_obj = tool
        elif callable(tool):
            tool_obj = Tool(tool)
        else:
            raise TypeError(f"Expected Tool or callable, got {type(tool)}")

        if deferred:
            self.deferred_tools.append(tool_obj)
            logger.info(f"Added deferred tool: {tool_obj.name} to agent {self.name}")
        else:
            self.tools.append(tool_obj)
            logger.info(f"Added tool: {tool_obj.name} to agent {self.name}")

        return tool_obj

    def _search_tools(self, query: str, limit: int = 5) -> str:
        """Search deferred tools and activate matching ones.

        Args:
            query: Search query to match against tool names and descriptions
            limit: Maximum number of tools to activate

        Returns:
            Confirmation message listing activated tools
        """
        matches = self._find_matching_tools(query, limit)

        if not matches:
            return "No matching tools found."

        activated = []
        for tool in matches:
            if tool not in self.tools:
                self.tools.append(tool)
                activated.append(tool.name)

        if activated:
            return f"Activated tools: {', '.join(activated)}"
        return f"Tools already active: {', '.join(t.name for t in matches)}"

    def _find_matching_tools(self, query: str, limit: int) -> List[Tool]:
        """Find deferred tools matching the query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching Tool objects
        """
        query_terms = query.lower().split()
        scored = []

        for tool in self.deferred_tools:
            text = f"{tool.name} {tool.description}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, tool))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [tool for _, tool in scored[:limit]]

    def _ensure_search_tool(self) -> None:
        """Add search_tools to active tools if not present."""
        if not any(t.name == "search_tools" for t in self.tools):
            search_tool = Tool(
                self._search_tools,
                description="Search for and activate tools by query. Use when you need a tool that isn't currently available.",
                name="search_tools"
            )
            self.tools.append(search_tool)
            logger.info(f"Added search_tools to agent {self.name}")

    def with_tools(
        self,
        tools: Optional[List[Union[Tool, Callable]]] = None,
        defer: Optional[List[Union[Tool, Callable]]] = None
    ) -> 'Agent':
        """Add tools and return self for chaining.

        Args:
            tools: List of Tool objects or callable functions (always active)
            defer: List of Tool objects or callable functions (searchable on-demand)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("MyAgent").with_tools([my_func])  # Active tools
            >>> agent = Agent("MyAgent").with_tools(defer=[many_funcs])  # Deferred
            >>> agent = Agent("MyAgent").with_tools([core], defer=[many])  # Both
        """
        if tools:
            for tool in tools:
                self._add_tool_internal(tool, deferred=False)

        if defer:
            for tool in defer:
                self._add_tool_internal(tool, deferred=True)
            self._ensure_search_tool()

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

    def with_skills(self, skills: List[Skill]) -> 'Agent':
        """Add agent skills with progressive loading.
        
        Skills use a 3-level loading system:
        - Level 1: Metadata (always loaded in system prompt, minimal context)
        - Level 2: Instructions (loaded when skill triggered)
        - Level 3: Resources (loaded on-demand)
        
        Args:
            skills: List of Skill objects to add
            
        Returns:
            Self for method chaining
            
        Example:
            >>> from pathlib import Path
            >>> pdf_skill = Skill(
            ...     name="pdf-processing",
            ...     description="Extract text and tables from PDF files",
            ...     instructions=Path("skills/pdf/SKILL.md"),
            ...     resources={"forms": Path("skills/pdf/FORMS.md")}
            ... )
            >>> agent = Agent("assistant").with_skills([pdf_skill])
        """
        self.skills.extend(skills)
        
        # Auto-add get_skill_resource tool if not present
        if skills and not any(t.name == "get_skill_resource" for t in self.tools):
            def get_skill_resource(skill_name: str, resource_key: str) -> str:
                """Load a skill resource file on-demand.
                
                Args:
                    skill_name: Name of the skill
                    resource_key: Resource identifier
                    
                Returns:
                    Resource content
                """
                skill = next((s for s in self.skills if s.name == skill_name), None)
                if not skill:
                    return f"Error: Skill '{skill_name}' not found"
                try:
                    return skill.load_resource(resource_key)
                except KeyError as e:
                    available = skill.list_resources()
                    return f"Error: {str(e)}. Available resources: {available}"
            
            self._add_tool_internal(Tool(
                get_skill_resource,
                description="Load additional documentation or resources from an activated skill",
                name="get_skill_resource"
            ))
            logger.info(f"Added get_skill_resource tool for {len(skills)} skills")
        
        logger.info(f"Added {len(skills)} skills to agent {self.name}")
        return self

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
        """Format tools and skills into a string for the prompt."""
        prompt_parts = []
        
        # Add skill metadata (Level 1: always loaded, minimal context)
        if self.skills:
            prompt_parts.append("Available Skills:\n")
            for skill in self.skills:
                prompt_parts.append(skill.metadata())
                prompt_parts.append("")  # Blank line
            prompt_parts.append("")  # Extra blank line
        
        # Add tool descriptions
        prompt_parts.append("Available tools:\n")
        for tool in self.tools:
            prompt_parts.append(f"Tool: {tool.name}")
            prompt_parts.append(f"Description: {tool.description}")
            prompt_parts.append(f"Parameters: {json.dumps(tool.parameters, indent=2)}\n")
        
        return "\n".join(prompt_parts)

    async def _call_llm(self, prompt: str) -> str:
        """Make an async API call to OpenAI-compatible endpoint."""
        with self.observer.trace(
            EventType.LLM_REQUEST,
            {"model": self.model, "prompt_length": len(prompt)}
        ):
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
                    with self.observer.trace(
                        EventType.TOOL_CALL,
                        {"tool_name": tool_name, "params": parameters}
                    ):
                        result = tool.function(**parameters)
                        # Check if result is a coroutine (async function)
                        if asyncio.iscoroutine(result):
                            result = await result
                        return result
                except Exception as e:
                    logger.error(f"Error calling tool {tool_name}: {str(e)}")
                    raise
        raise ValueError(f"Tool {tool_name} not found")

    def _match_skills(self, prompt: str) -> List[Skill]:
        """Determine which skills are relevant to the prompt.
        
        Uses keyword matching against skill descriptions to activate
        skills on-demand (Level 2 loading).
        
        Args:
            prompt: User input or task description
            
        Returns:
            List of matched Skill objects
        """
        matched = []
        prompt_lower = prompt.lower()
        
        for skill in self.skills:
            # Extract keywords from description (simple word matching)
            desc_words = skill.description.lower().split()
            
            # Check if any description keywords appear in prompt
            if any(word in prompt_lower for word in desc_words if len(word) > 3):
                matched.append(skill)
                logger.info(f"Matched skill: {skill.name} for prompt: {prompt[:50]}...")
        
        return matched

    async def infer(self, user_input: str) -> Dict[str, Any]:
        """Infer tool and parameters from natural language input.

        Runs a multi-turn agentic loop:
        1. LLM evaluates which tool to use
        2. If search_tools called, activate found tools and continue
        3. If regular tool called, execute and check if more work needed
        4. If no tool selected (text response), return final result

        Args:
            user_input: Natural language query

        Returns:
            Dict with tool_used, parameters, reasoning, and result
        """
        # Track inference start
        self.observer.record(EventType.INFERENCE_START, {"query": user_input})
        # Store user input in memory
        if self.memory_enabled:
            self.memory.remember(
                content=f"User: {user_input}",
                memory_type='conversation',
                metadata={'role': 'user'},
                importance=0.5
            )

        # Auto-match skills based on user input (Level 2 activation)
        active_skills = self._match_skills(user_input)
        if active_skills:
            logger.info(f"Activated {len(active_skills)} skill(s): {[s.name for s in active_skills]}")

        turn_history = []
        final_response = None

        for turn in range(self.max_turns):
            # Build context from previous turns (includes skill instructions if active)
            context = self._build_turn_context(user_input, turn_history, active_skills)

            evaluation = await self.evaluate_tool_use(context)

            if not evaluation.get("selected_tool"):
                # No tool selected = task complete or no match
                if turn_history:
                    final_response = turn_history[-1]
                else:
                    final_response = {"error": "No appropriate tool found"}
                break

            tool_name = evaluation["selected_tool"]
            parameters = evaluation["parameters"]

            try:
                result = await self.call(tool_name, parameters)
            except Exception as e:
                result = f"Error: {str(e)}"

            turn_result = {
                "turn": turn + 1,
                "tool_used": tool_name,
                "parameters": parameters,
                "reasoning": evaluation.get("reasoning", ""),
                "result": result
            }
            turn_history.append(turn_result)

            # If this was search_tools, continue to next turn
            if tool_name == "search_tools":
                continue

            # For other tools, check if we should continue
            # Currently: one non-search tool call = done
            final_response = turn_result
            break

        if final_response is None:
            final_response = {
                "error": f"Max turns ({self.max_turns}) reached",
                "history": turn_history
            }

        # Store agent response in memory
        if self.memory_enabled and "tool_used" in final_response:
            self.memory.remember(
                content=f"Agent: Used {final_response['tool_used']} - {final_response.get('reasoning', '')}",
                memory_type='conversation',
                metadata={
                    'role': 'agent',
                    'tool': final_response['tool_used'],
                    'parameters': final_response.get('parameters', {})
                },
                importance=0.6
            )

        self.conversation_history.append({
            "user_input": user_input,
            "response": final_response,
            "turns": len(turn_history)
        })
        
        # Track inference end
        self.observer.record(
            EventType.INFERENCE_END,
            {
                "query": user_input,
                "turns": len(turn_history),
                "tool_used": final_response.get('tool_used')
            }
        )

        return final_response

    def _build_turn_context(self, user_input: str, turn_history: List[Dict[str, Any]], 
                            active_skills: Optional[List[Skill]] = None) -> str:
        """Build context string for multi-turn inference.
        
        Args:
            user_input: Original user input
            turn_history: List of previous turn results
            active_skills: Skills that have been activated for this request
            
        Returns:
            Context string for LLM
        """
        context_parts = []
        
        # Add skill instructions (Level 2 loading) if skills are active
        if active_skills:
            context_parts.append("=== Active Skills ===")
            for skill in active_skills:
                context_parts.append(f"\n## Skill: {skill.name}")
                context_parts.append(skill.load_instructions())
                context_parts.append("")
            context_parts.append("=== End Skills ===\n")
        
        # Add original request
        if not turn_history:
            return "\n".join(context_parts) + user_input if context_parts else user_input
        
        # Multi-turn context
        context_parts.append(f"Original request: {user_input}")
        context_parts.append("")
        context_parts.append("Previous actions:")
        for turn in turn_history:
            context_parts.append(
                f"- Called {turn['tool_used']}: {turn['result']}"
            )
        context_parts.append("")
        context_parts.append("Continue with the task. What tool should be used next?")
        
        return "\n".join(context_parts)

    async def ralph(
        self,
        prompt_file: str,
        max_iterations: int = 50,
        timeout_minutes: int = 30,
        checkpoint_every: int = 5,
        on_iteration=None
    ):
        """Run agent in Ralph mode (autonomous loop).
        
        Ralph continuously reads a prompt file and executes until
        all checkpoints are complete or limits are reached.
        
        Args:
            prompt_file: Path to PROMPT.md file with goal and checkpoints
            max_iterations: Maximum loop iterations (safety limit)
            timeout_minutes: Maximum runtime in minutes
            checkpoint_every: Save state every N iterations
            on_iteration: Optional callback(iteration, result_dict)
            
        Returns:
            Execution summary dict
            
        Example:
            >>> result = await agent.ralph("PROMPT.md", max_iterations=50)
        """
        from .ralph import RalphRunner, RalphConfig
        
        config = RalphConfig(
            prompt_file=prompt_file,
            max_iterations=max_iterations,
            timeout_minutes=timeout_minutes,
            checkpoint_every=checkpoint_every
        )
        
        runner = RalphRunner(self, config)
        return await runner.run(on_iteration=on_iteration)