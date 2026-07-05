import json
import asyncio
import aiohttp
from typing import AsyncIterator, List, Dict, Any, Optional, Callable, Type, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

from .structured import (
    build_response_format,
    parse_and_validate,
    format_validation_error,
    StructuredOutputError,
)
from .multimodal import build_content_parts

from .tools import Tool, ToolPermission
from .safety import check_lethal_trifecta, spotlight_untrusted
from .hooks import (
    HookAction, HookResult, HookSet, PermissionApprovalRequired,
    _call_maybe_async, PreToolHook, PostToolHook, OnStopHook,
)
from ..mcp.config import load_mcp_servers
from ..mcp.tool import MCPToolManager
from ..memory.memory import Memory
from ..workflow.workflow import Step
from ..skills.skill import Skill, load_skill
from ..middleware.observe import Observer, EventType, get_config
from ..middleware.guardrails import Guardrail, GuardrailSet, GuardrailError
from ..middleware.middleware import BaseMiddleware, MiddlewareChain, CallContext
from ..middleware.notify import NotifyMiddleware

# Optional embedding imports for semantic matching
try:
    from ..cache.embeddings import EmbeddingProvider as _EmbeddingProvider, cosine_similarity as _cosine_similarity
except ImportError:  # pragma: no cover
    _EmbeddingProvider = None  # type: ignore[assignment,misc]
    _cosine_similarity = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


async def get_ollama_models(api_base: str = "http://localhost:11434") -> List[str]:
    """Get list of available Ollama models.

    Args:
        api_base: Base URL for Ollama API

    Returns:
        List of model names, or empty list if unable to fetch
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{api_base.rstrip('/')}/api/tags",
                timeout=aiohttp.ClientTimeout(total=2)
            ) as response:
                response.raise_for_status()
                models_data = await response.json()
                models = [model["name"] for model in models_data.get("models", [])]
                return models
    except Exception as e:
        logger.warning(f"Unable to fetch Ollama models: {e}")
        return []


def _get_ollama_models_sync(api_base: str = "http://localhost:11434") -> List[str]:
    """Sync wrapper to get Ollama models (used during __init__)."""
    try:
        import urllib.request
        import json as _json
        req = urllib.request.Request(f"{api_base.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            models_data = _json.loads(resp.read())
            return [model["name"] for model in models_data.get("models", [])]
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
    models = _get_ollama_models_sync(api_base)
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
                 api_key: Optional[str] = None, max_turns: int = 10,
                 cache: bool = False, cache_ttl: int = 3600,
                 enable_rationale_recording: bool = False,
                 codemode: bool = False,
                 auto_discover_rules: bool = True,
                 rules_dir: Optional[str] = None):
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
            cache: Enable LLM response caching (default: False)
            cache_ttl: Cache time-to-live in seconds (default: 3600 = 1 hour)
            enable_rationale_recording: Whether to automatically expose the record_rationale tool to the agent (default: False)
            codemode: If True, the LLM writes Python code to call tools instead of
                      making individual JSON tool calls. More token-efficient for multi-step
                      tasks and leverages LLMs' strength at writing code. (default: False)
            auto_discover_rules: If True, auto-discover AGENTS.md and CLAUDE.md files
                                 hierarchically from the project root and apply them as
                                 rules. Checks AGENTS.md, .agents/AGENTS.md, CLAUDE.md,
                                 .claude/CLAUDE.md in order. (default: True)
            rules_dir: Directory to start rule discovery from (default: current working
                       directory). Only used when auto_discover_rules is True.
        """
        self.name = name
        self.codemode = codemode
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
        
        # Initialize cache if enabled
        self.cache_enabled = cache
        if cache:
            from ..cache.tiered import TieredCache
            from ..cache.storage import MemoryBackend, SQLiteBackend
            self.cache = TieredCache(
                backends=[MemoryBackend(), SQLiteBackend()],
                ttl=cache_ttl,
            )
        else:
            self.cache = None
        self._cache_sync = None

        # Guardrails and middleware
        self._input_guardrails: Optional[GuardrailSet] = None
        self._output_guardrails: Optional[GuardrailSet] = None
        self._middleware_chain: Optional[MiddlewareChain] = None
        self._max_corrections: int = 0  # disabled until with_guardrails sets it
        self._allow_dangerous: bool = False

        # Hooks (pre_tool, post_tool, on_stop)
        self._hooks: Optional[HookSet] = None

        # Permission mode: 'auto' (default), 'plan', 'ask-writes'
        self._permission_mode: str = "auto"
        self._can_use_tool: Optional[Callable] = None

        # Sandbox for tool isolation
        self._sandbox = None
        self._sandbox_limits = None
        self._worktree_config = None
        self._context_config = None  # Context compaction config

        # Reusable aiohttp session for LLM calls
        self._llm_session: Optional[aiohttp.ClientSession] = None

        # Store MCP config for deferred async loading
        self._pending_mcp_config = mcp_config_path if (load_mcp_tools and mcp_config_path) else None

        # Auto-discover AGENTS.md / CLAUDE.md rules
        self._auto_discover_rules = auto_discover_rules
        self._rules_dir = rules_dir
        if auto_discover_rules:
            self._apply_discovered_rules(rules_dir)

        if enable_rationale_recording:
            self._add_tool_internal(Tool(
                self.record_rationale,
                description="Record architectural decisions, rationale, or reasons why a specific change or action was taken. Use this to leave an audit trail for future agents.",
                name="record_rationale"
            ))

    @classmethod
    async def from_config(cls, path: str) -> 'Agent':
        """Load an agent declaratively from a JSON or YAML configuration file.
        
        Requires the `[yaml]` optional dependency if using .yaml files.
        
        Args:
            path: String path resolving to the configuration file.
            
        Returns:
            A fully constructed asynchronous Agent instance.
        """
        from .config import AgentConfig
        
        cfg = AgentConfig.load(path)
        
        agent = cls(name=cfg.name, model=cfg.model)
        
        if cfg.system_prompt:
            agent.context = cfg.system_prompt
        
        if cfg.rules:
            agent.with_rules(cfg.rules)
        
        if cfg.cache:
            agent.with_cache(preset=cfg.cache.preset, ttl=cfg.cache.ttl)
            
        if cfg.notify:
            agent.with_notifier(targets=cfg.notify)
            
        if cfg.skills:
            agent.with_skills(cfg.skills)
            
        if cfg.mcp and cfg.mcp.urls:
            await agent.with_mcp(cfg.mcp.urls)
            
        return agent
        
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

    def _find_matching_tools(self, query: str, limit: int,
                             embedding_provider: Optional[Any] = None) -> List[Tool]:
        """Find deferred tools matching the query.

        Uses semantic (embedding-based) matching when *embedding_provider* is
        supplied and functional, falling back to keyword scoring otherwise.

        Args:
            query: Search query
            limit: Maximum results
            embedding_provider: Optional EmbeddingProvider for semantic ranking.

        Returns:
            List of matching Tool objects
        """
        # --- Try semantic matching first ---------------------------------
        if embedding_provider is not None and _cosine_similarity is not None:
            try:
                semantic_results = self._semantic_match_tools(
                    query, limit, embedding_provider
                )
                if semantic_results:
                    return semantic_results
            except Exception as e:
                logger.debug("Semantic tool matching failed, using keyword fallback: %s", e)

        # --- Keyword-based fallback -------------------------------------
        query_terms = query.lower().split()
        scored = []

        for tool in self.deferred_tools:
            text = f"{tool.name} {tool.description}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, tool))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [tool for _, tool in scored[:limit]]

    def _semantic_match_tools(self, query: str, limit: int,
                              embedding_provider: Any) -> List[Tool]:
        """Rank deferred tools by cosine-similarity of their descriptions.

        This is a sync helper that bridges into the async embedding API.
        Returns an empty list on failure so the caller can fall back.
        """
        async def _embed_and_rank():
            query_vec = await embedding_provider.embed(query)
            scored = []
            for tool in self.deferred_tools:
                text = f"{tool.name} {tool.description}"
                tool_vec = await embedding_provider.embed(text)
                score = _cosine_similarity(query_vec, tool_vec)
                scored.append((score, tool))
            scored.sort(reverse=True, key=lambda x: x[0])
            # Return tools with positive similarity
            return [tool for score, tool in scored[:limit] if score > 0]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _embed_and_rank()).result()
        else:
            return asyncio.run(_embed_and_rank())

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

        # Check for lethal trifecta across all registered tools
        all_tools = self.tools + self.deferred_tools
        report = check_lethal_trifecta(all_tools)
        if report.has_trifecta:
            self._trifecta_report = report

        return self

    def with_cache(self, preset: Optional[str] = None, ttl: int = 3600,
                   similarity_threshold: float = 0.95, embedding_provider: str = "local",
                   embedding_model: Optional[str] = None, redis_url: str = "redis://localhost:6379",
                   sync_enabled: bool = False, sync_path: Optional[str] = None,
                   sync_interval: int = 300) -> 'Agent':
        """Configure smart caching with presets.

        Presets:
            - "basic": memory + sqlite (default, same as cache=True)
            - "smart": memory + sqlite + local semantic matching
            - "offline": memory + sqlite + filesystem + semantic + background sync
            - "distributed": memory + redis + api semantic matching
        """
        from ..cache.storage import MemoryBackend, SQLiteBackend, RedisBackend, FilesystemBackend
        from ..cache.tiered import TieredCache

        if preset is None:
            preset = "basic"

        backends = []
        semantic_index = None

        if preset == "basic":
            backends = [MemoryBackend(), SQLiteBackend()]
        elif preset == "smart":
            backends = [MemoryBackend(), SQLiteBackend()]
            semantic_index = self._build_semantic_index(
                embedding_provider, embedding_model, similarity_threshold
            )
        elif preset == "offline":
            backends = [MemoryBackend(), SQLiteBackend(), FilesystemBackend()]
            semantic_index = self._build_semantic_index(
                embedding_provider, embedding_model, similarity_threshold
            )
            sync_enabled = True
        elif preset == "distributed":
            backends = [MemoryBackend(), RedisBackend(url=redis_url)]
            semantic_index = self._build_semantic_index(
                "api", embedding_model, similarity_threshold
            )

        self.cache = TieredCache(backends=backends, ttl=ttl, semantic_index=semantic_index)
        self.cache_enabled = True

        if sync_enabled:
            from ..cache.sync import CacheSync
            sqlite_backend = next((b for b in backends if isinstance(b, SQLiteBackend)), None)
            if sqlite_backend:
                if sync_path is None:
                    from pathlib import Path
                    sync_path = str(Path.home() / ".agentu" / "cache_sync")
                self._cache_sync = CacheSync(
                    source_backend=sqlite_backend,
                    sync_path=sync_path,
                    sync_interval=sync_interval,
                )

        return self

    def with_guardrails(
        self,
        input_guardrails: Optional[List[Guardrail]] = None,
        output_guardrails: Optional[List[Guardrail]] = None,
        max_corrections: int = 2,
    ) -> 'Agent':
        """Add guardrails for input/output validation.

        When output guardrails fail, the agent can self-correct by feeding
        the violation back to the LLM and retrying (up to max_corrections times).

        Args:
            input_guardrails: Guardrails to check user input before LLM call
            output_guardrails: Guardrails to check LLM output before returning
            max_corrections: Max self-correction attempts on output guardrail failure (default: 2).
                             Set to 0 to disable self-correction (raise immediately).

        Returns:
            Self for method chaining

        Example:
            >>> from agentu.guardrails import PII, ContentFilter, MaxLength
            >>> agent = Agent("bot").with_guardrails(
            ...     input_guardrails=[PII(), MaxLength(max_chars=5000)],
            ...     output_guardrails=[ContentFilter(block=["violence"])],
            ...     max_corrections=2,
            ... )
        """
        if input_guardrails:
            self._input_guardrails = GuardrailSet(input_guardrails)
        if output_guardrails:
            self._output_guardrails = GuardrailSet(output_guardrails)
        self._max_corrections = max_corrections
        return self

    def with_rules(self, path: str = "AGENTS.md") -> 'Agent':
        """Load feedforward rules from a markdown file.

        Rules are prepended to the system context and sent with every LLM call.
        Convention: place an AGENTS.md at your project root.

        Args:
            path: Path to rules file (default: AGENTS.md)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_rules("AGENTS.md")
        """
        from pathlib import Path as _Path

        rules_path = _Path(path)
        if not rules_path.exists():
            logger.warning(f"Rules file not found: {path}")
            return self

        rules_content = rules_path.read_text().strip()

        # Prepend rules to existing context
        if self.context:
            self.context = f"=== Project Rules ===\n{rules_content}\n=== End Rules ===\n\n{self.context}"
        else:
            self.context = f"=== Project Rules ===\n{rules_content}\n=== End Rules ==="

        logger.info(f"Loaded {len(rules_content)} chars of rules from {path}")
        return self

    def _apply_discovered_rules(self, rules_dir: Optional[str] = None) -> None:
        """Auto-discover and apply AGENTS.md / CLAUDE.md rules.

        Called during ``__init__`` when ``auto_discover_rules=True``.
        Searches the project root (or ``rules_dir``) for rule files in this
        priority order at each directory level:

        1. ``AGENTS.md``
        2. ``.agents/AGENTS.md``
        3. ``CLAUDE.md``
        4. ``.claude/CLAUDE.md``

        Discovered rules are prepended to the agent's context, just like
        :meth:`with_rules`.

        Args:
            rules_dir: Directory to start searching from (default: cwd).
        """
        from ..discovery import discover_and_format_rules

        rules_content = discover_and_format_rules(
            start_dir=rules_dir,
            recursive=False,  # Only check root level by default
            max_depth=0,
        )

        if not rules_content:
            return

        # Prepend rules to existing context (same format as with_rules)
        if self.context:
            self.context = (
                f"=== Project Rules (auto-discovered) ===\n"
                f"{rules_content}\n"
                f"=== End Rules ===\n\n{self.context}"
            )
        else:
            self.context = (
                f"=== Project Rules (auto-discovered) ===\n"
                f"{rules_content}\n"
                f"=== End Rules ==="
            )

        logger.info(
            f"Auto-discovered and applied {len(rules_content)} chars of rules"
        )

    def with_permissions(
        self,
        mode: str = "auto",
        can_use_tool: Optional[Callable] = None,
        allow_dangerous: bool = False,
    ) -> 'Agent':
        """Configure tool permission policy.

        Permission modes:
            - ``"auto"``: All tools execute freely (current default behaviour).
            - ``"plan"``: WRITE and DANGEROUS tools are blocked. The agent can
              only call READONLY tools.
            - ``"ask-writes"``: WRITE tools raise
              :class:`PermissionApprovalRequired` so the caller can prompt
              the user before allowing execution.

        The ``plan`` mode is implemented as a built-in *pre_tool* hook that
        denies WRITE tools, so it composes naturally with custom hooks.

        Args:
            mode: Permission mode — ``"auto"``, ``"plan"``, or ``"ask-writes"``.
            can_use_tool: Optional async/sync callback
                ``(tool_name, params, context) -> 'allow' | 'deny' | 'ask'``.
                Overrides the default mode logic for individual calls.
            allow_dangerous: If True, allow DANGEROUS tools to execute.

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_permissions(allow_dangerous=True)
            >>> agent = Agent("bot").with_permissions(mode="plan")
            >>> agent = Agent("bot").with_permissions(
            ...     mode="ask-writes",
            ...     can_use_tool=lambda name, params, ctx: "allow",
            ... )
        """
        if mode not in ("auto", "plan", "ask-writes"):
            raise ValueError(f"Invalid permission mode: {mode!r}. Use 'auto', 'plan', or 'ask-writes'.")
        self._permission_mode = mode
        self._can_use_tool = can_use_tool
        self._allow_dangerous = allow_dangerous

        # 'plan' mode installs a pre_tool hook that denies WRITE tools
        if mode == "plan":
            def _plan_mode_hook(tool_name: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> HookResult:
                tool_obj = context.get("tool")
                if tool_obj and tool_obj.permission in (ToolPermission.WRITE, ToolPermission.DANGEROUS):
                    return HookResult(
                        action=HookAction.DENY,
                        reason=f"Tool '{tool_name}' is blocked in plan mode (permission={tool_obj.permission.value}). "
                               f"Only READONLY tools are allowed.",
                    )
                return HookResult(action=HookAction.ALLOW)

            self.with_hooks(pre_tool=_plan_mode_hook)

        return self

    def with_hooks(
        self,
        pre_tool: Optional[Union[PreToolHook, List[PreToolHook]]] = None,
        post_tool: Optional[Union[PostToolHook, List[PostToolHook]]] = None,
        on_stop: Optional[Union[OnStopHook, List[OnStopHook]]] = None,
    ) -> 'Agent':
        """Register lifecycle hooks on this agent.

        Hooks can be sync or async callables.  Multiple calls to
        ``with_hooks`` are additive — hooks accumulate.

        Args:
            pre_tool: Called **before** each tool execution.
                Signature: ``(tool_name, parameters, context) -> HookResult``
                *context* contains ``{"tool": <Tool object>}``.
                Return ``HookResult(action=HookAction.DENY, reason=…)`` to
                block the call, or ``HookResult(action=HookAction.MODIFY,
                modified_params=…)`` to rewrite parameters.
            post_tool: Called **after** each tool execution.
                Signature: ``(tool_name, parameters, result) -> result``
                The returned value replaces the tool result.
            on_stop: Called when the agent loop finishes.
                Signature: ``(final_response) -> final_response``

        Returns:
            Self for method chaining

        Example:
            >>> def audit(tool_name, params, ctx):
            ...     print(f"Calling {tool_name}")
            ...     return HookResult()  # allow
            >>> agent = Agent("bot").with_hooks(pre_tool=audit)
        """
        if self._hooks is None:
            self._hooks = HookSet()

        # Normalise single callable into a list
        if pre_tool is not None:
            hooks = pre_tool if isinstance(pre_tool, list) else [pre_tool]
            self._hooks.pre_tool_hooks.extend(hooks)
        if post_tool is not None:
            hooks = post_tool if isinstance(post_tool, list) else [post_tool]
            self._hooks.post_tool_hooks.extend(hooks)
        if on_stop is not None:
            hooks = on_stop if isinstance(on_stop, list) else [on_stop]
            self._hooks.on_stop_hooks.extend(hooks)

        return self

    def with_sandbox(
        self,
        read_tools: Optional[List[Union[Tool, Callable]]] = None,
        write_tools: Optional[List[Union[Tool, Callable]]] = None,
        backend=None,
        timeout: float = 30.0,
        max_memory_mb: Optional[int] = 256,
        allow_network: bool = True,
        codemode: bool = False,
    ) -> 'Agent':
        """Set up a sandboxed execution environment.

        Tools in `read_tools` get READONLY permission (no side effects).
        Tools in `write_tools` get WRITE permission (has side effects).
        All tools run in an isolated subprocess.

        When codemode=True, the LLM writes Python code that calls tools via
        a typed API, instead of making individual JSON tool calls. The code
        runs in the sandbox with the same isolation and timeout guarantees.

        Args:
            read_tools: Tools with no side effects (READONLY)
            write_tools: Tools with side effects (WRITE)
            backend: A SandboxBackend instance. Defaults to SubprocessSandbox.
            timeout: Max seconds per tool call (default: 30)
            max_memory_mb: Memory limit in MB (default: 256, None=unlimited)
            allow_network: Allow outbound network access (default: True).
                Set to False to block HTTP via proxy env vars.
            codemode: If True, enable code mode for this sandbox (default: False)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_sandbox(
            ...     read_tools=[search, get_weather],
            ...     write_tools=[save_file, send_email],
            ...     timeout=10,
            ...     allow_network=False,
            ... )
        """
        from ..runtime.sandbox import SubprocessSandbox, SandboxLimits

        # Register read tools with READONLY permission
        if read_tools:
            for t in read_tools:
                if isinstance(t, Tool):
                    t.permission = ToolPermission.READONLY
                    self._add_tool_internal(t)
                else:
                    self._add_tool_internal(Tool(t, permission=ToolPermission.READONLY))

        # Register write tools with WRITE permission
        if write_tools:
            for t in write_tools:
                if isinstance(t, Tool):
                    t.permission = ToolPermission.WRITE
                    self._add_tool_internal(t)
                else:
                    self._add_tool_internal(Tool(t, permission=ToolPermission.WRITE))

        # Configure sandbox backend
        self._sandbox = backend or SubprocessSandbox()
        self._sandbox_limits = SandboxLimits(
            timeout_seconds=timeout,
            max_memory_mb=max_memory_mb,
            allow_network=allow_network,
        )
        if codemode:
            self.codemode = True
        logger.info(f"Sandbox enabled: {type(self._sandbox).__name__} (timeout={timeout}s, codemode={self.codemode})")
        return self

    def use(self, *middlewares: BaseMiddleware) -> 'Agent':
        """Add middleware to the processing pipeline.

        Middleware runs in order for `before` hooks and reverse order
        for `after` hooks (like Express.js).

        Args:
            *middlewares: Middleware instances to add

        Returns:
            Self for method chaining

        Example:
            >>> from agentu.middleware import CostTracker, LoggerMiddleware, RetryMiddleware
            >>> agent = Agent("bot").use(
            ...     CostTracker(),
            ...     LoggerMiddleware(),
            ...     RetryMiddleware(max_retries=3)
            ... )
        """
        if self._middleware_chain is None:
            self._middleware_chain = MiddlewareChain()
        for mw in middlewares:
            self._middleware_chain.add(mw)
        return self

    def with_notifier(self, targets: List[str], title: Optional[str] = None) -> 'Agent':
        """Quickly add notification middleware to the agent.
        
        Requires the `[notify]` extra to be installed.
        
        Args:
            targets: List of Apprise notification URLs
            title: Optional title for the notification
            
        Returns:
            Self for method chaining
        """
        return self.use(NotifyMiddleware(targets=targets, title=title))

    def _build_semantic_index(self, provider_type: str, model: Optional[str],
                              threshold: float):
        from ..cache.semantic import SemanticIndex
        from ..cache.embeddings import LocalEmbedding, APIEmbedding

        if provider_type == "local":
            provider = LocalEmbedding(model_name=model or "all-MiniLM-L6-v2")
            if not provider.available():
                import logging
                logging.getLogger(__name__).warning(
                    "sentence-transformers not installed, semantic caching disabled"
                )
                return None
        else:
            provider = APIEmbedding(
                api_base=self.api_base, model=model or "nomic-embed-text", api_key=self.api_key
            )

        return SemanticIndex(embedding_provider=provider, threshold=threshold)

    async def with_mcp(self, servers: List[Union[str, Dict[str, Any]]]) -> 'Agent':
        """Connect to MCP servers and load their tools (chainable, async).

        Args:
            servers: List of MCP server configurations. Each item can be:
                - String URL: "http://localhost:3000"
                - Dict with url and headers: {"url": "...", "headers": {...}}
                - Config file path: "~/.agentu/mcp_config.json"

        Returns:
            Self for method chaining

        Example:
            >>> agent = await Agent("bot").with_mcp([
            ...     "http://localhost:3000",
            ...     {"url": "https://api.com/mcp", "headers": {"Auth": "Bearer xyz"}}
            ... ])
        """
        from ..mcp.config import load_mcp_servers
        from ..mcp.transport import MCPServerConfig

        for server in servers:
            try:
                # Handle config file path
                if isinstance(server, str) and server.endswith('.json'):
                    server_configs = load_mcp_servers(server)
                    for server_name, server_config in server_configs.items():
                        adapter = self.mcp_manager.add_server(server_config)
                        tools = await adapter.load_tools()
                        for tool in tools:
                            self._add_tool_internal(tool)
                        logger.info(f"Loaded {len(tools)} tools from MCP server: {server_name}")

                # Handle URL string
                elif isinstance(server, str):
                    from ..mcp.transport import TransportType
                    config = MCPServerConfig(
                        name=f"mcp_{len(self.mcp_manager.adapters)}",
                        transport_type=TransportType.HTTP,
                        url=server
                    )
                    adapter = self.mcp_manager.add_server(config)
                    tools = await adapter.load_tools()
                    for tool in tools:
                        self._add_tool_internal(tool)
                    logger.info(f"Loaded {len(tools)} tools from MCP server: {server}")

                # Handle dict with url and headers
                elif isinstance(server, dict):
                    from ..mcp.transport import TransportType, AuthConfig
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
                    tools = await adapter.load_tools()
                    for tool in tools:
                        self._add_tool_internal(tool)
                    logger.info(f"Loaded {len(tools)} tools from MCP server: {url}")

                else:
                    raise TypeError(f"Invalid MCP server type: {type(server)}")

            except Exception as e:
                logger.error(f"Error connecting to MCP server {server}: {str(e)}")
                raise

        return self

    async def close_mcp_connections(self):
        """Close all MCP server connections."""
        await self.mcp_manager.close_all()

    async def close(self):
        """Close all async resources (LLM session, MCP connections)."""
        if self._llm_session and not self._llm_session.closed:
            await self._llm_session.close()
            self._llm_session = None
        await self.close_mcp_connections()

    async def __aenter__(self) -> 'Agent':
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, closing all resources."""
        await self.close()

    async def with_skills(self, skills: List[Union[Skill, str]], skill_ttl: Optional[int] = 86400) -> 'Agent':
        """Add agent skills with progressive loading.
        
        Skills use a 3-level loading system:
        - Level 1: Metadata (always loaded in system prompt, minimal context)
        - Level 2: Instructions (loaded when skill triggered)
        - Level 3: Resources (loaded on-demand)
        
        Args:
            skills: List of Skill objects, GitHub URLs, or local paths
            skill_ttl: Cache time-to-live in seconds for GitHub skills (default: 86400 = 24 hours)
                       None means cache forever, 0 means always fetch fresh
            
        Returns:
            Self for method chaining
            
        Example:
            >>> # From GitHub URL (auto-refreshes every 24 hours)
            >>> agent = await Agent("assistant").with_skills([
            ...     "https://github.com/hemanth/agentu-skills/tree/main/pdf-processor"
            ... ])
            
            >>> # Custom TTL (refresh every hour)
            >>> agent = await Agent("assistant").with_skills([...], skill_ttl=3600)
            
            >>> # Cache forever (never auto-refresh)
            >>> agent = await Agent("assistant").with_skills([...], skill_ttl=None)
            
            >>> # Always fetch fresh
            >>> agent = await Agent("assistant").with_skills([...], skill_ttl=0)
            
            >>> # From local path
            >>> agent = await Agent("assistant").with_skills(["./skills/my-skill"])
            
            >>> # Using Skill object directly
            >>> pdf_skill = Skill(
            ...     name="pdf-processing",
            ...     description="Extract text and tables from PDF files",
            ...     instructions="skills/pdf/SKILL.md"
            ... )
            >>> agent = await Agent("assistant").with_skills([pdf_skill])
        """
        # Resolve all skills concurrently (strings become Skill objects)
        resolved_skills = await asyncio.gather(*[load_skill(s, ttl=skill_ttl) for s in skills])
        self.skills.extend(resolved_skills)
        
        # Auto-add get_skill_resource tool if not present
        if resolved_skills and not any(t.name == "get_skill_resource" for t in self.tools):
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
            logger.info(f"Added get_skill_resource tool for {len(resolved_skills)} skills")
        
        logger.info(f"Added {len(resolved_skills)} skills to agent {self.name}")
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

    def record_rationale(self, action: str, reasoning: str) -> str:
        """Record an architectural decision or reasoning for a specific action.

        Args:
            action: The action that was taken.
            reasoning: The rationale or reasoning behind the action.

        Returns:
            Confirmation string.
        """
        self.observer.record(
            EventType.RATIONALE,
            metadata={"action": action, "reasoning": reasoning}
        )

        if self.memory_enabled:
            self.remember(
                f"Rationale for {action}: {reasoning}",
                memory_type="rationale",
                importance=0.9
            )

        return f"Rationale for '{action}' recorded successfully."

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
              limit: int = 5, include_short_term: bool = True,
              semantic: bool = False, semantic_threshold: float = 0.0):
        """Recall memories.

        Args:
            query: Search query (if None, returns recent memories)
            memory_type: Filter by memory type
            limit: Maximum number of results
            include_short_term: Whether to include short-term memories
            semantic: If True, use embedding-based similarity search
                instead of substring matching (requires embedding provider)
            semantic_threshold: Minimum similarity score for semantic recall

        Returns:
            List of MemoryEntry objects
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return []

        return self.memory.recall(
            query, memory_type, limit, include_short_term,
            semantic=semantic, semantic_threshold=semantic_threshold,
        )

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

    def _generate_type_stubs(self) -> str:
        """Generate Python type stubs from registered tools for codemode.

        Converts Tool objects into typed Python function signatures with
        docstrings, so the LLM can write code that calls them naturally.

        Returns:
            Python code string defining the `tools` namespace
        """
        import inspect as _inspect

        lines = ["class tools:"]
        lines.append('    """Available tools. Call these as tools.name(args)."""')

        if not self.tools:
            lines.append("    pass")
            return "\n".join(lines)

        for tool in self.tools:
            # Build parameter signature from tool's extracted parameters
            params = []
            for param_name, param_type in tool.parameters.items():
                # param_type is like "str" or "int: (default: 5)"
                if ": (default:" in str(param_type):
                    type_part = str(param_type).split(":")[0].strip()
                    default_part = str(param_type).split("default:")[1].rstrip(")").strip()
                    params.append(f"{param_name}: {type_part} = {default_part}")
                else:
                    params.append(f"{param_name}: {str(param_type)}")

            param_str = ", ".join(params)
            lines.append("")
            lines.append("    @staticmethod")
            lines.append(f"    def {tool.name}({param_str}):")
            # Add docstring from tool description
            lines.append(f'        """{tool.description}"""')
            lines.append("        ...")

        return "\n".join(lines)

    def _build_codemode_prompt(self, user_input: str) -> str:
        """Build the prompt for codemode: ask LLM to write Python code.

        Args:
            user_input: The user's natural language request

        Returns:
            Prompt string for the LLM
        """
        type_stubs = self._generate_type_stubs()

        return f"""{self.context}

You have access to the following Python API:

```python
{type_stubs}
```

Write Python code that accomplishes the user's request by calling these tools.
The `tools` object is already available in your execution environment.

Rules:
- Call tools as: tools.tool_name(arg1, arg2)
- Use print() to output results the user should see
- You can chain multiple tool calls, use variables, loops, try/except
- Only use the tools defined above — no imports, no file I/O, no network access
- Output ONLY a Python code block, no explanation

User request: {user_input}"""

    async def _exec_codemode(self, code: str) -> str:
        """Execute LLM-generated code with the tools namespace.

        Two execution modes:
        - With sandbox: entire code runs in subprocess (os/sys allowed)
        - Without sandbox: runs in-process with restricted builtins

        Args:
            code: Python code string from the LLM

        Returns:
            Captured stdout output from the code execution
        """
        # Branch: sandbox = subprocess isolation, no sandbox = in-process restricted
        if self._sandbox is not None:
            return await self._exec_codemode_sandboxed(code)
        return await self._exec_codemode_inprocess(code)

    async def _exec_codemode_sandboxed(self, code: str) -> str:
        """Execute codemode code in a sandbox subprocess.

        Serializes tool functions to source, builds a standalone script
        with a tools namespace, and runs the entire thing in subprocess.
        No import restrictions -- the subprocess IS the security boundary.
        """
        import inspect as _inspect
        import textwrap

        # Serialize tool functions to source code
        tool_sources = []
        tool_names = []
        for tool in self.tools:
            try:
                source = _inspect.getsource(tool.function)
                source = textwrap.dedent(source)
                tool_sources.append(source)
                tool_names.append(tool.name)
            except (OSError, TypeError):
                logger.warning(
                    f"Tool '{tool.name}' cannot be serialized for sandbox codemode. Skipping."
                )

        # Build the standalone script
        bridge_attrs = "\n".join(
            f"    {name} = staticmethod({name})"
            for name in tool_names
        )

        script = (
            "import sys\nimport os\nimport json\n\n"
            + "\n\n".join(tool_sources)
            + f"\n\nclass tools:\n{bridge_attrs}\n\n"
            + code
            + "\n"
        )

        # Execute in sandbox subprocess
        sandbox_result = await self._sandbox.execute(script, self._sandbox_limits)

        # Record in observer
        self.observer.record(EventType.TOOL_CALL, {
            "tool_name": "codemode_exec",
            "codemode": True,
            "sandboxed": True,
            "code_length": len(code),
            "output_length": len(sandbox_result.output),
            "error": sandbox_result.error,
        })

        if sandbox_result.timed_out:
            return f"Error: Code execution timed out after {self._sandbox_limits.timeout_seconds}s"

        if not sandbox_result.success:
            output = sandbox_result.output.strip()
            error = sandbox_result.error or "Unknown error"
            if output:
                return f"Output:\n{output}\n\nError:\n{error}"
            return f"Error: {error}"

        output = sandbox_result.output.strip()
        return output if output else "(no output)"

    async def _exec_codemode_inprocess(self, code: str) -> str:
        """Execute codemode code in-process with restricted builtins.

        No os/sys/subprocess allowed. Safe stdlib imports only.
        """
        import io
        import sys
        import contextlib
        import inspect as _inspect

        # Build the tools proxy object
        class ToolsBridge:
            """Proxy that routes tool.name() calls to real tool functions."""
            pass

        bridge = ToolsBridge()
        for tool in self.tools:
            # Handle both sync and async tool functions
            func = tool.function
            if _inspect.iscoroutinefunction(func):
                # Wrap async functions to run synchronously within exec
                import functools
                def make_sync_wrapper(async_fn):
                    @functools.wraps(async_fn)
                    def wrapper(*args, **kwargs):
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # We're inside an async context, use a thread
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                return pool.submit(
                                    asyncio.run, async_fn(*args, **kwargs)
                                ).result()
                        return loop.run_until_complete(async_fn(*args, **kwargs))
                    return wrapper
                setattr(bridge, tool.name, make_sync_wrapper(func))
            else:
                setattr(bridge, tool.name, func)

        # Safe import: allow standard library modules, block dangerous ones
        _SAFE_MODULES = frozenset({
            "math", "json", "re", "collections", "itertools", "functools",
            "string", "datetime", "decimal", "fractions", "statistics",
            "textwrap", "copy", "operator", "random", "hashlib", "base64",
            "urllib.parse", "html", "csv", "io",
        })
        _BLOCKED_MODULES = frozenset({
            "os", "sys", "subprocess", "shutil", "socket", "http",
            "ftplib", "smtplib", "ctypes", "signal", "multiprocessing",
            "threading", "importlib", "pathlib", "tempfile", "glob",
            "webbrowser", "code", "codeop", "compile", "compileall",
        })
        _import_cache = {}

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            top = name.split(".")[0]
            if top in _BLOCKED_MODULES or name in _BLOCKED_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed in codemode")
            if name not in _SAFE_MODULES and top not in _SAFE_MODULES:
                raise ImportError(
                    f"Import of '{name}' is not allowed in codemode. "
                    f"Allowed modules: {', '.join(sorted(_SAFE_MODULES))}"
                )
            if name not in _import_cache:
                import importlib
                _import_cache[name] = importlib.import_module(name)
            return _import_cache[name]

        # Execute with captured stdout
        namespace = {"tools": bridge, "__builtins__": {
            "__import__": _safe_import,
            "print": print, "len": len, "range": range, "str": str,
            "int": int, "float": float, "bool": bool, "list": list,
            "dict": dict, "tuple": tuple, "set": set, "sorted": sorted,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
            "isinstance": isinstance, "type": type, "hasattr": hasattr,
            "getattr": getattr, "True": True, "False": False, "None": None,
            "Exception": Exception, "ValueError": ValueError,
            "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "AttributeError": AttributeError,
            "ImportError": ImportError, "RuntimeError": RuntimeError,
        }}

        stdout_capture = io.StringIO()
        error = None

        try:
            timeout = self._sandbox_limits.timeout_seconds if self._sandbox_limits else 30.0

            def _run_code():
                with contextlib.redirect_stdout(stdout_capture):
                    exec(code, namespace)

            # Run in a thread with timeout to prevent blocking the event loop
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, _run_code),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            error = f"Code execution timed out after {timeout}s"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        output = stdout_capture.getvalue()

        # Record in observer
        self.observer.record(EventType.TOOL_CALL, {
            "tool_name": "codemode_exec",
            "codemode": True,
            "sandboxed": False,
            "code_length": len(code),
            "output_length": len(output),
            "error": error,
        })

        if error:
            return f"Output:\n{output}\n\nError:\n{error}" if output else f"Error: {error}"

        return output if output else "(no output)"

    async def _get_llm_session(self) -> aiohttp.ClientSession:
        """Get or create reusable aiohttp session for LLM calls."""
        if self._llm_session is None or self._llm_session.closed:
            self._llm_session = aiohttp.ClientSession()
        return self._llm_session

    async def _raw_llm_call(
        self,
        prompt: str,
        output_schema: Optional[Type] = None,
        images: Optional[List[str]] = None,
        max_retries: int = 2,
    ) -> str:
        """Make the raw HTTP call to the LLM API (no middleware/guardrails).
        
        Retries automatically on transient failures (429, 500, 502, 503, 504)
        with exponential backoff.
        """
        RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

        with self.observer.trace(
            EventType.LLM_REQUEST,
            {"model": self.model, "prompt_length": len(prompt)}
        ):
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Build message content (plain text or multi-part with images)
            content = build_content_parts(prompt, images)
            body: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "stream": False,
            }

            # Add structured output response_format if schema provided
            if output_schema is not None:
                body["response_format"] = build_response_format(output_schema)

            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    session = await self._get_llm_session()
                    async with session.post(
                        f"{self.api_base}/chat/completions",
                        json=body,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status in RETRYABLE_STATUSES and attempt < max_retries:
                            delay = (2 ** attempt) * 1.0  # 1s, 2s
                            logger.warning(
                                f"LLM API returned {response.status}, retrying in {delay}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue

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
                    last_error = e
                    if attempt < max_retries:
                        delay = (2 ** attempt) * 1.0
                        logger.warning(
                            f"LLM API connection error: {e}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"LLM API failed after {max_retries + 1} attempts: {e}")
                        raise

            # Should not reach here, but just in case
            raise last_error or Exception("LLM call failed after retries")

    async def _call_llm(
        self,
        prompt: str,
        output_schema: Optional[Type] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Make an LLM call with guardrails and middleware."""
        # Check input guardrails
        if self._input_guardrails:
            self._input_guardrails.check_or_raise(prompt, direction="input")

        # Check cache first if enabled
        if self.cache_enabled and self.cache:
            cached = await self.cache.get(prompt, self.model, temperature=self.temperature)
            if cached is not None:
                logger.debug(f"Cache hit for prompt (len={len(prompt)})")
                return cached

        # Run through middleware pipeline if present
        if self._middleware_chain:
            context = CallContext(
                prompt=prompt,
                namespace=self.model,
                temperature=self.temperature,
            )
            # Middleware wraps _raw_llm_call; pass schema/images via closure
            async def _call(p: str) -> str:
                return await self._raw_llm_call(p, output_schema=output_schema, images=images)
            full_response = await self._middleware_chain.execute(context, _call)
        else:
            full_response = await self._raw_llm_call(prompt, output_schema=output_schema, images=images)

        # Output guardrails with self-correction loop
        if self._output_guardrails:
            for correction in range(self._max_corrections + 1):
                failures = self._output_guardrails.check(full_response)
                if not failures:
                    break  # All guardrails passed

                if correction == self._max_corrections:
                    # Exhausted correction attempts, raise
                    self._output_guardrails.check_or_raise(full_response, direction="output")

                # Feed violation back to LLM for self-correction
                violation_feedback = "\n".join(f"- {f.reason}" for f in failures)
                correction_prompt = (
                    f"Your previous response was rejected for these reasons:\n{violation_feedback}\n\n"
                    f"Original request: {prompt}\n\n"
                    f"Regenerate a response that avoids these violations."
                )
                self.observer.record(EventType.SELF_CORRECTION, {
                    "attempt": correction + 1,
                    "max_attempts": self._max_corrections,
                    "violations": [f.reason for f in failures],
                })
                logger.info(
                    f"Self-correcting (attempt {correction + 1}/{self._max_corrections}): "
                    f"{', '.join(f.reason for f in failures)}"
                )
                full_response = await self._raw_llm_call(
                    correction_prompt, output_schema=output_schema, images=images
                )

        # Store in cache if enabled
        if self.cache_enabled and self.cache:
            await self.cache.set(prompt, self.model, full_response, temperature=self.temperature)

        return full_response

    async def _structured_output_with_retries(
        self,
        user_input: str,
        output_type: Type,
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Call the LLM and validate output against a Pydantic model, retrying on failure.

        Uses the same ``max_corrections`` limit configured via
        :meth:`with_guardrails` (default 0 = no retries).

        On each validation failure the error is formatted into a clear
        correction prompt and fed back to the LLM.

        Args:
            user_input: Original user prompt.
            output_type: Pydantic BaseModel class to validate against.
            images: Optional image sources.

        Returns:
            Dict with ``result`` (raw JSON), ``structured`` (validated
            instance), and ``attempts`` (number of LLM calls made).

        Raises:
            StructuredOutputError: If validation still fails after all
                retry attempts.
        """
        max_retries = self._max_corrections  # reuse guardrail setting
        last_error: Optional[Exception] = None
        raw = ""

        for attempt in range(max_retries + 1):
            if attempt == 0:
                # First attempt: normal call with schema hint
                raw = await self._call_llm(
                    user_input, output_schema=output_type, images=images,
                )
            else:
                # Retry: feed validation error back to LLM
                assert last_error is not None
                error_feedback = format_validation_error(last_error, output_type)
                correction_prompt = (
                    f"Your previous response was invalid:\n{error_feedback}\n\n"
                    f"Original request: {user_input}\n\n"
                    f"Regenerate a valid JSON response."
                )
                self.observer.record(EventType.SELF_CORRECTION, {
                    "attempt": attempt,
                    "max_attempts": max_retries,
                    "error": str(last_error)[:500],
                    "type": "structured_output",
                })
                logger.info(
                    f"Structured output retry (attempt {attempt}/{max_retries}): "
                    f"{str(last_error)[:200]}"
                )
                raw = await self._call_llm(
                    correction_prompt, output_schema=output_type, images=images,
                )

            try:
                validated = parse_and_validate(raw, output_type)
            except (ValueError, Exception) as exc:
                last_error = exc
                if attempt == max_retries:
                    # Exhausted retries
                    raise StructuredOutputError(
                        f"Structured output validation failed after "
                        f"{attempt + 1} attempt(s): {exc}",
                        raw_output=raw,
                        model=output_type,
                        attempts=attempt + 1,
                        last_error=str(exc),
                    ) from exc
                continue

            # Success
            result: Dict[str, Any] = {
                "result": raw,
                "structured": validated,
                "attempts": attempt + 1,
            }

            self.conversation_history.append({
                "user_input": user_input,
                "response": result,
                "turns": attempt + 1,
            })
            self.observer.record(
                EventType.INFERENCE_END,
                {
                    "query": user_input,
                    "turns": attempt + 1,
                    "structured": True,
                    "output_type": getattr(output_type, "__name__", str(output_type)),
                },
            )
            return result

        # Should be unreachable, but satisfy the type checker
        raise StructuredOutputError(  # pragma: no cover
            "Structured output validation failed",
            raw_output=raw,
            model=output_type,
            attempts=max_retries + 1,
            last_error=str(last_error),
        )

    async def _stream_llm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Stream LLM response chunks via SSE.

        Yields text chunks as they arrive from the OpenAI-compatible API.
        """
        with self.observer.trace(
            EventType.LLM_REQUEST,
            {"model": self.model, "prompt_length": len(prompt), "streaming": True}
        ):
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                content = build_content_parts(prompt, images)
                session = await self._get_llm_session()
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": content}],
                        "temperature": self.temperature,
                        "stream": True
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded or not decoded.startswith("data: "):
                            continue
                        data = decoded[6:]  # strip "data: "
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content_chunk = delta.get("content")
                            if content_chunk:
                                yield content_chunk
                        except json.JSONDecodeError:
                            continue

            except aiohttp.ClientError as e:
                logger.error(f"Error streaming LLM API: {str(e)}")
                raise

    async def stream(
        self,
        user_input: str,
        images: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Stream inference response as text chunks.

        Unlike infer(), this streams the raw LLM response without tool execution.
        Useful for conversational responses where you want real-time output.
        Respects input guardrails before streaming and output guardrails after.

        Args:
            user_input: Natural language query
            images: Optional list of image sources (URL, data URI, or local file path)

        Yields:
            Text chunks as they arrive from the LLM

        Example:
            >>> async for chunk in agent.stream("Explain quantum computing"):
            ...     print(chunk, end="", flush=True)
        """
        # Check input guardrails before streaming
        if self._input_guardrails:
            self._input_guardrails.check_or_raise(user_input, direction="input")

        self.observer.record(EventType.INFERENCE_START, {"query": user_input, "streaming": True})

        # Run middleware before hook
        if self._middleware_chain:
            ctx = CallContext(prompt=user_input, namespace=self.model, temperature=self.temperature)
            ctx = await self._middleware_chain.run_before(ctx)

        # Load deferred MCP tools on first use
        if self._pending_mcp_config:
            await self.with_mcp([self._pending_mcp_config])
            self._pending_mcp_config = None

        # Build prompt with context, memory, and skills
        active_skills = self._match_skills(user_input)
        context = self._build_turn_context(user_input, [], active_skills)

        full_response = []
        async for chunk in self._stream_llm(context, images=images):
            full_response.append(chunk)
            yield chunk

        # Store in memory
        collected = "".join(full_response)
        if self.memory_enabled:
            await asyncio.to_thread(
                self.memory.remember,
                content=f"User: {user_input}",
                memory_type='conversation',
                metadata={'role': 'user'},
                importance=0.5
            )
            await asyncio.to_thread(
                self.memory.remember,
                content=f"Agent: {collected}",
                memory_type='conversation',
                metadata={'role': 'agent', 'streaming': True},
                importance=0.6
            )

        # Run middleware after hook
        if self._middleware_chain:
            ctx = CallContext(prompt=user_input, namespace=self.model, temperature=self.temperature)
            collected = await self._middleware_chain.run_after(ctx, collected)

        # Check output guardrails on the collected response
        if self._output_guardrails:
            self._output_guardrails.check_or_raise(collected, direction="output")

        # Cache the full response
        if self.cache_enabled and self.cache:
            await self.cache.set(context, self.model, collected, temperature=self.temperature)

        self.conversation_history.append({
            "user_input": user_input,
            "response": {"result": collected, "streaming": True},
            "turns": 1
        })

        self.observer.record(
            EventType.INFERENCE_END,
            {"query": user_input, "streaming": True, "response_length": len(collected)}
        )

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
            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove opening fence (with optional language tag)
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            # Try to find JSON object if there's extra text around it
            if not cleaned.startswith("{"):
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                if start != -1 and end > start:
                    cleaned = cleaned[start:end]
            return json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "selected_tool": None,
                "parameters": {},
                "reasoning": f"Error parsing response: {str(e)}"
            }

    async def call(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a specific tool with given parameters.

        If a sandbox is enabled via with_sandbox(), tools execute in subprocess
        isolation. Otherwise they run in-process.

        Hooks and permission modes are applied in this order:

        1. DANGEROUS permission check (always)
        2. ``can_use_tool`` callback (if set)
        3. ``ask-writes`` mode check
        4. Pre-tool hooks
        5. Tool execution
        6. Post-tool hooks

        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool

        Returns:
            Tool execution result

        Raises:
            PermissionError: If tool is DANGEROUS and not explicitly allowed
            PermissionApprovalRequired: In 'ask-writes' mode for WRITE tools
        """
        for tool in self.tools:
            if tool.name == tool_name:
                # 1. Check DANGEROUS permission (unchanged default behaviour)
                if tool.permission == ToolPermission.DANGEROUS and not self._allow_dangerous:
                    self.observer.record(EventType.TOOL_BLOCKED, {
                        "tool_name": tool_name,
                        "permission": tool.permission.value,
                        "reason": "dangerous_not_allowed",
                    })
                    raise PermissionError(
                        f"Tool '{tool_name}' is marked as DANGEROUS and blocked. "
                        f"Use agent.with_permissions(allow_dangerous=True) to enable."
                    )

                # 2. can_use_tool callback (custom per-call decision)
                if self._can_use_tool is not None:
                    decision = await _call_maybe_async(
                        self._can_use_tool, tool_name, parameters,
                        {"tool": tool},
                    )
                    if decision == "deny":
                        self.observer.record(EventType.TOOL_BLOCKED, {
                            "tool_name": tool_name,
                            "reason": "can_use_tool_denied",
                        })
                        raise PermissionError(
                            f"Tool '{tool_name}' denied by can_use_tool callback."
                        )
                    if decision == "ask":
                        raise PermissionApprovalRequired(
                            tool_name=tool_name,
                            parameters=parameters,
                            reason=f"Tool '{tool_name}' requires approval (can_use_tool returned 'ask').",
                        )

                # 3. ask-writes mode: WRITE tools need approval
                if (
                    self._permission_mode == "ask-writes"
                    and tool.permission == ToolPermission.WRITE
                    and self._can_use_tool is None  # skip if callback already handled
                ):
                    raise PermissionApprovalRequired(
                        tool_name=tool_name,
                        parameters=parameters,
                        reason=f"Tool '{tool_name}' is a WRITE tool and requires approval in ask-writes mode.",
                    )

                # 4. Pre-tool hooks
                if self._hooks and self._hooks.pre_tool_hooks:
                    hook_result = await self._hooks.run_pre_tool(
                        tool_name, parameters, {"tool": tool},
                    )
                    if hook_result.action == HookAction.DENY:
                        self.observer.record(EventType.TOOL_BLOCKED, {
                            "tool_name": tool_name,
                            "reason": f"pre_tool_hook_denied: {hook_result.reason}",
                        })
                        # Return denial reason as a string so the model can
                        # see it and adjust, rather than raising an exception
                        # that would abort the entire agent loop.
                        return f"DENIED: {hook_result.reason}"
                    if hook_result.action == HookAction.MODIFY:
                        if hook_result.modified_params is not None:
                            parameters = hook_result.modified_params

                try:
                    with self.observer.trace(
                        EventType.TOOL_CALL,
                        {
                            "tool_name": tool_name,
                            "permission": tool.permission.value,
                            "sandboxed": self._sandbox is not None,
                            "params": parameters,
                        }
                    ):
                        # Sandboxed execution path
                        if self._sandbox is not None:
                            result = await self._call_sandboxed(tool, parameters)
                        else:
                            # In-process execution (default)
                            result = tool.function(**parameters)
                            if asyncio.iscoroutine(result):
                                result = await result

                        # 6. Post-tool hooks
                        if self._hooks and self._hooks.post_tool_hooks:
                            result = await self._hooks.run_post_tool(
                                tool_name, parameters, result,
                            )

                        # 7. Spotlight untrusted content
                        if tool.ingests_untrusted and isinstance(result, str):
                            result = spotlight_untrusted(result)

                        return result
                except Exception as e:
                    logger.error(f"Error calling tool {tool_name}: {str(e)}")
                    raise
        raise ValueError(f"Tool {tool_name} not found")

    async def _call_sandboxed(self, tool, parameters: Dict[str, Any]) -> Any:
        """Execute a tool in the sandbox backend.

        Serializes the tool function to source code, runs it in subprocess,
        and parses the JSON result.
        """
        import inspect as _inspect
        from ..runtime.sandbox import build_tool_code

        try:
            func_source = _inspect.getsource(tool.function)
        except (OSError, TypeError):
            # Can't get source (builtins, lambdas, etc.) -- fall back to in-process
            logger.warning(
                f"Tool '{tool.name}' cannot be sandboxed (no source). Running in-process."
            )
            result = tool.function(**parameters)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        # Dedent the source in case it's a nested function
        import textwrap
        func_source = textwrap.dedent(func_source)

        code = build_tool_code(func_source, tool.function.__name__, parameters)
        sandbox_result = await self._sandbox.execute(code, self._sandbox_limits)

        # Record sandbox execution details in observer
        self.observer.record(EventType.TOOL_CALL, {
            "tool_name": tool.name,
            "sandbox_exit_code": sandbox_result.exit_code,
            "sandbox_timed_out": sandbox_result.timed_out,
            "sandbox_stderr": sandbox_result.error,
        })

        if sandbox_result.timed_out:
            raise TimeoutError(
                f"Tool '{tool.name}' timed out after {self._sandbox_limits.timeout_seconds}s"
            )

        if not sandbox_result.success:
            raise RuntimeError(
                f"Tool '{tool.name}' failed in sandbox: {sandbox_result.error}"
            )

        # Parse JSON output from subprocess
        output = sandbox_result.output.strip()
        if output:
            try:
                parsed = json.loads(output)
                return parsed.get("result", output)
            except json.JSONDecodeError:
                return output
        return None

    def _match_skills(self, prompt: str,
                      embedding_provider: Optional[Any] = None,
                      semantic_threshold: float = 0.35) -> List[Skill]:
        """Determine which skills are relevant to the prompt.
        
        Uses keyword matching against skill descriptions to activate
        skills on-demand (Level 2 loading).  When *embedding_provider* is
        supplied, skills are first ranked by semantic similarity; keyword
        matching is used as fallback.
        
        Args:
            prompt: User input or task description
            embedding_provider: Optional EmbeddingProvider for semantic ranking.
            semantic_threshold: Minimum cosine-similarity to consider a skill
                matched when using semantic ranking (default: 0.35).
            
        Returns:
            List of matched Skill objects
        """
        # --- Try semantic matching first ---------------------------------
        if (embedding_provider is not None and _cosine_similarity is not None
                and self.skills):
            try:
                semantic_matched = self._semantic_match_skills(
                    prompt, embedding_provider, semantic_threshold
                )
                if semantic_matched:
                    return semantic_matched
            except Exception as e:
                logger.debug("Semantic skill matching failed, using keyword fallback: %s", e)

        # --- Keyword-based fallback -------------------------------------
        matched = []
        prompt_lower = prompt.lower()
        
        for skill in self.skills:
            desc_lower = skill.description.lower()
            name_lower = skill.name.lower()
            searchable = f"{name_lower} {desc_lower}"
            searchable_words = searchable.split()
            prompt_words = prompt_lower.split()
            
            # Match via substring or shared stem (5+ char common prefix)
            word_in_prompt = any(w in prompt_lower for w in searchable_words if len(w) > 3)
            word_in_desc = any(w in searchable for w in prompt_words if len(w) > 3)
            stem_match = any(
                min(len(sw), len(pw)) >= 5 and sw[:5] == pw[:5]
                for sw in searchable_words if len(sw) > 3
                for pw in prompt_words if len(pw) > 3
            )
            
            if word_in_prompt or word_in_desc or stem_match:
                matched.append(skill)
                logger.info(f"Matched skill: {skill.name} for prompt: {prompt[:50]}...")
        
        return matched

    def _semantic_match_skills(self, prompt: str, embedding_provider: Any,
                               threshold: float) -> List[Skill]:
        """Rank skills by cosine-similarity of their descriptions to the prompt.

        Returns an empty list on failure so the caller can fall back.
        """
        async def _embed_and_rank():
            prompt_vec = await embedding_provider.embed(prompt)
            scored = []
            for skill in self.skills:
                text = f"{skill.name} {skill.description}"
                skill_vec = await embedding_provider.embed(text)
                score = _cosine_similarity(prompt_vec, skill_vec)
                if score >= threshold:
                    scored.append((score, skill))
                    logger.info(
                        "Semantic matched skill: %s (score=%.3f) for prompt: %s...",
                        skill.name, score, prompt[:50]
                    )
            scored.sort(reverse=True, key=lambda x: x[0])
            return [skill for _, skill in scored]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _embed_and_rank()).result()
        else:
            return asyncio.run(_embed_and_rank())

    async def infer(
        self,
        user_input: str,
        output_schema: Optional[Type] = None,
        output_type: Optional[Type] = None,
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Infer tool and parameters from natural language input.

        Runs a multi-turn agentic loop:
        1. LLM evaluates which tool to use
        2. If search_tools called, activate found tools and continue
        3. If regular tool called, execute and check if more work needed
        4. If no tool selected (text response), return final result

        When codemode is enabled on the agent, the LLM writes Python code
        that calls tools via a typed API instead of making individual
        JSON tool calls.

        When output_schema is set and no tools are registered, runs a
        direct structured LLM call instead of the agentic tool loop.

        When output_type is set, the LLM output is validated against the
        given Pydantic BaseModel class and retried on validation failure
        (up to max_corrections times). Returns a validated Pydantic
        instance in the 'structured' key. Mutually exclusive with
        output_schema.

        Args:
            user_input: Natural language query
            output_schema: Optional Pydantic BaseModel class or JSON schema dict
                for structured output (raw JSON, no retry)
            output_type: Optional Pydantic BaseModel class for validated
                structured output with auto-retry on validation failure
            images: Optional list of image sources (URL, data URI, or local file path)

        Returns:
            Dict with tool_used, parameters, reasoning, and result.
            When output_schema or output_type is set, includes 'structured'
            key with validated instance.

        Raises:
            ValueError: If both output_schema and output_type are provided.
            StructuredOutputError: If output_type validation fails after all
                retry attempts.
        """
        if output_schema is not None and output_type is not None:
            raise ValueError(
                "output_schema and output_type are mutually exclusive. "
                "Use output_type for Pydantic-validated structured output "
                "with auto-retry, or output_schema for raw JSON schema."
            )
        # Track inference start
        self.observer.record(EventType.INFERENCE_START, {
            "query": user_input, "codemode": self.codemode,
        })

        # Apply worktree isolation if configured
        worktree_manager = None
        if self._worktree_config:
            from ..workflow.worktree import WorktreeManager
            worktree_manager = WorktreeManager(
                branch=self._worktree_config.get('branch'),
                cleanup=self._worktree_config.get('cleanup', True),
            )
            worktree_path = await worktree_manager.create(agent_name=self.name)
            if worktree_path:
                self.observer.record(
                    EventType.WORKTREE_CREATE,
                    metadata={"agent": self.name, "path": worktree_path}
                )

        try:
            return await self._infer_inner(user_input, output_schema, output_type, images)
        finally:
            if worktree_manager:
                await worktree_manager.remove()

    async def _infer_inner(
        self,
        user_input: str,
        output_schema: Optional[Type] = None,
        output_type: Optional[Type] = None,
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Inner inference logic, separated for worktree wrapping."""

        # Load deferred MCP tools on first inference
        if self._pending_mcp_config:
            await self.with_mcp([self._pending_mcp_config])
            self._pending_mcp_config = None

        # output_type fast path: validated Pydantic output with retry
        if output_type is not None and not self.tools:
            return await self._structured_output_with_retries(
                user_input, output_type, images=images,
            )

        # Structured output fast path: direct LLM call with schema
        if output_schema is not None and not self.tools:
            raw = await self._call_llm(user_input, output_schema=output_schema, images=images)
            validated = parse_and_validate(raw, output_schema)
            result = {
                "result": raw,
                "structured": validated,
            }
            self.conversation_history.append({
                "user_input": user_input,
                "response": result,
                "turns": 1,
            })
            self.observer.record(
                EventType.INFERENCE_END,
                {"query": user_input, "turns": 1, "structured": True}
            )
            return result

        # Store user input in memory
        if self.memory_enabled:
            await asyncio.to_thread(
                self.memory.remember,
                content=f"User: {user_input}",
                memory_type='conversation',
                metadata={'role': 'user'},
                importance=0.5
            )

        # ===== CODEMODE PATH =====
        if self.codemode and self.tools:
            prompt = self._build_codemode_prompt(user_input)
            max_corrections = 2
            last_code = None
            exec_result = None

            for attempt in range(max_corrections + 1):
                if attempt == 0:
                    response = await self._call_llm(prompt, images=images)
                else:
                    # Self-correction: feed error back to LLM
                    correction_prompt = (
                        f"{prompt}\n\n"
                        f"Your previous code failed with this error:\n"
                        f"```\n{exec_result}\n```\n\n"
                        f"Previous code:\n```python\n{last_code}\n```\n\n"
                        f"Fix the code and try again. Output ONLY the corrected Python code."
                    )
                    self.observer.record(EventType.SELF_CORRECTION, {
                        "attempt": attempt,
                        "max_attempts": max_corrections,
                        "error": exec_result[:500],
                    })
                    logger.info(
                        f"Codemode self-correcting (attempt {attempt}/{max_corrections})"
                    )
                    response = await self._call_llm(correction_prompt, images=images)

                # Extract code from LLM response (strip markdown fences)
                code = response.strip()
                if code.startswith("```"):
                    # Remove opening fence (with optional language tag like ```python)
                    code = code.split("\n", 1)[1] if "\n" in code else code[3:]
                if code.endswith("```"):
                    code = code[:-3]
                code = code.strip()
                last_code = code

                # Execute the code
                try:
                    exec_result = await self._exec_codemode(code)
                except Exception as e:
                    exec_result = f"Error: {str(e)}"

                # If no error in output, we're done
                if not exec_result.startswith("Error:") and "Error:\n" not in exec_result:
                    break

            final_response = {
                "tool_used": "codemode",
                "parameters": {"code": last_code},
                "reasoning": "LLM generated Python code to call tools",
                "result": exec_result,
                "codemode": True,
                "attempts": attempt + 1,
            }

            # Store in memory
            if self.memory_enabled:
                await asyncio.to_thread(
                    self.memory.remember,
                    content=f"Agent: Executed code via codemode - {exec_result[:200]}",
                    memory_type='conversation',
                    metadata={'role': 'agent', 'tool': 'codemode', 'code': last_code[:500]},
                    importance=0.6
                )

            self.conversation_history.append({
                "user_input": user_input,
                "response": final_response,
                "turns": attempt + 1,
            })

            self.observer.record(
                EventType.INFERENCE_END,
                {"query": user_input, "turns": attempt + 1, "codemode": True}
            )

            return final_response

        # ===== TRADITIONAL TOOL-CALLING PATH =====

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

            # Check for text_response (model is done, no more tool calls)
            if evaluation.get("text_response"):
                final_response = {
                    "text_response": evaluation["text_response"],
                    "history": turn_history,
                }
                break

            if not evaluation.get("selected_tool"):
                # No tool selected = task complete or no match
                if turn_history:
                    final_response = turn_history[-1]
                else:
                    final_response = {"error": "No appropriate tool found"}
                break

            tool_name = evaluation["selected_tool"]
            parameters = evaluation["parameters"]

            # Execute the tool call(s)
            # Support parallel tool calls if the model returns additional_tools
            additional_tools = evaluation.get("additional_tools", [])
            if additional_tools:
                # Parallel execution with asyncio.gather
                async def _safe_call(name, params):
                    try:
                        return name, params, await self.call(name, params), None
                    except Exception as e:
                        return name, params, None, str(e)

                tasks = [_safe_call(tool_name, parameters)]
                for extra in additional_tools:
                    tasks.append(_safe_call(extra["tool"], extra.get("parameters", {})))

                results = await asyncio.gather(*tasks)

                for t_name, t_params, t_result, t_error in results:
                    turn_result = {
                        "turn": turn + 1,
                        "tool_used": t_name,
                        "parameters": t_params,
                        "reasoning": evaluation.get("reasoning", ""),
                        "result": t_result if t_error is None else f"Error: {t_error}",
                    }
                    turn_history.append(turn_result)
            else:
                # Single tool call (most common path)
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

            # search_tools is always a continuation signal
            if tool_name == "search_tools":
                continue

            # Multi-turn: continue looping — the model decides when to stop
            # by returning no tool or a text_response

        if final_response is None:
            final_response = {
                "error": f"Max turns ({self.max_turns}) reached",
                "history": turn_history,
            }
            # Include last turn result if available
            if turn_history:
                final_response["tool_used"] = turn_history[-1].get("tool_used")
                final_response["result"] = turn_history[-1].get("result")

        # Store agent response in memory
        if self.memory_enabled and "tool_used" in final_response:
            await asyncio.to_thread(
                self.memory.remember,
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

        # Apply context compaction if configured
        if self._context_config:
            from .context import compact_context, estimate_history_tokens
            current_tokens = estimate_history_tokens(self.conversation_history)
            if current_tokens > self._context_config.max_tokens * 0.8:  # 80% threshold
                self.conversation_history, stats = await compact_context(
                    self.conversation_history,
                    self._context_config,
                    llm_call=self._call_llm,
                    system_prompt=self.context,
                )
                if stats.tokens_saved > 0:
                    self.observer.record(
                        EventType.CONTEXT_COMPACTION,
                        {
                            "tokens_saved": stats.tokens_saved,
                            "turns_before": stats.total_turns,
                            "turns_after": len(self.conversation_history),
                        }
                    )
        
        # Track inference end
        self.observer.record(
            EventType.INFERENCE_END,
            {
                "query": user_input,
                "turns": len(turn_history),
                "tool_used": final_response.get('tool_used')
            }
        )

        # On-stop hooks
        if self._hooks and self._hooks.on_stop_hooks:
            final_response = await self._hooks.run_on_stop(final_response)

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
        from ..workflow.ralph import RalphRunner, RalphConfig
        
        config = RalphConfig(
            prompt_file=prompt_file,
            max_iterations=max_iterations,
            timeout_minutes=timeout_minutes,
            checkpoint_every=checkpoint_every
        )
        
        runner = RalphRunner(self, config)
        return await runner.run(on_iteration=on_iteration)

    # ── Loop Engineering: Schedule ──────────────────────────────────

    def with_schedule(
        self,
        every: Optional[int] = None,
        cron: Optional[str] = None,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        ralph: Optional[str] = None,
        max_runs: Optional[int] = None,
    ) -> 'Agent':
        """Schedule this agent to run on a cadence.

        Supports interval (every N minutes) or cron expressions.
        Each run produces a finding in the triage inbox.

        Args:
            every: Run every N minutes (interval mode)
            cron: Cron expression, e.g. "0 9 * * *" (cron mode)
            prompt: Static prompt to run each cycle
            prompt_file: Path to prompt file (re-read each cycle)
            ralph: Path to PROMPT.md for Ralph autonomous loop
            max_runs: Maximum number of runs (None = unlimited)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("triage").with_schedule(every=30, prompt="Check issues")
            >>> agent = Agent("ops").with_schedule(cron="0 9 * * *", prompt_file="TRIAGE.md")
        """
        from ..workflow.schedule import Scheduler, ScheduleConfig, ScheduleStore

        if every is None and cron is None:
            raise ValueError("Must specify either 'every' (minutes) or 'cron' expression")

        config = ScheduleConfig(
            every=every,
            cron=cron,
            prompt=prompt,
            prompt_file=prompt_file,
            ralph=ralph,
            max_runs=max_runs,
        )

        if not hasattr(self, '_schedulers'):
            self._schedulers = []
        if not hasattr(self, '_schedule_store'):
            self._schedule_store = ScheduleStore()

        scheduler = Scheduler(self, config, store=self._schedule_store)
        self._schedulers.append(scheduler)

        logger.info(f"Added schedule {config.id} to agent {self.name}")
        return self

    async def start(self):
        """Start all configured schedules.

        Runs all schedules concurrently. Blocks until all schedules
        complete or are stopped via stop().

        Example:
            >>> agent.with_schedule(every=30, prompt="Check status")
            >>> await agent.start()
        """
        if not hasattr(self, '_schedulers') or not self._schedulers:
            raise RuntimeError("No schedules configured. Use with_schedule() first.")

        tasks = [asyncio.create_task(s.start()) for s in self._schedulers]
        await asyncio.gather(*tasks)

    def stop(self):
        """Stop all running schedules gracefully."""
        if hasattr(self, '_schedulers'):
            for scheduler in self._schedulers:
                scheduler.stop()

    def findings(self, status: str = "pending", limit: int = 50):
        """Get findings from scheduled runs.

        Args:
            status: Filter by status: "pending", "archived", "dismissed"
            limit: Maximum number of findings to return

        Returns:
            List of Finding dicts

        Example:
            >>> findings = agent.findings()
            >>> findings = agent.findings(status="archived")
        """
        if not hasattr(self, '_schedule_store'):
            return []
        from ..workflow.schedule import ScheduleStore
        return [f.to_dict() for f in self._schedule_store.get_findings(status=status, limit=limit)]

    # ── Loop Engineering: Sub-agents ────────────────────────────────

    def with_otel(
        self,
        endpoint: Optional[str] = None,
        service_name: Optional[str] = None,
    ) -> 'Agent':
        """Enable OpenTelemetry GenAI span export for this agent.

        Requires the ``[otel]`` extra to be installed::

            pip install agentu[otel]

        When the ``opentelemetry`` SDK is not available the method is
        a silent no-op — no errors are raised.

        Args:
            endpoint: OTLP HTTP exporter endpoint (default: env var or
                ``http://localhost:4318``).
            service_name: OTel resource ``service.name`` attribute
                (default: ``"agentu"``).

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_otel(
            ...     endpoint="http://localhost:4318",
            ...     service_name="my-agent",
            ... )
        """
        from ..middleware.otel import OTelExporter

        exporter = OTelExporter(
            service_name=service_name or "agentu",
            endpoint=endpoint,
            model=self.model,
            observer=self.observer,
        )
        exporter.attach(self.observer)
        self._otel_exporter = exporter
        return self

    def with_backend(
        self,
        backend: Union[str, Any],
    ) -> 'Agent':
        """Set the key-value storage backend for sessions, checkpoints, and memory.

        Args:
            backend: Either a Redis URL string (``redis://host:6379/0``)
                or a :class:`StorageBackend` instance.

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_backend("redis://localhost:6379")
            >>> # or with a custom backend
            >>> agent = Agent("bot").with_backend(MyCustomBackend())
        """
        if isinstance(backend, str):
            # Treat as Redis URL — lazy-create on first use
            self._backend_url = backend
            self._storage_backend = None  # Created async on first access
        else:
            self._storage_backend = backend
            self._backend_url = None
        return self

    def with_vectors(
        self,
        backend: Union[str, Any],
        dimension: int = 384,
    ) -> 'Agent':
        """Set the vector storage backend for semantic search.

        Args:
            backend: Either a PostgreSQL DSN (``postgresql://host/db``)
                or a :class:`VectorBackend` instance.
            dimension: Embedding dimension (default: 384 for MiniLM).

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("bot").with_vectors("postgresql://localhost/agentu")
            >>> # or with a custom backend
            >>> from agentu.storage import InMemoryVectorBackend
            >>> agent = Agent("bot").with_vectors(InMemoryVectorBackend())
        """
        if isinstance(backend, str):
            self._vector_dsn = backend
            self._vector_dimension = dimension
            self._vector_backend = None  # Created async on first access
        else:
            self._vector_backend = backend
            self._vector_dsn = None
        return self

    def with_subagents(
        self,
        agents: Union[str, List[Dict[str, Any]]],
    ) -> 'Agent':
        """Configure sub-agents for delegation.

        Sub-agents are lightweight agent roles (maker, checker) that
        can be composed into structured patterns like maker-checker.

        Args:
            agents: Either a path to .agents/ directory containing
                    YAML/JSON configs, or a list of dicts with
                    sub-agent definitions.

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("lead").with_subagents([
            ...     {"name": "coder", "instructions": "Write code.", "role": "maker"},
            ...     {"name": "reviewer", "instructions": "Review code.", "role": "checker"},
            ... ])
            >>> agent = Agent("lead").with_subagents(".agents/")
        """
        from ..workflow.subagent import load_subagent_configs

        self._subagent_configs = load_subagent_configs(agents)
        logger.info(f"Loaded {len(self._subagent_configs)} sub-agent configs")
        return self

    async def delegate(
        self,
        task: str,
        max_corrections: int = 1,
        judges: int = 1,
    ) -> Dict[str, Any]:
        """Delegate a task to sub-agents using maker-checker pattern.

        The maker sub-agent executes the task, then the checker reviews
        the output. If rejected, the maker gets correction attempts.

        When ``judges > 1``, multiple checker instances vote on the
        maker output using a panel consensus model.  The output is
        approved only when a majority of judges approve.

        Args:
            task: The task to delegate
            max_corrections: Max correction attempts on rejection (default: 1)
            judges: Number of checker instances to use as a judge panel
                (default: 1 — single checker, original behaviour).

        Returns:
            Dict with: result, review, approved, corrections, maker, checker

        Example:
            >>> result = await agent.delegate("Refactor the auth module")
            >>> if result["approved"]:
            ...     print("Changes approved by reviewer")
            >>> # Use a panel of 3 judges for higher confidence
            >>> result = await agent.delegate("Refactor auth", judges=3)
        """
        from ..workflow.subagent import _build_subagent, run_maker_checker, run_judge_panel

        if not hasattr(self, '_subagent_configs') or not self._subagent_configs:
            raise RuntimeError("No sub-agents configured. Use with_subagents() first.")

        # Build agent instances from configs
        makers = []
        checkers = []
        for config in self._subagent_configs:
            agent = _build_subagent(config, self)

            # If worktree is enabled, create isolated worktrees
            if self._worktree_config:
                from ..workflow.worktree import WorktreeManager
                wt = WorktreeManager(
                    cleanup=self._worktree_config.get('cleanup', True)
                )
                path = await wt.create(agent_name=config.name)
                if path:
                    self.observer.record(
                        EventType.WORKTREE_CREATE,
                        metadata={"agent": config.name, "path": path}
                    )

            if config.role == "checker":
                checkers.append(agent)
            else:
                makers.append(agent)

        if judges > 1 and checkers:
            return await run_judge_panel(
                task=task,
                makers=makers,
                checkers=checkers,
                num_judges=judges,
                max_corrections=max_corrections,
                parent_observer=self.observer,
            )

        return await run_maker_checker(
            task=task,
            makers=makers,
            checkers=checkers,
            max_corrections=max_corrections,
            parent_observer=self.observer,
        )

    async def best_of(
        self,
        n: int,
        task: str,
        judge: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run N instances of this agent in parallel and pick the best.

        Each instance runs the same task independently (using worktree
        isolation when configured).  A judge agent then evaluates all
        results and selects the best one.

        Args:
            n: Number of parallel instances to run.
            task: The task to run.
            judge: Optional Agent instance to judge results.  If
                ``None``, the first result is returned (no judging).

        Returns:
            Dict with ``result``, ``all_results``, ``chosen_index``,
            and ``judge_reasoning``.

        Example:
            >>> result = await agent.best_of(3, "Write a haiku about code")
        """
        from ..workflow.subagent import _build_subagent, run_best_of

        return await run_best_of(
            agent=self,
            n=n,
            task=task,
            judge=judge,
            parent_observer=self.observer,
        )


    # ── Context Management ──────────────────────────────────────────

    def with_context(
        self,
        max_tokens: int = 128_000,
        compaction: str = "auto",
        keep_recent: int = 5,
        max_result_chars: int = 2000,
    ) -> 'Agent':
        """Configure context management and compaction.

        Prevents context window overflow in long-running loops by
        applying tiered compaction strategies:
          1. Truncate old tool results
          2. LLM-summarize older turns
          3. Keep system prompt + recent turns intact

        Args:
            max_tokens: Approximate max tokens for context (default: 128K)
            compaction: Strategy - 'none', 'auto', 'truncate', 'summarize'
            keep_recent: Number of recent turns to always preserve
            max_result_chars: Max chars per tool result before truncation

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("assistant").with_context(
            ...     max_tokens=100_000,
            ...     compaction="auto",
            ... )
        """
        from .context import ContextConfig, CompactionMode
        self._context_config = ContextConfig(
            max_tokens=max_tokens,
            compaction=CompactionMode(compaction),
            keep_recent=keep_recent,
            max_result_chars=max_result_chars,
        )
        logger.info(f"Context management enabled: max_tokens={max_tokens}, compaction={compaction}")
        return self

    # ── Loop Engineering: Worktree ──────────────────────────────────

    def with_worktree(
        self,
        branch: Optional[str] = None,
        cleanup: bool = True,
    ) -> 'Agent':
        """Enable git worktree isolation for this agent.

        When enabled, infer() and delegate() run in isolated git
        worktrees so parallel agents don't collide.

        Args:
            branch: Branch name (default: auto-generated)
            cleanup: Auto-remove worktree after use (default: True)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("builder").with_worktree()
            >>> result = await agent.infer("Refactor auth")
        """
        self._worktree_config = {
            "branch": branch,
            "cleanup": cleanup,
        }
        logger.info(f"Worktree isolation enabled for agent {self.name}")
        return self