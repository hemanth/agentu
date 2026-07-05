"""agentu - A flexible Python package for creating AI agents with customizable tools."""

# ── Core (always available) ──────────────────────────────────────
from ._core.agent import Agent
from ._core.tools import Tool, ToolPermission
from ._core.hooks import HookAction, HookResult, HookSet, PermissionApprovalRequired
from ._core.config import AgentConfig
from ._core.structured import (
    pydantic_to_json_schema, build_response_format,
    parse_and_validate, format_validation_error, StructuredOutputError,
)
from ._core.multimodal import build_content_parts, resolve_image, detect_mime_type

# ── Skills & Discovery ───────────────────────────────────────────
from .skills.skill import Skill, load_skill
from .skills.loader import SkillLoader, parse_skill_md_frontmatter, load_skill_from_md
from .skills.search import SearchAgent, search_tool
from .discovery import discover_rules, discover_and_format_rules

# ── MCP ──────────────────────────────────────────────────────────
from .mcp.config import MCPConfigLoader, load_mcp_servers
from .mcp.transport import MCPServerConfig, AuthConfig, TransportType
from .mcp.tool import MCPToolAdapter, MCPToolManager

# ── Memory ───────────────────────────────────────────────────────
from .memory.memory import Memory, MemoryEntry, ShortTermMemory, LongTermMemory
from .memory.storage import MemoryStorage, JSONStorage, SQLiteStorage, create_storage

# ── Runtime ──────────────────────────────────────────────────────
from .runtime.serve import serve, AgentServer, create_server
from .runtime.session import Session, SessionManager
from .runtime.sandbox import SubprocessSandbox, SandboxLimits, SandboxResult, SandboxBackend

# ── Eval ─────────────────────────────────────────────────────────
from .eval.eval import evaluate, EvalResult, FailedCase

# ── Workflow ─────────────────────────────────────────────────────
from .workflow.ralph import ralph, ralph_resume, RalphRunner, RalphConfig
from .workflow.workflow import Step, SequentialStep, ParallelStep, WorkflowCheckpoint, resume_workflow
from .workflow.schedule import Scheduler, ScheduleConfig, ScheduleStore, Finding, CronParser
from .workflow.subagent import SubAgentConfig, load_subagent_configs, run_maker_checker
from .workflow.worktree import WorktreeManager

# ── Cache ────────────────────────────────────────────────────────
from .cache.cache import LLMCache, CacheStats
from .cache.storage import CacheStorageBackend, MemoryBackend, SQLiteBackend, FilesystemBackend
from .cache.embeddings import EmbeddingProvider, FakeEmbedding, cosine_similarity
from .cache.semantic import SemanticIndex
from .cache.tiered import TieredCache
from .cache.sync import CacheSync

# ── Middleware ───────────────────────────────────────────────────
from .middleware import guardrails as guardrails
from .middleware import observe as observe
from .middleware.guardrails import (
    Guardrail, GuardrailResult, GuardrailSet, GuardrailError,
    PII, ContentFilter, MaxLength, JSONSchema,
)
from .middleware.middleware import (
    Middleware, BaseMiddleware, MiddlewareChain, CallContext,
    CostTracker, LoggerMiddleware, RetryMiddleware,
)
from .middleware.notify import NotifyMiddleware

# ── Storage (core protocols + in-memory, always available) ──────
from .storage import (
    StorageBackend, VectorBackend,
    InMemoryBackend, InMemoryVectorBackend,
)

# ── Optional dependencies (lazy imports, won't crash if missing) ─
# These are loaded on first access via __getattr__ below.
# Users can still do `from agentu import RedisStorageBackend` etc.

_LAZY_IMPORTS = {
    # Redis (requires: pip install agentu[redis])
    "RedisBackend": (".cache.storage", "RedisBackend"),
    "RedisSessionStore": (".runtime.redis_backend", "RedisSessionStore"),
    "RedisStorageBackend": (".storage", "RedisStorageBackend"),
    "TaskQueue": (".runtime.tasks", "TaskQueue"),
    "TaskStatus": (".runtime.tasks", "TaskStatus"),
    "TaskInfo": (".runtime.tasks", "TaskInfo"),
    # LanceDB (requires: pip install agentu[vectors])
    "LanceDBBackend": (".storage", "LanceDBBackend"),
    # Embeddings (requires: pip install agentu[semantic])
    "LocalEmbedding": (".cache.embeddings", "LocalEmbedding"),
    "APIEmbedding": (".cache.embeddings", "APIEmbedding"),
}


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        # Cache it so subsequent accesses are fast
        globals()[name] = value
        return value
    raise AttributeError(f"module 'agentu' has no attribute {name!r}")


__version__ = "1.20.0"
__all__ = [
    # Core
    "Agent", "AgentConfig", "Tool", "ToolPermission",
    "HookAction", "HookResult", "HookSet", "PermissionApprovalRequired",
    # Skills
    "Skill", "load_skill", "SkillLoader",
    "parse_skill_md_frontmatter", "load_skill_from_md",
    "SearchAgent", "search_tool",
    "discover_rules", "discover_and_format_rules",
    # MCP
    "MCPConfigLoader", "load_mcp_servers",
    "MCPServerConfig", "AuthConfig", "TransportType",
    "MCPToolAdapter", "MCPToolManager",
    # Memory
    "Memory", "MemoryEntry", "ShortTermMemory", "LongTermMemory",
    "MemoryStorage", "JSONStorage", "SQLiteStorage", "create_storage",
    # Runtime
    "serve", "AgentServer", "create_server",
    "Session", "SessionManager",
    "SubprocessSandbox", "SandboxLimits", "SandboxResult", "SandboxBackend",
    # Eval
    "evaluate", "EvalResult", "FailedCase",
    # Workflow
    "ralph", "ralph_resume", "RalphRunner", "RalphConfig",
    "Step", "SequentialStep", "ParallelStep",
    "WorkflowCheckpoint", "resume_workflow",
    "Scheduler", "ScheduleConfig", "ScheduleStore", "Finding", "CronParser",
    "SubAgentConfig", "load_subagent_configs", "run_maker_checker",
    "WorktreeManager",
    # Cache
    "LLMCache", "CacheStats",
    "CacheStorageBackend", "MemoryBackend", "SQLiteBackend", "FilesystemBackend",
    "EmbeddingProvider", "FakeEmbedding", "cosine_similarity",
    "SemanticIndex", "TieredCache", "CacheSync",
    # Middleware
    "observe", "guardrails",
    "Guardrail", "GuardrailResult", "GuardrailSet", "GuardrailError",
    "PII", "ContentFilter", "MaxLength", "JSONSchema",
    "Middleware", "BaseMiddleware", "MiddlewareChain", "CallContext",
    "CostTracker", "LoggerMiddleware", "RetryMiddleware", "NotifyMiddleware",
    # Structured Output
    "pydantic_to_json_schema", "build_response_format",
    "parse_and_validate", "format_validation_error", "StructuredOutputError",
    # Multi-modal
    "build_content_parts", "resolve_image", "detect_mime_type",
    # Storage
    "StorageBackend", "VectorBackend",
    "InMemoryBackend", "InMemoryVectorBackend",
    # Lazy (optional)
    "RedisBackend", "RedisSessionStore", "RedisStorageBackend",
    "TaskQueue", "TaskStatus", "TaskInfo",
    "LanceDBBackend",
    "LocalEmbedding", "APIEmbedding",
]