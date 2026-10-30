"""agentu - A harness-engineered AI agent runtime with tool isolation, self-correction, and permission scoping."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

# ── Core value types (instant load, zero network/SDK imports) ────
from ._core.result import InferResult
from ._core.tools import Tool, ToolPermission
from ._core.hooks import HookAction, HookResult, HookSet, PermissionApprovalRequired

if TYPE_CHECKING:
    from ._core.agent import Agent
    from ._core.structured import (
        pydantic_to_json_schema,
        build_response_format,
        parse_and_validate,
        format_validation_error,
        StructuredOutputError,
    )
    from .drivers import (
        BaseDriver,
        DriverResponse,
        GeneratedImage,
        ImageGenerationResponse,
        OpenAIDriver,
        GeminiDriver,
        ClaudeDriver,
        OllamaDriver,
        get_driver_for_model,
        is_auto_routed_model,
        detect_env_default_model,
    )
    from ._core.config import AgentConfig
    from ._core.multimodal import (
        build_content_parts,
        resolve_image,
        resolve_media,
        detect_mime_type,
        detect_media_kind,
        convert_media_to_markdown,
        detect_model_capabilities,
    )
    from .skills.skill import Skill, load_skill
    from .skills.loader import SkillLoader, parse_skill_md_frontmatter, load_skill_from_md
    from .skills.search import SearchAgent, search_tool
    from .discovery import discover_rules, discover_and_format_rules
    from .plugin.loader import PluginLoader
    from .mcp.config import MCPConfigLoader, load_mcp_servers
    from .mcp.transport import MCPServerConfig, AuthConfig, TransportType
    from .mcp.tool import MCPToolAdapter, MCPToolManager
    from .memory.memory import Memory, MemoryEntry, ShortTermMemory, LongTermMemory
    from .memory.storage import MemoryStorage, JSONStorage, SQLiteStorage, create_storage
    from .runtime.serve import serve, AgentServer, create_server
    from .runtime.session import Session, SessionManager
    from .runtime.sandbox import SubprocessSandbox, SandboxLimits, SandboxResult, SandboxBackend
    from .runtime.durable import AgentLock
    from .runtime.redis_backend import RedisSessionStore
    from .runtime.tasks import TaskQueue, TaskStatus, TaskInfo
    from .eval.eval import evaluate, EvalResult, FailedCase
    from .workflow.ralph import ralph, ralph_resume, RalphRunner, RalphConfig
    from .workflow.workflow import Step, SequentialStep, ParallelStep, WorkflowCheckpoint, resume_workflow
    from .workflow.schedule import Scheduler, ScheduleConfig, ScheduleStore, Finding, CronParser
    from .workflow.subagent import SubAgentConfig, load_subagent_configs, run_maker_checker
    from .workflow.worktree import WorktreeManager
    from .cache.cache import LLMCache, CacheStats
    from .cache.storage import CacheStorageBackend, MemoryBackend, SQLiteBackend, FilesystemBackend, RedisBackend
    from .cache.embeddings import EmbeddingProvider, FakeEmbedding, LocalEmbedding, APIEmbedding, cosine_similarity
    from .cache.semantic import SemanticIndex
    from .cache.tiered import TieredCache
    from .cache.sync import CacheSync
    from .middleware import guardrails as guardrails, observe as observe
    from .middleware.guardrails import (
        Guardrail, GuardrailResult, GuardrailSet, GuardrailError,
        PII, ContentFilter, MaxLength, JSONSchema,
    )
    from .middleware.middleware import (
        Middleware, BaseMiddleware, MiddlewareChain, CallContext,
        CostTracker, LoggerMiddleware, RetryMiddleware,
    )
    from .middleware.notify import NotifyMiddleware
    from .storage import (
        StorageBackend, VectorBackend,
        InMemoryBackend, InMemoryVectorBackend,
        RedisStorageBackend, LanceDBBackend,
    )

# ── On-demand module loading (zero startup cost, 100% API compatible) ─
_LAZY_IMPORTS = {
    # Core Agent, Structured & Drivers
    "Agent": ("._core.agent", "Agent"),
    "pydantic_to_json_schema": ("._core.structured", "pydantic_to_json_schema"),
    "build_response_format": ("._core.structured", "build_response_format"),
    "parse_and_validate": ("._core.structured", "parse_and_validate"),
    "format_validation_error": ("._core.structured", "format_validation_error"),
    "StructuredOutputError": ("._core.structured", "StructuredOutputError"),
    "BaseDriver": (".drivers", "BaseDriver"),
    "DriverResponse": (".drivers", "DriverResponse"),
    "GeneratedImage": (".drivers", "GeneratedImage"),
    "ImageGenerationResponse": (".drivers", "ImageGenerationResponse"),
    "OpenAIDriver": (".drivers", "OpenAIDriver"),
    "GeminiDriver": (".drivers", "GeminiDriver"),
    "ClaudeDriver": (".drivers", "ClaudeDriver"),
    "OllamaDriver": (".drivers", "OllamaDriver"),
    "get_driver_for_model": (".drivers", "get_driver_for_model"),
    "is_auto_routed_model": (".drivers", "is_auto_routed_model"),
    "detect_env_default_model": (".drivers", "detect_env_default_model"),
    # Config & Multimodal helpers
    "AgentConfig": ("._core.config", "AgentConfig"),
    "build_content_parts": ("._core.multimodal", "build_content_parts"),
    "resolve_image": ("._core.multimodal", "resolve_image"),
    "resolve_media": ("._core.multimodal", "resolve_media"),
    "detect_mime_type": ("._core.multimodal", "detect_mime_type"),
    "detect_media_kind": ("._core.multimodal", "detect_media_kind"),
    "convert_media_to_markdown": ("._core.multimodal", "convert_media_to_markdown"),
    "detect_model_capabilities": ("._core.multimodal", "detect_model_capabilities"),
    # Skills & Discovery
    "Skill": (".skills.skill", "Skill"),
    "load_skill": (".skills.skill", "load_skill"),
    "SkillLoader": (".skills.loader", "SkillLoader"),
    "parse_skill_md_frontmatter": (".skills.loader", "parse_skill_md_frontmatter"),
    "load_skill_from_md": (".skills.loader", "load_skill_from_md"),
    "SearchAgent": (".skills.search", "SearchAgent"),
    "search_tool": (".skills.search", "search_tool"),
    "discover_rules": (".discovery", "discover_rules"),
    "discover_and_format_rules": (".discovery", "discover_and_format_rules"),
    "PluginLoader": (".plugin.loader", "PluginLoader"),
    # MCP
    "MCPConfigLoader": (".mcp.config", "MCPConfigLoader"),
    "load_mcp_servers": (".mcp.config", "load_mcp_servers"),
    "MCPServerConfig": (".mcp.transport", "MCPServerConfig"),
    "AuthConfig": (".mcp.transport", "AuthConfig"),
    "TransportType": (".mcp.transport", "TransportType"),
    "MCPToolAdapter": (".mcp.tool", "MCPToolAdapter"),
    "MCPToolManager": (".mcp.tool", "MCPToolManager"),
    # Memory
    "Memory": (".memory.memory", "Memory"),
    "MemoryEntry": (".memory.memory", "MemoryEntry"),
    "ShortTermMemory": (".memory.memory", "ShortTermMemory"),
    "LongTermMemory": (".memory.memory", "LongTermMemory"),
    "MemoryStorage": (".memory.storage", "MemoryStorage"),
    "JSONStorage": (".memory.storage", "JSONStorage"),
    "SQLiteStorage": (".memory.storage", "SQLiteStorage"),
    "create_storage": (".memory.storage", "create_storage"),
    # Runtime & Server ([serve] extra for serve / AgentServer / create_server)
    "serve": (".runtime.serve", "serve"),
    "AgentServer": (".runtime.serve", "AgentServer"),
    "create_server": (".runtime.serve", "create_server"),
    "Session": (".runtime.session", "Session"),
    "SessionManager": (".runtime.session", "SessionManager"),
    "SubprocessSandbox": (".runtime.sandbox", "SubprocessSandbox"),
    "SandboxLimits": (".runtime.sandbox", "SandboxLimits"),
    "SandboxResult": (".runtime.sandbox", "SandboxResult"),
    "SandboxBackend": (".runtime.sandbox", "SandboxBackend"),
    "AgentLock": (".runtime.durable", "AgentLock"),
    # Eval
    "evaluate": (".eval.eval", "evaluate"),
    "EvalResult": (".eval.eval", "EvalResult"),
    "FailedCase": (".eval.eval", "FailedCase"),
    # Workflow
    "ralph": (".workflow.ralph", "ralph"),
    "ralph_resume": (".workflow.ralph", "ralph_resume"),
    "RalphRunner": (".workflow.ralph", "RalphRunner"),
    "RalphConfig": (".workflow.ralph", "RalphConfig"),
    "Step": (".workflow.workflow", "Step"),
    "SequentialStep": (".workflow.workflow", "SequentialStep"),
    "ParallelStep": (".workflow.workflow", "ParallelStep"),
    "WorkflowCheckpoint": (".workflow.workflow", "WorkflowCheckpoint"),
    "resume_workflow": (".workflow.workflow", "resume_workflow"),
    "Scheduler": (".workflow.schedule", "Scheduler"),
    "ScheduleConfig": (".workflow.schedule", "ScheduleConfig"),
    "ScheduleStore": (".workflow.schedule", "ScheduleStore"),
    "Finding": (".workflow.schedule", "Finding"),
    "CronParser": (".workflow.schedule", "CronParser"),
    "SubAgentConfig": (".workflow.subagent", "SubAgentConfig"),
    "load_subagent_configs": (".workflow.subagent", "load_subagent_configs"),
    "run_maker_checker": (".workflow.subagent", "run_maker_checker"),
    "WorktreeManager": (".workflow.worktree", "WorktreeManager"),
    # Cache
    "LLMCache": (".cache.cache", "LLMCache"),
    "CacheStats": (".cache.cache", "CacheStats"),
    "CacheStorageBackend": (".cache.storage", "CacheStorageBackend"),
    "MemoryBackend": (".cache.storage", "MemoryBackend"),
    "SQLiteBackend": (".cache.storage", "SQLiteBackend"),
    "FilesystemBackend": (".cache.storage", "FilesystemBackend"),
    "EmbeddingProvider": (".cache.embeddings", "EmbeddingProvider"),
    "FakeEmbedding": (".cache.embeddings", "FakeEmbedding"),
    "cosine_similarity": (".cache.embeddings", "cosine_similarity"),
    "SemanticIndex": (".cache.semantic", "SemanticIndex"),
    "TieredCache": (".cache.tiered", "TieredCache"),
    "CacheSync": (".cache.sync", "CacheSync"),
    # Middleware
    "guardrails": (".middleware", "guardrails"),
    "observe": (".middleware", "observe"),
    "Guardrail": (".middleware.guardrails", "Guardrail"),
    "GuardrailResult": (".middleware.guardrails", "GuardrailResult"),
    "GuardrailSet": (".middleware.guardrails", "GuardrailSet"),
    "GuardrailError": (".middleware.guardrails", "GuardrailError"),
    "PII": (".middleware.guardrails", "PII"),
    "ContentFilter": (".middleware.guardrails", "ContentFilter"),
    "MaxLength": (".middleware.guardrails", "MaxLength"),
    "JSONSchema": (".middleware.guardrails", "JSONSchema"),
    "Middleware": (".middleware.middleware", "Middleware"),
    "BaseMiddleware": (".middleware.middleware", "BaseMiddleware"),
    "MiddlewareChain": (".middleware.middleware", "MiddlewareChain"),
    "CallContext": (".middleware.middleware", "CallContext"),
    "CostTracker": (".middleware.middleware", "CostTracker"),
    "LoggerMiddleware": (".middleware.middleware", "LoggerMiddleware"),
    "RetryMiddleware": (".middleware.middleware", "RetryMiddleware"),
    "NotifyMiddleware": (".middleware.notify", "NotifyMiddleware"),
    # Storage
    "StorageBackend": (".storage", "StorageBackend"),
    "VectorBackend": (".storage", "VectorBackend"),
    "InMemoryBackend": (".storage", "InMemoryBackend"),
    "InMemoryVectorBackend": (".storage", "InMemoryVectorBackend"),
    # Optional extras (redis, lancedb, sentence-transformers)
    "RedisBackend": (".cache.storage", "RedisBackend"),
    "RedisSessionStore": (".runtime.redis_backend", "RedisSessionStore"),
    "RedisStorageBackend": (".storage", "RedisStorageBackend"),
    "TaskQueue": (".runtime.tasks", "TaskQueue"),
    "TaskStatus": (".runtime.tasks", "TaskStatus"),
    "TaskInfo": (".runtime.tasks", "TaskInfo"),
    "LanceDBBackend": (".storage", "LanceDBBackend"),
    "LocalEmbedding": (".cache.embeddings", "LocalEmbedding"),
    "APIEmbedding": (".cache.embeddings", "APIEmbedding"),
}


def __getattr__(name: str):
    """Lazy import for non-core submodules and optional dependencies."""
    if name == "__version__":
        try:
            ver = version("agentu")
        except PackageNotFoundError:  # Source tree used without an installed distribution.
            ver = "0+unknown"
        globals()["__version__"] = ver
        return ver
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'agentu' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__) | {"__version__"})

__all__ = [
    # Core
    "Agent", "AgentConfig", "Tool", "ToolPermission",
    "HookAction", "HookResult", "HookSet", "PermissionApprovalRequired",
    # Skills
    "Skill", "load_skill", "SkillLoader",
    "parse_skill_md_frontmatter", "load_skill_from_md",
    "SearchAgent", "search_tool", "PluginLoader",
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
    "Session", "SessionManager", "AgentLock",
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
    # Structured Output & Typed Generics
    "InferResult",
    "pydantic_to_json_schema", "build_response_format",
    "parse_and_validate", "format_validation_error", "StructuredOutputError",
    # Multi-modal
    "build_content_parts", "resolve_image", "resolve_media",
    "detect_mime_type", "detect_media_kind",
    "convert_media_to_markdown", "detect_model_capabilities",
    # Drivers
    "BaseDriver", "DriverResponse",
    "GeneratedImage", "ImageGenerationResponse",
    "OpenAIDriver", "GeminiDriver", "ClaudeDriver", "OllamaDriver",
    "get_driver_for_model", "is_auto_routed_model", "detect_env_default_model",
    # Storage
    "StorageBackend", "VectorBackend",
    "InMemoryBackend", "InMemoryVectorBackend",
    # Lazy (optional)
    "RedisBackend", "RedisSessionStore", "RedisStorageBackend",
    "TaskQueue", "TaskStatus", "TaskInfo",
    "LanceDBBackend",
    "LocalEmbedding", "APIEmbedding",
]
