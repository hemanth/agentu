"""agentu - A flexible Python package for creating AI agents with customizable tools."""

from ._core.agent import Agent
from ._core.tools import Tool, ToolPermission
from ._core.hooks import HookAction, HookResult, HookSet, PermissionApprovalRequired
from .skills.skill import Skill, load_skill
from .skills.loader import SkillLoader, parse_skill_md_frontmatter, load_skill_from_md
from .skills.search import SearchAgent, search_tool
from .discovery import discover_rules, discover_and_format_rules
from .mcp.config import MCPConfigLoader, load_mcp_servers
from .mcp.transport import MCPServerConfig, AuthConfig, TransportType
from .mcp.tool import MCPToolAdapter, MCPToolManager
from .memory.memory import Memory, MemoryEntry, ShortTermMemory, LongTermMemory
from .memory.storage import MemoryStorage, JSONStorage, SQLiteStorage, create_storage
from .runtime.serve import serve, AgentServer
from .runtime.session import Session, SessionManager
from .runtime.sandbox import SubprocessSandbox, SandboxLimits, SandboxResult, SandboxBackend
from .eval.eval import evaluate, EvalResult, FailedCase
from .workflow.ralph import ralph, ralph_resume, RalphRunner, RalphConfig
from .cache.cache import LLMCache, CacheStats
from .cache.storage import CacheStorageBackend, MemoryBackend, SQLiteBackend, RedisBackend, FilesystemBackend
from .cache.embeddings import EmbeddingProvider, LocalEmbedding, APIEmbedding, FakeEmbedding, cosine_similarity
from .cache.semantic import SemanticIndex
from .cache.tiered import TieredCache
from .cache.sync import CacheSync
from .workflow.workflow import Step, SequentialStep, ParallelStep, WorkflowCheckpoint, resume_workflow
from .workflow.schedule import Scheduler, ScheduleConfig, ScheduleStore, Finding, CronParser
from .workflow.subagent import SubAgentConfig, load_subagent_configs, run_maker_checker
from .workflow.worktree import WorktreeManager
from ._core.structured import pydantic_to_json_schema, build_response_format, parse_and_validate
from ._core.multimodal import build_content_parts, resolve_image, detect_mime_type
from .middleware import guardrails as guardrails
from .middleware import observe as observe

# Backward-compat re-exports (individual class imports still work)
from .middleware.guardrails import (
    Guardrail, GuardrailResult, GuardrailSet, GuardrailError,
    PII, ContentFilter, MaxLength, JSONSchema,
)
from .middleware.middleware import (
    Middleware, BaseMiddleware, MiddlewareChain, CallContext,
    CostTracker, LoggerMiddleware, RetryMiddleware,
)
from .middleware.notify import NotifyMiddleware
from ._core.config import AgentConfig

__version__ = "1.20.0"
__all__ = [
    "Agent",
    "AgentConfig",
    "Tool",
    "Skill",
    "load_skill",
    "SkillLoader",
    "parse_skill_md_frontmatter",
    "load_skill_from_md",
    "discover_rules",
    "discover_and_format_rules",
    "SearchAgent",
    "search_tool",
    "MCPConfigLoader",
    "load_mcp_servers",
    "MCPServerConfig",
    "AuthConfig",
    "TransportType",
    "MCPToolAdapter",
    "MCPToolManager",
    "Memory",
    "MemoryEntry",
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryStorage",
    "JSONStorage",
    "SQLiteStorage",
    "create_storage",
    "serve",
    "AgentServer",
    "Session",
    "SessionManager",
    "evaluate",
    "EvalResult",
    "FailedCase",
    "ralph",
    "ralph_resume",
    "RalphRunner",
    "RalphConfig",
    "LLMCache",
    "CacheStats",
    "CacheStorageBackend",
    "MemoryBackend",
    "SQLiteBackend",
    "RedisBackend",
    "FilesystemBackend",
    "EmbeddingProvider",
    "LocalEmbedding",
    "APIEmbedding",
    "FakeEmbedding",
    "cosine_similarity",
    "SemanticIndex",
    "TieredCache",
    "CacheSync",
    "Step",
    "SequentialStep",
    "ParallelStep",
    "WorkflowCheckpoint",
    "resume_workflow",
    # Loop Engineering
    "Scheduler",
    "ScheduleConfig",
    "ScheduleStore",
    "Finding",
    "CronParser",
    "SubAgentConfig",
    "load_subagent_configs",
    "run_maker_checker",
    "WorktreeManager",
    "observe",
    "guardrails",
    # Guardrails
    "Guardrail",
    "GuardrailResult",
    "GuardrailSet",
    "GuardrailError",
    "PII",
    "ContentFilter",
    "MaxLength",
    "JSONSchema",
    # Middleware
    "Middleware",
    "BaseMiddleware",
    "MiddlewareChain",
    "CallContext",
    "CostTracker",
    "LoggerMiddleware",
    "RetryMiddleware",
    "NotifyMiddleware",
    # Structured Output
    "pydantic_to_json_schema",
    "build_response_format",
    "parse_and_validate",
    # Multi-modal
    "build_content_parts",
    "resolve_image",
    "detect_mime_type",
    # Hooks & Permissions
    "HookAction",
    "HookResult",
    "HookSet",
    "PermissionApprovalRequired",
]