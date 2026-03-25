"""agentu - A flexible Python package for creating AI agents with customizable tools."""

from ._core.agent import Agent
from ._core.tools import Tool
from .skills.skill import Skill, load_skill
from .skills.search import SearchAgent, search_tool
from .mcp.config import MCPConfigLoader, load_mcp_servers
from .mcp.transport import MCPServerConfig, AuthConfig, TransportType
from .mcp.tool import MCPToolAdapter, MCPToolManager
from .memory.memory import Memory, MemoryEntry, ShortTermMemory, LongTermMemory
from .memory.storage import MemoryStorage, JSONStorage, SQLiteStorage, create_storage
from .runtime.serve import serve, AgentServer
from .runtime.session import Session, SessionManager
from .eval.eval import evaluate, EvalResult, FailedCase
from .workflow.ralph import ralph, ralph_resume, RalphRunner, RalphConfig
from .cache.cache import LLMCache, CacheStats
from .cache.storage import CacheStorageBackend, MemoryBackend, SQLiteBackend, RedisBackend, FilesystemBackend
from .cache.embeddings import EmbeddingProvider, LocalEmbedding, APIEmbedding, FakeEmbedding, cosine_similarity
from .cache.semantic import SemanticIndex
from .cache.tiered import TieredCache
from .cache.sync import CacheSync
from .workflow.workflow import Step, SequentialStep, ParallelStep, WorkflowCheckpoint, resume_workflow
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

__version__ = "1.14.0"
__all__ = [
    "Agent",
    "Tool",
    "Skill",
    "load_skill",
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
    # Structured Output
    "pydantic_to_json_schema",
    "build_response_format",
    "parse_and_validate",
    # Multi-modal
    "build_content_parts",
    "resolve_image",
    "detect_mime_type",
]