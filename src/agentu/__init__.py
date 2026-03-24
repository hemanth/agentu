"""agentu - A flexible Python package for creating AI agents with customizable tools."""

from .agent import Agent
from .tools import Tool
from .skill import Skill, load_skill
from .search import SearchAgent, search_tool
from .mcp_config import MCPConfigLoader, load_mcp_servers
from .mcp_transport import MCPServerConfig, AuthConfig, TransportType
from .mcp_tool import MCPToolAdapter, MCPToolManager
from .memory import Memory, MemoryEntry, ShortTermMemory, LongTermMemory
from .memory_storage import MemoryStorage, JSONStorage, SQLiteStorage, create_storage
from .serve import serve, AgentServer
from .session import Session, SessionManager
from .eval import evaluate, EvalResult, FailedCase
from .ralph import ralph, ralph_resume, RalphRunner, RalphConfig
from .cache import LLMCache, CacheStats
from .cache_storage_backends import CacheStorageBackend, MemoryBackend, SQLiteBackend, RedisBackend, FilesystemBackend
from .cache_embeddings import EmbeddingProvider, LocalEmbedding, APIEmbedding, FakeEmbedding, cosine_similarity
from .cache_semantic import SemanticIndex
from .cache_tiered import TieredCache
from .cache_sync import CacheSync
from .workflow import Step, SequentialStep, ParallelStep, WorkflowCheckpoint, resume_workflow
from . import guardrails, middleware, observe

# Backward-compat re-exports (individual class imports still work)
from .guardrails import (
    Guardrail, GuardrailResult, GuardrailSet, GuardrailError,
    PII, ContentFilter, MaxLength, JSONSchema,
)
from .middleware import (
    Middleware, BaseMiddleware, MiddlewareChain, CallContext,
    CostTracker, LoggerMiddleware, RetryMiddleware,
)

__version__ = "1.13.0"
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
    "middleware",
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
]