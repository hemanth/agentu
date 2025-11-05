"""AgentU - A flexible Python package for creating AI agents with customizable tools."""

from .agent import Agent
from .tools import Tool
from .search import SearchAgent, search_tool
from .mcp_config import MCPConfigLoader, load_mcp_servers
from .mcp_transport import MCPServerConfig, AuthConfig, TransportType
from .mcp_tool import MCPToolAdapter, MCPToolManager
from .memory import Memory, MemoryEntry, ShortTermMemory, LongTermMemory
from .memory_storage import MemoryStorage, JSONStorage, SQLiteStorage, create_storage
from .orchestrator import (
    Orchestrator,
    AgentRole,
    ExecutionMode,
    Task,
    Message,
    AgentCapability,
    create_specialized_agent
)

__version__ = "0.3.0"
__all__ = [
    "Agent",
    "Tool",
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
    "Orchestrator",
    "AgentRole",
    "ExecutionMode",
    "Task",
    "Message",
    "AgentCapability",
    "create_specialized_agent",
]