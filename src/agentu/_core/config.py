"""Declarative agent configuration parser using Pydantic."""
from typing import List, Optional
from pydantic import BaseModel, Field

try:
    import yaml
except ImportError:
    yaml = None


class CacheConfig(BaseModel):
    preset: str = Field(default="smart", description="Caching preset (basic, smart, offline, distributed)")
    ttl: int = Field(default=3600, description="Time to live in seconds")


class MCPConfig(BaseModel):
    urls: List[str] = Field(default_factory=list, description="List of MCP server connection URLs")


class AgentConfig(BaseModel):
    """Pydantic model that defines the schema for declarative agent payloads."""
    name: str = Field(..., description="Namespace identity of the agent")
    model: str = Field(default="openai/gpt-4o", description="LiteLLM provider formatted string")
    system_prompt: Optional[str] = Field(default=None, description="System instructions baseline")
    
    skills: Optional[List[str]] = Field(default_factory=list, description="Array of GitHub skill URIs")
    notify: Optional[List[str]] = Field(default_factory=list, description="Array of Apprise webhook URLs")
    
    mcp: Optional[MCPConfig] = None
    cache: Optional[CacheConfig] = None

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        """Reads a JSON or YAML file into the rigidly-typed AgentConfig structure.
        
        Raises:
            ImportError: if pyyaml is missing and a .yaml file is provided
            ValidationError: Built-in to pydantic if the schema deviates
        """
        import json
        
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                data = json.load(f)
            else:
                if yaml is None:
                    raise ImportError(
                        "The 'PyYAML' library is required to load .yaml configs. "
                        "Please run `pip install agentu[yaml]`"
                    )
                data = yaml.safe_load(f)
            
        return cls(**data)
