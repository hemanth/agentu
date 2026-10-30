"""Zero-dependency declarative agent configuration parser."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class CacheConfig:
    preset: str = "smart"
    ttl: int = 3600


@dataclass
class MCPConfig:
    urls: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Declarative configuration schema for agentu payloads (JSON or YAML)."""

    name: str
    model: str = "openai/gpt-4o"
    system_prompt: Optional[str] = None
    rules: Optional[str] = None
    skills: Optional[List[str]] = field(default_factory=list)
    notify: Optional[List[str]] = field(default_factory=list)
    mcp: Optional[MCPConfig] = None
    cache: Optional[CacheConfig] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("AgentConfig.name must be a non-empty string")
        if isinstance(self.mcp, dict):
            self.mcp = MCPConfig(**{k: v for k, v in self.mcp.items() if k in {"urls"}})
        if isinstance(self.cache, dict):
            self.cache = CacheConfig(**{k: v for k, v in self.cache.items() if k in {"preset", "ttl"}})
        if self.cache is not None and not isinstance(self.cache.ttl, int):
            self.cache.ttl = int(self.cache.ttl)

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        """Read a JSON or YAML file into a validated AgentConfig instance."""
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                data: Dict[str, Any] = json.load(f)
            else:
                if yaml is None:
                    raise ImportError(
                        "The 'PyYAML' library is required to load .yaml configs. "
                        "Please run `pip install agentu[yaml]`"
                    )
                data = yaml.safe_load(f)

        if not isinstance(data, dict) or "name" not in data:
            raise ValueError(f"Invalid agent configuration in {path}: missing required 'name' field")

        allowed = {"name", "model", "system_prompt", "rules", "skills", "notify", "mcp", "cache"}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)
