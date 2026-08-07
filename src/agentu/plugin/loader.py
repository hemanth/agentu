"""Agent Plugins v1.0.0 specification loader for agentu.

Specification: https://agent-plugins.org/specification
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Closed schema allowed top-level keys in plugin.json
ALLOWED_MANIFEST_KEYS = {
    "$schema", "name", "version", "description",
    "author", "homepage", "repository", "license",
    "keywords", "extensions"
}


class PluginLoader:
    """Loader for Agent Plugins (v1.0.0 spec)."""

    def __init__(self, plugin_path: Union[str, Path]):
        self.plugin_root = Path(plugin_path).resolve()
        self.manifest: Dict[str, Any] = {}
        self.skills: List[Path] = []
        self.mcp_config: Optional[Path] = None

    def is_contained(self, path: Path) -> bool:
        """Verify path stays within plugin root (containment rule §4.1)."""
        try:
            resolved = path.resolve()
            return resolved == self.plugin_root or self.plugin_root in resolved.parents
        except Exception:
            return False

    def load(self) -> "PluginLoader":
        """Validate and discover plugin components according to Agent Plugins 1.0.0 spec."""
        if not self.plugin_root.exists() or not self.plugin_root.is_dir():
            raise FileNotFoundError(f"Plugin root directory does not exist: {self.plugin_root}")

        # 1. Manifest (§5)
        manifest_file = self.plugin_root / "plugin.json"
        if not manifest_file.exists() or not self.is_contained(manifest_file):
            raise FileNotFoundError(f"Missing or out-of-bounds plugin.json at {manifest_file}")

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse plugin.json: {e}")

        if not isinstance(self.manifest, dict):
            raise ValueError("plugin.json must be a JSON object")

        if "name" not in self.manifest or not isinstance(self.manifest["name"], str) or not self.manifest["name"].strip():
            raise ValueError("plugin.json missing required non-empty 'name' field")

        # Report unknown fields (do not crash, per spec §5.2)
        unknown_keys = set(self.manifest.keys()) - ALLOWED_MANIFEST_KEYS
        if unknown_keys:
            logger.warning(f"Plugin '{self.manifest['name']}' has unknown top-level keys in plugin.json: {unknown_keys}")

        # 2. Component discovery — Skills (§6 & §7.1)
        skills_dir = self.plugin_root / "skills"
        if skills_dir.exists() and skills_dir.is_dir() and self.is_contained(skills_dir):
            for skill_subdir in skills_dir.iterdir():
                if skill_subdir.is_dir() and self.is_contained(skill_subdir):
                    skill_file = skill_subdir / "SKILL.md"
                    if skill_file.exists() and skill_file.is_file() and self.is_contained(skill_file):
                        self.skills.append(skill_subdir)

        # 3. Component discovery — MCP Servers (§6 & §7.2)
        mcp_file = self.plugin_root / "mcp.json"
        if mcp_file.exists() and mcp_file.is_file() and self.is_contained(mcp_file):
            self.mcp_config = mcp_file

        return self
