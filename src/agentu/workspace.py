"""Workspace loader for agentu.

Provides :func:`load_workspace` — the main entry point for loading an agent
from a ``.agentu/`` directory convention.  The workspace directory is expected
to contain an ``agent.yaml`` configuration file and, optionally, a ``tools/``
directory with Python tool modules and a ``context/`` directory with context
files.

Typical layout::

    .agentu/
    ├── agent.yaml
    ├── tools/
    │   ├── search.py
    │   └── file_ops.py
    └── context/
        ├── guidelines.md
        └── api-docs.md

Usage::

    from agentu.workspace import load_workspace

    config, tools, context = load_workspace('.agentu')
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._core.tools import Tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceConfig:
    """Parsed representation of ``agent.yaml``.

    Fields map directly to :class:`Agent` constructor kwargs, making it
    straightforward to hydrate an agent from a workspace directory.
    """

    name: str

    # Model / inference
    model: Optional[str] = None
    temperature: float = 0.7
    system_prompt: Optional[str] = None  # inline or loaded from file
    max_turns: int = 10
    api_base: str = "http://localhost:11434/v1"
    api_key: Optional[str] = None
    codemode: bool = False

    # Memory
    enable_memory: bool = True
    memory_path: Optional[str] = None
    short_term_size: int = 10

    # Tools
    tools_dir: Optional[str] = None  # path to tools/ directory

    # Context
    context_files: List[str] = field(default_factory=list)  # paths to context files

    # Storage
    backend_url: Optional[str] = None  # redis://...
    vectors_path: Optional[str] = None  # ./vectors

    # Hooks
    permissions: Optional[Dict[str, Any]] = None  # allow_dangerous, etc.

    # Cache
    cache: bool = False
    cache_ttl: int = 3600


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

def parse_agent_yaml(path: str) -> WorkspaceConfig:
    """Parse an ``agent.yaml`` file into a :class:`WorkspaceConfig`.

    All relative file paths found in the YAML are resolved relative to the
    directory containing the YAML file itself.

    Args:
        path: Absolute or relative path to the ``agent.yaml`` file.

    Returns:
        A fully populated :class:`WorkspaceConfig` instance.

    Raises:
        ImportError: If ``pyyaml`` is not installed.
        FileNotFoundError: If *path* does not exist.
        ValueError: If required fields (e.g. ``name``) are missing.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for workspace loading. "
            "Install it with: pip install agentu[yaml]"
        )

    yaml_path = Path(path).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"agent.yaml not found at {yaml_path}")

    workspace_dir = yaml_path.parent

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    # ---- helpers --------------------------------------------------------
    def _resolve(rel: str) -> str:
        """Resolve a path relative to the workspace directory."""
        return str((workspace_dir / rel).resolve())

    # ---- required fields ------------------------------------------------
    name = raw.get("name")
    if not name:
        raise ValueError("agent.yaml must include a 'name' field")

    # ---- system_prompt (inline string or file reference) ----------------
    system_prompt: Optional[str] = None
    sp_raw = raw.get("system_prompt")
    if isinstance(sp_raw, dict) and "file" in sp_raw:
        prompt_path = Path(_resolve(sp_raw["file"]))
        if prompt_path.is_file():
            system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning("system_prompt file not found: %s", prompt_path)
    elif isinstance(sp_raw, str):
        system_prompt = sp_raw

    # ---- memory ---------------------------------------------------------
    memory_raw: Dict[str, Any] = raw.get("memory", {})
    enable_memory = memory_raw.get("enabled", True)
    memory_path_raw = memory_raw.get("path")
    memory_path = _resolve(memory_path_raw) if memory_path_raw else None
    short_term_size = memory_raw.get("short_term_size", 10)

    # ---- tools ----------------------------------------------------------
    tools_raw: Dict[str, Any] = raw.get("tools", {})
    tools_discover = tools_raw.get("discover")
    tools_dir = _resolve(tools_discover) if tools_discover else None

    # ---- context --------------------------------------------------------
    context_raw: Dict[str, Any] = raw.get("context", {})
    context_files_raw: List[str] = context_raw.get("files", [])
    context_files = [_resolve(f) for f in context_files_raw]

    # ---- backend --------------------------------------------------------
    backend_raw: Dict[str, Any] = raw.get("backend", {})
    backend_url = backend_raw.get("storage")
    vectors_path_raw = backend_raw.get("vectors")
    vectors_path = _resolve(vectors_path_raw) if vectors_path_raw else None

    # ---- permissions ----------------------------------------------------
    permissions = raw.get("permissions") or None

    # ---- cache ----------------------------------------------------------
    cache_raw: Dict[str, Any] = raw.get("cache", {})
    cache_enabled = cache_raw.get("enabled", False)
    cache_ttl = cache_raw.get("ttl", 3600)

    return WorkspaceConfig(
        name=name,
        model=raw.get("model"),
        temperature=raw.get("temperature", 0.7),
        system_prompt=system_prompt,
        max_turns=raw.get("max_turns", 10),
        api_base=raw.get("api_base", "http://localhost:11434/v1"),
        api_key=raw.get("api_key"),
        codemode=raw.get("codemode", False),
        enable_memory=enable_memory,
        memory_path=memory_path,
        short_term_size=short_term_size,
        tools_dir=tools_dir,
        context_files=context_files,
        backend_url=backend_url,
        vectors_path=vectors_path,
        permissions=permissions,
        cache=cache_enabled,
        cache_ttl=cache_ttl,
    )


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def discover_tools(tools_dir: str) -> List[Tool]:
    """Scan a directory for Python files and create :class:`Tool` objects.

    Each ``.py`` file in *tools_dir* is dynamically imported.  All **public**
    functions (those not starting with ``_``) that have a docstring are wrapped
    in a :class:`Tool`.  Functions without docstrings are assumed to be
    internal helpers and are silently skipped.

    ``__init__.py`` and ``__pycache__`` entries are ignored.

    Args:
        tools_dir: Absolute path to the tools directory.

    Returns:
        A list of :class:`Tool` instances discovered from the directory.
    """
    tools_path = Path(tools_dir).resolve()
    if not tools_path.is_dir():
        logger.warning("Tools directory does not exist: %s", tools_path)
        return []

    discovered: List[Tool] = []

    for entry in sorted(tools_path.iterdir()):
        # Skip non-Python files, __init__.py, and __pycache__
        if entry.name == "__init__.py":
            continue
        if entry.name == "__pycache__":
            continue
        if entry.suffix != ".py":
            continue

        module_name = f"agentu_workspace_tools.{entry.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, str(entry))
            if spec is None or spec.loader is None:
                logger.warning("Could not create module spec for %s", entry)
                continue

            module = importlib.util.module_from_spec(spec)
            # Make the module importable by other workspace tools
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception:
            logger.exception("Failed to load tool module: %s", entry.name)
            continue

        # Scan module for public, documented functions
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue

            obj = getattr(module, attr_name)
            if not callable(obj):
                continue

            # Skip classes and objects that aren't plain functions
            if not hasattr(obj, "__code__"):
                continue

            # Require a docstring — undocumented functions are helpers
            if not obj.__doc__:
                continue

            try:
                tool = Tool(obj)
                discovered.append(tool)
                logger.info(
                    "Discovered tool '%s' from %s", tool.name, entry.name
                )
            except Exception:
                logger.exception(
                    "Failed to create Tool from %s.%s", entry.name, attr_name
                )

    logger.info("Discovered %d tool(s) from %s", len(discovered), tools_path)
    return discovered


# ---------------------------------------------------------------------------
# Context file loading
# ---------------------------------------------------------------------------

def load_context_files(file_paths: List[str]) -> str:
    """Read and concatenate context files with section headers.

    Each file is rendered as::

        --- context: <filename> ---
        <file contents>

    Files that do not exist are skipped with a warning.

    Args:
        file_paths: List of absolute paths to context files.

    Returns:
        A single string with all context files concatenated.
    """
    sections: List[str] = []

    for fpath in file_paths:
        p = Path(fpath)
        if not p.is_file():
            logger.warning("Context file not found, skipping: %s", p)
            continue

        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed to read context file: %s", p)
            continue

        header = f"--- context: {p.name} ---"
        sections.append(f"{header}\n{content}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_workspace(
    workspace_path: str,
) -> Tuple[WorkspaceConfig, List[Tool], str]:
    """Load a workspace from a directory containing ``agent.yaml``.

    This is the main entry point for workspace-based agent loading.  It:

    1. Resolves *workspace_path* to an absolute path.
    2. Parses ``agent.yaml`` via :func:`parse_agent_yaml`.
    3. Discovers tools from the configured ``tools.discover`` directory.
    4. Loads context files from the configured ``context.files`` list.

    Args:
        workspace_path: Path to the workspace directory (e.g. ``.agentu``).

    Returns:
        A 3-tuple of ``(config, tools, context_string)``.

    Raises:
        FileNotFoundError: If *workspace_path* or ``agent.yaml`` is missing.
    """
    ws = Path(workspace_path).resolve()
    if not ws.is_dir():
        raise FileNotFoundError(f"Workspace directory not found: {ws}")

    yaml_file = ws / "agent.yaml"
    if not yaml_file.is_file():
        raise FileNotFoundError(
            f"agent.yaml not found in workspace: {ws}"
        )

    logger.info("Loading workspace from %s", ws)

    # 1. Parse config
    config = parse_agent_yaml(str(yaml_file))

    # 2. Discover tools
    tools: List[Tool] = []
    if config.tools_dir:
        tools = discover_tools(config.tools_dir)

    # 3. Load context
    context = ""
    if config.context_files:
        context = load_context_files(config.context_files)

    logger.info(
        "Workspace loaded: name=%s, tools=%d, context_len=%d",
        config.name,
        len(tools),
        len(context),
    )

    return config, tools, context
