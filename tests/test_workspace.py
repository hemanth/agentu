"""Tests for workspace module — Agent.from_workspace() and friends.

Covers: YAML parsing, tool discovery, context loading, workspace integration,
error handling, and edge cases.
"""

import os
import json
import asyncio
import textwrap
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from agentu import Agent


def _make_workspace(tmp_path, agent_yaml=None, tools=None, context=None):
    """Helper: create a minimal workspace directory structure.

    Args:
        tmp_path: pytest tmp_path fixture
        agent_yaml: YAML string for agent.yaml (or None to skip)
        tools: dict of {filename: source_code} for tools/ directory
        context: dict of {filename: content} for context/ directory

    Returns:
        Path to the workspace directory
    """
    ws = tmp_path / ".agentu"
    ws.mkdir()

    if agent_yaml is not None:
        (ws / "agent.yaml").write_text(agent_yaml)

    if tools:
        tools_dir = ws / "tools"
        tools_dir.mkdir()
        for filename, code in tools.items():
            (tools_dir / filename).write_text(code)

    if context:
        ctx_dir = ws / "context"
        ctx_dir.mkdir()
        for filename, content in context.items():
            (ctx_dir / filename).write_text(content)

    return str(ws)


# ── YAML Parsing Tests ────────────────────────────────────

class TestParseAgentYaml:
    """Tests for parse_agent_yaml function."""

    def test_minimal_config(self, tmp_path):
        """Minimal agent.yaml with only name."""
        from agentu.workspace import parse_agent_yaml
        ws = _make_workspace(tmp_path, agent_yaml="name: mybot\n")
        config = parse_agent_yaml(os.path.join(ws, "agent.yaml"))
        assert config.name == "mybot"
        assert config.model is None
        assert config.temperature == 0.7
        assert config.max_turns == 10
        assert config.enable_memory is True

    def test_full_config(self, tmp_path):
        """Full agent.yaml with all fields."""
        from agentu.workspace import parse_agent_yaml
        yaml_content = textwrap.dedent("""\
            name: researcher
            model: gemini-2.5-flash
            temperature: 0.3
            max_turns: 20
            codemode: true
            api_base: https://api.example.com/v1
            api_key: sk-test-key
            system_prompt: "You are a researcher."

            memory:
              enabled: true
              path: ./memory.db
              short_term_size: 20

            tools:
              discover: ./tools

            context:
              files:
                - ./context/readme.md
                - ./context/docs.md

            backend:
              storage: redis://localhost:6379
              vectors: ./vectors

            permissions:
              allow_dangerous: true

            cache:
              enabled: true
              ttl: 7200
        """)
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content)
        config = parse_agent_yaml(os.path.join(ws, "agent.yaml"))
        assert config.name == "researcher"
        assert config.model == "gemini-2.5-flash"
        assert config.temperature == 0.3
        assert config.max_turns == 20
        assert config.codemode is True
        assert config.api_base == "https://api.example.com/v1"
        assert config.api_key == "sk-test-key"
        assert config.system_prompt == "You are a researcher."
        assert config.enable_memory is True
        assert config.short_term_size == 20
        assert config.backend_url == "redis://localhost:6379"
        assert config.vectors_path is not None
        assert config.cache is True
        assert config.cache_ttl == 7200
        assert config.permissions == {"allow_dangerous": True}

    def test_missing_name_raises(self, tmp_path):
        """agent.yaml without name should raise ValueError."""
        from agentu.workspace import parse_agent_yaml
        ws = _make_workspace(tmp_path, agent_yaml="model: gpt-4\n")
        with pytest.raises(ValueError, match="name"):
            parse_agent_yaml(os.path.join(ws, "agent.yaml"))

    def test_system_prompt_from_file(self, tmp_path):
        """system_prompt can reference a file."""
        from agentu.workspace import parse_agent_yaml
        ws_path = tmp_path / ".agentu"
        ws_path.mkdir()
        (ws_path / "prompt.md").write_text("You are a helpful assistant.")
        yaml_content = textwrap.dedent("""\
            name: bot
            system_prompt:
              file: ./prompt.md
        """)
        (ws_path / "agent.yaml").write_text(yaml_content)
        config = parse_agent_yaml(str(ws_path / "agent.yaml"))
        assert config.system_prompt == "You are a helpful assistant."

    def test_file_not_found_raises(self, tmp_path):
        """Missing agent.yaml should raise FileNotFoundError."""
        from agentu.workspace import parse_agent_yaml
        with pytest.raises(FileNotFoundError):
            parse_agent_yaml(str(tmp_path / "nonexistent.yaml"))

    def test_memory_defaults(self, tmp_path):
        """Memory section can be omitted (defaults apply)."""
        from agentu.workspace import parse_agent_yaml
        ws = _make_workspace(tmp_path, agent_yaml="name: bot\n")
        config = parse_agent_yaml(os.path.join(ws, "agent.yaml"))
        assert config.enable_memory is True
        assert config.memory_path is None
        assert config.short_term_size == 10

    def test_memory_disabled(self, tmp_path):
        """Memory can be explicitly disabled."""
        from agentu.workspace import parse_agent_yaml
        yaml_content = textwrap.dedent("""\
            name: bot
            memory:
              enabled: false
        """)
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content)
        config = parse_agent_yaml(os.path.join(ws, "agent.yaml"))
        assert config.enable_memory is False

    def test_paths_resolved_relative_to_workspace(self, tmp_path):
        """Relative paths in YAML are resolved relative to workspace dir."""
        from agentu.workspace import parse_agent_yaml
        yaml_content = textwrap.dedent("""\
            name: bot
            tools:
              discover: ./tools
            context:
              files:
                - ./context/readme.md
            backend:
              vectors: ./vectors
        """)
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content)
        config = parse_agent_yaml(os.path.join(ws, "agent.yaml"))
        # Paths should be absolute (resolved from workspace dir)
        assert os.path.isabs(config.tools_dir)
        assert all(os.path.isabs(p) for p in config.context_files)
        assert os.path.isabs(config.vectors_path)


# ── Tool Discovery Tests ─────────────────────────────────

class TestDiscoverTools:
    """Tests for discover_tools function."""

    def test_discover_simple_tool(self, tmp_path):
        """Discover a simple function from a .py file."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "math_tool.py").write_text(textwrap.dedent('''\
            def add(x: int, y: int) -> int:
                """Add two numbers together."""
                return x + y
        '''))
        tools = discover_tools(str(tools_dir))
        assert len(tools) == 1
        assert tools[0].name == "add"
        assert "Add two numbers" in tools[0].description

    def test_discover_multiple_functions(self, tmp_path):
        """Multiple public functions in one file are all discovered."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "utils.py").write_text(textwrap.dedent('''\
            def greet(name: str) -> str:
                """Greet someone by name."""
                return f"Hello, {name}!"

            def farewell(name: str) -> str:
                """Say goodbye to someone."""
                return f"Goodbye, {name}!"
        '''))
        tools = discover_tools(str(tools_dir))
        names = {t.name for t in tools}
        assert names == {"greet", "farewell"}

    def test_skip_private_functions(self, tmp_path):
        """Functions starting with _ are skipped."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "helper.py").write_text(textwrap.dedent('''\
            def public_tool(x: int) -> int:
                """A public tool."""
                return _helper(x)

            def _helper(x: int) -> int:
                return x * 2
        '''))
        tools = discover_tools(str(tools_dir))
        assert len(tools) == 1
        assert tools[0].name == "public_tool"

    def test_skip_no_docstring(self, tmp_path):
        """Functions without docstrings are skipped."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "nodoc.py").write_text(textwrap.dedent('''\
            def has_doc(x: int) -> int:
                """This has a docstring."""
                return x

            def no_doc(x: int) -> int:
                return x
        '''))
        tools = discover_tools(str(tools_dir))
        assert len(tools) == 1
        assert tools[0].name == "has_doc"

    def test_skip_init_and_pycache(self, tmp_path):
        """__init__.py and __pycache__ are ignored."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "__init__.py").write_text("")
        cache_dir = tools_dir / "__pycache__"
        cache_dir.mkdir()
        (tools_dir / "real_tool.py").write_text(textwrap.dedent('''\
            def real(x: int) -> int:
                """A real tool."""
                return x
        '''))
        tools = discover_tools(str(tools_dir))
        assert len(tools) == 1
        assert tools[0].name == "real"

    def test_empty_directory(self, tmp_path):
        """Empty tools directory returns no tools."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        tools = discover_tools(str(tools_dir))
        assert tools == []

    def test_nonexistent_directory(self, tmp_path):
        """Non-existent tools directory returns empty list."""
        from agentu.workspace import discover_tools
        tools = discover_tools(str(tmp_path / "nonexistent"))
        assert tools == []

    def test_skip_non_py_files(self, tmp_path):
        """Non-.py files are ignored."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "readme.md").write_text("# Tools")
        (tools_dir / "config.json").write_text("{}")
        (tools_dir / "tool.py").write_text(textwrap.dedent('''\
            def real_tool() -> str:
                """The only real tool."""
                return "hello"
        '''))
        tools = discover_tools(str(tools_dir))
        assert len(tools) == 1

    def test_tool_is_callable(self, tmp_path):
        """Discovered tools should be callable."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "calc.py").write_text(textwrap.dedent('''\
            def multiply(x: int, y: int) -> int:
                """Multiply two numbers."""
                return x * y
        '''))
        tools = discover_tools(str(tools_dir))
        assert len(tools) == 1
        assert tools[0].function(3, 4) == 12

    def test_multiple_files(self, tmp_path):
        """Tools from multiple .py files are all discovered."""
        from agentu.workspace import discover_tools
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "a.py").write_text(textwrap.dedent('''\
            def tool_a() -> str:
                """Tool A."""
                return "a"
        '''))
        (tools_dir / "b.py").write_text(textwrap.dedent('''\
            def tool_b() -> str:
                """Tool B."""
                return "b"
        '''))
        tools = discover_tools(str(tools_dir))
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}


# ── Context Loading Tests ─────────────────────────────────

class TestLoadContextFiles:
    """Tests for load_context_files function."""

    def test_load_single_file(self, tmp_path):
        """Load a single context file."""
        from agentu.workspace import load_context_files
        ctx_file = tmp_path / "readme.md"
        ctx_file.write_text("# Project Overview\nThis is a test project.")
        result = load_context_files([str(ctx_file)])
        assert "Project Overview" in result
        assert "readme.md" in result

    def test_load_multiple_files(self, tmp_path):
        """Load multiple context files with section headers."""
        from agentu.workspace import load_context_files
        (tmp_path / "a.md").write_text("Content A")
        (tmp_path / "b.md").write_text("Content B")
        result = load_context_files([
            str(tmp_path / "a.md"),
            str(tmp_path / "b.md"),
        ])
        assert "Content A" in result
        assert "Content B" in result
        assert "a.md" in result
        assert "b.md" in result

    def test_skip_missing_files(self, tmp_path):
        """Missing files are skipped with a warning (not an error)."""
        from agentu.workspace import load_context_files
        (tmp_path / "exists.md").write_text("Real content")
        result = load_context_files([
            str(tmp_path / "exists.md"),
            str(tmp_path / "missing.md"),
        ])
        assert "Real content" in result
        # Should not crash, just skip the missing one

    def test_empty_list(self):
        """Empty file list returns empty string."""
        from agentu.workspace import load_context_files
        result = load_context_files([])
        assert result == ""


# ── Workspace Loading Tests ───────────────────────────────

class TestLoadWorkspace:
    """Tests for load_workspace function."""

    def test_load_minimal_workspace(self, tmp_path):
        """Load workspace with just agent.yaml."""
        from agentu.workspace import load_workspace
        ws = _make_workspace(tmp_path, agent_yaml="name: testbot\n")
        config, tools, context = load_workspace(ws)
        assert config.name == "testbot"
        assert tools == []
        assert context == ""

    def test_load_workspace_with_tools(self, tmp_path):
        """Load workspace with tools directory."""
        from agentu.workspace import load_workspace
        yaml_content = textwrap.dedent("""\
            name: toolbot
            tools:
              discover: ./tools
        """)
        tools = {
            "calc.py": textwrap.dedent('''\
                def add(x: int, y: int) -> int:
                    """Add two numbers."""
                    return x + y
            ''')
        }
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content, tools=tools)
        config, discovered_tools, context = load_workspace(ws)
        assert config.name == "toolbot"
        assert len(discovered_tools) == 1
        assert discovered_tools[0].name == "add"

    def test_load_workspace_with_context(self, tmp_path):
        """Load workspace with context files."""
        from agentu.workspace import load_workspace
        yaml_content = textwrap.dedent("""\
            name: ctxbot
            context:
              files:
                - ./context/readme.md
        """)
        context = {"readme.md": "# Important context\nThis is important."}
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content, context=context)
        config, tools, ctx_string = load_workspace(ws)
        assert "Important context" in ctx_string

    def test_missing_agent_yaml_raises(self, tmp_path):
        """Workspace without agent.yaml should raise FileNotFoundError."""
        from agentu.workspace import load_workspace
        ws = tmp_path / ".agentu"
        ws.mkdir()
        with pytest.raises(FileNotFoundError, match="agent.yaml"):
            load_workspace(str(ws))


# ── Agent.from_workspace Integration Tests ────────────────

class TestAgentFromWorkspace:
    """Integration tests for Agent.from_workspace()."""

    @pytest.mark.asyncio
    async def test_from_workspace_minimal(self, tmp_path):
        """Create agent from minimal workspace."""
        ws = _make_workspace(tmp_path, agent_yaml="name: wsbot\n")
        agent = await Agent.from_workspace(ws)
        assert agent.name == "wsbot"
        assert agent._workspace_path == ws

    @pytest.mark.asyncio
    async def test_from_workspace_with_model(self, tmp_path):
        """Workspace sets model correctly."""
        yaml_content = textwrap.dedent("""\
            name: smartbot
            model: gemini-2.5-flash
            temperature: 0.2
        """)
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content)
        agent = await Agent.from_workspace(ws)
        assert agent.name == "smartbot"
        assert agent.model == "gemini-2.5-flash"
        assert agent.temperature == 0.2

    @pytest.mark.asyncio
    async def test_from_workspace_with_tools(self, tmp_path):
        """Tools from workspace directory are registered on the agent."""
        yaml_content = textwrap.dedent("""\
            name: toolbot
            tools:
              discover: ./tools
        """)
        tools = {
            "math.py": textwrap.dedent('''\
                def add(x: int, y: int) -> int:
                    """Add two numbers."""
                    return x + y

                def subtract(x: int, y: int) -> int:
                    """Subtract y from x."""
                    return x - y
            ''')
        }
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content, tools=tools)
        agent = await Agent.from_workspace(ws)
        tool_names = {t.name for t in agent.tools}
        assert "add" in tool_names
        assert "subtract" in tool_names

    @pytest.mark.asyncio
    async def test_from_workspace_with_context(self, tmp_path):
        """Context files are loaded into agent.context."""
        yaml_content = textwrap.dedent("""\
            name: ctxbot
            context:
              files:
                - ./context/rules.md
        """)
        context = {"rules.md": "Always be polite."}
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content, context=context)
        agent = await Agent.from_workspace(ws)
        assert "Always be polite" in agent.context

    @pytest.mark.asyncio
    async def test_from_workspace_system_prompt_plus_context(self, tmp_path):
        """Inline system_prompt is combined with context files."""
        yaml_content = textwrap.dedent("""\
            name: bot
            system_prompt: "You are a researcher."
            context:
              files:
                - ./context/docs.md
        """)
        context = {"docs.md": "API documentation here."}
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content, context=context)
        agent = await Agent.from_workspace(ws)
        assert "You are a researcher" in agent.context
        assert "API documentation" in agent.context

    @pytest.mark.asyncio
    async def test_from_workspace_with_backend(self, tmp_path):
        """Backend URLs are passed through to the agent."""
        yaml_content = textwrap.dedent("""\
            name: bot
            backend:
              storage: redis://localhost:6379
              vectors: ./vectors
        """)
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content)
        agent = await Agent.from_workspace(ws)
        assert agent._backend_url == "redis://localhost:6379"
        assert agent._vector_dsn is not None

    @pytest.mark.asyncio
    async def test_from_workspace_memory_disabled(self, tmp_path):
        """Memory can be disabled via workspace config."""
        yaml_content = textwrap.dedent("""\
            name: bot
            memory:
              enabled: false
        """)
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content)
        agent = await Agent.from_workspace(ws)
        assert agent.memory_enabled is False

    @pytest.mark.asyncio
    async def test_from_workspace_codemode(self, tmp_path):
        """Codemode is set from workspace config."""
        yaml_content = textwrap.dedent("""\
            name: coder
            codemode: true
        """)
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content)
        agent = await Agent.from_workspace(ws)
        assert agent.codemode is True

    @pytest.mark.asyncio
    async def test_from_workspace_cache(self, tmp_path):
        """Cache settings are applied from workspace config."""
        yaml_content = textwrap.dedent("""\
            name: bot
            cache:
              enabled: true
              ttl: 1800
        """)
        ws = _make_workspace(tmp_path, agent_yaml=yaml_content)
        agent = await Agent.from_workspace(ws)
        assert agent.cache_enabled is True

    @pytest.mark.asyncio
    async def test_from_workspace_default_path(self, tmp_path, monkeypatch):
        """Default workspace path is .agentu/ in current directory."""
        # Create .agentu in tmp_path
        ws = tmp_path / ".agentu"
        ws.mkdir()
        (ws / "agent.yaml").write_text("name: defaultbot\n")
        monkeypatch.chdir(tmp_path)
        agent = await Agent.from_workspace()
        assert agent.name == "defaultbot"

