"""Tests for AGENTS.md / CLAUDE.md auto-discovery."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from agentu.discovery import discover_rules, discover_and_format_rules
from agentu import Agent


# ── discover_rules ──────────────────────────────────────────────────


@pytest.fixture
def project_dir():
    """Create a temporary project directory."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp)


class TestDiscoverRules:
    """Tests for discover_rules function."""

    def test_discover_agents_md(self, project_dir):
        """AGENTS.md at the root should be discovered."""
        (project_dir / "AGENTS.md").write_text("# Project Rules\nUse type hints.")
        rules = discover_rules(start_dir=str(project_dir))
        assert len(rules) == 1
        assert "Use type hints" in rules[0]

    def test_discover_dot_agents_agents_md(self, project_dir):
        """Files in .agents/ should be discovered."""
        dot_agents = project_dir / ".agents"
        dot_agents.mkdir()
        (dot_agents / "AGENTS.md").write_text("# Rules\nFollow PEP 8.")
        rules = discover_rules(start_dir=str(project_dir))
        assert len(rules) == 1
        assert "Follow PEP 8" in rules[0]

    def test_discover_claude_md(self, project_dir):
        """CLAUDE.md at the root should be discovered."""
        (project_dir / "CLAUDE.md").write_text("# Claude Rules\nBe concise.")
        rules = discover_rules(start_dir=str(project_dir))
        assert len(rules) == 1
        assert "Be concise" in rules[0]

    def test_discover_dot_claude_claude_md(self, project_dir):
        """Files in .claude/ should be discovered."""
        dot_claude = project_dir / ".claude"
        dot_claude.mkdir()
        (dot_claude / "CLAUDE.md").write_text("# Dot Claude\nWrite tests.")
        rules = discover_rules(start_dir=str(project_dir))
        assert len(rules) == 1
        assert "Write tests" in rules[0]

    def test_priority_agents_md_over_claude_md(self, project_dir):
        """AGENTS.md should take priority over CLAUDE.md at the same level."""
        (project_dir / "AGENTS.md").write_text("AGENTS rules")
        (project_dir / "CLAUDE.md").write_text("CLAUDE rules")
        rules = discover_rules(start_dir=str(project_dir))
        # At the root level, only the first match (AGENTS.md) should be returned
        assert len(rules) == 1
        assert "AGENTS rules" in rules[0]

    def test_priority_agents_md_over_dot_agents(self, project_dir):
        """Root AGENTS.md should win over .agents/AGENTS.md."""
        (project_dir / "AGENTS.md").write_text("Root rules")
        dot_agents = project_dir / ".agents"
        dot_agents.mkdir()
        (dot_agents / "AGENTS.md").write_text("Dot agents rules")
        rules = discover_rules(start_dir=str(project_dir))
        assert len(rules) == 1
        assert "Root rules" in rules[0]

    def test_recursive_discovery(self, project_dir):
        """Rules in subdirectories should be found when recursive=True."""
        (project_dir / "AGENTS.md").write_text("Root rules")
        sub = project_dir / "src"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Subdirectory rules")
        rules = discover_rules(start_dir=str(project_dir), recursive=True)
        assert len(rules) == 2
        assert "Root rules" in rules[0]
        assert "Subdirectory rules" in rules[1]

    def test_non_recursive_discovery(self, project_dir):
        """With recursive=False, only root level should be checked."""
        (project_dir / "AGENTS.md").write_text("Root rules")
        sub = project_dir / "src"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Should not be found")
        rules = discover_rules(start_dir=str(project_dir), recursive=False)
        assert len(rules) == 1
        assert "Root rules" in rules[0]

    def test_max_depth_limits_recursion(self, project_dir):
        """max_depth should limit how deep we search."""
        (project_dir / "AGENTS.md").write_text("Root")
        level1 = project_dir / "a"
        level1.mkdir()
        (level1 / "AGENTS.md").write_text("Level 1")
        level2 = level1 / "b"
        level2.mkdir()
        (level2 / "AGENTS.md").write_text("Level 2")
        rules = discover_rules(start_dir=str(project_dir), recursive=True, max_depth=1)
        assert len(rules) == 2
        assert "Root" in rules[0]
        assert "Level 1" in rules[1]

    def test_empty_project(self, project_dir):
        """Empty project should return empty list."""
        rules = discover_rules(start_dir=str(project_dir))
        assert rules == []

    def test_nonexistent_directory(self):
        """Nonexistent directory should return empty list."""
        rules = discover_rules(start_dir="/nonexistent/path/abc123")
        assert rules == []

    def test_empty_files_skipped(self, project_dir):
        """Empty rule files should be skipped."""
        (project_dir / "AGENTS.md").write_text("")
        rules = discover_rules(start_dir=str(project_dir))
        assert rules == []

    def test_whitespace_only_files_skipped(self, project_dir):
        """Files with only whitespace should be skipped."""
        (project_dir / "AGENTS.md").write_text("   \n\n  ")
        rules = discover_rules(start_dir=str(project_dir))
        assert rules == []

    def test_hidden_directories_not_recursed(self, project_dir):
        """Hidden directories (starting with .) should not be recursed into
        for subdirectory discovery (but .agents/ and .claude/ are checked
        via the candidate list)."""
        hidden = project_dir / ".hidden"
        hidden.mkdir()
        (hidden / "AGENTS.md").write_text("Should not be found via recursion")
        rules = discover_rules(start_dir=str(project_dir), recursive=True)
        assert rules == []


# ── discover_and_format_rules ───────────────────────────────────────


class TestDiscoverAndFormatRules:
    """Tests for discover_and_format_rules convenience function."""

    def test_single_file_returns_content(self, project_dir):
        (project_dir / "AGENTS.md").write_text("Single rule file")
        result = discover_and_format_rules(start_dir=str(project_dir))
        assert result == "Single rule file"

    def test_multiple_files_joined(self, project_dir):
        (project_dir / "AGENTS.md").write_text("Root")
        sub = project_dir / "src"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Sub")
        result = discover_and_format_rules(
            start_dir=str(project_dir), recursive=True
        )
        assert "Root" in result
        assert "Sub" in result

    def test_no_files_returns_none(self, project_dir):
        result = discover_and_format_rules(start_dir=str(project_dir))
        assert result is None


# ── Agent integration ───────────────────────────────────────────────


class TestAgentAutoDiscoverRules:
    """Tests for Agent auto_discover_rules integration."""

    def test_auto_discover_enabled_with_agents_md(self, project_dir):
        """Agent should auto-discover and apply AGENTS.md."""
        (project_dir / "AGENTS.md").write_text("Always use docstrings.")
        agent = Agent(
            "test",
            auto_discover_rules=True,
            rules_dir=str(project_dir),
        )
        assert "Always use docstrings" in agent.context
        assert "auto-discovered" in agent.context

    def test_auto_discover_enabled_with_claude_md(self, project_dir):
        """Agent should auto-discover CLAUDE.md when AGENTS.md is absent."""
        (project_dir / "CLAUDE.md").write_text("Write clean code.")
        agent = Agent(
            "test",
            auto_discover_rules=True,
            rules_dir=str(project_dir),
        )
        assert "Write clean code" in agent.context

    def test_auto_discover_with_dot_agents(self, project_dir):
        """Agent should find .agents/AGENTS.md."""
        dot_agents = project_dir / ".agents"
        dot_agents.mkdir()
        (dot_agents / "AGENTS.md").write_text("Use typing module.")
        agent = Agent(
            "test",
            auto_discover_rules=True,
            rules_dir=str(project_dir),
        )
        assert "Use typing module" in agent.context

    def test_auto_discover_disabled(self, project_dir):
        """When disabled, no rules should be auto-loaded."""
        (project_dir / "AGENTS.md").write_text("This should not appear.")
        agent = Agent(
            "test",
            auto_discover_rules=False,
            rules_dir=str(project_dir),
        )
        assert "This should not appear" not in agent.context

    def test_auto_discover_no_files(self, project_dir):
        """No rule files should result in empty context."""
        agent = Agent(
            "test",
            auto_discover_rules=True,
            rules_dir=str(project_dir),
        )
        # Context should be empty since no rules found and no context set
        assert agent.context == ""

    def test_auto_discover_preserves_existing_context(self, project_dir):
        """Auto-discovered rules should be in context after init."""
        (project_dir / "AGENTS.md").write_text("Rule 1")
        agent = Agent(
            "test",
            auto_discover_rules=True,
            rules_dir=str(project_dir),
        )
        # Rules should be in context right after init
        assert "Rule 1" in agent.context
        # set_context replaces context, so rules are lost (expected behavior)
        agent.set_context("You are a helpful assistant.")
        assert agent.context == "You are a helpful assistant."

    def test_with_rules_still_works(self, project_dir):
        """Explicit with_rules should still work on top of auto-discovery."""
        (project_dir / "AGENTS.md").write_text("Auto-discovered rule")
        explicit_rules = project_dir / "CUSTOM.md"
        explicit_rules.write_text("Explicit custom rule")
        agent = Agent(
            "test",
            auto_discover_rules=True,
            rules_dir=str(project_dir),
        )
        agent.with_rules(str(explicit_rules))
        assert "Auto-discovered rule" in agent.context
        assert "Explicit custom rule" in agent.context

    def test_auto_discover_default_is_enabled(self):
        """auto_discover_rules should default to True."""
        # Can't easily test the default behavior since it depends on cwd,
        # but we can at least verify the attribute is set
        agent = Agent("test")
        assert agent._auto_discover_rules is True

    def test_auto_discover_rules_dir_parameter(self, project_dir):
        """rules_dir parameter should control where to search."""
        sub = project_dir / "custom_root"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Custom root rule")
        agent = Agent(
            "test",
            auto_discover_rules=True,
            rules_dir=str(sub),
        )
        assert "Custom root rule" in agent.context

    def test_dot_claude_claude_md(self, project_dir):
        """Agent should find .claude/CLAUDE.md."""
        dot_claude = project_dir / ".claude"
        dot_claude.mkdir()
        (dot_claude / "CLAUDE.md").write_text("From .claude dir")
        agent = Agent(
            "test",
            auto_discover_rules=True,
            rules_dir=str(project_dir),
        )
        assert "From .claude dir" in agent.context
