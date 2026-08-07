"""Tests for Agent Plugins (v1.0.0 spec) support."""
import json
import pytest
from pathlib import Path
from agentu import Agent, PluginLoader


def _make_agent(name="test_agent"):
    return Agent(name, model="test-model", auto_discover_rules=False)


@pytest.fixture
def temp_plugin(tmp_path):
    plugin_dir = tmp_path / "sample-plugin"
    plugin_dir.mkdir()

    # plugin.json
    manifest = {
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        "name": "sample-plugin",
        "version": "1.0.0",
        "description": "A sample test plugin"
    }
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")

    # skills/summarize/SKILL.md
    skill_dir = plugin_dir / "skills" / "summarize"
    skill_dir.mkdir(parents=True)
    skill_md = """---
name: summarize
description: Summarize documents
---
# Summarize Skill
Instructions on how to summarize text.
"""
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    # mcp.json
    mcp_config = {
        "mcpServers": {
            "test-mcp": {
                "url": "http://localhost:8000/sse"
            }
        }
    }
    (plugin_dir / "mcp.json").write_text(json.dumps(mcp_config), encoding="utf-8")

    return plugin_dir


class TestPluginLoader:

    def test_loader_valid_plugin(self, temp_plugin):
        loader = PluginLoader(temp_plugin).load()
        assert loader.manifest["name"] == "sample-plugin"
        assert len(loader.skills) == 1
        assert loader.skills[0].name == "summarize"
        assert loader.mcp_config is not None
        assert loader.mcp_config.name == "mcp.json"

    def test_missing_plugin_json(self, tmp_path):
        bad_dir = tmp_path / "no-manifest"
        bad_dir.mkdir()
        loader = PluginLoader(bad_dir)
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_invalid_plugin_json_no_name(self, tmp_path):
        bad_dir = tmp_path / "bad-manifest"
        bad_dir.mkdir()
        (bad_dir / "plugin.json").write_text('{"version": "1.0"}', encoding="utf-8")
        loader = PluginLoader(bad_dir)
        with pytest.raises(ValueError, match="missing required non-empty 'name'"):
            loader.load()

    def test_containment_check(self, temp_plugin):
        loader = PluginLoader(temp_plugin)
        assert loader.is_contained(temp_plugin / "plugin.json")
        assert not loader.is_contained(temp_plugin / ".." / "outside.txt")


class TestAgentWithPlugin:

    @pytest.mark.asyncio
    async def test_with_plugin_loads_skills_and_mcp(self, temp_plugin):
        agent = _make_agent()
        await agent.with_plugin(temp_plugin)

        assert len(agent.skills) == 1
        assert agent.skills[0].name == "summarize"

    @pytest.mark.asyncio
    async def test_with_plugins_multiple(self, temp_plugin, tmp_path):
        plugin2 = tmp_path / "plugin-two"
        plugin2.mkdir()
        manifest2 = {"name": "plugin-two"}
        (plugin2 / "plugin.json").write_text(json.dumps(manifest2), encoding="utf-8")

        agent = _make_agent()
        await agent.with_plugins([temp_plugin, plugin2])
        assert len(agent.skills) == 1
