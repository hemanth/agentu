"""Tests for SKILL.md format support and SkillLoader auto-discovery."""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from agentu.skills.loader import (
    parse_skill_md_frontmatter,
    load_skill_from_md,
    SkillLoader,
)
from agentu.skills.skill import Skill, load_skill


# ── Frontmatter parsing ────────────────────────────────────────────


class TestParseSkillMdFrontmatter:
    """Tests for parse_skill_md_frontmatter."""

    def test_basic_frontmatter(self):
        content = "---\nname: my-skill\ndescription: Does cool things\n---\n# Body\nHello"
        fm, body = parse_skill_md_frontmatter(content)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "Does cool things"
        assert "# Body" in body
        assert "Hello" in body

    def test_no_frontmatter(self):
        content = "# Just a markdown file\nNo frontmatter here."
        fm, body = parse_skill_md_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_empty_content(self):
        fm, body = parse_skill_md_frontmatter("")
        assert fm == {}
        assert body == ""

    def test_frontmatter_with_extra_fields(self):
        content = "---\nname: skill-x\ndescription: X\nversion: 1.0\nauthor: test\n---\nBody"
        fm, body = parse_skill_md_frontmatter(content)
        assert fm["name"] == "skill-x"
        assert fm["description"] == "X"
        assert fm["version"] == "1.0"
        assert fm["author"] == "test"
        assert body.strip() == "Body"

    def test_frontmatter_with_colon_in_value(self):
        content = "---\nname: my-skill\ndescription: Uses tools: hammer, screwdriver\n---\nBody"
        fm, body = parse_skill_md_frontmatter(content)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "Uses tools: hammer, screwdriver"

    def test_frontmatter_ignores_comments(self):
        content = "---\n# This is a comment\nname: skill\ndescription: desc\n---\nBody"
        fm, _ = parse_skill_md_frontmatter(content)
        assert fm["name"] == "skill"
        assert fm["description"] == "desc"

    def test_frontmatter_only_no_body(self):
        content = "---\nname: minimal\ndescription: bare\n---\n"
        fm, body = parse_skill_md_frontmatter(content)
        assert fm["name"] == "minimal"
        assert body.strip() == ""


# ── load_skill_from_md ──────────────────────────────────────────────


@pytest.fixture
def skill_dir():
    """Create a temp directory with a SKILL.md file."""
    tmp = Path(tempfile.mkdtemp())
    skill_md = tmp / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: test-loader-skill\n"
        "description: A skill loaded via SKILL.md frontmatter\n"
        "---\n\n"
        "# Instructions\n\nDo the thing.\n"
    )
    yield tmp
    shutil.rmtree(tmp)


class TestLoadSkillFromMd:
    """Tests for load_skill_from_md."""

    def test_loads_name_and_description_from_frontmatter(self, skill_dir):
        skill = load_skill_from_md(skill_dir / "SKILL.md")
        assert skill.name == "test-loader-skill"
        assert skill.description == "A skill loaded via SKILL.md frontmatter"

    def test_instructions_readable(self, skill_dir):
        skill = load_skill_from_md(skill_dir / "SKILL.md")
        instructions = skill.load_instructions()
        assert "Do the thing" in instructions

    def test_fallback_name_from_directory(self, skill_dir):
        """When frontmatter has no name, directory name is used."""
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: no name field\n---\n# Body"
        )
        skill = load_skill_from_md(skill_dir / "SKILL.md")
        assert skill.name == skill_dir.name

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_skill_from_md(Path("/nonexistent/SKILL.md"))

    def test_companion_skill_json_resources(self, skill_dir):
        """Resources from a companion skill.json should be loaded."""
        resource_file = skill_dir / "REFERENCE.md"
        resource_file.write_text("# Reference\nSome reference content.")
        skill_json = skill_dir / "skill.json"
        skill_json.write_text(json.dumps({
            "resources": {"reference": "REFERENCE.md"}
        }))
        skill = load_skill_from_md(skill_dir / "SKILL.md")
        assert "reference" in skill.list_resources()
        assert "Reference" in skill.load_resource("reference")

    def test_skill_json_name_fallback(self, skill_dir):
        """If frontmatter has no name, skill.json name is used."""
        (skill_dir / "SKILL.md").write_text("---\n---\n# Body")
        (skill_dir / "skill.json").write_text(json.dumps({
            "name": "json-name",
            "description": "from json"
        }))
        skill = load_skill_from_md(skill_dir / "SKILL.md")
        assert skill.name == "json-name"
        assert skill.description == "from json"


# ── load_skill with SKILL.md frontmatter ─────────────────────────


@pytest.mark.asyncio
async def test_load_skill_parses_frontmatter(skill_dir):
    """load_skill() should parse YAML frontmatter from SKILL.md when no skill.json exists."""
    skill = await load_skill(str(skill_dir))
    assert skill.name == "test-loader-skill"
    assert skill.description == "A skill loaded via SKILL.md frontmatter"


@pytest.mark.asyncio
async def test_load_skill_prefers_skill_json_over_frontmatter(skill_dir):
    """skill.json metadata should take precedence over SKILL.md frontmatter."""
    (skill_dir / "skill.json").write_text(json.dumps({
        "name": "json-priority-name",
        "description": "JSON takes precedence"
    }))
    skill = await load_skill(str(skill_dir))
    assert skill.name == "json-priority-name"
    assert skill.description == "JSON takes precedence"


# ── SkillLoader auto-discovery ──────────────────────────────────────


@pytest.fixture
def project_with_skills():
    """Create a project directory with skills in conventional locations."""
    tmp = Path(tempfile.mkdtemp())

    # .agents/skills/skill-a/SKILL.md
    skill_a_dir = tmp / ".agents" / "skills" / "skill-a"
    skill_a_dir.mkdir(parents=True)
    (skill_a_dir / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: First discovered skill\n---\n# A\nDo A."
    )

    # .claude/skills/skill-b/SKILL.md
    skill_b_dir = tmp / ".claude" / "skills" / "skill-b"
    skill_b_dir.mkdir(parents=True)
    (skill_b_dir / "SKILL.md").write_text(
        "---\nname: skill-b\ndescription: Second discovered skill\n---\n# B\nDo B."
    )

    yield tmp
    shutil.rmtree(tmp)


class TestSkillLoader:
    """Tests for SkillLoader auto-discovery."""

    def test_discover_from_agents_dir(self, project_with_skills):
        loader = SkillLoader(project_root=str(project_with_skills))
        skills = loader.discover()
        names = {s.name for s in skills}
        assert "skill-a" in names

    def test_discover_from_claude_dir(self, project_with_skills):
        loader = SkillLoader(project_root=str(project_with_skills))
        skills = loader.discover()
        names = {s.name for s in skills}
        assert "skill-b" in names

    @patch("agentu.skills.loader._SKILL_DISCOVERY_DIR_HOME", Path("/nonexistent"))
    def test_discover_all_skills(self, project_with_skills):
        loader = SkillLoader(project_root=str(project_with_skills))
        skills = loader.discover()
        assert len(skills) == 2

    def test_discover_deduplicates_by_name(self, project_with_skills):
        """If the same skill name appears in multiple directories, only keep first."""
        # Add a duplicate skill-a in .claude/skills
        dup_dir = project_with_skills / ".claude" / "skills" / "skill-a-dup"
        dup_dir.mkdir(parents=True)
        (dup_dir / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: Duplicate\n---\n# Dup"
        )
        loader = SkillLoader(project_root=str(project_with_skills))
        skills = loader.discover()
        a_skills = [s for s in skills if s.name == "skill-a"]
        assert len(a_skills) == 1
        assert a_skills[0].description == "First discovered skill"

    def test_discover_extra_dirs(self):
        """Extra directories should also be searched."""
        tmp = Path(tempfile.mkdtemp())
        try:
            skill_dir = tmp / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: extra-skill\ndescription: From extra dir\n---\n# Extra"
            )
            loader = SkillLoader(extra_dirs=[str(tmp)])
            skills = loader.discover()
            names = {s.name for s in skills}
            assert "extra-skill" in names
        finally:
            shutil.rmtree(tmp)

    @patch("agentu.skills.loader._SKILL_DISCOVERY_DIR_HOME", Path("/nonexistent"))
    def test_discover_empty_project(self):
        """Empty project should return empty list."""
        tmp = Path(tempfile.mkdtemp())
        try:
            loader = SkillLoader(project_root=str(tmp))
            skills = loader.discover()
            assert skills == []
        finally:
            shutil.rmtree(tmp)

    def test_discover_no_project_root(self):
        """No project root should still work (just search home/extras)."""
        loader = SkillLoader()
        # Should not raise; may return skills from ~/.agents/skills/ if they exist
        skills = loader.discover()
        assert isinstance(skills, list)

    def test_discovered_skill_has_working_instructions(self, project_with_skills):
        loader = SkillLoader(project_root=str(project_with_skills))
        skills = loader.discover()
        skill_a = next(s for s in skills if s.name == "skill-a")
        instructions = skill_a.load_instructions()
        assert "Do A" in instructions

    def test_discovered_skill_metadata(self, project_with_skills):
        loader = SkillLoader(project_root=str(project_with_skills))
        skills = loader.discover()
        skill_a = next(s for s in skills if s.name == "skill-a")
        metadata = skill_a.metadata()
        assert "name: skill-a" in metadata
        assert "description: First discovered skill" in metadata
