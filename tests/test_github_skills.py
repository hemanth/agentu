"""Tests for GitHub skill loading."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import URLError

from agentu.skill import (
    Skill,
    load_skill,
    _parse_github_url,
    _fetch_github_skill,
    SKILL_CACHE_DIR
)


class TestParseGitHubUrl:
    """Tests for GitHub URL parsing."""
    
    def test_https_with_tree(self):
        url = "https://github.com/hemanth/agentu-skills/tree/main/pdf-processor"
        owner, repo, branch, path = _parse_github_url(url)
        assert owner == "hemanth"
        assert repo == "agentu-skills"
        assert branch == "main"
        assert path == "pdf-processor"
    
    def test_https_with_tree_nested_path(self):
        url = "https://github.com/org/repo/tree/develop/skills/advanced/processor"
        owner, repo, branch, path = _parse_github_url(url)
        assert owner == "org"
        assert repo == "repo"
        assert branch == "develop"
        assert path == "skills/advanced/processor"
    
    def test_https_without_tree_defaults_to_main(self):
        url = "https://github.com/hemanth/skills/pdf-processor"
        owner, repo, branch, path = _parse_github_url(url)
        assert owner == "hemanth"
        assert repo == "skills"
        assert branch == "main"
        assert path == "pdf-processor"
    
    def test_ssh_format(self):
        url = "git@github.com:hemanth/skills.git/pdf-processor"
        owner, repo, branch, path = _parse_github_url(url)
        assert owner == "hemanth"
        assert repo == "skills"
        assert branch == "main"
        assert path == "pdf-processor"
    
    def test_ssh_format_without_git_extension(self):
        url = "git@github.com:hemanth/skills/pdf"
        owner, repo, branch, path = _parse_github_url(url)
        assert owner == "hemanth"
        assert repo == "skills"
        assert branch == "main"
        assert path == "pdf"
    
    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Could not parse GitHub URL"):
            _parse_github_url("https://gitlab.com/user/repo")


class TestLoadSkill:
    """Tests for the load_skill function."""
    
    def test_load_skill_from_skill_object(self):
        """Passing a Skill object should return it unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_md = Path(tmpdir) / "SKILL.md"
            skill_md.write_text("# Test Skill\nInstructions here.")
            
            skill = Skill(
                name="test",
                description="Test skill",
                instructions=str(skill_md)
            )
            
            result = load_skill(skill)
            assert result is skill
    
    def test_load_skill_from_local_path(self):
        """Load skill from local directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()
            
            # Create SKILL.md
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("# My Skill\nDo things.")
            
            # Create skill.json
            skill_json = skill_dir / "skill.json"
            skill_json.write_text(json.dumps({
                "name": "my-skill",
                "description": "A test skill"
            }))
            
            result = load_skill(str(skill_dir))
            
            assert result.name == "my-skill"
            assert result.description == "A test skill"
            assert "My Skill" in result.load_instructions()
    
    def test_load_skill_from_local_path_without_json(self):
        """Load skill from local path without skill.json uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "auto-skill"
            skill_dir.mkdir()
            
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("# Auto Skill")
            
            result = load_skill(str(skill_dir))
            
            assert result.name == "auto-skill"
            assert "auto-skill" in result.description or "Local skill" in result.description
    
    def test_load_skill_missing_skill_md_raises(self):
        """Error when SKILL.md is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "bad-skill"
            skill_dir.mkdir()
            
            with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
                load_skill(str(skill_dir))
    
    def test_load_skill_invalid_type_raises(self):
        """Error on invalid input type."""
        with pytest.raises(TypeError, match="Expected Skill or str"):
            load_skill(123)


class TestFetchGitHubSkill:
    """Tests for GitHub skill fetching (mocked)."""
    
    @patch('agentu.skill.urlopen')
    def test_fetch_github_skill_success(self, mock_urlopen):
        """Successfully fetch skill from GitHub."""
        # Mock skill.json response
        skill_json_response = MagicMock()
        skill_json_response.read.return_value = json.dumps({
            "name": "test-skill",
            "description": "A test skill from GitHub"
        }).encode()
        skill_json_response.__enter__ = MagicMock(return_value=skill_json_response)
        skill_json_response.__exit__ = MagicMock(return_value=False)
        
        # Mock SKILL.md response
        skill_md_response = MagicMock()
        skill_md_response.read.return_value = b"# Test Skill\nInstructions from GitHub."
        skill_md_response.__enter__ = MagicMock(return_value=skill_md_response)
        skill_md_response.__exit__ = MagicMock(return_value=False)
        
        mock_urlopen.side_effect = [skill_json_response, skill_md_response]
        
        url = "https://github.com/hemanth/skills/tree/main/test-skill"
        skill = _fetch_github_skill(url)
        
        assert skill.name == "test-skill"
        assert skill.description == "A test skill from GitHub"
        assert "Test Skill" in skill.load_instructions()
    
    @patch('agentu.skill.urlopen')
    def test_fetch_github_skill_no_skill_json(self, mock_urlopen):
        """Fetch skill without skill.json uses defaults from path."""
        # First call (skill.json) raises URLError
        # Second call (SKILL.md) succeeds
        skill_md_response = MagicMock()
        skill_md_response.read.return_value = b"# Default Skill"
        skill_md_response.__enter__ = MagicMock(return_value=skill_md_response)
        skill_md_response.__exit__ = MagicMock(return_value=False)
        
        mock_urlopen.side_effect = [URLError("Not found"), skill_md_response]
        
        url = "https://github.com/owner/repo/tree/main/my-skill"
        skill = _fetch_github_skill(url)
        
        assert skill.name == "my-skill"
        assert "owner/repo" in skill.description
    
    @patch('agentu.skill.urlopen')
    def test_fetch_github_skill_missing_skill_md_raises(self, mock_urlopen):
        """Error when SKILL.md cannot be fetched."""
        # Both calls raise URLError
        mock_urlopen.side_effect = URLError("Not found")
        
        url = "https://github.com/owner/repo/tree/main/bad-skill"
        
        with pytest.raises(FileNotFoundError, match="Could not fetch SKILL.md"):
            _fetch_github_skill(url)


class TestSkillWithResources:
    """Tests for skills with resources."""
    
    def test_load_skill_with_resources(self):
        """Load local skill with resources defined in skill.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "resource-skill"
            skill_dir.mkdir()
            
            # Create SKILL.md
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("# Resource Skill")
            
            # Create resource file
            resource_file = skill_dir / "templates.md"
            resource_file.write_text("# Templates\n- Template 1\n- Template 2")
            
            # Create skill.json with resources
            skill_json = skill_dir / "skill.json"
            skill_json.write_text(json.dumps({
                "name": "resource-skill",
                "description": "Skill with resources",
                "resources": {
                    "templates": "templates.md"
                }
            }))
            
            result = load_skill(str(skill_dir))
            
            assert result.name == "resource-skill"
            assert "templates" in result.list_resources()
            assert "Template 1" in result.load_resource("templates")
