"""Unit tests for the Skill class."""

import pytest
from pathlib import Path
import tempfile
import shutil
from agentu import Skill


@pytest.fixture
def temp_skill_dir():
    """Create a temporary directory with skill files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create SKILL.md
    skill_md = temp_dir / "SKILL.md"
    skill_md.write_text("""---
name: test-skill
description: A test skill for unit testing
---

# Test Skill
This is a test skill with instructions.
""")
    
    # Create a resource file
    resource_md = temp_dir / "RESOURCE.md"
    resource_md.write_text("# Resource\nThis is a resource file.")
    
    # Create a resource directory
    scripts_dir = temp_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "test.py").write_text("print('test')")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_skill_creation(temp_skill_dir):
    """Test creating a skill."""
    skill = Skill(
        name="test-skill",
        description="A test skill",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    assert skill.name == "test-skill"
    assert skill.description == "A test skill"
    assert skill.instructions.exists()


def test_skill_with_resources(temp_skill_dir):
    """Test creating a skill with resources."""
    skill = Skill(
        name="test-skill",
        description="A test skill",
        instructions=temp_skill_dir / "SKILL.md",
        resources={
            "resource": temp_skill_dir / "RESOURCE.md",
            "scripts": temp_skill_dir / "scripts"
        }
    )
    
    assert len(skill.list_resources()) == 2
    assert "resource" in skill.list_resources()
    assert "scripts" in skill.list_resources()


def test_skill_metadata(temp_skill_dir):
    """Test Level 1: metadata generation."""
    skill = Skill(
        name="test-skill",
        description="A test skill for testing",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    metadata = skill.metadata()
    
    assert "---" in metadata
    assert "name: test-skill" in metadata
    assert "description: A test skill for testing" in metadata
    assert len(metadata) < 200  # Metadata should be very small


def test_skill_load_instructions(temp_skill_dir):
    """Test Level 2: instruction loading."""
    skill = Skill(
        name="test-skill",
        description="A test skill",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    instructions = skill.load_instructions()
    
    assert "Test Skill" in instructions
    assert "test skill with instructions" in instructions
    assert len(instructions) > len(skill.metadata())  # Instructions > metadata


def test_skill_load_resource(temp_skill_dir):
    """Test Level 3: resource loading."""
    skill = Skill(
        name="test-skill",
        description="A test skill",
        instructions=temp_skill_dir / "SKILL.md",
        resources={"resource": temp_skill_dir / "RESOURCE.md"}
    )
    
    resource_content = skill.load_resource("resource")
    
    assert "Resource" in resource_content
    assert "resource file" in resource_content


def test_skill_load_directory_resource(temp_skill_dir):
    """Test loading a directory resource."""
    skill = Skill(
        name="test-skill",
        description="A test skill",
        instructions=temp_skill_dir / "SKILL.md",
        resources={"scripts": temp_skill_dir / "scripts"}
    )
    
    dir_content = skill.load_resource("scripts")
    
    assert "Directory" in dir_content
    assert "test.py" in dir_content


def test_skill_missing_resource(temp_skill_dir):
    """Test error handling for missing resources."""
    skill = Skill(
        name="test-skill",
        description="A test skill",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    with pytest.raises(KeyError, match="Resource 'nonexistent' not found"):
        skill.load_resource("nonexistent")


def test_skill_missing_instructions():
    """Test error handling for missing instructions file."""
    with pytest.raises(FileNotFoundError):
        Skill(
            name="test-skill",
            description="A test skill",
            instructions=Path("/nonexistent/SKILL.md")
        )


def test_skill_missing_resource_file(temp_skill_dir):
    """Test error handling for missing resource file."""
    with pytest.raises(FileNotFoundError):
        Skill(
            name="test-skill",
            description="A test skill",
            instructions=temp_skill_dir / "SKILL.md",
            resources={"missing": Path("/nonexistent/file.md")}
        )
