"""Integration tests for Skills with Agent."""

import pytest
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentu import Agent, Skill, Tool


@pytest.fixture
def temp_skill_dir():
    """Create a temporary directory with skill files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    skill_md = temp_dir / "SKILL.md"
    skill_md.write_text("""---
name: calculator
description: Perform mathematical calculations
---

# Calculator Skill
Use Python for calculations:
``` python
result = x + y
```
""")
    
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_agent_with_skills(temp_skill_dir):
    """Test adding skills to an agent."""
    skill = Skill(
        name="calculator",
        description="Perform calculations",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    agent = Agent("test-agent").with_skills([skill])
    
    assert len(agent.skills) == 1
    assert agent.skills[0].name == "calculator"


def test_agent_skill_resource_tool_added(temp_skill_dir):
    """Test that get_skill_resource tool is auto-added."""
    skill = Skill(
        name="test-skill",
        description="Test skill",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    agent = Agent("test-agent").with_skills([skill])
    
    # Check that get_skill_resource tool was added
    tool_names = [t.name for t in agent.tools]
    assert "get_skill_resource" in tool_names


def test_agent_multiple_skills(temp_skill_dir):
    """Test adding multiple skills."""
    skill1 = Skill(
        name="skill1",
        description="First skill",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    skill2_md = temp_skill_dir / "SKILL2.md"
    skill2_md.write_text("---\nname: skill2\n---\n# Skill 2")
    
    skill2 = Skill(
        name="skill2",
        description="Second skill",
        instructions=skill2_md
    )
    
    agent = Agent("test-agent").with_skills([skill1, skill2])
    
    assert len(agent.skills) == 2
    assert agent.skills[0].name == "skill1"
    assert agent.skills[1].name == "skill2"


def test_agent_skills_in_prompt(temp_skill_dir):
    """Test that skill metadata appears in formatted prompt."""
    skill = Skill(
        name="test-skill",
        description="A test skill for demonstrations",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    agent = Agent("test-agent").with_skills([skill])
    prompt = agent._format_tools_for_prompt()
    
    # Check that skill metadata is in the prompt
    assert "Available Skills" in prompt
    assert "test-skill" in prompt
    assert "A test skill for demonstrations" in prompt


def test_agent_with_tools_and_skills(temp_skill_dir):
    """Test agent with both tools and skills."""
    def my_tool(x: int) -> int:
        """A simple tool."""
        return x * 2
    
    skill = Skill(
        name="test-skill",
        description="Test skill",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    agent = (Agent("test-agent")
             .with_tools([my_tool])
             .with_skills([skill]))
    
    assert len(agent.tools) >= 1  # at least my_tool (get_skill_resource is auto-added)
    assert len(agent.skills) == 1
    
    # Both should appear in prompt
    prompt = agent._format_tools_for_prompt()
    assert "Available Skills" in prompt
    assert "Available tools" in prompt


def test_skill_chainability(temp_skill_dir):
    """Test that with_skills returns self for chaining."""
    skill = Skill(
        name="test-skill",
        description="Test skill",
        instructions=temp_skill_dir / "SKILL.md"
    )
    
    agent = Agent("test-agent")
    result = agent.with_skills([skill])
    
    assert result is agent  # Should return self for chaining
