"""Unit tests for auto-loading skills during inference."""

import pytest
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentu import Agent, Skill


@pytest.fixture
def temp_skill_dir():
    """Create temporary skill directories."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # PDF skill
    pdf_md = temp_dir / "pdf.md"
    pdf_md.write_text("""---
name: pdf
description: Extract text and tables from PDF files
---
# PDF Skill
Use pdfplumber for PDFs.
""")
    
    # Calculator skill
    calc_md = temp_dir / "calc.md"
    calc_md.write_text("""---
name: calculator
description: Perform mathematical calculations
---
# Calculator Skill
Use Python for math.
""")
    
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_match_skills_pdf_query(temp_skill_dir):
    """Test that PDF-related queries match PDF skill."""
    pdf_skill = Skill(
        name="pdf",
        description="Extract text and tables from PDF files",
        instructions=temp_skill_dir / "pdf.md"
    )
    
    agent = Agent("test").with_skills([pdf_skill])
    
    # Should match on "PDF" keyword
    matched = agent._match_skills("How do I extract text from a PDF?")
    assert len(matched) == 1
    assert matched[0].name == "pdf"


def test_match_skills_no_match(temp_skill_dir):
    """Test that unrelated queries don't match skills."""
    pdf_skill = Skill(
        name="pdf",
        description="Extract text and tables from PDF files",
        instructions=temp_skill_dir / "pdf.md"
    )
    
    agent = Agent("test").with_skills([pdf_skill])
    
    # Should NOT match weather query
    matched = agent._match_skills("What is the weather today?")
    assert len(matched) == 0


def test_match_skills_multiple(temp_skill_dir):
    """Test matching multiple skills."""
    pdf_skill = Skill(
        name="pdf",
        description="Extract text and tables from PDF files",
        instructions=temp_skill_dir / "pdf.md"
    )
    
    calc_skill = Skill(
        name="calculator",
        description="Perform mathematical calculations",
        instructions=temp_skill_dir / "calc.md"
    )
    
    agent = Agent("test").with_skills([pdf_skill, calc_skill])
    
    # Should only match calculator
    matched = agent._match_skills("Calculate 5 + 3")
    assert len(matched) == 1
    assert matched[0].name == "calculator"


def test_build_context_without_skills(temp_skill_dir):
    """Test context building without active skills."""
    agent = Agent("test")
    
    context = agent._build_turn_context("Hello world", [])
    
    assert context == "Hello world"
    assert "Active Skills" not in context


def test_build_context_with_skills(temp_skill_dir):
    """Test context building with active skills."""
    pdf_skill = Skill(
        name="pdf",
        description="Extract text and tables from PDF files",
        instructions=temp_skill_dir / "pdf.md"
    )
    
    agent = Agent("test").with_skills([pdf_skill])
    
    context = agent._build_turn_context("Extract PDF", [], [pdf_skill])
    
    assert "Active Skills" in context
    assert "pdf" in context
    assert "pdfplumber" in context  # From instructions
    assert len(context) > 100  # Should include full instructions


def test_context_size_difference(temp_skill_dir):
    """Test that skills significantly increase context when active."""
    pdf_skill = Skill(
        name="pdf",  
        description="Extract text and tables from PDF files",
        instructions=temp_skill_dir / "pdf.md"
    )
    
    agent = Agent("test").with_skills([pdf_skill])
    
    without_skill = agent._build_turn_context("Query", [])
    with_skill = agent._build_turn_context("Query", [], [pdf_skill])
    
    # With skill should be much larger
    assert len(with_skill) > len(without_skill) + 50
