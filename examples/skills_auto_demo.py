"""
Demo: Auto-Loading Skills During Inference

This demonstrates Phase 2 - skills automatically activate
when the user's prompt matches skill descriptions.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentu import Agent, Skill


# Mock tool that would normally extract PDF text
def mock_extract_pdf(path: str) -> str:
    """Mock PDF extraction (in real usage, this would use pdfplumber)."""
    return f"Extracted text from {path}: Lorem ipsum dolor sit amet..."


async def main():
    # Create PDF skill
    pdf_skill = Skill(
        name="pdf-processing",
        description="Extract text and tables from PDF files, fill forms, merge documents",
        instructions=Path("examples/skills/pdf-processing/SKILL.md"),
        resources={"forms": Path("examples/skills/pdf-processing/FORMS.md")}
    )
    
    # Create agent with skill AND a tool
    agent = (Agent("pdf-assistant")
             .with_tools([mock_extract_pdf])
             .with_skills([pdf_skill]))
    
    print("=" * 70)
    print("Phase 2: Auto-Loading Skills Demo")
    print("=" * 70)
    print()
    
    # Test 1: Skill should auto-activate on PDF-related query
    print("Test 1: PDF-related query")
    print("-" * 70)
    user_query = "How do I extract text from a PDF document?"
    print(f"User: {user_query}")
    print()
    
    # Manually test skill matching (infer() would do this automatically)
    matched = agent._match_skills(user_query)
    print(f"✓ Auto-matched {len(matched)} skill(s): {[s.name for s in matched]}")
    
    if matched:
        print(f"✓ Skill '{matched[0].name}' will auto-load instructions")
        print(f"  Instructions size: {len(matched[0].load_instructions())} chars")
    print()
    
    # Test 2: Non-PDF query should NOT activate skill
    print("Test 2: Non-PDF query")
    print("-" * 70)
    non_pdf_query = "What is the weather today?"
    print(f"User: {non_pdf_query}")
    print()
    
    matched = agent._match_skills(non_pdf_query)
    print(f"✓ Matched {len(matched)} skill(s)")
    if not matched:
        print("  (No skills activated - context stays minimal!)")
    print()
    
    # Test 3: Show context difference
    print("Test 3: Context Comparison")
    print("-" * 70)
    
    # Without skill
    context_without = agent._build_turn_context(non_pdf_query, [])
    print(f"Context WITHOUT skill: {len(context_without)} chars")
    print(f"  Preview: {context_without[:80]}...")
    print()
    
    # With skill
    pdf_matched = agent._match_skills(user_query)
    context_with = agent._build_turn_context(user_query, [], pdf_matched)
    print(f"Context WITH skill: {len(context_with)} chars")
    print(f"  Preview: {context_with[:80]}...")
    print()
    
    print(f"Difference: {len(context_with) - len(context_without)} chars")
    print(f"  (Skill instructions loaded automatically!)")
    print()
    
    print("=" * 70)
    print("Key Insight")
    print("=" * 70)
    print("Skills are now TRANSPARENT to the user:")
    print("  • User asks about PDFs → PDF skill auto-loads")
    print("  • User asks about weather → No skill loaded")
    print("  • Agent gets the right expertise automatically!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
