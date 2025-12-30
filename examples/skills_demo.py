"""
Demo: Agent Skills with Progressive Loading

This example demonstrates the new Skills system that enables
progressive disclosure of domain expertise without context bloat.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for running without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentu import Agent, Skill


async def main():
    # Create a PDF processing skill with 3-level loading
    pdf_skill = Skill(
        name="pdf-processing",
        description="Extract text and tables from PDF files, fill forms, merge documents",
        instructions=Path("examples/skills/pdf-processing/SKILL.md"),
        resources={
            "forms": Path("examples/skills/pdf-processing/FORMS.md")
        }
    )
    
    # Create agent with skill attached
    agent = Agent("pdf-assistant").with_skills([pdf_skill])
    
    print("=" * 60)
    print("agentu Skills Demo")
    print("=" * 60)
    print()
    
    # Level 1: Metadata is ALWAYS loaded (zero context penalty)
    print("✓ Skill metadata loaded in system prompt")
    print(f"  Skill: {pdf_skill.name}")
    print(f"  Description: {pdf_skill.description}")
    print()
    
    # Level 2: Instructions loaded ONLY when skill triggered
    print("Simulating skill activation...")
    print("User: 'How do I extract text from a PDF?'")
    print()
    
    instructions = pdf_skill.load_instructions()
    print("✓ Skill instructions loaded (Level 2)")
    print(f"  Size: {len(instructions)} characters")
    print(f"  Preview: {instructions[:200]}...")
    print()
    
    # Level 3: Resources loaded ON-DEMAND
    print("If agent needs form-filling details...")
    forms_guide = pdf_skill.load_resource("forms")
    print("✓ Forms resource loaded (Level 3)")
    print(f"  Size: {len(forms_guide)} characters")
    print(f"  Available resources: {pdf_skill.list_resources()}")
    print()
    
    print("=" * 60)
    print("Progressive Loading Benefits")
    print("=" * 60)
    print(f"  • Metadata: ~100 chars (always loaded)")
    print(f"  • Instructions: ~{len(instructions)} chars (loaded when triggered)")
    print(f"  • Resources: ~{len(forms_guide)} chars (loaded as needed)")
    print()
    print("Total potential context: ~{} chars".format(
        100 + len(instructions) + len(forms_guide)
    ))
    print("Context WITHOUT skill trigger: ~100 chars")
    print(f"Savings: {100*(1 - 100/(100+len(instructions)+len(forms_guide))):.1f}%")
    print()
    
    # Demonstrate tool access
    print("=" * 60)
    print("Skill Resource Tool")
    print("=" * 60)
    print("Agent can access additional resources via get_skill_resource tool")
    print(f"  Tool available: {'get_skill_resource' in [t.name for t in agent.tools]}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
