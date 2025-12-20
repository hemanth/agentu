"""
Agent Skills with progressive loading.

Skills provide domain-specific expertise through a 3-level loading system:
- Level 1: Metadata (always loaded, minimal context)
- Level 2: Instructions (loaded when triggered)
- Level 3: Resources (loaded on-demand)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List


@dataclass
class Skill:
    """
    An agent skill with progressive disclosure.
    
    Skills package domain expertise as filesystem-based resources that load
    incrementally based on task relevance, minimizing context consumption.
    
    Args:
        name: Unique skill identifier (e.g., "pdf-processing")
        description: When to use this skill (triggers activation)
        instructions: Path to SKILL.md with procedural knowledge
        resources: Optional dict mapping resource keys to file paths
        
    Example:
        >>> pdf_skill = Skill(
        ...     name="pdf-processing",
        ...     description="Extract text and tables from PDF files",
        ...     instructions="skills/pdf/SKILL.md",
        ...     resources={"forms": "skills/pdf/FORMS.md"}
        ... )
    """
    name: str
    description: str
    instructions: str
    resources: Optional[Dict[str, str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate."""
        # Convert instructions string to Path
        self.instructions = Path(self.instructions)
        
        if not self.instructions.exists():
            raise FileNotFoundError(f"Skill instructions not found: {self.instructions}")
        
        # Convert resource strings to Path objects
        if self.resources:
            converted = {}
            for key, path_str in self.resources.items():
                path = Path(path_str)
                if not path.exists():
                    raise FileNotFoundError(f"Skill resource '{key}' not found: {path}")
                converted[key] = path
            self.resources = converted
    
    def metadata(self) -> str:
        """
        Returns Level 1 YAML frontmatter for system prompt.
        
        This lightweight metadata enables skill discovery without context overhead.
        An agent can have dozens of skills, but only this minimal info is always loaded.
        
        Returns:
            YAML-formatted metadata string
        """
        return f"---\nname: {self.name}\ndescription: {self.description}\n---"
    
    def load_instructions(self) -> str:
        """
        Load Level 2 instructions from SKILL.md.
        
        Called when the skill is triggered by a matching user request.
        Contains procedural knowledge, workflows, and best practices.
        
        Returns:
            Markdown content from instructions file
        """
        return self.instructions.read_text()
    
    def load_resource(self, key: str) -> str:
        """
        Load Level 3 resource on-demand.
        
        Resources are loaded only when explicitly referenced in instructions.
        This enables skills to bundle extensive documentation, schemas, or
        templates without bloating the context window.
        
        Args:
            key: Resource identifier from resources dict
            
        Returns:
            Resource content (file text or directory listing)
            
        Raises:
            KeyError: If resource key doesn't exist
        """
        if not self.resources or key not in self.resources:
            raise KeyError(f"Resource '{key}' not found in skill '{self.name}'")
        
        path = self.resources[key]
        
        if path.is_dir():
            # Return directory structure
            files = list(path.glob("*"))
            return f"Directory '{key}' contains: {[f.name for f in files]}"
        
        return path.read_text()
    
    def list_resources(self) -> List[str]:
        """
        List available resource keys.
        
        Returns:
            List of resource identifiers
        """
        return list(self.resources.keys()) if self.resources else []
