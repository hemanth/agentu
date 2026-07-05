"""
SKILL.md open standard loader.

Parses SKILL.md files with YAML frontmatter (name, description fields)
as the primary skill format, with backwards compatibility for skill.json.

Also provides auto-discovery of skills from conventional directories:
- .agents/skills/
- ~/.agents/skills/
- .claude/skills/
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .skill import Skill

logger = logging.getLogger(__name__)

# Standard directories to search for skills (relative to project root)
_SKILL_DISCOVERY_DIRS_RELATIVE = [
    ".agents/skills",
    ".claude/skills",
]

# User-level skill directory
_SKILL_DISCOVERY_DIR_HOME = Path.home() / ".agents" / "skills"


def parse_skill_md_frontmatter(content: str) -> Tuple[Dict[str, str], str]:
    """Parse YAML frontmatter from a SKILL.md file.

    The expected format is:
        ---
        name: skillname
        description: what it does
        ---
        <markdown body>

    Args:
        content: Raw text content of a SKILL.md file.

    Returns:
        A tuple of (frontmatter_dict, body_text).
        frontmatter_dict contains parsed key-value pairs (name, description, etc.).
        body_text is the markdown content after the frontmatter.
    """
    frontmatter: Dict[str, str] = {}
    body = content

    # Match YAML frontmatter delimited by --- lines
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if match:
        fm_block = match.group(1)
        body = match.group(2)

        # Parse simple key: value lines (covers the standard fields)
        for line in fm_block.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            colon_idx = line.find(":")
            if colon_idx > 0:
                key = line[:colon_idx].strip()
                value = line[colon_idx + 1:].strip()
                frontmatter[key] = value

    return frontmatter, body


def load_skill_from_md(
    skill_md_path: Path,
    base_dir: Optional[Path] = None,
) -> Skill:
    """Load a Skill from a SKILL.md file with YAML frontmatter.

    If the SKILL.md contains ``name`` and ``description`` in its frontmatter,
    those are used.  Otherwise, fallback values are derived from the directory
    name / file path.

    A ``skill.json`` in the same directory is also consulted for ``resources``
    mapping (backwards compatibility).

    Args:
        skill_md_path: Path to the SKILL.md file.
        base_dir: Optional base directory for resolving resource paths.
                  Defaults to the parent directory of skill_md_path.

    Returns:
        A fully constructed Skill object.

    Raises:
        FileNotFoundError: If skill_md_path does not exist.
    """
    skill_md_path = Path(skill_md_path)
    if not skill_md_path.exists():
        raise FileNotFoundError(f"SKILL.md not found: {skill_md_path}")

    if base_dir is None:
        base_dir = skill_md_path.parent

    content = skill_md_path.read_text()
    frontmatter, _ = parse_skill_md_frontmatter(content)

    name = frontmatter.get("name", base_dir.name)
    description = frontmatter.get("description", f"Skill from {base_dir}")

    # Check for a companion skill.json for resource mapping (backwards compat)
    resources: Dict[str, str] = {}
    skill_json_path = base_dir / "skill.json"
    if skill_json_path.exists():
        import json
        try:
            metadata = json.loads(skill_json_path.read_text())
            if "resources" in metadata:
                for key, resource_path in metadata["resources"].items():
                    full_path = base_dir / resource_path
                    if full_path.exists():
                        resources[key] = str(full_path)
            # Also pick up name/description from skill.json if not in frontmatter
            if "name" not in frontmatter and "name" in metadata:
                name = metadata["name"]
            if "description" not in frontmatter and "description" in metadata:
                description = metadata["description"]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse skill.json at {skill_json_path}: {e}")

    return Skill(
        name=name,
        description=description,
        instructions=str(skill_md_path),
        resources=resources if resources else None,
    )


@dataclass
class SkillLoader:
    """Auto-discover and load skills from conventional directories.

    Searches for SKILL.md files (with YAML frontmatter) and falls back to
    skill.json for metadata.  Skills are discovered from:
    - ``<project_root>/.agents/skills/``
    - ``<project_root>/.claude/skills/``
    - ``~/.agents/skills/`` (user-level)
    - Any additional directories passed via ``extra_dirs``.

    Example::

        loader = SkillLoader(project_root="/path/to/project")
        skills = loader.discover()
        agent = await Agent("bot").with_skills(skills)
    """
    project_root: Optional[str] = None
    extra_dirs: List[str] = field(default_factory=list)

    def _get_search_dirs(self) -> List[Path]:
        """Build the ordered list of directories to search for skills."""
        dirs: List[Path] = []

        if self.project_root:
            root = Path(self.project_root)
            for rel in _SKILL_DISCOVERY_DIRS_RELATIVE:
                candidate = root / rel
                if candidate.is_dir():
                    dirs.append(candidate)

        # User-level directory
        if _SKILL_DISCOVERY_DIR_HOME.is_dir():
            dirs.append(_SKILL_DISCOVERY_DIR_HOME)

        # Extra directories
        for d in self.extra_dirs:
            p = Path(d)
            if p.is_dir():
                dirs.append(p)

        return dirs

    def discover(self) -> List[Skill]:
        """Discover skills from all configured directories.

        Each immediate subdirectory containing a ``SKILL.md`` is treated as
        a skill.  The YAML frontmatter is parsed for ``name`` and
        ``description``.

        Returns:
            List of discovered Skill objects.
        """
        skills: List[Skill] = []
        seen_names: set = set()

        for search_dir in self._get_search_dirs():
            logger.info(f"Searching for skills in {search_dir}")
            for child in sorted(search_dir.iterdir()):
                if not child.is_dir():
                    # Also handle a bare SKILL.md file directly in the skills dir
                    if child.name == "SKILL.md":
                        try:
                            skill = load_skill_from_md(child, base_dir=search_dir)
                            if skill.name not in seen_names:
                                skills.append(skill)
                                seen_names.add(skill.name)
                                logger.info(f"Discovered skill: {skill.name} from {child}")
                        except Exception as e:
                            logger.warning(f"Failed to load skill from {child}: {e}")
                    continue

                # Look for SKILL.md inside the subdirectory
                skill_md = child / "SKILL.md"
                if skill_md.exists():
                    try:
                        skill = load_skill_from_md(skill_md)
                        if skill.name not in seen_names:
                            skills.append(skill)
                            seen_names.add(skill.name)
                            logger.info(f"Discovered skill: {skill.name} from {skill_md}")
                    except Exception as e:
                        logger.warning(f"Failed to load skill from {skill_md}: {e}")
                    continue

                # Fallback: look for skill.json (backwards compatibility)
                skill_json = child / "skill.json"
                if skill_json.exists():
                    try:
                        import json
                        metadata = json.loads(skill_json.read_text())
                        name = metadata.get("name", child.name)
                        desc = metadata.get("description", f"Skill from {child}")
                        # Still require a SKILL.md for instructions
                        # If none, create a minimal skill with just metadata
                        if name not in seen_names:
                            logger.info(
                                f"Found skill.json without SKILL.md in {child}, "
                                "skipping (SKILL.md required for instructions)"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to load skill.json from {child}: {e}")

        logger.info(f"Discovered {len(skills)} skills total")
        return skills
