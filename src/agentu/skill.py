"""
Agent Skills with progressive loading.

Skills provide domain-specific expertise through a 3-level loading system:
- Level 1: Metadata (always loaded, minimal context)
- Level 2: Instructions (loaded when triggered)
- Level 3: Resources (loaded on-demand)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Union
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Cache directory for GitHub skills
SKILL_CACHE_DIR = Path.home() / ".agentu" / "skills"


def _parse_github_url(url: str) -> tuple:
    """Parse GitHub URL or shorthand to extract owner, repo, branch, and path.
    
    Supports formats:
    - owner/repo/path (shorthand, defaults to main branch)
    - owner/repo/path@branch (shorthand with branch)
    - https://github.com/owner/repo/tree/branch/path/to/skill
    - https://github.com/owner/repo/tree/main/skill-name
    - git@github.com:owner/repo.git/path (SSH format)
    
    Returns:
        Tuple of (owner, repo, branch, path)
    """
    # HTTPS format: https://github.com/owner/repo/tree/branch/path
    https_pattern = r'https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)'
    match = re.match(https_pattern, url)
    if match:
        return match.groups()
    
    # Short HTTPS without tree: https://github.com/owner/repo/path
    short_pattern = r'https://github\.com/([^/]+)/([^/]+)/(?!tree)(.+)'
    match = re.match(short_pattern, url)
    if match:
        owner, repo, path = match.groups()
        return (owner, repo, 'main', path)
    
    # SSH format: git@github.com:owner/repo.git/path
    ssh_pattern = r'git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/(.+)'
    match = re.match(ssh_pattern, url)
    if match:
        owner, repo, path = match.groups()
        return (owner, repo, 'main', path)
    
    # Shorthand format: owner/repo/path or owner/repo/path@branch
    # Must have at least 3 parts (owner/repo/path)
    shorthand_pattern = r'^([^/@]+)/([^/@]+)/([^@]+?)(?:@([^/]+))?$'
    match = re.match(shorthand_pattern, url)
    if match:
        owner, repo, path, branch = match.groups()
        return (owner, repo, branch or 'main', path)
    
    raise ValueError(f"Could not parse GitHub URL or shorthand: {url}")


def _fetch_github_skill(url: str, ttl: Optional[int] = 86400) -> 'Skill':
    """Fetch a skill from GitHub and cache it locally.
    
    Args:
        url: GitHub URL to the skill directory
        ttl: Cache time-to-live in seconds (default: 86400 = 24 hours)
             None means cache forever, 0 means always fetch fresh
        
    Returns:
        Skill object loaded from GitHub
    """
    import time
    
    owner, repo, branch, path = _parse_github_url(url)
    
    # Create cache directory
    cache_path = SKILL_CACHE_DIR / owner / repo / branch / path
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Check if cache exists and is still valid
    instructions_file = cache_path / "SKILL.md"
    cache_meta_file = cache_path / ".cache_meta"
    
    should_fetch = True
    if ttl is not None and ttl > 0 and instructions_file.exists() and cache_meta_file.exists():
        try:
            cache_time = float(cache_meta_file.read_text().strip())
            age = time.time() - cache_time
            if age < ttl:
                should_fetch = False
                logger.info(f"Using cached skill from {cache_path} (age: {int(age)}s, ttl: {ttl}s)")
        except (ValueError, OSError):
            pass  # Invalid cache meta, refetch
    elif ttl is None and instructions_file.exists():
        # TTL=None means cache forever
        should_fetch = False
        logger.info(f"Using forever-cached skill from {cache_path}")
    
    if not should_fetch:
        # Load from cache
        try:
            skill_json_path = cache_path / "skill.json"
            if skill_json_path.exists():
                metadata = json.loads(skill_json_path.read_text())
            else:
                metadata = {
                    "name": path.split("/")[-1],
                    "description": f"Skill from {owner}/{repo}"
                }
            
            # Load cached resources
            resources = {}
            if "resources" in metadata:
                for key, resource_path in metadata["resources"].items():
                    resource_file = cache_path / resource_path
                    if resource_file.exists():
                        resources[key] = str(resource_file)
            
            return Skill(
                name=metadata.get("name", path.split("/")[-1]),
                description=metadata.get("description", f"Skill from {owner}/{repo}"),
                instructions=str(instructions_file),
                resources=resources if resources else None,
                _skip_validation=True
            )
        except Exception as e:
            logger.warning(f"Failed to load cached skill, refetching: {e}")
            should_fetch = True
    
    # Fetch from GitHub
    raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    
    # Fetch skill.json for metadata
    skill_json_url = f"{raw_base}/skill.json"
    skill_md_url = f"{raw_base}/SKILL.md"
    
    try:
        # Try to fetch skill.json first
        with urlopen(skill_json_url, timeout=10) as response:
            metadata = json.loads(response.read().decode())
            # Cache skill.json
            (cache_path / "skill.json").write_text(json.dumps(metadata, indent=2))
    except URLError:
        # No skill.json, use defaults from path
        metadata = {
            "name": path.split("/")[-1],
            "description": f"Skill from {owner}/{repo}"
        }
    
    # Fetch SKILL.md instructions
    try:
        with urlopen(skill_md_url, timeout=10) as response:
            instructions_content = response.read().decode()
            instructions_file.write_text(instructions_content)
            logger.info(f"Cached skill instructions: {instructions_file}")
    except URLError as e:
        raise FileNotFoundError(f"Could not fetch SKILL.md from {skill_md_url}: {e}")
    
    # Update cache metadata with current time
    cache_meta_file.write_text(str(time.time()))
    
    # Fetch resources if specified in metadata
    resources = {}
    if "resources" in metadata:
        for key, resource_path in metadata["resources"].items():
            resource_url = f"{raw_base}/{resource_path}"
            resource_file = cache_path / resource_path
            resource_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with urlopen(resource_url, timeout=10) as response:
                    resource_file.write_text(response.read().decode())
                resources[key] = str(resource_file)
                logger.info(f"Cached skill resource '{key}': {resource_file}")
            except URLError:
                logger.warning(f"Could not fetch resource '{key}' from {resource_url}")
    
    return Skill(
        name=metadata.get("name", path.split("/")[-1]),
        description=metadata.get("description", f"Skill from {owner}/{repo}"),
        instructions=str(instructions_file),
        resources=resources if resources else None,
        _skip_validation=True  # Already validated during fetch
    )


def load_skill(source: Union[str, 'Skill'], ttl: Optional[int] = 86400) -> 'Skill':
    """Load a skill from various sources.
    
    Args:
        source: Either a Skill object, GitHub URL, or local path
        ttl: Cache time-to-live in seconds for GitHub skills (default: 86400 = 24 hours)
             None means cache forever, 0 means always fetch fresh
        
    Returns:
        Skill object
        
    Examples:
        >>> skill = load_skill("hemanth/agentu-skills/pdf-processor")  # shorthand
        >>> skill = load_skill("owner/repo/skill@v1.0")  # with branch
        >>> skill = load_skill("https://github.com/hemanth/agentu-skills/tree/main/pdf")
        >>> skill = load_skill("./skills/my-skill")
        >>> skill = load_skill(existing_skill_object)
    """
    if isinstance(source, Skill):
        return source
    
    if not isinstance(source, str):
        raise TypeError(f"Expected Skill or str, got {type(source)}")
    
    # GitHub URL (full URL format)
    if source.startswith("https://github.com/") or source.startswith("git@github.com:"):
        return _fetch_github_skill(source, ttl=ttl)
    
    # Check if it's a shorthand GitHub format (owner/repo/path) 
    # Must have at least 2 slashes and not start with ./ or /
    if not source.startswith(('./','/')):
        parts = source.split('/')
        if len(parts) >= 3:
            # Looks like shorthand: owner/repo/path or owner/repo/path@branch
            return _fetch_github_skill(source, ttl=ttl)
    
    # Local path - look for SKILL.md or skill.json
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Skill path not found: {path}")
    
    skill_md = path / "SKILL.md"
    skill_json = path / "skill.json"
    
    if skill_json.exists():
        metadata = json.loads(skill_json.read_text())
    else:
        metadata = {
            "name": path.name,
            "description": f"Local skill from {path}"
        }
    
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {path}")
    
    # Load resources
    resources = {}
    if "resources" in metadata:
        for key, resource_path in metadata["resources"].items():
            full_path = path / resource_path
            if full_path.exists():
                resources[key] = str(full_path)
    
    return Skill(
        name=metadata.get("name", path.name),
        description=metadata.get("description", f"Skill from {path}"),
        instructions=str(skill_md),
        resources=resources if resources else None
    )


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
        
        # Or load from GitHub:
        >>> skill = load_skill("https://github.com/hemanth/agentu-skills/tree/main/pdf")
    """
    name: str
    description: str
    instructions: str
    resources: Optional[Dict[str, str]] = field(default_factory=dict)
    _skip_validation: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate."""
        # Skip validation for GitHub-fetched skills (already validated)
        if self._skip_validation:
            self.instructions = Path(self.instructions)
            if self.resources:
                self.resources = {k: Path(v) for k, v in self.resources.items()}
            return
            
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
