"""
AGENTS.md / CLAUDE.md auto-discovery.

Hierarchically discovers and loads rule files from project directories,
supporting the open standard filenames:
- AGENTS.md
- .agents/AGENTS.md
- CLAUDE.md
- .claude/CLAUDE.md

Rules from all discovered files are aggregated (repo root first, then
subdirectories in depth-first order).
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Candidate filenames, checked in priority order at each directory level.
# The first match at each level wins (no double-loading at the same level).
_RULE_FILE_CANDIDATES = [
    "AGENTS.md",
    Path(".agents") / "AGENTS.md",
    "CLAUDE.md",
    Path(".claude") / "CLAUDE.md",
]


def discover_rules(
    start_dir: Optional[str] = None,
    recursive: bool = True,
    max_depth: int = 3,
) -> List[str]:
    """Discover AGENTS.md / CLAUDE.md files hierarchically.

    Searches ``start_dir`` (defaults to cwd) and, if ``recursive`` is True,
    its subdirectories up to ``max_depth`` levels deep.

    At each directory level, files are checked in this priority order:
    1. ``AGENTS.md``
    2. ``.agents/AGENTS.md``
    3. ``CLAUDE.md``
    4. ``.claude/CLAUDE.md``

    The *first* match at each directory level is used (no duplicates from
    the same level).  Results are ordered root-first.

    Args:
        start_dir: Directory to start searching from (default: cwd).
        recursive: Whether to search subdirectories (default: True).
        max_depth: Maximum subdirectory depth to search (default: 3).

    Returns:
        List of file contents (strings) from all discovered rule files,
        ordered from root to deepest subdirectory.
    """
    root = Path(start_dir) if start_dir else Path.cwd()
    if not root.is_dir():
        logger.warning(f"Start directory does not exist: {root}")
        return []

    results: List[str] = []
    _search_dir(root, results, current_depth=0, max_depth=max_depth, recursive=recursive)
    return results


def _search_dir(
    directory: Path,
    results: List[str],
    current_depth: int,
    max_depth: int,
    recursive: bool,
) -> None:
    """Recursively search a directory for rule files."""
    # Check candidates at this level
    found_at_level = False
    for candidate in _RULE_FILE_CANDIDATES:
        full_path = directory / candidate
        if full_path.is_file():
            try:
                content = full_path.read_text().strip()
                if content:
                    results.append(content)
                    logger.info(f"Discovered rules file: {full_path} ({len(content)} chars)")
                    found_at_level = True
                    break  # First match at this level wins
            except OSError as e:
                logger.warning(f"Failed to read rules file {full_path}: {e}")

    # Recurse into subdirectories
    if recursive and current_depth < max_depth:
        try:
            for child in sorted(directory.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    _search_dir(child, results, current_depth + 1, max_depth, recursive)
        except PermissionError:
            pass


def discover_and_format_rules(
    start_dir: Optional[str] = None,
    recursive: bool = True,
    max_depth: int = 3,
) -> Optional[str]:
    """Discover rule files and format them for agent context injection.

    Convenience wrapper that calls :func:`discover_rules` and joins the
    results into a single string suitable for prepending to a system prompt.

    Args:
        start_dir: Directory to start searching from (default: cwd).
        recursive: Whether to search subdirectories (default: True).
        max_depth: Maximum subdirectory depth (default: 3).

    Returns:
        Formatted rules string, or None if no rules were found.
    """
    rule_contents = discover_rules(
        start_dir=start_dir,
        recursive=recursive,
        max_depth=max_depth,
    )

    if not rule_contents:
        return None

    # Join multiple rule files with separators
    if len(rule_contents) == 1:
        return rule_contents[0]

    parts = []
    for i, content in enumerate(rule_contents):
        if i > 0:
            parts.append("")  # blank line separator
        parts.append(content)

    return "\n".join(parts)
