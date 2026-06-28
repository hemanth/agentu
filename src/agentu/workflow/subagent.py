"""
Sub-agent system for agentu loop engineering.

Define agent roles (maker, checker) and run them in structured patterns.

Usage:
    agent = Agent("lead").with_subagents([
        {"name": "coder", "instructions": "Write clean code.", "role": "maker"},
        {"name": "reviewer", "instructions": "Review for bugs.", "role": "checker"},
    ])
    result = await agent.delegate("Refactor the auth module")
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""
    name: str
    instructions: str = ""
    description: str = ""
    role: str = "maker"  # maker | checker
    model: Optional[str] = None  # None = inherit from parent
    temperature: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubAgentConfig':
        """Create from a dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            instructions=data.get("instructions", ""),
            description=data.get("description", ""),
            role=data.get("role", "maker"),
            model=data.get("model"),
            temperature=data.get("temperature"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'SubAgentConfig':
        """Load from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Sub-agent config not found: {path}")

        text = p.read_text()

        if p.suffix in ('.yaml', '.yml'):
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to load .yaml sub-agent configs. "
                    "Install with: pip install agentu[yaml]"
                )
            data = yaml.safe_load(text)
        elif p.suffix == '.json':
            data = json.loads(text)
        else:
            raise ValueError(f"Unsupported config format: {p.suffix}")

        return cls.from_dict(data)


def load_subagent_configs(source: Union[str, List[Dict[str, Any]]]) -> List[SubAgentConfig]:
    """Load sub-agent configurations from a directory path or list of dicts.

    Args:
        source: Either a path to a directory containing YAML/JSON configs,
                or a list of dicts with sub-agent definitions.

    Returns:
        List of SubAgentConfig objects.
    """
    if isinstance(source, list):
        return [SubAgentConfig.from_dict(d) for d in source]

    # Directory path
    dir_path = Path(source)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Sub-agent directory not found: {source}")

    configs = []
    for f in sorted(dir_path.iterdir()):
        if f.suffix in ('.yaml', '.yml', '.json') and not f.name.startswith('.'):
            try:
                configs.append(SubAgentConfig.from_yaml(str(f)))
                logger.info(f"Loaded sub-agent config: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load sub-agent config {f.name}: {e}")

    if not configs:
        logger.warning(f"No sub-agent configs found in {source}")

    return configs


def _build_subagent(config: SubAgentConfig, parent_agent) -> Any:
    """Build an Agent instance from a SubAgentConfig.

    The sub-agent inherits the parent's model and api settings
    unless overridden in the config.

    Args:
        config: Sub-agent configuration
        parent_agent: Parent Agent to inherit settings from

    Returns:
        Agent instance configured as a sub-agent
    """
    # Import here to avoid circular dependency
    from .._core.agent import Agent

    model = config.model or parent_agent.model
    temperature = config.temperature or parent_agent.temperature

    agent = Agent(
        name=config.name,
        model=model,
        temperature=temperature,
        api_base=parent_agent.api_base,
        api_key=parent_agent.api_key,
        enable_memory=False,  # sub-agents don't need persistent memory
    )

    # Set instructions as system context
    if config.instructions:
        agent.context = config.instructions

    # Copy parent's tools to sub-agent
    for tool in parent_agent.tools:
        if tool.name not in ("search_tools", "get_skill_resource"):
            agent.tools.append(tool)

    return agent


async def run_maker_checker(
    task: str,
    makers: List[Any],
    checkers: List[Any],
    max_corrections: int = 1,
    parent_observer=None,
) -> Dict[str, Any]:
    """Run the maker-checker delegation pattern.

    Flow:
        1. First maker runs the task
        2. First checker reviews the output
        3. If rejected and corrections remain, maker retries with feedback
        4. Returns result with review metadata

    Args:
        task: The task to delegate
        makers: List of maker Agent instances
        checkers: List of checker Agent instances
        max_corrections: Max correction attempts (default: 1)
        parent_observer: Optional observer for event tracking

    Returns:
        Dict with result, review, and approval status
    """
    if not makers:
        raise ValueError("At least one maker sub-agent is required")

    maker = makers[0]
    checker = checkers[0] if checkers else None

    # Emit delegate start event
    if parent_observer:
        from ..middleware.observe import EventType
        parent_observer.record(
            EventType.DELEGATE_START,
            metadata={"task": task[:200], "maker": maker.name,
                       "checker": checker.name if checker else None}
        )

    # Step 1: Maker runs the task
    logger.info(f"Delegate: Maker '{maker.name}' starting task")
    maker_result = await maker.infer(task)
    maker_output = str(maker_result)

    # No checker — return maker's output directly
    if not checker:
        return {
            "result": maker_output,
            "review": None,
            "approved": True,
            "corrections": 0,
            "maker": maker.name,
            "checker": None,
        }

    # Step 2-3: Checker reviews, with correction loop
    corrections = 0
    current_output = maker_output

    for attempt in range(max_corrections + 1):
        # Checker reviews
        review_prompt = (
            f"Review the following output for correctness, quality, and completeness.\n\n"
            f"## Original Task\n{task}\n\n"
            f"## Output to Review\n{current_output}\n\n"
            f"Respond with your assessment. Start with 'APPROVED:' if the output is acceptable, "
            f"or 'NEEDS REVISION:' followed by specific feedback if changes are needed."
        )

        logger.info(f"Delegate: Checker '{checker.name}' reviewing (attempt {attempt + 1})")
        review_result = await checker.infer(review_prompt)
        review_output = str(review_result)

        # Emit review event
        if parent_observer:
            from ..middleware.observe import EventType
            parent_observer.record(
                EventType.DELEGATE_REVIEW,
                metadata={
                    "maker": maker.name,
                    "checker": checker.name,
                    "attempt": attempt + 1,
                    "approved": review_output.strip().upper().startswith("APPROVED"),
                }
            )

        # Check if approved
        if review_output.strip().upper().startswith("APPROVED"):
            return {
                "result": current_output,
                "review": review_output,
                "approved": True,
                "corrections": corrections,
                "maker": maker.name,
                "checker": checker.name,
            }

        # Not approved — attempt correction if allowed
        if attempt < max_corrections:
            corrections += 1
            correction_prompt = (
                f"Your previous output was reviewed and needs revision.\n\n"
                f"## Original Task\n{task}\n\n"
                f"## Your Previous Output\n{current_output}\n\n"
                f"## Reviewer Feedback\n{review_output}\n\n"
                f"Please revise your output based on the feedback."
            )

            logger.info(f"Delegate: Maker '{maker.name}' correcting (attempt {corrections})")
            correction_result = await maker.infer(correction_prompt)
            current_output = str(correction_result)

    # Max corrections reached, return with rejection
    return {
        "result": current_output,
        "review": review_output,
        "approved": False,
        "corrections": corrections,
        "maker": maker.name,
        "checker": checker.name,
    }
