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
    max_tokens: Optional[int] = None  # Per-subagent token budget

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
            max_tokens=data.get("max_tokens"),
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


async def run_judge_panel(
    task: str,
    makers: List[Any],
    checkers: List[Any],
    num_judges: int = 3,
    max_corrections: int = 1,
    parent_observer=None,
) -> Dict[str, Any]:
    """Run the maker-checker pattern with a panel of judges.

    Instead of a single checker, ``num_judges`` checker instances vote
    concurrently on the maker output.  The output is approved when a
    strict majority of judges approve.

    Args:
        task: The task to delegate
        makers: List of maker Agent instances
        checkers: List of checker Agent instances (used as templates)
        num_judges: Number of judges in the panel (default: 3)
        max_corrections: Max correction attempts (default: 1)
        parent_observer: Optional observer for event tracking

    Returns:
        Dict with result, reviews, votes, approved, and maker/checker info
    """
    if not makers:
        raise ValueError("At least one maker sub-agent is required")
    if not checkers:
        raise ValueError("At least one checker sub-agent is required for judge panel")

    maker = makers[0]
    checker_template = checkers[0]

    # Emit delegate start event
    if parent_observer:
        from ..middleware.observe import EventType
        parent_observer.record(
            EventType.DELEGATE_START,
            metadata={
                "task": task[:200],
                "maker": maker.name,
                "checker": checker_template.name,
                "judges": num_judges,
            }
        )

    # Step 1: Maker runs the task
    logger.info(f"Judge Panel: Maker '{maker.name}' starting task")
    maker_result = await maker.infer(task)
    maker_output = str(maker_result)

    # Step 2: Panel of judges review concurrently
    review_prompt = (
        f"Review the following output for correctness, quality, and completeness.\n\n"
        f"## Original Task\n{task}\n\n"
        f"## Output to Review\n{maker_output}\n\n"
        f"Respond with your assessment. Start with 'APPROVED:' if the output is acceptable, "
        f"or 'NEEDS REVISION:' followed by specific feedback if changes are needed."
    )

    async def _single_review(judge_idx: int) -> Dict[str, Any]:
        """Run a single judge's review."""
        logger.info(f"Judge Panel: Judge {judge_idx + 1}/{num_judges} reviewing")
        result = await checker_template.infer(review_prompt)
        review_text = str(result)
        approved = review_text.strip().upper().startswith("APPROVED")
        return {"review": review_text, "approved": approved, "judge": judge_idx}

    # Run all judges concurrently
    judge_tasks = [_single_review(i) for i in range(num_judges)]
    reviews = await asyncio.gather(*judge_tasks)

    # Count votes
    approvals = sum(1 for r in reviews if r["approved"])
    majority = num_judges / 2
    panel_approved = approvals > majority

    # Emit review event
    if parent_observer:
        from ..middleware.observe import EventType
        parent_observer.record(
            EventType.DELEGATE_REVIEW,
            metadata={
                "maker": maker.name,
                "checker": checker_template.name,
                "judges": num_judges,
                "approvals": approvals,
                "approved": panel_approved,
            }
        )

    logger.info(
        f"Judge Panel: {approvals}/{num_judges} approved "
        f"({'APPROVED' if panel_approved else 'REJECTED'})"
    )

    return {
        "result": maker_output,
        "reviews": [r["review"] for r in reviews],
        "votes": {"approved": approvals, "rejected": num_judges - approvals},
        "approved": panel_approved,
        "corrections": 0,
        "maker": maker.name,
        "checker": checker_template.name,
        "judges": num_judges,
    }


async def run_best_of(
    agent: Any,
    n: int,
    task: str,
    judge: Optional[Any] = None,
    parent_observer=None,
) -> Dict[str, Any]:
    """Run N instances of an agent in parallel and pick the best result.

    Each instance runs the same task independently.  When a judge is
    provided, it evaluates all results and selects the best one.

    Args:
        agent: The Agent to run N instances of
        n: Number of parallel instances
        task: The task to run
        judge: Optional Agent to judge results
        parent_observer: Optional observer for event tracking

    Returns:
        Dict with result, all_results, chosen_index, judge_reasoning
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    logger.info(f"Best-of-{n}: Running {n} instances of '{agent.name}'")

    # Run all instances concurrently
    async def _run_instance(idx: int) -> Dict[str, Any]:
        result = await agent.infer(task)
        return {"index": idx, "result": str(result)}

    instance_tasks = [_run_instance(i) for i in range(n)]
    all_results = await asyncio.gather(*instance_tasks)

    # No judge — return the first result
    if judge is None or n == 1:
        return {
            "result": all_results[0]["result"],
            "all_results": [r["result"] for r in all_results],
            "chosen_index": 0,
            "judge_reasoning": None,
        }

    # Have the judge pick the best
    results_text = "\n\n".join(
        f"### Result {i + 1}\n{r['result']}" for i, r in enumerate(all_results)
    )
    judge_prompt = (
        f"You are judging {n} responses to the same task. "
        f"Pick the best one.\n\n"
        f"## Task\n{task}\n\n"
        f"## Candidate Results\n{results_text}\n\n"
        f"Respond with the number of the best result (1-{n}) "
        f"followed by your reasoning. Format: BEST: <number> - <reason>"
    )

    logger.info(f"Best-of-{n}: Judge '{judge.name}' evaluating {n} results")
    judge_result = await judge.infer(judge_prompt)
    judge_text = str(judge_result)

    # Parse the judge's choice
    chosen_index = 0  # Default to first
    reasoning = judge_text
    try:
        import re
        match = re.search(r"BEST:\s*(\d+)", judge_text, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1  # Convert 1-indexed to 0-indexed
            if 0 <= idx < n:
                chosen_index = idx
            reasoning = judge_text.split("-", 1)[1].strip() if "-" in judge_text else judge_text
    except (ValueError, IndexError):
        pass

    if parent_observer:
        from ..middleware.observe import EventType
        parent_observer.record(
            EventType.DELEGATE_REVIEW,
            metadata={
                "mode": "best_of",
                "n": n,
                "chosen_index": chosen_index,
                "judge": judge.name,
            }
        )

    return {
        "result": all_results[chosen_index]["result"],
        "all_results": [r["result"] for r in all_results],
        "chosen_index": chosen_index,
        "judge_reasoning": reasoning,
    }


async def run_judge_panel(
    task: str,
    makers: List[Any],
    checkers: List[Any],
    num_judges: int = 3,
    max_corrections: int = 1,
    parent_observer=None,
) -> Dict[str, Any]:
    """Run maker-checker with a panel of judges for consensus.

    Instead of a single checker, multiple checker instances independently
    review the maker output. The output is approved only when a majority
    of judges approve.

    Args:
        task: The task to delegate
        makers: List of maker Agent instances
        checkers: List of checker Agent instances (at least one required)
        num_judges: Number of judge instances for voting
        max_corrections: Max correction attempts
        parent_observer: Optional observer for event tracking

    Returns:
        Dict with result, reviews, approval status, and vote tally
    """
    if not makers:
        raise ValueError("At least one maker sub-agent is required")
    if not checkers:
        raise ValueError("At least one checker sub-agent is required for judge panel")

    maker = makers[0]
    checker_template = checkers[0]

    if parent_observer:
        from ..middleware.observe import EventType
        parent_observer.record(
            EventType.DELEGATE_START,
            metadata={
                "task": task[:200],
                "maker": maker.name,
                "checker": checker_template.name,
                "judges": num_judges,
                "mode": "judge_panel",
            }
        )

    # Step 1: Maker runs the task
    logger.info(f"Judge panel: Maker '{maker.name}' starting task")
    maker_result = await maker.infer(task)
    maker_output = str(maker_result)

    # Step 2: Judge panel reviews with correction loop
    corrections = 0

    for attempt in range(max_corrections + 1):
        review_prompt = (
            f"Review the following output for correctness, quality, and completeness.\n\n"
            f"## Original Task\n{task}\n\n"
            f"## Output to Review\n{maker_output}\n\n"
            f"Respond with your assessment. Start with 'APPROVED:' if the output is acceptable, "
            f"or 'NEEDS REVISION:' followed by specific feedback if changes are needed."
        )

        # Run all judges in parallel
        logger.info(
            f"Judge panel: {num_judges} judges reviewing (attempt {attempt + 1})"
        )

        async def _judge_review(judge_id: int) -> Dict[str, Any]:
            result = await checker_template.infer(review_prompt)
            output = str(result)
            approved = output.strip().upper().startswith("APPROVED")
            return {
                "judge_id": judge_id,
                "output": output,
                "approved": approved,
            }

        reviews = await asyncio.gather(
            *[_judge_review(i) for i in range(num_judges)]
        )

        approvals = sum(1 for r in reviews if r["approved"])
        majority = num_judges // 2 + 1
        consensus = approvals >= majority

        if parent_observer:
            from ..middleware.observe import EventType
            parent_observer.record(
                EventType.DELEGATE_REVIEW,
                metadata={
                    "maker": maker.name,
                    "checker": checker_template.name,
                    "attempt": attempt + 1,
                    "approvals": approvals,
                    "rejections": num_judges - approvals,
                    "consensus": consensus,
                    "mode": "judge_panel",
                }
            )

        if consensus:
            return {
                "result": maker_output,
                "reviews": reviews,
                "approved": True,
                "corrections": corrections,
                "maker": maker.name,
                "checker": checker_template.name,
                "votes": {"approved": approvals, "rejected": num_judges - approvals},
            }

        # Majority rejected — collect feedback
        corrections += 1
        if attempt < max_corrections:
            # Aggregate rejection feedback
            feedback = "\n".join(
                f"Judge {r['judge_id']}: {r['output']}"
                for r in reviews
                if not r["approved"]
            )

            correction_prompt = (
                f"Your previous output was reviewed by {num_judges} judges. "
                f"{num_judges - approvals} of {num_judges} rejected it.\n\n"
                f"## Feedback\n{feedback}\n\n"
                f"## Original Task\n{task}\n\n"
                f"Please revise your output based on the feedback."
            )

            logger.info(f"Judge panel: Maker revising (correction {corrections})")
            maker_result = await maker.infer(correction_prompt)
            maker_output = str(maker_result)

    # Max corrections reached without consensus
    return {
        "result": maker_output,
        "reviews": reviews,
        "approved": False,
        "corrections": corrections,
        "maker": maker.name,
        "checker": checker_template.name,
        "votes": {"approved": approvals, "rejected": num_judges - approvals},
    }

