"""Evaluation system for testing agent performance.

Provides simple interface for testing agents against test cases with
automatic matching strategies and performance metrics.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import logging

from .agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class FailedCase:
    """Represents a failed test case."""
    query: str
    expected: Any
    actual: Any
    reason: str
    tool_used: Optional[str] = None
    turns: int = 0


@dataclass
class EvalResult:
    """Results from running evaluation."""
    passed: int
    failed: int
    total: int
    failures: List[FailedCase] = field(default_factory=list)
    avg_turns: float = 0.0
    total_tokens: int = 0
    duration: float = 0.0
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "accuracy": self.accuracy,
            "avg_turns": self.avg_turns,
            "total_tokens": self.total_tokens,
            "duration": self.duration,
            "failures": [
                {
                    "query": f.query,
                   "expected": str(f.expected),
                    "actual": str(f.actual),
                    "reason": f.reason,
                    "tool_used": f.tool_used,
                    "turns": f.turns
                }
                for f in self.failures
            ]
        }
    
    def to_json(self) -> str:
        """Export as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        """Human-readable summary with colors."""
        try:
            # Try to use colors
            GREEN = '\033[92m'
            RED = '\033[91m'
            BOLD = '\033[1m'
            RESET = '\033[0m'
        except:
            GREEN = RED = BOLD = RESET = ''
        
        lines = []
        
        # Summary line
        checkmark = f"{GREEN}✓{RESET}" if self.accuracy == 100 else f"{RED}✗{RESET}"
        lines.append(f"{BOLD}Results:{RESET}")
        lines.append(f"{checkmark} Passed: {self.passed}/{self.total} ({BOLD}{self.accuracy:.1f}%{RESET})")
        lines.append(f"  Avg turns: {self.avg_turns:.1f}")
        
        if self.duration > 0:
            lines.append(f"  Duration: {self.duration:.2f}s")
        
        # Individual test results
        if self.failures:
            lines.append(f"\n{BOLD}Failures:{RESET}")
            for fail in self.failures:
                lines.append(f"  {RED}✗{RESET} {fail.query}")
                lines.append(f"    Expected: {fail.expected}")
                lines.append(f"    Got: {fail.actual}")
                lines.append(f"    Reason: {fail.reason}")
                if fail.tool_used:
                    lines.append(f"    Tool: {fail.tool_used} ({fail.turns} turns)")
        
        return "\n".join(lines)


def _exact_match(expected: Any, actual: Any) -> bool:
    """Exact equality match."""
    return expected == actual


def _substring_match(expected: str, actual: Any) -> bool:
    """Substring match (case-insensitive)."""
    if not isinstance(expected, str):
        return False
    actual_str = str(actual).lower()
    expected_str = expected.lower()
    return expected_str in actual_str


async def _llm_judge_match(expected: Any, actual: Any, agent: Agent) -> tuple[bool, str]:
    """Use LLM to judge semantic similarity.
    
    Returns:
        Tuple of (matches: bool, reason: str)
    """
    prompt = f"""Compare these two responses and determine if they are semantically equivalent:

Expected: {expected}
Actual: {actual}

Respond with ONLY 'YES' or 'NO' followed by a brief reason.
Format: YES/NO - reason"""

    try:
        response = await agent._call_llm(prompt)
        response_lower = response.lower().strip()
        
        matches = response_lower.startswith('yes')
        reason = response.split('-', 1)[1].strip() if '-' in response else response
        
        return matches, reason
    except Exception as e:
        logger.error(f"LLM judge error: {e}")
        return False, f"LLM judge failed: {str(e)}"


async def _evaluate_case(
    agent: Agent,
    case: Dict[str, Any],
    use_llm_judge: bool = False
) -> tuple[bool, Optional[FailedCase], int]:
    """Evaluate a single test case.
    
    Returns:
        Tuple of (passed: bool, failure: Optional[FailedCase], turns: int)
    """
    query = case["ask"]
    expected = case["expect"]
    expected_tool = case.get("expect_tool")
    max_turns = case.get("max_turns", 10)
    timeout = case.get("timeout", 30)
    custom_validator = case.get("validator")
    
    # Run agent inference with timeout
    try:
        response = await asyncio.wait_for(
            agent.infer(query),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return False, FailedCase(
            query=query,
            expected=expected,
            actual="<timeout>",
            reason=f"Exceeded {timeout}s timeout"
        ), 0
    except Exception as e:
        return False, FailedCase(
            query=query,
            expected=expected,
            actual=f"<error: {e}>",
            reason=str(e)
        ), 0
    
    # Extract result and metadata
    actual = response.get("result")
    tool_used = response.get("tool_used")
    turns = response.get("turns", 1)
    
    # Check max turns
    if turns > max_turns:
        return False, FailedCase(
            query=query,
            expected=expected,
            actual=actual,
            reason=f"Exceeded max turns ({turns} > {max_turns})",
            tool_used=tool_used,
            turns=turns
        ), turns
    
    # Check expected tool
    if expected_tool and tool_used != expected_tool:
        return False, FailedCase(
            query=query,
            expected=expected,
            actual=actual,
            reason=f"Wrong tool used: {tool_used} (expected {expected_tool})",
            tool_used=tool_used,
            turns=turns
        ), turns
    
    # Validate result
    passed = False
    reason = ""
    
    if custom_validator:
        # Custom validation function
        try:
            passed = custom_validator(expected, actual)
            reason = "Custom validator" if passed else "Custom validator failed"
        except Exception as e:
            passed = False
            reason = f"Validator error: {e}"
    
    elif use_llm_judge:
        # LLM-as-judge
        passed, reason = await _llm_judge_match(expected, actual, agent)
    
    elif isinstance(expected, str) and not _exact_match(expected, actual):
        # Try substring match for strings
        passed = _substring_match(expected, actual)
        reason = "Substring match" if passed else "Not found in response"
    
    else:
        # Exact match
        passed = _exact_match(expected, actual)
        reason = "Exact match" if passed else "Values don't match"
    
    if passed:
        return True, None, turns
    else:
        return False, FailedCase(
            query=query,
            expected=expected,
            actual=actual,
            reason=reason,
            tool_used=tool_used,
            turns=turns
        ), turns


async def evaluate(
    agent: Agent,
    test_cases: List[Dict[str, Any]],
    use_llm_judge: bool = False
) -> EvalResult:
    """Evaluate agent performance against test cases.
    
    Args:
        agent: Agent to evaluate
        test_cases: List of test case dictionaries with format:
            {
                "ask": "query",
                "expect": "expected result",
                "expect_tool": "tool_name",  # optional
                "max_turns": 3,              # optional
                "timeout": 10,               # optional
                "validator": callable        # optional
            }
        use_llm_judge: Whether to use LLM for semantic matching
    
    Returns:
        EvalResult with metrics and failures
    
    Example:
        >>> cases = [
        ...     {"ask": "What's 5 + 3?", "expect": 8},
        ...     {"ask": "Weather in SF?", "expect": "sunny"}
        ... ]
        >>> results = await evaluate(agent, cases)
        >>> print(f"Accuracy: {results.accuracy}%")
    """
    start_time = time.time()
    
    passed = 0
    failed = 0
    failures = []
    total_turns = 0
    
    for case in test_cases:
        case_passed, failure, turns = await _evaluate_case(
            agent, case, use_llm_judge
        )
        
        total_turns += turns
        
        if case_passed:
            passed += 1
            logger.info(f"✓ {case['ask']}")
        else:
            failed += 1
            failures.append(failure)
            logger.warning(f"✗ {case['ask']}: {failure.reason}")
    
    total = len(test_cases)
    duration = time.time() - start_time
    avg_turns = total_turns / total if total > 0 else 0.0
    
    return EvalResult(
        passed=passed,
        failed=failed,
        total=total,
        failures=failures,
        avg_turns=avg_turns,
        duration=duration
    )
