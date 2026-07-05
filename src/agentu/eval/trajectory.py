"""Trajectory evaluation for testing agent tool-call sequences.

Provides assertions over recorded Observer traces to verify that agents
follow expected tool-call patterns — useful for regression testing and
guardrail enforcement.

Usage::

    from agentu.eval.trajectory import TrajectoryAssertion, to_eval_case

    # Validate a recorded trace
    assertion = TrajectoryAssertion(
        expected_tools=["search", "summarize"],
        max_tool_calls=5,
        no_redundant_calls=True,
    )
    result = assertion.check(observer)

    # Convert a trace into a replayable EvalCase
    case = to_eval_case(observer, expected_output="42")
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..middleware.observe import Observer, EventType, Event

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    """Result of a trajectory assertion check."""
    passed: bool
    violations: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    total_calls: int = 0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Trajectory: {status} ({self.total_calls} tool calls)"]
        if self.violations:
            for v in self.violations:
                lines.append(f"  ✗ {v}")
        return "\n".join(lines)


@dataclass
class TrajectoryAssertion:
    """Assert properties over a recorded agent trajectory.

    A trajectory is the sequence of tool calls recorded by an
    :class:`Observer` during inference.

    Args:
        expected_tools: Expected ordered tool-call sequence.  When set,
            the actual tool sequence must contain these tools in order
            (not necessarily contiguously).
        max_tool_calls: Maximum number of tool calls allowed.  ``None``
            means unlimited.
        no_redundant_calls: If ``True``, flag consecutive duplicate
            tool calls with identical names as violations.
        required_tools: Tools that must appear at least once (unordered).
        forbidden_tools: Tools that must *not* appear.
    """
    expected_tools: Optional[List[str]] = None
    max_tool_calls: Optional[int] = None
    no_redundant_calls: bool = False
    required_tools: Optional[List[str]] = None
    forbidden_tools: Optional[List[str]] = None

    def check(self, observer: Observer) -> TrajectoryResult:
        """Run assertions against an Observer's recorded events.

        Args:
            observer: The Observer containing recorded events.

        Returns:
            A :class:`TrajectoryResult` with pass/fail and violations.
        """
        # Extract tool call names from events
        tool_calls = self._extract_tool_calls(observer.events)
        total_calls = len(tool_calls)
        violations: List[str] = []

        # Check expected tool sequence (subsequence match)
        if self.expected_tools is not None:
            if not self._is_subsequence(self.expected_tools, tool_calls):
                violations.append(
                    f"Expected tool sequence {self.expected_tools} not found "
                    f"in actual sequence {tool_calls}"
                )

        # Check max tool calls
        if self.max_tool_calls is not None and total_calls > self.max_tool_calls:
            violations.append(
                f"Too many tool calls: {total_calls} > {self.max_tool_calls}"
            )

        # Check for redundant consecutive calls
        if self.no_redundant_calls and tool_calls:
            redundant = []
            for i in range(1, len(tool_calls)):
                if tool_calls[i] == tool_calls[i - 1]:
                    redundant.append(f"{tool_calls[i]} at position {i}")
            if redundant:
                violations.append(
                    f"Redundant consecutive calls: {', '.join(redundant)}"
                )

        # Check required tools
        if self.required_tools:
            tool_set = set(tool_calls)
            missing = [t for t in self.required_tools if t not in tool_set]
            if missing:
                violations.append(f"Required tools missing: {missing}")

        # Check forbidden tools
        if self.forbidden_tools:
            tool_set = set(tool_calls)
            found = [t for t in self.forbidden_tools if t in tool_set]
            if found:
                violations.append(f"Forbidden tools used: {found}")

        return TrajectoryResult(
            passed=len(violations) == 0,
            violations=violations,
            tool_calls=tool_calls,
            total_calls=total_calls,
        )

    def _extract_tool_calls(self, events: List[Event]) -> List[str]:
        """Extract ordered list of tool names from events."""
        tool_calls = []
        for event in events:
            event_type = event.event_type
            if event_type == EventType.TOOL_CALL:
                tool_name = event.metadata.get("tool_name", "unknown")
                tool_calls.append(tool_name)
        return tool_calls

    @staticmethod
    def _is_subsequence(expected: List[str], actual: List[str]) -> bool:
        """Check if expected is a subsequence of actual."""
        it = iter(actual)
        return all(tool in it for tool in expected)


def to_eval_case(
    observer: Observer,
    expected_output: Optional[str] = None,
    query: Optional[str] = None,
    trajectory_assertion: Optional[TrajectoryAssertion] = None,
) -> Dict[str, Any]:
    """Convert a recorded Observer trace into an eval case dictionary.

    The returned dict is compatible with :func:`agentu.eval.evaluate`.

    Args:
        observer: Observer with a recorded trace.
        expected_output: Expected output string for the case.
        query: The original query.  If ``None``, extracted from the
            first ``INFERENCE_START`` event.
        trajectory_assertion: Optional trajectory assertion to attach.

    Returns:
        A dict with ``ask``, ``expect``, and optionally a ``validator``
        that also checks trajectory assertions.

    Example::

        case = to_eval_case(observer, expected_output="42")
        results = await evaluate(agent, [case])
    """
    # Extract query from recorded events
    if query is None:
        for event in observer.events:
            if event.event_type == EventType.INFERENCE_START:
                query = event.metadata.get("query", "")
                break
        if query is None:
            query = ""

    case: Dict[str, Any] = {
        "ask": query,
        "expect": expected_output or "",
    }

    # If a trajectory assertion is provided, build a custom validator
    if trajectory_assertion is not None:
        def _trajectory_validator(expected: Any, actual: Any) -> bool:
            """Validate both output and trajectory."""
            # Check trajectory
            traj_result = trajectory_assertion.check(observer)
            if not traj_result.passed:
                logger.warning(
                    "Trajectory assertion failed: %s",
                    "; ".join(traj_result.violations),
                )
                return False

            # Check output (substring match)
            if expected and str(expected).lower() not in str(actual).lower():
                return False

            return True

        case["validator"] = _trajectory_validator

    return case
