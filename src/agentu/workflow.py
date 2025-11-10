"""Workflow system for composing agent pipelines using operators.

Operators:
    >> : Sequential execution (data flows from left to right)
    &  : Parallel execution (run concurrently, merge results)

Example:
    # Sequential
    workflow = researcher("Find trends") >> analyst("Analyze") >> writer("Write report")

    # Parallel
    workflow = researcher("AI") & researcher("ML") & researcher("Crypto")

    # Combined
    workflow = (researcher("AI") & researcher("ML")) >> analyst("Compare results")
"""

import asyncio
import logging
from typing import Any, Callable, Union, List, Optional

logger = logging.getLogger(__name__)


class Step:
    """A single workflow step that executes an agent with a task.

    Can be composed with other steps using:
        >> : Sequential (this then that)
        &  : Parallel (this and that concurrently)
    """

    def __init__(self, agent: Any, task: Union[str, Callable]):
        """Create a workflow step.

        Args:
            agent: Agent instance to execute
            task: Task string or lambda function
                 - str: Static task, context auto-injected if available
                 - callable: Takes previous result, returns task string
        """
        self.agent = agent
        self.task = task

    def __rshift__(self, other: 'Step') -> 'SequentialStep':
        """>> operator: Sequential execution.

        Args:
            other: Next step to execute after this one

        Returns:
            SequentialStep that runs this then other
        """
        return SequentialStep(self, other)

    def __and__(self, other: 'Step') -> 'ParallelStep':
        """& operator: Parallel execution.

        Args:
            other: Step to run concurrently with this one

        Returns:
            ParallelStep that runs both concurrently
        """
        return ParallelStep(self, other)

    async def run(self, context: Any = None) -> Any:
        """Execute this step.

        Args:
            context: Result from previous step (if any)

        Returns:
            Result from agent execution
        """
        try:
            # Determine the prompt
            if callable(self.task):
                # User-provided lambda - full control
                prompt = self.task(context)
            elif context is not None:
                # Auto-inject previous result
                prompt = f"{self.task}\n\n--- Context from previous step ---\n{context}"
            else:
                # First step - no context
                prompt = self.task

            logger.info(f"Executing step with agent: {self.agent.name}")
            result = await self.agent.infer(prompt)
            return result

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {"error": str(e), "step": "failed"}


class SequentialStep:
    """Sequential composition of two steps (left >> right).

    Executes left step first, passes result to right step.
    """

    def __init__(self, left: Step, right: Step):
        """Create sequential composition.

        Args:
            left: First step to execute
            right: Second step to execute (receives left's result)
        """
        self.left = left
        self.right = right

    def __rshift__(self, other: Step) -> 'SequentialStep':
        """Chain another step sequentially."""
        return SequentialStep(self, other)

    def __and__(self, other: Step) -> 'ParallelStep':
        """Run this sequence in parallel with another step."""
        return ParallelStep(self, other)

    async def run(self, context: Any = None) -> Any:
        """Execute steps sequentially.

        Args:
            context: Initial context (if any)

        Returns:
            Result from final step
        """
        # Execute left step
        left_result = await self.left.run(context)

        # Pass result to right step
        right_result = await self.right.run(left_result)

        return right_result


class ParallelStep:
    """Parallel composition of multiple steps (step1 & step2 & ...).

    Executes all steps concurrently, returns list of results.
    """

    def __init__(self, *steps: Step):
        """Create parallel composition.

        Args:
            *steps: Steps to execute concurrently
        """
        # Flatten nested ParallelSteps
        self.steps = []
        for step in steps:
            if isinstance(step, ParallelStep):
                self.steps.extend(step.steps)
            else:
                self.steps.append(step)

    def __and__(self, other: Step) -> 'ParallelStep':
        """Add another step to run in parallel."""
        return ParallelStep(*self.steps, other)

    def __rshift__(self, other: Step) -> SequentialStep:
        """Chain a step after all parallel steps complete."""
        return SequentialStep(self, other)

    async def run(self, context: Any = None) -> List[Any]:
        """Execute all steps concurrently.

        Args:
            context: Shared context for all steps

        Returns:
            List of results from all steps
        """
        logger.info(f"Executing {len(self.steps)} steps in parallel")

        # Run all steps concurrently
        results = await asyncio.gather(
            *[step.run(context) for step in self.steps],
            return_exceptions=True
        )

        # Convert exceptions to error dicts
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "error": str(result),
                    "step": i,
                    "failed": True
                })
            else:
                formatted_results.append(result)

        return formatted_results
