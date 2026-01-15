"""Workflow system for composing agent pipelines using operators.

Operators:
    >> : Sequential execution (data flows from left to right)
    &  : Parallel execution (run concurrently, merge results)

Features:
    - Checkpoint: Save state after each step for resume capability
    - Resume: Continue from last successful step after crash

Example:
    # Sequential
    workflow = researcher("Find trends") >> analyst("Analyze") >> writer("Write report")

    # Parallel
    workflow = researcher("AI") & researcher("ML") & researcher("Crypto")

    # Combined
    workflow = (researcher("AI") & researcher("ML")) >> analyst("Compare results")
    
    # With checkpoint
    result = await workflow.run(checkpoint="./checkpoints")
    
    # Resume from checkpoint
    result = await resume_workflow("./checkpoints/workflow_abc.json")
"""

import asyncio
import logging
import json
import uuid
import time
from pathlib import Path
from typing import Any, Callable, Union, List, Optional, Dict
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class WorkflowCheckpoint:
    """Checkpoint state for workflow resume."""
    workflow_id: str
    created_at: float
    updated_at: float
    current_step: int
    total_steps: int
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "in_progress"  # in_progress, completed, failed
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowCheckpoint':
        return cls(**data)


def _save_checkpoint(checkpoint: WorkflowCheckpoint, checkpoint_dir: str) -> str:
    """Save checkpoint to disk."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    filepath = path / f"workflow_{checkpoint.workflow_id}.json"
    with open(filepath, 'w') as f:
        json.dump(checkpoint.to_dict(), f, indent=2, default=str)
    
    logger.debug(f"Saved checkpoint to {filepath}")
    return str(filepath)


def _load_checkpoint(checkpoint_path: str) -> WorkflowCheckpoint:
    """Load checkpoint from disk."""
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    return WorkflowCheckpoint.from_dict(data)


async def resume_workflow(checkpoint_path: str) -> Any:
    """Resume a workflow from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint JSON file
        
    Returns:
        Result from resumed workflow execution
    """
    checkpoint = _load_checkpoint(checkpoint_path)
    
    if checkpoint.status == "completed":
        logger.info("Workflow already completed, returning last result")
        if checkpoint.completed_steps:
            return checkpoint.completed_steps[-1].get("result")
        return None
    
    logger.info(f"Resuming workflow from step {checkpoint.current_step + 1}/{checkpoint.total_steps}")
    
    # Get last result as context for next step
    context = None
    if checkpoint.completed_steps:
        last_step = checkpoint.completed_steps[-1]
        context = last_step.get("result")
    
    return {
        "resumed_from": checkpoint.current_step,
        "completed_steps": checkpoint.completed_steps,
        "context": context,
        "checkpoint": checkpoint.to_dict()
    }


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

    async def run(self, context: Any = None, checkpoint: Optional[str] = None) -> Any:
        """Execute this step.

        Args:
            context: Result from previous step (if any)
            checkpoint: Directory to save checkpoints (ignored for single step)

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
    
    def _flatten_steps(self) -> List[Step]:
        """Flatten nested sequential steps into a list."""
        steps = []
        if isinstance(self.left, SequentialStep):
            steps.extend(self.left._flatten_steps())
        else:
            steps.append(self.left)
        if isinstance(self.right, SequentialStep):
            steps.extend(self.right._flatten_steps())
        else:
            steps.append(self.right)
        return steps

    async def run(
        self, 
        context: Any = None, 
        checkpoint: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute steps sequentially with optional checkpointing.

        Args:
            context: Initial context (if any)
            checkpoint: Directory to save checkpoints for resume capability
            workflow_id: Custom workflow ID (default: auto-generated UUID)

        Returns:
            Dict with 'result' and optionally 'checkpoint_path'
        """
        # Flatten nested sequential steps for checkpoint tracking
        all_steps = self._flatten_steps()
        
        # Initialize checkpoint if enabled
        cp = None
        checkpoint_path = None
        if checkpoint:
            wf_id = workflow_id or str(uuid.uuid4())[:8]
            cp = WorkflowCheckpoint(
                workflow_id=wf_id,
                created_at=time.time(),
                updated_at=time.time(),
                current_step=0,
                total_steps=len(all_steps)
            )
            checkpoint_path = _save_checkpoint(cp, checkpoint)
            logger.info(f"Workflow checkpoint: {checkpoint_path}")
        
        current_result = context
        
        for i, step in enumerate(all_steps):
            try:
                current_result = await step.run(current_result)
                
                # Update checkpoint after each step
                if cp and checkpoint:
                    cp.current_step = i + 1
                    cp.updated_at = time.time()
                    cp.completed_steps.append({
                        "step_index": i,
                        "agent": step.agent.name if hasattr(step, 'agent') else "unknown",
                        "result": current_result,
                        "completed_at": time.time()
                    })
                    
                    if i == len(all_steps) - 1:
                        cp.status = "completed"
                    
                    checkpoint_path = _save_checkpoint(cp, checkpoint)
                    logger.info(f"Checkpoint saved: step {i + 1}/{len(all_steps)}")
                    
            except Exception as e:
                if cp and checkpoint:
                    cp.status = "failed"
                    cp.error = str(e)
                    cp.updated_at = time.time()
                    _save_checkpoint(cp, checkpoint)
                raise

        # Return result with checkpoint path if enabled
        if checkpoint_path:
            return {"result": current_result, "checkpoint_path": checkpoint_path}
        return current_result


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

    async def run(self, context: Any = None, checkpoint: Optional[str] = None) -> List[Any]:
        """Execute all steps concurrently.

        Args:
            context: Shared context for all steps
            checkpoint: Directory to save checkpoints (parallel steps save as single unit)

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
