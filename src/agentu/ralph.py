"""
Ralph: Autonomous Agent Loop

Inspired by ghuntley.com/ralph - a technique for running AI agents
in a continuous loop until a goal is reached.

Usage:
    await agent.ralph("PROMPT.md", max_iterations=50, timeout_minutes=30)
"""

import asyncio
import time
import re
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RalphState:
    """State tracking for Ralph loop execution."""
    iteration: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    checkpoints_completed: List[str] = field(default_factory=list)
    last_result: Optional[str] = None
    total_tokens: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "started_at": self.started_at.isoformat(),
            "checkpoints_completed": self.checkpoints_completed,
            "last_result": self.last_result,
            "total_tokens": self.total_tokens,
            "errors": self.errors
        }


@dataclass 
class RalphConfig:
    """Configuration for Ralph autonomous loop."""
    prompt_file: str
    max_iterations: int = 50
    timeout_minutes: int = 30
    checkpoint_every: int = 5
    cost_limit_usd: Optional[float] = None
    pause_on_error: bool = True
    
    
class RalphRunner:
    """Runs an agent in an autonomous loop (Ralph mode)."""
    
    def __init__(self, agent, config: RalphConfig):
        self.agent = agent
        self.config = config
        self.state = RalphState()
        self._stop_requested = False
        
    def stop(self):
        """Request graceful stop of the loop."""
        self._stop_requested = True
        logger.info("Ralph: Stop requested, will halt after current iteration")
        
    async def run(
        self, 
        on_iteration: Optional[Callable[[int, Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the autonomous loop.
        
        Args:
            on_iteration: Optional callback called after each iteration
            
        Returns:
            Final state and results
        """
        logger.info(f"Ralph: Starting autonomous loop with {self.config.prompt_file}")
        
        start_time = time.time()
        timeout_seconds = self.config.timeout_minutes * 60
        
        try:
            for i in range(self.config.max_iterations):
                self.state.iteration = i + 1
                
                # Check stop conditions
                if self._stop_requested:
                    logger.info("Ralph: Stopping due to user request")
                    break
                    
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(f"Ralph: Timeout reached ({self.config.timeout_minutes} min)")
                    break
                
                # Read prompt file (re-read each iteration to pick up changes)
                prompt = self._read_prompt()
                if prompt is None:
                    break
                    
                # Check if goal already reached
                if self._check_completion(prompt):
                    logger.info("Ralph: All checkpoints complete, goal reached!")
                    break
                
                # Run agent inference
                try:
                    result = await self.agent.infer(prompt)
                    self.state.last_result = str(result)
                    
                    # Update state in prompt file
                    self._update_prompt_state(prompt)
                    
                except Exception as e:
                    error_msg = f"Iteration {i+1}: {str(e)}"
                    self.state.errors.append(error_msg)
                    logger.error(f"Ralph: {error_msg}")
                    
                    if self.config.pause_on_error:
                        logger.info("Ralph: Pausing due to error (pause_on_error=True)")
                        break
                
                # Callback
                if on_iteration:
                    on_iteration(i + 1, {
                        "result": self.state.last_result,
                        "elapsed_seconds": elapsed
                    })
                
                # Checkpoint
                if (i + 1) % self.config.checkpoint_every == 0:
                    self._save_checkpoint()
                    
        except KeyboardInterrupt:
            logger.info("Ralph: Interrupted by user")
            
        return self._build_summary()
    
    def _read_prompt(self) -> Optional[str]:
        """Read the prompt file."""
        try:
            path = Path(self.config.prompt_file)
            return path.read_text()
        except FileNotFoundError:
            logger.error(f"Ralph: Prompt file not found: {self.config.prompt_file}")
            return None
        except Exception as e:
            logger.error(f"Ralph: Error reading prompt: {e}")
            return None
            
    def _check_completion(self, prompt: str) -> bool:
        """Check if all checkpoints in prompt are complete."""
        # Look for checkbox pattern: - [x] or - [ ]
        incomplete = re.findall(r'- \[ \]', prompt)
        complete = re.findall(r'- \[x\]', prompt, re.IGNORECASE)
        
        if not incomplete and not complete:
            # No checkboxes found, can't determine completion
            return False
            
        if not incomplete:
            # All checkboxes are complete
            return True
            
        return False
    
    def _update_prompt_state(self, original_prompt: str):
        """Update the 'Current State' section in prompt file."""
        try:
            path = Path(self.config.prompt_file)
            
            # Update or add Current State section
            state_section = f"""## Current State
Last iteration: {self.state.iteration}
Last result: {self.state.last_result[:100] if self.state.last_result else 'None'}...
Errors: {len(self.state.errors)}
"""
            
            # Check if Current State section exists
            if "## Current State" in original_prompt:
                # Replace existing section
                updated = re.sub(
                    r'## Current State\n.*?(?=\n##|\Z)',
                    state_section,
                    original_prompt,
                    flags=re.DOTALL
                )
            else:
                # Append section
                updated = original_prompt + "\n" + state_section
                
            path.write_text(updated)
            
        except Exception as e:
            logger.warning(f"Ralph: Could not update prompt state: {e}")
    
    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        try:
            checkpoint_path = Path(self.config.prompt_file).parent / ".ralph_checkpoint.json"
            import json
            checkpoint_path.write_text(json.dumps(self.state.to_dict(), indent=2))
            logger.info(f"Ralph: Checkpoint saved at iteration {self.state.iteration}")
        except Exception as e:
            logger.warning(f"Ralph: Could not save checkpoint: {e}")
    
    def _build_summary(self) -> Dict[str, Any]:
        """Build final execution summary."""
        return {
            "iterations": self.state.iteration,
            "started_at": self.state.started_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "checkpoints_completed": self.state.checkpoints_completed,
            "errors": self.state.errors,
            "last_result": self.state.last_result,
            "stopped_by": (
                "user_request" if self._stop_requested else
                "completion" if self._check_completion(self._read_prompt() or "") else
                "max_iterations_or_timeout"
            )
        }


async def ralph(
    agent,
    prompt_file: str,
    max_iterations: int = 50,
    timeout_minutes: int = 30,
    checkpoint_every: int = 5,
    on_iteration: Optional[Callable[[int, Dict], None]] = None
) -> Dict[str, Any]:
    """
    Run an agent in Ralph mode (autonomous loop).
    
    Args:
        agent: The Agent instance to run
        prompt_file: Path to PROMPT.md file
        max_iterations: Maximum loop iterations (safety limit)
        timeout_minutes: Maximum runtime in minutes
        checkpoint_every: Save state every N iterations
        on_iteration: Optional callback for progress reporting
        
    Returns:
        Execution summary dict
        
    Example:
        >>> from agentu import Agent, ralph
        >>> agent = Agent("builder").with_tools([...])
        >>> result = await ralph(agent, "PROMPT.md", max_iterations=50)
    """
    config = RalphConfig(
        prompt_file=prompt_file,
        max_iterations=max_iterations,
        timeout_minutes=timeout_minutes,
        checkpoint_every=checkpoint_every
    )
    
    runner = RalphRunner(agent, config)
    return await runner.run(on_iteration=on_iteration)
