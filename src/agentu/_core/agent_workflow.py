"""WorkflowMixin – workflow, scheduling, and sub-agent methods extracted from Agent."""

import asyncio
import logging
from typing import Dict, Any, Optional, Union, List

from ..middleware.observe import EventType

logger = logging.getLogger(__name__)


class WorkflowMixin:
    """Mixin providing workflow orchestration for the Agent class.

    Methods here assume they are mixed into an Agent instance that has
    ``name``, ``observer``, ``_worktree_config``, and related attributes.
    """

    def with_subagents(
        self,
        agents: Union[str, List[Dict[str, Any]]],
    ) -> 'WorkflowMixin':
        """Configure sub-agents for delegation.

        Sub-agents are lightweight agent roles (maker, checker) that
        can be composed into structured patterns like maker-checker.

        Args:
            agents: Either a path to .agents/ directory containing
                    YAML/JSON configs, or a list of dicts with
                    sub-agent definitions.

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("lead").with_subagents([
            ...     {"name": "coder", "instructions": "Write code.", "role": "maker"},
            ...     {"name": "reviewer", "instructions": "Review code.", "role": "checker"},
            ... ])
            >>> agent = Agent("lead").with_subagents(".agents/")
        """
        from ..workflow.subagent import load_subagent_configs

        self._subagent_configs = load_subagent_configs(agents)
        logger.info(f"Loaded {len(self._subagent_configs)} sub-agent configs")
        return self

    def with_worktree(
        self,
        branch: Optional[str] = None,
        cleanup: bool = True,
    ) -> 'WorkflowMixin':
        """Enable git worktree isolation for this agent.

        When enabled, infer() and delegate() run in isolated git
        worktrees so parallel agents don't collide.

        Args:
            branch: Branch name (default: auto-generated)
            cleanup: Auto-remove worktree after use (default: True)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("builder").with_worktree()
            >>> result = await agent.infer("Refactor auth")
        """
        self._worktree_config = {
            "branch": branch,
            "cleanup": cleanup,
        }
        logger.info(f"Worktree isolation enabled for agent {self.name}")
        return self

    def with_schedule(
        self,
        every: Optional[int] = None,
        cron: Optional[str] = None,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        ralph: Optional[str] = None,
        max_runs: Optional[int] = None,
    ) -> 'WorkflowMixin':
        """Schedule this agent to run on a cadence.

        Supports interval (every N minutes) or cron expressions.
        Each run produces a finding in the triage inbox.

        Args:
            every: Run every N minutes (interval mode)
            cron: Cron expression, e.g. "0 9 * * *" (cron mode)
            prompt: Static prompt to run each cycle
            prompt_file: Path to prompt file (re-read each cycle)
            ralph: Path to PROMPT.md for Ralph autonomous loop
            max_runs: Maximum number of runs (None = unlimited)

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("triage").with_schedule(every=30, prompt="Check issues")
            >>> agent = Agent("ops").with_schedule(cron="0 9 * * *", prompt_file="TRIAGE.md")
        """
        from ..workflow.schedule import Scheduler, ScheduleConfig, ScheduleStore

        if every is None and cron is None:
            raise ValueError("Must specify either 'every' (minutes) or 'cron' expression")

        config = ScheduleConfig(
            every=every,
            cron=cron,
            prompt=prompt,
            prompt_file=prompt_file,
            ralph=ralph,
            max_runs=max_runs,
        )

        if not hasattr(self, '_schedulers'):
            self._schedulers = []
        if not hasattr(self, '_schedule_store'):
            self._schedule_store = ScheduleStore()

        scheduler = Scheduler(self, config, store=self._schedule_store)
        self._schedulers.append(scheduler)

        logger.info(f"Added schedule {config.id} to agent {self.name}")
        return self

    async def start(self):
        """Start all configured schedules.

        Runs all schedules concurrently. Blocks until all schedules
        complete or are stopped via stop().

        Example:
            >>> agent.with_schedule(every=30, prompt="Check status")
            >>> await agent.start()
        """
        if not hasattr(self, '_schedulers') or not self._schedulers:
            raise RuntimeError("No schedules configured. Use with_schedule() first.")

        tasks = [asyncio.create_task(s.start()) for s in self._schedulers]
        await asyncio.gather(*tasks)

    def stop(self):
        """Stop all running schedules gracefully."""
        if hasattr(self, '_schedulers'):
            for scheduler in self._schedulers:
                scheduler.stop()

    def findings(self, status: str = "pending", limit: int = 50):
        """Get findings from scheduled runs.

        Args:
            status: Filter by status: "pending", "archived", "dismissed"
            limit: Maximum number of findings to return

        Returns:
            List of Finding dicts

        Example:
            >>> findings = agent.findings()
            >>> findings = agent.findings(status="archived")
        """
        if not hasattr(self, '_schedule_store'):
            return []
        from ..workflow.schedule import ScheduleStore
        return [f.to_dict() for f in self._schedule_store.get_findings(status=status, limit=limit)]

    async def delegate(
        self,
        task: str,
        max_corrections: int = 1,
        judges: int = 1,
    ) -> Dict[str, Any]:
        """Delegate a task to sub-agents using maker-checker pattern.

        The maker sub-agent executes the task, then the checker reviews
        the output. If rejected, the maker gets correction attempts.

        When ``judges > 1``, multiple checker instances vote on the
        maker output using a panel consensus model.  The output is
        approved only when a majority of judges approve.

        Args:
            task: The task to delegate
            max_corrections: Max correction attempts on rejection (default: 1)
            judges: Number of checker instances to use as a judge panel
                (default: 1 — single checker, original behaviour).

        Returns:
            Dict with: result, review, approved, corrections, maker, checker

        Example:
            >>> result = await agent.delegate("Refactor the auth module")
            >>> if result["approved"]:
            ...     print("Changes approved by reviewer")
            >>> # Use a panel of 3 judges for higher confidence
            >>> result = await agent.delegate("Refactor auth", judges=3)
        """
        from ..workflow.subagent import _build_subagent, run_maker_checker, run_judge_panel

        if not hasattr(self, '_subagent_configs') or not self._subagent_configs:
            raise RuntimeError("No sub-agents configured. Use with_subagents() first.")

        # Build agent instances from configs
        makers = []
        checkers = []
        for config in self._subagent_configs:
            agent = _build_subagent(config, self)

            # If worktree is enabled, create isolated worktrees
            if self._worktree_config:
                from ..workflow.worktree import WorktreeManager
                wt = WorktreeManager(
                    cleanup=self._worktree_config.get('cleanup', True)
                )
                path = await wt.create(agent_name=config.name)
                if path:
                    self.observer.record(
                        EventType.WORKTREE_CREATE,
                        metadata={"agent": config.name, "path": path}
                    )

            if config.role == "checker":
                checkers.append(agent)
            else:
                makers.append(agent)

        if judges > 1 and checkers:
            return await run_judge_panel(
                task=task,
                makers=makers,
                checkers=checkers,
                num_judges=judges,
                max_corrections=max_corrections,
                parent_observer=self.observer,
            )

        return await run_maker_checker(
            task=task,
            makers=makers,
            checkers=checkers,
            max_corrections=max_corrections,
            parent_observer=self.observer,
        )

    async def best_of(
        self,
        n: int,
        task: str,
        judge: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run N instances of this agent in parallel and pick the best.

        Each instance runs the same task independently (using worktree
        isolation when configured).  A judge agent then evaluates all
        results and selects the best one.

        Args:
            n: Number of parallel instances to run.
            task: The task to run.
            judge: Optional Agent instance to judge results.  If
                ``None``, the first result is returned (no judging).

        Returns:
            Dict with ``result``, ``all_results``, ``chosen_index``,
            and ``judge_reasoning``.

        Example:
            >>> result = await agent.best_of(3, "Write a haiku about code")
        """
        from ..workflow.subagent import _build_subagent, run_best_of

        return await run_best_of(
            agent=self,
            n=n,
            task=task,
            judge=judge,
            parent_observer=self.observer,
        )

    async def ralph(
        self,
        prompt_file: str,
        max_iterations: int = 50,
        timeout_minutes: int = 30,
        checkpoint_every: int = 5,
        on_iteration=None
    ):
        """Run agent in Ralph mode (autonomous loop).
        
        Ralph continuously reads a prompt file and executes until
        all checkpoints are complete or limits are reached.
        
        Args:
            prompt_file: Path to PROMPT.md file with goal and checkpoints
            max_iterations: Maximum loop iterations (safety limit)
            timeout_minutes: Maximum runtime in minutes
            checkpoint_every: Save state every N iterations
            on_iteration: Optional callback(iteration, result_dict)
            
        Returns:
            Execution summary dict
            
        Example:
            >>> result = await agent.ralph("PROMPT.md", max_iterations=50)
        """
        from ..workflow.ralph import RalphRunner, RalphConfig
        
        config = RalphConfig(
            prompt_file=prompt_file,
            max_iterations=max_iterations,
            timeout_minutes=timeout_minutes,
            checkpoint_every=checkpoint_every
        )
        
        runner = RalphRunner(self, config)
        return await runner.run(on_iteration=on_iteration)
