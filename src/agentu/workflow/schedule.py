"""
Scheduled Automations for agentu loop engineering.

Run agents on a cadence with interval or cron scheduling.
Findings are persisted to SQLite for triage.

Usage:
    agent = Agent("triage").with_schedule(every=30, prompt="Check issues")
    await agent.start()
    findings = agent.findings()
"""

import asyncio
import time
import re
import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FindingStatus(Enum):
    """Status of a scheduled run finding."""
    PENDING = "pending"
    ARCHIVED = "archived"
    DISMISSED = "dismissed"


@dataclass
class Finding:
    """A single finding from a scheduled run."""
    id: str
    schedule_id: str
    result: str
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0.0
    iteration: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "schedule_id": self.schedule_id,
            "result": self.result,
            "status": self.status,
            "created_at": self.created_at,
            "duration_ms": self.duration_ms,
            "iteration": self.iteration,
            "error": self.error,
        }


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled automation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    every: Optional[int] = None  # interval in minutes
    cron: Optional[str] = None  # cron expression
    prompt: Optional[str] = None
    prompt_file: Optional[str] = None
    ralph: Optional[str] = None  # path to PROMPT.md for ralph mode
    max_runs: Optional[int] = None


class CronParser:
    """Lightweight cron expression parser.

    Supports standard 5-field cron expressions:
        minute hour day-of-month month day-of-week

    Field syntax:
        * — every value
        */N — every Nth value
        N — exact value
        N,M — specific values
        N-M — range of values
    """

    @staticmethod
    def matches(expression: str, dt: Optional[datetime] = None) -> bool:
        """Check if a datetime matches a cron expression.

        Args:
            expression: 5-field cron expression
            dt: Datetime to check (default: now)

        Returns:
            True if the datetime matches the expression
        """
        if dt is None:
            dt = datetime.now()

        fields = expression.strip().split()
        if len(fields) != 5:
            raise ValueError(f"Cron expression must have 5 fields, got {len(fields)}: {expression}")

        values = [dt.minute, dt.hour, dt.day, dt.month, dt.weekday()]
        # cron uses 0=Sunday, Python uses 0=Monday. Convert.
        values[4] = (values[4] + 1) % 7

        ranges = [
            (0, 59),   # minute
            (0, 23),   # hour
            (1, 31),   # day of month
            (1, 12),   # month
            (0, 6),    # day of week (0=Sunday)
        ]

        for field_str, value, (low, high) in zip(fields, values, ranges):
            if not CronParser._field_matches(field_str, value, low, high):
                return False
        return True

    @staticmethod
    def _field_matches(field_str: str, value: int, low: int, high: int) -> bool:
        """Check if a single cron field matches a value."""
        if field_str == "*":
            return True

        for part in field_str.split(","):
            # Step: */N or N-M/S
            if "/" in part:
                base, step_str = part.split("/", 1)
                step = int(step_str)
                if base == "*":
                    if (value - low) % step == 0:
                        return True
                elif "-" in base:
                    start, end = map(int, base.split("-", 1))
                    if start <= value <= end and (value - start) % step == 0:
                        return True
            # Range: N-M
            elif "-" in part:
                start, end = map(int, part.split("-", 1))
                if start <= value <= end:
                    return True
            # Exact: N
            else:
                if int(part) == value:
                    return True

        return False

    @staticmethod
    def next_minute_boundary() -> float:
        """Seconds until the next minute boundary."""
        now = time.time()
        return 60 - (now % 60)


class ScheduleStore:
    """SQLite-backed storage for schedules and findings."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".agentu" / "schedules.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schedules (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_run_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY,
                    schedule_id TEXT NOT NULL,
                    result TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    duration_ms REAL DEFAULT 0,
                    iteration INTEGER DEFAULT 0,
                    error TEXT,
                    FOREIGN KEY (schedule_id) REFERENCES schedules(id)
                )
            """)
            conn.commit()

    def save_schedule(self, agent_name: str, config: ScheduleConfig):
        """Persist a schedule configuration."""
        config_data = {
            "id": config.id,
            "every": config.every,
            "cron": config.cron,
            "prompt": config.prompt,
            "prompt_file": config.prompt_file,
            "ralph": config.ralph,
            "max_runs": config.max_runs,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO schedules (id, agent_name, config_json) VALUES (?, ?, ?)",
                (config.id, agent_name, json.dumps(config_data))
            )
            conn.commit()

    def save_finding(self, finding: Finding):
        """Persist a finding."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO findings 
                   (id, schedule_id, result, status, created_at, duration_ms, iteration, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (finding.id, finding.schedule_id, finding.result, finding.status,
                 finding.created_at, finding.duration_ms, finding.iteration, finding.error)
            )
            conn.commit()

    def get_findings(self, schedule_id: Optional[str] = None,
                     status: str = "pending", limit: int = 50) -> List[Finding]:
        """Retrieve findings filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if schedule_id:
                rows = conn.execute(
                    "SELECT * FROM findings WHERE schedule_id = ? AND status = ? ORDER BY created_at DESC LIMIT ?",
                    (schedule_id, status, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM findings WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                ).fetchall()

        return [Finding(
            id=r["id"], schedule_id=r["schedule_id"], result=r["result"] or "",
            status=r["status"], created_at=r["created_at"],
            duration_ms=r["duration_ms"], iteration=r["iteration"],
            error=r["error"]
        ) for r in rows]

    def update_finding_status(self, finding_id: str, status: str):
        """Update a finding's status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE findings SET status = ? WHERE id = ?",
                (status, finding_id)
            )
            conn.commit()

    def mark_schedule_inactive(self, schedule_id: str):
        """Mark a schedule as inactive."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE schedules SET active = 0 WHERE id = ?",
                (schedule_id,)
            )
            conn.commit()

    def update_last_run(self, schedule_id: str):
        """Update the last run timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE schedules SET last_run_at = ? WHERE id = ?",
                (datetime.now().isoformat(), schedule_id)
            )
            conn.commit()


class Scheduler:
    """Runs an agent on a schedule (interval or cron).

    Internal class — users interact via Agent.with_schedule().
    """

    def __init__(self, agent, config: ScheduleConfig,
                 store: Optional[ScheduleStore] = None):
        self.agent = agent
        self.config = config
        self.store = store or ScheduleStore()
        self._stop_event = asyncio.Event()
        self._iteration = 0
        self._task: Optional[asyncio.Task] = None

    def stop(self):
        """Request graceful stop."""
        self._stop_event.set()
        logger.info(f"Schedule {self.config.id}: Stop requested")

    async def start(self):
        """Start the schedule loop."""
        # Persist schedule
        self.store.save_schedule(self.agent.name, self.config)

        # Emit observer event
        if hasattr(self.agent, 'observer'):
            from ..middleware.observe import EventType
            self.agent.observer.record(
                EventType.SCHEDULE_RUN,
                metadata={"schedule_id": self.config.id, "action": "start"}
            )

        logger.info(f"Schedule {self.config.id}: Starting "
                     f"({'every ' + str(self.config.every) + ' min' if self.config.every else 'cron ' + str(self.config.cron)})")

        if self.config.every is not None:
            await self._run_interval()
        elif self.config.cron is not None:
            await self._run_cron()

    async def _run_interval(self):
        """Run on a fixed interval."""
        interval_seconds = self.config.every * 60

        while not self._stop_event.is_set():
            if self.config.max_runs and self._iteration >= self.config.max_runs:
                logger.info(f"Schedule {self.config.id}: Max runs ({self.config.max_runs}) reached")
                break

            await self._execute_run()

            # Wait for interval or stop
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval_seconds)
                break  # stop was requested
            except asyncio.TimeoutError:
                pass  # timeout = time to run again

    async def _run_cron(self):
        """Run on a cron schedule."""
        while not self._stop_event.is_set():
            if self.config.max_runs and self._iteration >= self.config.max_runs:
                logger.info(f"Schedule {self.config.id}: Max runs ({self.config.max_runs}) reached")
                break

            # Wait until next minute boundary
            wait_seconds = CronParser.next_minute_boundary()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_seconds)
                break  # stop was requested
            except asyncio.TimeoutError:
                pass

            # Check if cron matches now
            if CronParser.matches(self.config.cron):
                await self._execute_run()

    async def _execute_run(self):
        """Execute a single scheduled run."""
        self._iteration += 1
        start_time = time.time()
        finding_id = str(uuid.uuid4())[:8]
        result_str = ""
        error_str = None

        try:
            # Determine what to run
            if self.config.ralph:
                from .ralph import ralph
                result = await ralph(self.agent, self.config.ralph, max_iterations=50)
                result_str = json.dumps(result, default=str)
            elif self.config.prompt_file:
                prompt = Path(self.config.prompt_file).read_text()
                result = await self.agent.infer(prompt)
                result_str = str(result)
            elif self.config.prompt:
                result = await self.agent.infer(self.config.prompt)
                result_str = str(result)
            else:
                logger.warning(f"Schedule {self.config.id}: No prompt, prompt_file, or ralph configured")
                return

        except Exception as e:
            error_str = str(e)
            result_str = f"Error: {error_str}"
            logger.error(f"Schedule {self.config.id}: Run {self._iteration} failed: {e}")

        elapsed_ms = (time.time() - start_time) * 1000

        # Create finding
        finding = Finding(
            id=finding_id,
            schedule_id=self.config.id,
            result=result_str,
            status="pending" if result_str.strip() else "archived",
            duration_ms=elapsed_ms,
            iteration=self._iteration,
            error=error_str,
        )

        # Persist finding
        self.store.save_finding(finding)
        self.store.update_last_run(self.config.id)

        # Emit observer event
        if hasattr(self.agent, 'observer'):
            from ..middleware.observe import EventType
            self.agent.observer.record(
                EventType.SCHEDULE_FINDING,
                metadata={
                    "schedule_id": self.config.id,
                    "finding_id": finding_id,
                    "iteration": self._iteration,
                    "duration_ms": elapsed_ms,
                    "has_error": error_str is not None,
                }
            )

        logger.info(f"Schedule {self.config.id}: Run {self._iteration} complete "
                     f"({elapsed_ms:.0f}ms, status={finding.status})")
