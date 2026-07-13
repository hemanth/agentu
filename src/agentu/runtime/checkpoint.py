"""Checkpoint storage for agentu sessions.

Provides SQLite-backed persistence of session state so that sessions
can be suspended and later resumed (or forked into new sessions).
"""

import json
import sqlite3
import uuid
import time
import os
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """Serialisable snapshot of a session's state.

    Attributes:
        session_id: Unique session identifier.
        agent_name: Name of the agent that owned the session.
        conversation_history: List of conversation entries (dicts).
        metadata: Arbitrary key/value metadata attached to the session.
        turn_count: Number of user turns completed so far.
        created_at: Epoch when the *session* was originally created.
        checkpointed_at: Epoch when this checkpoint was taken.
        parent_session_id: If this was forked, the ID of the source session.
        pending_tool_calls: In-flight tool call state at time of checkpoint.
            Present when auto-checkpoint fires before a tool call. Contains
            the user query, completed turn history, and the tool about to
            be executed. ``None`` for clean (between-turn) checkpoints.
    """

    session_id: str
    agent_name: str
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    turn_count: int
    created_at: float
    checkpointed_at: float
    parent_session_id: Optional[str] = None
    pending_tool_calls: Optional[Dict[str, Any]] = None

    @property
    def was_interrupted(self) -> bool:
        """True if this checkpoint captured mid-tool-call state."""
        return self.pending_tool_calls is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Reconstruct from a dictionary."""
        # Handle checkpoints saved before pending_tool_calls existed
        if 'pending_tool_calls' not in data:
            data['pending_tool_calls'] = None
        return cls(**data)


class CheckpointStore:
    """SQLite-backed storage for session checkpoints.

    Each checkpoint is a full snapshot of a session's conversational
    state.  The store supports saving, loading, listing, deleting, and
    forking checkpoints.

    Args:
        db_path: Filesystem path for the SQLite database.  Parent
            directories are created automatically.
    """

    def __init__(self, db_path: str = ".checkpoints/checkpoints.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # ── schema ──────────────────────────────────────────────

    def _create_tables(self) -> None:
        """Create the checkpoints table and indexes."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                agent_name      TEXT    NOT NULL,
                conversation    TEXT    NOT NULL,
                metadata        TEXT    NOT NULL,
                turn_count      INTEGER NOT NULL DEFAULT 0,
                created_at      REAL    NOT NULL,
                checkpointed_at REAL    NOT NULL,
                parent_session_id TEXT,
                pending_tool_calls TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cp_session_id
            ON checkpoints(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cp_checkpointed_at
            ON checkpoints(checkpointed_at DESC)
        """)
        # Migrate existing tables that lack the new column
        try:
            cursor.execute(
                "ALTER TABLE checkpoints ADD COLUMN pending_tool_calls TEXT"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        self.conn.commit()

    # ── public API ──────────────────────────────────────────

    def save(self, data: CheckpointData) -> int:
        """Persist a checkpoint.

        Args:
            data: The checkpoint snapshot to store.

        Returns:
            The database row id of the saved checkpoint.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO checkpoints
                    (session_id, agent_name, conversation, metadata,
                     turn_count, created_at, checkpointed_at,
                     parent_session_id, pending_tool_calls)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.session_id,
                    data.agent_name,
                    json.dumps(data.conversation_history),
                    json.dumps(data.metadata),
                    data.turn_count,
                    data.created_at,
                    data.checkpointed_at,
                    data.parent_session_id,
                    json.dumps(data.pending_tool_calls) if data.pending_tool_calls else None,
                ),
            )
            self.conn.commit()
            row_id = cursor.lastrowid
            logger.info(
                "Saved checkpoint for session %s (row %d%s)",
                data.session_id,
                row_id,
                ", pending_tool" if data.pending_tool_calls else "",
            )
            return row_id  # type: ignore[return-value]
        except Exception:
            self.conn.rollback()
            logger.error("Failed to save checkpoint for %s", data.session_id)
            raise

    def load(self, session_id: str) -> Optional[CheckpointData]:
        """Load the most recent checkpoint for a session.

        Args:
            session_id: The session to look up.

        Returns:
            The newest ``CheckpointData`` for the session, or ``None``
            if no checkpoint exists.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT session_id, agent_name, conversation, metadata,
                   turn_count, created_at, checkpointed_at,
                   parent_session_id, pending_tool_calls
            FROM checkpoints
            WHERE session_id = ?
            ORDER BY checkpointed_at DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        pending_raw = row["pending_tool_calls"]
        pending = json.loads(pending_raw) if pending_raw else None

        return CheckpointData(
            session_id=row["session_id"],
            agent_name=row["agent_name"],
            conversation_history=json.loads(row["conversation"]),
            metadata=json.loads(row["metadata"]),
            turn_count=row["turn_count"],
            created_at=row["created_at"],
            checkpointed_at=row["checkpointed_at"],
            parent_session_id=row["parent_session_id"],
            pending_tool_calls=pending,
        )

    def list_checkpoints(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List stored checkpoints.

        Args:
            session_id: If provided, list only checkpoints for this session.
            limit: Maximum number of results.

        Returns:
            List of checkpoint summary dicts (no full conversation).
        """
        cursor = self.conn.cursor()
        if session_id:
            cursor.execute(
                """
                SELECT session_id, agent_name, turn_count,
                       created_at, checkpointed_at, parent_session_id
                FROM checkpoints
                WHERE session_id = ?
                ORDER BY checkpointed_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT session_id, agent_name, turn_count,
                       created_at, checkpointed_at, parent_session_id
                FROM checkpoints
                ORDER BY checkpointed_at DESC
                LIMIT ?
                """,
                (limit,),
            )

        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            results.append({
                "session_id": row["session_id"],
                "agent_name": row["agent_name"],
                "turn_count": row["turn_count"],
                "created_at": row["created_at"],
                "checkpointed_at": row["checkpointed_at"],
                "parent_session_id": row["parent_session_id"],
            })
        return results

    def delete(self, session_id: str) -> int:
        """Delete all checkpoints for a session.

        Args:
            session_id: The session whose checkpoints to remove.

        Returns:
            Number of rows deleted.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM checkpoints WHERE session_id = ?",
            (session_id,),
        )
        self.conn.commit()
        deleted = cursor.rowcount
        logger.info(
            "Deleted %d checkpoint(s) for session %s", deleted, session_id
        )
        return deleted

    def fork(self, source_session_id: str) -> Optional[CheckpointData]:
        """Create a forked checkpoint from an existing session.

        Loads the latest checkpoint of *source_session_id*, assigns it a
        new ``session_id``, records the parent, saves it, and returns
        the new checkpoint.

        Args:
            source_session_id: Session to fork from.

        Returns:
            New ``CheckpointData`` with a fresh session_id, or ``None``
            if no source checkpoint was found.
        """
        source = self.load(source_session_id)
        if source is None:
            logger.warning(
                "Cannot fork: no checkpoint for session %s",
                source_session_id,
            )
            return None

        forked = CheckpointData(
            session_id=str(uuid.uuid4()),
            agent_name=source.agent_name,
            conversation_history=list(source.conversation_history),
            metadata=dict(source.metadata),
            turn_count=source.turn_count,
            created_at=time.time(),
            checkpointed_at=time.time(),
            parent_session_id=source_session_id,
        )
        self.save(forked)
        logger.info(
            "Forked session %s → %s",
            source_session_id,
            forked.session_id,
        )
        return forked

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Closed checkpoint store at %s", self.db_path)
