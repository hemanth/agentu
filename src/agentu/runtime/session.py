"""Session-based stateful intelligence for agentu.

Provides server-managed conversation sessions where memory and context
are automatically preserved across requests, similar to Interactions API.
"""

import uuid
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

from .._core.agent import Agent
from ..memory.memory import Memory, MemoryEntry
from .checkpoint import CheckpointData, CheckpointStore

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a stateful conversation session.
    
    The session maintains:
    - Conversation history
    - Persistent memory
    - Agent state
    - Metadata (user_id, tags, etc.)
    """
    session_id: str
    agent:Agent
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    
    def __post_init__(self):
        """Initialize session with memory enabled."""
        if not self.agent.memory_enabled:
            self.agent.memory_enabled = True
            # Use agent's configured storage path, or default
            storage_path = f".sessions/{self.session_id}.db"
            self.agent.memory = Memory(
                short_term_size=20,  # Larger for sessions
                use_sqlite=True,
                storage_path=storage_path,
            )
        # Expose storage backend (if agent has one) for external consumers
        self._storage_backend = getattr(self.agent, '_storage_backend', None)
    
    async def send(self, message: str) -> Dict[str, Any]:
        """Send a message in this session.
        
        The agent automatically remembers context from previous turns.
        
        Args:
            message: User message
            
        Returns:
            Response with tool_used, result, and session metadata
        """
        self.last_accessed = time.time()
        self.turn_count += 1
        
        # Agent's infer() method already stores in memory
        response = await self.agent.infer(message)
        
        # Add session metadata to response
        response['session_info'] = {
            'session_id': self.session_id,
            'turn': self.turn_count,
            'memory_stats': self.agent.get_memory_stats()
        }
        
        return response
    
    def get_history(self, limit: int = 10) -> List[MemoryEntry]:
        """Get conversation history.
        
        Args:
            limit: Maximum number of entries
            
        Returns:
            List of conversation memory entries
        """
        return self.agent.recall(
            memory_type='conversation',
            limit=limit,
            include_short_term=True
        )
    
    def clear_history(self):
        """Clear conversation history but keep facts/tasks."""
        # Remove conversation memories
        if self.agent.memory:
            self.agent.memory.short_term.entries = [
                e for e in self.agent.memory.short_term.entries
                if e.memory_type != 'conversation'
            ]
            self.agent.memory.long_term.entries = [
                e for e in self.agent.memory.long_term.entries
                if e.memory_type != 'conversation'
            ]
        self.turn_count = 0
    
    def save(self):
        """Explicitly save session state."""
        if self.agent.memory_enabled:
            self.agent.save_memory()

    def checkpoint(
        self,
        store: Optional["CheckpointStore"] = None,
        fork: bool = False,
    ) -> "CheckpointData":
        """Serialise current session state to a checkpoint.

        Args:
            store: ``CheckpointStore`` to persist the snapshot to.
                If *None* and the agent has a storage backend configured
                (via ``with_backend()``), the checkpoint is stored there.
                Otherwise a default SQLite store at
                ``.checkpoints/checkpoints.db`` is used.
            fork: If *True*, the checkpoint is saved under a **new**
                session ID (a fork), and the returned data has a fresh
                ``session_id`` with ``parent_session_id`` pointing back
                to the original.

        Returns:
            The persisted ``CheckpointData``.
        """
        if store is None:
            store = CheckpointStore()

        # Gather conversation history from memory
        history: List[Dict[str, Any]] = []
        if self.agent.memory and self.agent.memory_enabled:
            entries = self.agent.recall(
                memory_type='conversation',
                limit=1000,
                include_short_term=True,
            )
            history = [e.to_dict() for e in entries]

        session_id = self.session_id
        parent_id: Optional[str] = None
        if fork:
            parent_id = self.session_id
            session_id = str(uuid.uuid4())

        data = CheckpointData(
            session_id=session_id,
            agent_name=self.agent.name,
            conversation_history=history,
            metadata=dict(self.metadata),
            turn_count=self.turn_count,
            created_at=self.created_at,
            checkpointed_at=time.time(),
            parent_session_id=parent_id,
        )

        # Persist to storage backend if available, else default store
        backend = getattr(self.agent, '_storage_backend', None)
        if backend is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            key = f"checkpoint:{data.session_id}"
            import json as _json
            payload = _json.dumps(data.to_dict())
            if loop and loop.is_running():
                # Schedule async set without blocking
                loop.create_task(backend.set(key, payload))
            else:
                asyncio.run(backend.set(key, payload))
            logger.info(
                "Checkpointed session %s to storage backend (fork=%s)",
                data.session_id, fork,
            )
        else:
            store.save(data)
            logger.info(
                "Checkpointed session %s (fork=%s)",
                data.session_id, fork,
            )
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Export session metadata (not full state)."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'turn_count': self.turn_count,
            'metadata': self.metadata,
            'memory_stats': self.agent.get_memory_stats()
        }


class SessionManager:
    """Manages multiple stateful sessions.
    
    This provides the "Interactions API" experience where
    sessions are managed server-side with automatic memory.
    
    Example:
        >>> manager = SessionManager()
        >>> session = manager.create_session(agent)
        >>> response = await session.send("What's the weather?")
        >>> # Later, same session remembers context
        >>> response = await session.send("What about tomorrow?")
    """
    
    def __init__(self, max_sessions: int = 1000, session_timeout: int = 3600):
        """Initialize session manager.
        
        Args:
            max_sessions: Maximum concurrent sessions
            session_timeout: Seconds before inactive session expires
        """
        self.sessions: Dict[str, Session] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
    
    def create_session(
        self,
        agent: Agent,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create a new stateful session.
        
        Args:
            agent: Agent instance to use (will be cloned per session)
            session_id: Optional custom session ID
            metadata: Optional metadata (user_id, tags, etc.)
            
        Returns:
            New Session object
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists, returning existing")
            return self.sessions[session_id]
        
        # Clean up old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_old_sessions()
        
        session = Session(
            session_id=session_id,
            agent=agent,  # In production, you'd clone the agent
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session: {session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get existing session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object if found, None otherwise
        """
        session = self.sessions.get(session_id)
        
        if session:
            # Check if expired
            if time.time() - session.last_accessed > self.session_timeout:
                logger.info(f"Session {session_id} expired")
                self.delete_session(session_id)
                return None
            
            session.last_accessed = time.time()
        
        return session

    def resume(
        self,
        session_id: str,
        agent: Agent,
        store: Optional[CheckpointStore] = None,
    ) -> Optional[Session]:
        """Resume a session from a persisted checkpoint.

        Loads the most recent checkpoint for *session_id*, creates a
        fresh ``Session`` wrapping the provided *agent*, and replays
        the saved conversation history into the agent's memory.

        Args:
            session_id: The session to resume.
            agent: An ``Agent`` instance to attach to the resumed
                session (should match the original agent's tools/model).
            store: ``CheckpointStore`` to read from.  If *None*, a
                default store at ``.checkpoints/checkpoints.db`` is used.

        Returns:
            The resumed ``Session``, or ``None`` if no checkpoint was
            found for *session_id*.
        """
        if store is None:
            store = CheckpointStore()

        # Try loading from agent's storage backend first
        data = None
        backend = getattr(agent, '_storage_backend', None)
        if backend is not None:
            import asyncio
            import json as _json
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            key = f"checkpoint:{session_id}"
            try:
                if loop and loop.is_running():
                    # Can't await in sync context, fall through to store
                    pass
                else:
                    raw = asyncio.run(backend.get(key))
                    if raw:
                        data = CheckpointData.from_dict(_json.loads(raw))
                        logger.info("Loaded checkpoint %s from storage backend", session_id)
            except Exception:
                logger.debug("Storage backend load failed, falling back to SQLite", exc_info=True)

        # Fallback to SQLite checkpoint store
        if data is None:
            data = store.load(session_id)

        if data is None:
            logger.warning("No checkpoint found for session %s", session_id)
            return None

        # If the session is already live, return it
        if session_id in self.sessions:
            logger.info("Session %s already active, returning it", session_id)
            return self.sessions[session_id]

        session = Session(
            session_id=data.session_id,
            agent=agent,
            created_at=data.created_at,
            metadata=data.metadata,
        )
        session.turn_count = data.turn_count

        # Replay conversation history into agent memory
        for entry_dict in data.conversation_history:
            if agent.memory:
                from ..memory.memory import MemoryEntry
                entry = MemoryEntry.from_dict(entry_dict)
                agent.memory.short_term.entries.append(entry)

        self.sessions[session_id] = session
        logger.info(
            "Resumed session %s (%d turns, %d history entries)",
            session_id,
            data.turn_count,
            len(data.conversation_history),
        )
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session to delete
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.save()  # Save before deletion
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all active sessions.
        
        Args:
            user_id: Optional filter by user_id in metadata
            
        Returns:
            List of session metadata dicts
        """
        sessions = []
        for session in self.sessions.values():
            if user_id and session.metadata.get('user_id') != user_id:
                continue
            sessions.append(session.to_dict())
        
        return sessions
    
    def _cleanup_old_sessions(self, count: int = 10):
        """Remove oldest inactive sessions.
        
        Args:
            count: Number of sessions to remove
        """
        # Sort by last accessed
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for session_id, session in sorted_sessions[:count]:
            logger.info(f"Cleaning up old session: {session_id}")
            session.save()
            del self.sessions[session_id]
    
    def save_all(self):
        """Save all active sessions."""
        for session in self.sessions.values():
            session.save()
        logger.info(f"Saved {len(self.sessions)} sessions")
    
    def stats(self) -> Dict[str, Any]:
        """Get manager statistics.
        
        Returns:
            Dict with session stats
        """
        return {
            'total_sessions': len(self.sessions),
            'max_sessions': self.max_sessions,
            'session_timeout': self.session_timeout,
            'sessions': [s.to_dict() for s in self.sessions.values()]
        }
