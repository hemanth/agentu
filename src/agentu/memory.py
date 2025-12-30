"""Memory module for agentu - provides short-term and long-term memory capabilities."""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .memory_storage import create_storage, MemoryStorage

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    memory_type: str  # 'conversation', 'fact', 'task', 'observation'
    importance: float = 0.5  # 0.0 to 1.0, used for memory consolidation
    access_count: int = 0
    last_accessed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        return cls(**data)


class ShortTermMemory:
    """Short-term memory (working memory) with limited capacity."""

    def __init__(self, max_size: int = 10):
        """Initialize short-term memory.

        Args:
            max_size: Maximum number of entries to keep in short-term memory
        """
        self.max_size = max_size
        self.entries: List[MemoryEntry] = []

    def add(self, content: str, memory_type: str = 'conversation',
            metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> MemoryEntry:
        """Add entry to short-term memory.

        Args:
            content: The content to remember
            memory_type: Type of memory ('conversation', 'fact', 'task', 'observation')
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            The created MemoryEntry
        """
        entry = MemoryEntry(
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
            memory_type=memory_type,
            importance=importance,
            last_accessed=time.time()
        )

        self.entries.append(entry)

        # Remove oldest if exceeds max_size
        if len(self.entries) > self.max_size:
            # Keep most important or most recent
            self.entries.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
            self.entries = self.entries[:self.max_size]

        return entry

    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """Get n most recent entries.

        Args:
            n: Number of entries to retrieve

        Returns:
            List of recent MemoryEntry objects
        """
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:n]

    def clear(self):
        """Clear all short-term memory."""
        self.entries.clear()

    def get_all(self) -> List[MemoryEntry]:
        """Get all entries."""
        return self.entries.copy()


class LongTermMemory:
    """Long-term memory with persistent storage and semantic organization."""

    def __init__(self, storage_path: Optional[str] = None, use_sqlite: bool = True):
        """Initialize long-term memory.

        Args:
            storage_path: Path to file for persistent storage (optional)
            use_sqlite: If True, use SQLite database; otherwise use JSON (default: True)
        """
        self.storage_path = storage_path
        self.use_sqlite = use_sqlite
        self.storage: Optional[MemoryStorage] = None
        self.entries: List[MemoryEntry] = []
        self.index_by_type: Dict[str, List[MemoryEntry]] = {}

        if storage_path:
            self.storage = create_storage(storage_path, use_sqlite=use_sqlite)
            self.load()

    def add(self, content: str, memory_type: str = 'fact',
            metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> MemoryEntry:
        """Add entry to long-term memory.

        Args:
            content: The content to remember
            memory_type: Type of memory
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            The created MemoryEntry
        """
        entry = MemoryEntry(
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
            memory_type=memory_type,
            importance=importance,
            last_accessed=time.time()
        )

        self.entries.append(entry)

        # Update index
        if memory_type not in self.index_by_type:
            self.index_by_type[memory_type] = []
        self.index_by_type[memory_type].append(entry)

        if self.storage_path:
            self.save()

        return entry

    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search memories by content.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching MemoryEntry objects
        """
        query_lower = query.lower()
        matches = []

        for entry in self.entries:
            if query_lower in entry.content.lower():
                entry.access_count += 1
                entry.last_accessed = time.time()
                matches.append(entry)

        # Sort by relevance (importance + access frequency)
        matches.sort(key=lambda x: (x.importance * (1 + x.access_count)), reverse=True)

        return matches[:limit]

    def get_by_type(self, memory_type: str, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Get memories by type.

        Args:
            memory_type: Type of memory to retrieve
            limit: Maximum number of results (None for all)

        Returns:
            List of MemoryEntry objects of the specified type
        """
        entries = self.index_by_type.get(memory_type, [])

        # Update access stats
        for entry in entries:
            entry.access_count += 1
            entry.last_accessed = time.time()

        # Sort by importance and recency
        entries = sorted(entries, key=lambda x: (x.importance, x.timestamp), reverse=True)

        return entries[:limit] if limit else entries

    def consolidate(self, importance_threshold: float = 0.3):
        """Remove low-importance, rarely accessed memories.

        Args:
            importance_threshold: Minimum importance to keep
        """
        before_count = len(self.entries)

        # Keep entries that are important or recently/frequently accessed
        current_time = time.time()
        self.entries = [
            entry for entry in self.entries
            if (entry.importance >= importance_threshold or
                entry.access_count > 5 or
                (current_time - entry.last_accessed) < 86400)  # accessed within 24 hours
        ]

        # Rebuild index
        self.index_by_type.clear()
        for entry in self.entries:
            if entry.memory_type not in self.index_by_type:
                self.index_by_type[entry.memory_type] = []
            self.index_by_type[entry.memory_type].append(entry)

        removed = before_count - len(self.entries)
        if removed > 0:
            logger.info(f"Consolidated long-term memory: removed {removed} entries")

        if self.storage_path:
            self.save()

    def save(self):
        """Save to persistent storage."""
        if not self.storage:
            return

        try:
            self.storage.save(self.entries)
        except Exception as e:
            logger.error(f"Error saving long-term memory: {str(e)}")

    def load(self):
        """Load from persistent storage."""
        if not self.storage:
            return

        try:
            entry_dicts = self.storage.load()
            self.entries = [MemoryEntry.from_dict(entry_data) for entry_data in entry_dicts]

            # Rebuild index
            self.index_by_type.clear()
            for entry in self.entries:
                if entry.memory_type not in self.index_by_type:
                    self.index_by_type[entry.memory_type] = []
                self.index_by_type[entry.memory_type].append(entry)

            logger.info(f"Loaded {len(self.entries)} memories from storage")
        except Exception as e:
            logger.error(f"Error loading long-term memory: {str(e)}")

    def close(self):
        """Close storage connection."""
        if self.storage:
            self.storage.close()

    def get_all(self) -> List[MemoryEntry]:
        """Get all entries."""
        return self.entries.copy()


class Memory:
    """Unified memory system combining short-term and long-term memory."""

    def __init__(self, short_term_size: int = 10, storage_path: Optional[str] = None,
                 auto_consolidate: bool = True, use_sqlite: bool = True):
        """Initialize memory system.

        Args:
            short_term_size: Size of short-term memory buffer
            storage_path: Path for persistent long-term memory storage
            auto_consolidate: Whether to automatically consolidate memories
            use_sqlite: If True, use SQLite database; otherwise use JSON (default: True)
        """
        self.short_term = ShortTermMemory(max_size=short_term_size)
        self.long_term = LongTermMemory(storage_path=storage_path, use_sqlite=use_sqlite)
        self.auto_consolidate = auto_consolidate

    def remember(self, content: str, memory_type: str = 'conversation',
                metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5,
                store_long_term: bool = False) -> MemoryEntry:
        """Store a memory.

        Args:
            content: The content to remember
            memory_type: Type of memory
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            store_long_term: If True, store directly in long-term memory

        Returns:
            The created MemoryEntry
        """
        # Always add to short-term
        entry = self.short_term.add(content, memory_type, metadata, importance)

        # Add to long-term if important or explicitly requested
        if store_long_term or importance >= 0.7:
            self.long_term.add(content, memory_type, metadata, importance)

        return entry

    def recall(self, query: Optional[str] = None, memory_type: Optional[str] = None,
              limit: int = 5, include_short_term: bool = True) -> List[MemoryEntry]:
        """Recall memories.

        Args:
            query: Search query (if None, returns recent memories)
            memory_type: Filter by memory type
            limit: Maximum number of results
            include_short_term: Whether to include short-term memories

        Returns:
            List of relevant MemoryEntry objects
        """
        results = []

        # Search long-term memory
        if query:
            results.extend(self.long_term.search(query, limit=limit))
        elif memory_type:
            results.extend(self.long_term.get_by_type(memory_type, limit=limit))
        else:
            # Get recent from long-term
            results.extend(sorted(self.long_term.get_all(),
                                key=lambda x: x.timestamp, reverse=True)[:limit])

        # Add short-term memories
        if include_short_term:
            st_memories = self.short_term.get_recent(n=limit)
            results.extend(st_memories)

        # Remove duplicates and sort by relevance
        seen = set()
        unique_results = []
        for entry in results:
            key = (entry.content, entry.timestamp)
            if key not in seen:
                seen.add(key)
                unique_results.append(entry)

        # Sort by importance and recency
        unique_results.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)

        return unique_results[:limit]

    def consolidate_to_long_term(self, importance_threshold: float = 0.6):
        """Move important short-term memories to long-term storage.

        Args:
            importance_threshold: Minimum importance to consolidate
        """
        for entry in self.short_term.get_all():
            if entry.importance >= importance_threshold:
                self.long_term.add(
                    content=entry.content,
                    memory_type=entry.memory_type,
                    metadata=entry.metadata,
                    importance=entry.importance
                )

        if self.auto_consolidate:
            self.long_term.consolidate()

    def get_context(self, max_entries: int = 5) -> str:
        """Get formatted context from recent memories.

        Args:
            max_entries: Maximum number of memory entries to include

        Returns:
            Formatted string with memory context
        """
        memories = self.recall(limit=max_entries, include_short_term=True)

        if not memories:
            return ""

        context_parts = ["Recent memories:"]
        for mem in memories:
            timestamp_str = datetime.fromtimestamp(mem.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            context_parts.append(f"- [{mem.memory_type}] {mem.content} ({timestamp_str})")

        return "\n".join(context_parts)

    def clear_short_term(self):
        """Clear short-term memory."""
        self.short_term.clear()

    def save(self):
        """Save long-term memory to storage."""
        self.long_term.save()

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        return {
            'short_term_size': len(self.short_term.entries),
            'long_term_size': len(self.long_term.entries),
            'memory_types': {
                mem_type: len(entries)
                for mem_type, entries in self.long_term.index_by_type.items()
            },
            'total_memories': len(self.short_term.entries) + len(self.long_term.entries)
        }
