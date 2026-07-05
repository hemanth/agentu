"""MemoryMixin – memory-related methods extracted from Agent."""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryMixin:
    """Mixin providing memory operations for the Agent class.

    Methods here assume they are mixed into an Agent instance that has
    ``memory_enabled``, ``memory``, and ``observer`` attributes.
    """

    def remember(self, content: str, memory_type: str = 'conversation',
                metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5,
                store_long_term: bool = False):
        """Store information in memory.

        Args:
            content: The content to remember
            memory_type: Type of memory ('conversation', 'fact', 'task', 'observation')
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            store_long_term: If True, store directly in long-term memory
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.remember(content, memory_type, metadata, importance, store_long_term)

    def recall(self, query: Optional[str] = None, memory_type: Optional[str] = None,
              limit: int = 5, include_short_term: bool = True,
              semantic: bool = False, semantic_threshold: float = 0.0):
        """Recall memories.

        Args:
            query: Search query (if None, returns recent memories)
            memory_type: Filter by memory type
            limit: Maximum number of results
            include_short_term: Whether to include short-term memories
            semantic: If True, use embedding-based similarity search
                instead of substring matching (requires embedding provider)
            semantic_threshold: Minimum similarity score for semantic recall

        Returns:
            List of MemoryEntry objects
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return []

        return self.memory.recall(
            query, memory_type, limit, include_short_term,
            semantic=semantic, semantic_threshold=semantic_threshold,
        )

    def get_memory_context(self, max_entries: int = 5) -> str:
        """Get formatted context from memories.

        Args:
            max_entries: Maximum number of memory entries to include

        Returns:
            Formatted string with memory context
        """
        if not self.memory_enabled:
            return ""

        return self.memory.get_context(max_entries)

    def consolidate_memory(self, importance_threshold: float = 0.6):
        """Consolidate short-term memories to long-term storage.

        Args:
            importance_threshold: Minimum importance to consolidate
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.consolidate_to_long_term(importance_threshold)

    def clear_short_term_memory(self):
        """Clear short-term memory."""
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.clear_short_term()

    def save_memory(self):
        """Save memory to persistent storage."""
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.save()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        if not self.memory_enabled:
            return {'memory_enabled': False}

        stats = self.memory.stats()
        stats['memory_enabled'] = True
        return stats
