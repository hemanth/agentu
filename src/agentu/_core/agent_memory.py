"""MemoryMixin – memory-related methods extracted from Agent."""

import asyncio
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


class MemoryMixin:
    """Mixin providing memory operations for the Agent class.

    Methods here assume they are mixed into an Agent instance that has
    ``memory_enabled``, ``memory``, and ``observer`` attributes.

    When the agent has a ``_vector_backend`` configured (via
    ``with_vectors()``), ``remember()`` and ``recall(semantic=True)``
    will use it for embedding-based storage and retrieval.
    """

    def _get_vector_backend_sync(self):
        """Resolve vector backend, lazily creating from DSN if needed.

        Returns the backend or None if not configured.
        LanceDBBackend.create() is synchronous, so this works in any
        context (sync or async).
        """
        # Direct backend (set via with_vectors(backend_obj))
        backend = getattr(self, '_vector_backend', None)
        if backend is not None:
            return backend
        # DSN configured but not yet created (with_vectors("./path"))
        dsn = getattr(self, '_vector_dsn', None)
        if dsn is not None:
            try:
                from ..storage import LanceDBBackend
                backend = LanceDBBackend.create(dsn)
                # Cache so we don't re-create on every call
                self._vector_backend = backend
                return backend
            except Exception:
                logger.debug("Failed to lazily create vector backend", exc_info=True)
        return None

    def remember(self, content: str, memory_type: str = 'conversation',
                metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5,
                store_long_term: bool = False, summary: Optional[str] = None,
                entities: Optional[List[str]] = None, topics: Optional[List[str]] = None,
                source: Optional[str] = None) -> None:
        """Store information in memory.

        The LLM automatically extracts summary, entities, topics, and
        importance from the content. Just pass raw text — the LLM does
        the rest. If you provide structured fields manually, the LLM
        call is skipped.

        Auto-extraction is skipped for ``memory_type='conversation'``
        (chat turns don't need it) and can be disabled globally with
        ``Agent(..., auto_extract_memory=False)``.

        Args:
            content: The content to remember
            memory_type: Type of memory ('conversation', 'fact', 'task', 'observation')
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            store_long_term: If True, store directly in long-term memory
            summary: Short summary (auto-extracted by default)
            entities: Key entities (auto-extracted by default)
            topics: Topic tags (auto-extracted by default)
            source: Optional source identifier
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        # Auto-extract structured metadata via LLM when:
        # 1. No structured fields were provided manually
        # 2. Not a conversation turn (those are just chat history)
        # 3. auto_extract_memory is not disabled (defaults to True)
        auto_extract = getattr(self, 'auto_extract_memory', True)
        needs_extraction = (
            auto_extract
            and not summary and not entities and not topics
            and memory_type != 'conversation'
        )
        if needs_extraction:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._remember_with_extraction(
                    content, memory_type, metadata, importance,
                    store_long_term, source,
                ))
                return  # async path handles storage
            except RuntimeError:
                # No event loop — try sync extraction
                try:
                    extracted = asyncio.run(self._extract_memory_metadata(content))
                    summary = extracted.get('summary', summary)
                    entities = extracted.get('entities', entities)
                    topics = extracted.get('topics', topics)
                    importance = extracted.get('importance', importance)
                except Exception:
                    logger.debug("LLM extraction failed, storing without metadata", exc_info=True)

        # Store in standard memory system
        entry = self.memory.remember(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            importance=importance,
            store_long_term=store_long_term,
            summary=summary,
            entities=entities,
            topics=topics,
            source=source
        )

        # Also store in vector backend for semantic search if available
        vector_backend = self._get_vector_backend_sync()
        if vector_backend is not None and (store_long_term or importance >= 0.7):
            self._store_to_vector_backend(content, entry, memory_type, metadata)

    async def _extract_memory_metadata(self, content: str) -> Dict[str, Any]:
        """Use the LLM to extract structured metadata from content.

        Returns a dict with ``summary``, ``entities``, ``topics``,
        and ``importance``.
        """
        import json as _json

        extraction_prompt = (
            "Extract structured metadata from this text. "
            "Return ONLY valid JSON with these fields:\n"
            '{"summary": "1-2 sentence summary", '
            '"entities": ["people", "orgs", "products", "concepts"], '
            '"topics": ["2-4 topic tags"], '
            '"importance": 0.0 to 1.0}\n\n'
            f"Text:\n{content[:3000]}"
        )

        try:
            raw = await self._call_llm(extraction_prompt)
            # Parse JSON from the response — handle markdown fences
            text = raw.strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1] if '\n' in text else text[3:]
                text = text.rsplit('```', 1)[0]
            result = _json.loads(text.strip())
            return {
                'summary': result.get('summary'),
                'entities': result.get('entities', []),
                'topics': result.get('topics', []),
                'importance': float(result.get('importance', 0.5)),
            }
        except Exception:
            logger.debug("Failed to parse LLM extraction response", exc_info=True)
            return {}

    async def _remember_with_extraction(
        self, content: str, memory_type: str,
        metadata: Optional[Dict[str, Any]], importance: float,
        store_long_term: bool, source: Optional[str],
    ) -> None:
        """Async path: extract metadata then store."""
        try:
            extracted = await self._extract_memory_metadata(content)
        except Exception:
            logger.debug("LLM extraction failed, storing without metadata", exc_info=True)
            extracted = {}

        entry = self.memory.remember(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            importance=extracted.get('importance', importance),
            store_long_term=store_long_term,
            summary=extracted.get('summary'),
            entities=extracted.get('entities'),
            topics=extracted.get('topics'),
            source=source,
        )

        vector_backend = self._get_vector_backend_sync()
        if vector_backend is not None and (store_long_term or importance >= 0.7):
            self._store_to_vector_backend(content, entry, memory_type, metadata)


    def _store_to_vector_backend(self, content: str, entry: Any,
                                  memory_type: str,
                                  metadata: Optional[Dict[str, Any]]) -> None:
        """Store a memory entry in the vector backend (fire-and-forget)."""
        embedding_provider = getattr(self.memory, 'embedding_provider', None)
        if embedding_provider is None:
            logger.debug("No embedding provider for vector backend storage")
            return

        # Resolve backend once (already cached by _get_vector_backend_sync)
        backend = self._get_vector_backend_sync()
        if backend is None:
            return

        async def _store():
            try:
                embedding = await embedding_provider.embed(content)
                meta = dict(metadata or {})
                meta['memory_type'] = memory_type
                meta['content'] = content
                key = f"mem:{getattr(entry, 'id', id(entry))}"
                await backend.store(key, embedding, meta)
            except Exception:
                logger.debug("Vector backend store failed", exc_info=True)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_store())
        except RuntimeError:
            asyncio.run(_store())

    def recall(self, query: Optional[str] = None, memory_type: Optional[str] = None,
              limit: int = 5, include_short_term: bool = True,
              semantic: bool = False, semantic_threshold: float = 0.0) -> list:
        """Recall memories.

        Args:
            query: Search query (if None, returns recent memories)
            memory_type: Filter by memory type
            limit: Maximum number of results
            include_short_term: Whether to include short-term memories
            semantic: If True, use embedding-based similarity search
                instead of substring matching (requires embedding provider
                or a vector backend configured via ``with_vectors()``)
            semantic_threshold: Minimum similarity score for semantic recall

        Returns:
            List of MemoryEntry objects
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return []

        # Try vector backend for semantic recall if available and
        # the in-memory semantic index isn't ready
        vector_backend = self._get_vector_backend_sync()
        if (semantic and query and vector_backend is not None
                and not getattr(self.memory.long_term, '_semantic_ready', False)):
            vector_results = self._recall_from_vector_backend(
                query, limit, semantic_threshold
            )
            if vector_results:
                return vector_results

        return self.memory.recall(
            query, memory_type, limit, include_short_term,
            semantic=semantic, semantic_threshold=semantic_threshold,
        )

    def _recall_from_vector_backend(self, query: str, limit: int,
                                     threshold: float) -> list:
        """Search the vector backend for semantically similar memories."""
        embedding_provider = getattr(self.memory, 'embedding_provider', None)
        if embedding_provider is None:
            return []

        # Resolve backend once (already cached)
        backend = self._get_vector_backend_sync()
        if backend is None:
            return []

        async def _search():
            query_embedding = await embedding_provider.embed(query)
            results = await backend.search(
                query_embedding, limit=limit
            )
            return results

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    results = pool.submit(asyncio.run, _search()).result()
            else:
                results = asyncio.run(_search())
        except Exception:
            logger.debug("Vector backend search failed", exc_info=True)
            return []

        # Convert vector backend results to MemoryEntry-like objects
        from ..memory.memory import MemoryEntry
        entries = []
        for key, score, meta in results:
            if score < threshold:
                continue
            content = meta.get('content', '')
            entry = MemoryEntry(
                content=content,
                memory_type=meta.get('memory_type', 'fact'),
                metadata=meta,
                importance=score,
            )
            entries.append(entry)
        return entries

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

    def consolidate_memory(self, importance_threshold: float = 0.6) -> None:
        """Consolidate short-term memories to long-term storage.

        Args:
            importance_threshold: Minimum importance to consolidate
        """
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.consolidate_to_long_term(importance_threshold)

    def clear_short_term_memory(self) -> None:
        """Clear short-term memory."""
        if not self.memory_enabled:
            logger.warning("Memory is not enabled for this agent")
            return

        self.memory.clear_short_term()

    def save_memory(self) -> None:
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

    def with_consolidation(self, every: int = 30) -> 'Agent':
        """Enable background memory consolidation.
        
        Runs on a timer, reviewing unconsolidated memories, finding
        connections and patterns, and generating synthesized insights.
        Like the human brain during sleep.
        
        Args:
            every: Minutes between consolidation runs (default: 30)
        
        Returns:
            Self for chaining.
        """
        consolidation_prompt = (
            "Review all unconsolidated memories. For each group of related memories:\n"
            "1. Find connections and patterns between them\n"
            "2. Generate a synthesized insight\n"
            "3. Use the consolidate_memories tool to store your findings\n"
            "Focus on cross-cutting themes that connect different memories."
        )
        self.with_schedule(every=every, prompt=consolidation_prompt)
        
        # Add the consolidation tool
        from .tools import Tool
        
        def consolidate_memories(
            insight: str,
            related_topics: list,
            source_summaries: list,
        ) -> dict:
            """Store a consolidation insight from reviewing memories.
            
            Args:
                insight: The cross-cutting pattern or insight discovered.
                related_topics: Topics that connect the consolidated memories.
                source_summaries: Summaries of the memories that were consolidated.
            
            Returns:
                Confirmation of stored consolidation.
            """
            # Store the insight as a high-importance memory
            self.remember(
                content=insight,
                memory_type='consolidation',
                importance=0.9,
                store_long_term=True,
                summary=insight,
                topics=related_topics,
                source='consolidation',
            )
            
            # Mark source memories as consolidated
            if self.memory:
                for entry in self.memory.short_term.entries:
                    if not entry.consolidated and entry.summary in source_summaries:
                        entry.consolidated = True
                for entry in self.memory.long_term.entries:
                    if not entry.consolidated and entry.summary in source_summaries:
                        entry.consolidated = True
            
            return {
                "status": "consolidated",
                "insight": insight,
                "topics": related_topics,
                "sources_processed": len(source_summaries),
            }
        
        self._add_tool_internal(Tool(consolidate_memories))
        return self

    def with_inbox(self, inbox_path: str = './inbox', poll_interval: int = 5) -> 'Agent':
        """Watch a directory for new files and ingest them as memories.
        
        Dropped files are processed by the agent and stored as memories.
        After processing, files are moved to a .processed/ subdirectory.
        
        Args:
            inbox_path: Directory to watch for new files.
            poll_interval: Seconds between directory polls.
        
        Returns:
            Self for chaining.
        """
        import os
        self._inbox_path = os.path.abspath(inbox_path)
        self._inbox_poll_interval = poll_interval
        self._inbox_processed = set()  # Track processed files
        return self

    async def _poll_inbox(self) -> None:
        """Poll the inbox directory for new files (called by the event loop)."""
        import os
        inbox = getattr(self, '_inbox_path', None)
        if not inbox or not os.path.isdir(inbox):
            return
        
        processed_dir = os.path.join(inbox, '.processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        processed = getattr(self, '_inbox_processed', set())
        
        for filename in os.listdir(inbox):
            filepath = os.path.join(inbox, filename)
            
            # Skip directories, hidden files, already processed
            if (os.path.isdir(filepath) or filename.startswith('.') 
                    or filepath in processed):
                continue
            
            try:
                # Read file content
                with open(filepath, 'r', errors='replace') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Ingest via the agent
                logger.info(f"Inbox: processing {filename}")
                await self.infer(
                    f"Process and remember this information from file '{filename}':\n\n{content[:4000]}"
                )
                
                # Move to processed
                import shutil
                dest = os.path.join(processed_dir, filename)
                shutil.move(filepath, dest)
                processed.add(filepath)
                logger.info(f"Inbox: processed {filename} → .processed/")
                
            except Exception as e:
                logger.error(f"Inbox: failed to process {filename}: {e}")
                processed.add(filepath)  # Don't retry on error

    async def start_inbox(self) -> None:
        """Start the inbox file watcher as a background task."""
        import asyncio
        interval = getattr(self, '_inbox_poll_interval', 5)
        
        async def _watch_loop():
            while True:
                await self._poll_inbox()
                await asyncio.sleep(interval)
        
        self._inbox_task = asyncio.create_task(_watch_loop())
        logger.info(f"Inbox watcher started: {self._inbox_path}")

    def stop_inbox(self) -> None:
        """Stop the inbox file watcher."""
        task = getattr(self, '_inbox_task', None)
        if task:
            task.cancel()
            self._inbox_task = None
            logger.info("Inbox watcher stopped")
