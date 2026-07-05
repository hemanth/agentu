"""ContextMixin – context management methods extracted from Agent."""

import logging

logger = logging.getLogger(__name__)


class ContextMixin:
    """Mixin providing context management for the Agent class.

    Methods here assume they are mixed into an Agent instance that has
    ``_context_config`` and ``context`` attributes.
    """

    def with_context(
        self,
        max_tokens: int = 128_000,
        compaction: str = "auto",
        keep_recent: int = 5,
        max_result_chars: int = 2000,
    ) -> 'ContextMixin':
        """Configure context management and compaction.

        Prevents context window overflow in long-running loops by
        applying tiered compaction strategies:
          1. Truncate old tool results
          2. LLM-summarize older turns
          3. Keep system prompt + recent turns intact

        Args:
            max_tokens: Approximate max tokens for context (default: 128K)
            compaction: Strategy - 'none', 'auto', 'truncate', 'summarize'
            keep_recent: Number of recent turns to always preserve
            max_result_chars: Max chars per tool result before truncation

        Returns:
            Self for method chaining

        Example:
            >>> agent = Agent("assistant").with_context(
            ...     max_tokens=100_000,
            ...     compaction="auto",
            ... )
        """
        from .context import ContextConfig, CompactionMode
        self._context_config = ContextConfig(
            max_tokens=max_tokens,
            compaction=CompactionMode(compaction),
            keep_recent=keep_recent,
            max_result_chars=max_result_chars,
        )
        logger.info(f"Context management enabled: max_tokens={max_tokens}, compaction={compaction}")
        return self

    def set_context(self, context: str) -> None:
        """Set the context for the agent."""
        self.context = context
