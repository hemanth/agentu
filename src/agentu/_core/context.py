"""Context management and compaction for agentu.

Prevents context window overflow in long-running loops (ralph, schedules)
by applying tiered compaction strategies.

Strategy:
    Tier 1: Truncate/clear old tool results (cheapest)
    Tier 2: LLM-summarize older conversation turns
    Tier 3: Keep system prompt + recent N turns intact (always)

Usage:
    agent = Agent("assistant").with_context(
        max_tokens=100_000,
        compaction="auto",
    )
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union

logger = logging.getLogger(__name__)

# Rough chars-to-tokens ratio for approximate counting
CHARS_PER_TOKEN = 4


class CompactionMode(Enum):
    """Compaction strategy modes."""
    NONE = "none"           # No compaction (legacy behavior)
    AUTO = "auto"           # Tiered compaction: truncate results → summarize → keep recent
    TRUNCATE = "truncate"   # Only truncate old tool results
    SUMMARIZE = "summarize" # Summarize older turns via LLM


@dataclass
class ContextConfig:
    """Configuration for context management.

    Args:
        max_tokens: Approximate max tokens for conversation context.
            When exceeded, compaction is triggered.
        compaction: Compaction strategy (none, auto, truncate, summarize).
        keep_recent: Number of recent turns to always keep intact.
        max_result_chars: Max characters per tool result before truncation.
        summary_prompt: Custom prompt for LLM summarization.
    """
    max_tokens: int = 128_000
    compaction: CompactionMode = CompactionMode.NONE
    keep_recent: int = 5
    max_result_chars: int = 2000
    summary_prompt: Optional[str] = None


@dataclass
class ContextStats:
    """Statistics about context usage."""
    total_tokens: int = 0
    total_turns: int = 0
    compactions_performed: int = 0
    tokens_saved: int = 0


def estimate_tokens(text: str) -> int:
    """Approximate token count from text length.

    Uses a simple chars/4 heuristic. Good enough for budget management
    without requiring a tokenizer dependency.
    """
    return len(text) // CHARS_PER_TOKEN


def estimate_history_tokens(history: List[Dict[str, Any]]) -> int:
    """Estimate total tokens in conversation history."""
    total = 0
    for entry in history:
        total += estimate_tokens(json.dumps(entry, default=str))
    return total


def truncate_tool_results(
    history: List[Dict[str, Any]],
    max_result_chars: int = 2000,
    keep_recent: int = 5,
) -> List[Dict[str, Any]]:
    """Tier 1 compaction: truncate large tool results in older turns.

    Keeps the most recent `keep_recent` turns intact; truncates
    tool result strings in older turns to `max_result_chars`.

    Args:
        history: Conversation history list
        max_result_chars: Max chars per tool result
        keep_recent: Number of recent entries to preserve

    Returns:
        Modified history (mutates in place for efficiency)
    """
    if len(history) <= keep_recent:
        return history

    compacted_count = 0
    for entry in history[:-keep_recent]:
        response = entry.get("response", {})
        if isinstance(response, dict):
            result = response.get("result")
            if isinstance(result, str) and len(result) > max_result_chars:
                response["result"] = (
                    result[:max_result_chars]
                    + f"\n... [truncated {len(result) - max_result_chars} chars]"
                )
                compacted_count += 1

            # Also truncate history within the response
            hist = response.get("history", [])
            if isinstance(hist, list):
                for h in hist:
                    if isinstance(h, dict):
                        r = h.get("result")
                        if isinstance(r, str) and len(r) > max_result_chars:
                            h["result"] = (
                                r[:max_result_chars]
                                + f"\n... [truncated {len(r) - max_result_chars} chars]"
                            )
                            compacted_count += 1

    if compacted_count > 0:
        logger.debug(f"Tier 1 compaction: truncated {compacted_count} tool results")

    return history


def drop_old_turns(
    history: List[Dict[str, Any]],
    max_tokens: int,
    keep_recent: int = 5,
) -> List[Dict[str, Any]]:
    """Emergency compaction: drop oldest turns to fit within budget.

    Used when summarization is not available or budget is very tight.

    Args:
        history: Conversation history
        max_tokens: Target token budget
        keep_recent: Minimum recent turns to keep

    Returns:
        Trimmed history
    """
    if len(history) <= keep_recent:
        return history

    while len(history) > keep_recent:
        current_tokens = estimate_history_tokens(history)
        if current_tokens <= max_tokens:
            break
        # Drop the oldest entry
        dropped = history.pop(0)
        logger.debug(
            f"Dropped oldest turn: {dropped.get('user_input', 'unknown')[:50]}..."
        )

    return history


async def summarize_turns(
    history: List[Dict[str, Any]],
    llm_call: Callable,
    keep_recent: int = 5,
    summary_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Tier 2 compaction: LLM-summarize older turns.

    Replaces old turns with a single summary turn, keeping
    the most recent `keep_recent` turns intact.

    Args:
        history: Conversation history
        llm_call: Async LLM call function (prompt -> response)
        keep_recent: Number of recent turns to keep intact
        summary_prompt: Custom summarization prompt

    Returns:
        Compacted history with summary replacing old turns
    """
    if len(history) <= keep_recent:
        return history

    old_turns = history[:-keep_recent]
    recent_turns = history[-keep_recent:]

    # Build summarization input
    turns_text = []
    for turn in old_turns:
        user_msg = turn.get("user_input", "")
        response = turn.get("response", {})
        tool = response.get("tool_used", "none") if isinstance(response, dict) else "none"
        result_preview = str(response.get("result", ""))[:200] if isinstance(response, dict) else str(response)[:200]
        turns_text.append(f"- User: {user_msg[:100]}\n  Tool: {tool}, Result: {result_preview}")

    prompt = summary_prompt or (
        "Summarize the following conversation turns into a concise context summary. "
        "Focus on key facts, decisions, and state changes. Be brief.\n\n"
    )
    prompt += "\n".join(turns_text)

    try:
        summary = await llm_call(prompt)
        summary_entry = {
            "user_input": "[Context Summary]",
            "response": {
                "text_response": summary,
                "summarized_turns": len(old_turns),
            },
            "turns": 0,
        }
        logger.info(f"Tier 2 compaction: summarized {len(old_turns)} turns into context summary")
        return [summary_entry] + recent_turns
    except Exception as e:
        logger.warning(f"Summarization failed, falling back to turn dropping: {e}")
        return recent_turns


async def compact_context(
    history: List[Dict[str, Any]],
    config: ContextConfig,
    llm_call: Optional[Callable] = None,
    system_prompt: str = "",
) -> tuple:
    """Apply tiered compaction to conversation history.

    Returns:
        Tuple of (compacted_history, stats)
    """
    stats = ContextStats(
        total_turns=len(history),
        total_tokens=estimate_history_tokens(history) + estimate_tokens(system_prompt),
    )

    if config.compaction == CompactionMode.NONE:
        return history, stats

    original_tokens = stats.total_tokens

    # Tier 1: Truncate old tool results
    if config.compaction in (CompactionMode.AUTO, CompactionMode.TRUNCATE):
        history = truncate_tool_results(
            history,
            max_result_chars=config.max_result_chars,
            keep_recent=config.keep_recent,
        )

    current_tokens = estimate_history_tokens(history) + estimate_tokens(system_prompt)

    # Check if we're within budget after tier 1
    if current_tokens <= config.max_tokens:
        stats.total_tokens = current_tokens
        stats.tokens_saved = original_tokens - current_tokens
        if stats.tokens_saved > 0:
            stats.compactions_performed = 1
        return history, stats

    # Tier 2: Summarize older turns (only if LLM is available)
    if config.compaction in (CompactionMode.AUTO, CompactionMode.SUMMARIZE):
        if llm_call is not None:
            history = await summarize_turns(
                history,
                llm_call=llm_call,
                keep_recent=config.keep_recent,
                summary_prompt=config.summary_prompt,
            )
        else:
            # No LLM available, fall back to dropping
            history = drop_old_turns(
                history,
                max_tokens=config.max_tokens - estimate_tokens(system_prompt),
                keep_recent=config.keep_recent,
            )

    final_tokens = estimate_history_tokens(history) + estimate_tokens(system_prompt)
    stats.total_tokens = final_tokens
    stats.tokens_saved = original_tokens - final_tokens
    stats.compactions_performed = 1 if stats.tokens_saved > 0 else 0
    stats.total_turns = len(history)

    return history, stats
