# agentu Enhancement Roadmap — Aligning with 2026 Agent Trends

**Date:** 2026-07-04 · **Baseline:** agentu v1.20.0

This document maps agentu's current capabilities against where the agent ecosystem has moved (MCP spec evolution, A2A interop, context engineering, harness engineering, agentic memory, safety architecture) and proposes concrete enhancements. Each item states the trend, what agentu has today, and what to build.

---

## Where agentu already stands strong

agentu is ahead of most small runtimes on several 2026 themes — worth calling out because these are marketing/positioning assets, not just features:

- **Code Mode** (`codemode=True`) — agents writing Python to call tools is now considered the single biggest context-economics lever (Anthropic's "Code execution with MCP" claims ~98% token reduction; Cloudflare Code Mode). agentu shipped this in 1.19.
- **Loop engineering** (schedules, maker-checker sub-agents, worktree isolation) — matches the "ambient agents" / background-agent trend (Claude Code Remote Tasks, LangGraph cron jobs).
- **Deferred tools + `search_tools`** — mirrors Anthropic's Tool Search Tool (`defer_loading`) pattern that reduces context ~85%.
- **Skills with progressive disclosure** — 3-level loading (metadata → instructions → resources) is exactly the SKILL.md progressive-disclosure model.
- **Guardrails with self-correction, tool permission tiers, subprocess sandbox** — the "harness is the moat" primitives.
- **Tiered semantic caching** — deepest, best-tested subsystem; few frameworks have this.

The gaps below are about *completing* these bets and adopting the interop standards that solidified in the last 12 months.

---

## Tier 1 — Standards currency (table stakes)

### 1.1 MCP: finish the client, track the v2 spec

**Trend.** MCP is now Linux Foundation–governed (Agentic AI Foundation), with spec releases 2025-06-18 (elicitation, structured tool output, OAuth resource-server split), 2025-11-25 (URL-mode elicitation, sampling-with-tools, tasks, CIMD auth), and a **2026-07-28 release candidate that makes the protocol core stateless** (no `initialize` handshake, no `Mcp-Session-Id`). Python SDK v2.0.0b1 already implements the RC.

**agentu today.** Hand-rolled HTTP transport (`src/agentu/mcp/transport.py`), bearer/API-key auth. **STDIO transport raises `NotImplementedError`** (transport.py:534) — this locks out the majority of MCP servers, which are stdio-first.

**Enhancements:**
- Implement STDIO transport (highest-impact single fix in the repo — most local MCP servers are stdio).
- Adopt Streamable HTTP as primary remote transport; avoid hardcoding session-id/initialize assumptions so the v2 stateless core is a small migration.
- Consider depending on the official `mcp` Python SDK (≥1.28 now, v2 in July) instead of maintaining a bespoke transport — the spec is moving too fast to track by hand.
- Client-side **elicitation** support (form mode + URL mode) so servers can ask the user for input mid-call.
- **OAuth 2.1 + CIMD** client auth (client_id as HTTPS metadata URL) — the recommended registration path, superseding DCR.
- Optional: MCP **tasks** (call-now/fetch-later) for long-running server operations, matching the SEP-2663 extension shape.

### 1.2 SKILL.md open standard compatibility

**Trend.** Agent Skills became a cross-vendor standard (agentskills.io, Dec 2025), adopted by OpenAI Codex with discovery from `.agents/skills`. Format: `SKILL.md` with YAML frontmatter (`name`, `description`), progressive disclosure.

**agentu today.** Skills use `skill.json` or directory-name metadata; GitHub fetching, TTL caching, progressive loading are all there — but the *format* is proprietary.

**Enhancements:**
- Parse standard `SKILL.md` frontmatter as the primary skill format (keep `skill.json` for back-compat).
- Auto-discover skills from `.agents/skills/`, `~/.agents/skills/`, and `.claude/skills/` so agentu agents can consume the same skill libraries as Claude Code and Codex.
- Upgrade `_match_skills` from keyword/stem matching to description-based matching using the existing cache-embedding infrastructure (see 2.3).

### 1.3 AGENTS.md convention

**Trend.** AGENTS.md is now an AAIF founding project (60k+ repos). `with_rules("AGENTS.md")` already exists — extend it to **auto-discover** AGENTS.md / CLAUDE.md hierarchically (repo root → subdirectory) instead of requiring an explicit path, matching how Claude Code and Codex load instruction files.

---

## Tier 2 — Core loop and context engineering

### 2.1 True multi-turn tool iteration in `infer()`

**Trend.** Every serious runtime runs an agentic loop: model → tool calls → results → model, until done.

**agentu today.** The traditional (non-codemode) `infer()` path **terminates after one non-search tool call** (`_core/agent.py`). Multi-step work only happens through codemode. This is the biggest core-loop gap.

**Enhancement.** Make the tool-calling path loop until the model stops requesting tools or `max_turns` is hit, with parallel tool-call execution (`asyncio.gather`) when the model requests several at once. This also unlocks trajectory evals (4.2).

### 2.2 Context management: compaction and token budgets

**Trend.** "Context engineering" replaced prompt engineering. Anthropic context editing (auto-clear stale tool results: +29% evals, −84% tokens on long tasks) and OpenAI's compaction endpoint are now provider-native; frameworks implement client-side fallbacks.

**agentu today.** Nothing — sessions accumulate history unboundedly; long ralph/schedule loops will hit context limits or degrade ("context rot").

**Enhancements:**
- Token accounting per session/agent (approximate counting is fine to start).
- Tiered compaction on threshold: (1) clear/truncate old tool results first, (2) LLM-summarize older turns second; keep system prompt + recent turns intact.
- `agent.with_context(max_tokens=..., compaction="auto")` builder, applied inside `infer()`, `stream()`, sessions, and ralph loops (ralph benefits most — it's the long-horizon path).

### 2.3 Semantic memory and semantic tool search

**Trend.** File-based agentic memory (Anthropic `memory_20250818` tool) + background "sleep-time" consolidation (Letta, OpenAI Dreaming) are the 2026 memory patterns.

**agentu today.** Memory recall is **substring matching only** (`memory/memory.py`) despite the embedding stack already existing in `cache/embeddings.py`. Tool search is keyword-overlap scoring. `consolidate_memory()` exists but is importance-threshold-based, not synthesis.

**Enhancements:**
- Wire `LocalEmbedding`/`APIEmbedding` + `SemanticIndex` (already built for cache) into `Memory.recall()` — cheap win, big quality jump.
- Reuse the same for deferred-tool search (embedding match on tool descriptions).
- **Memory-as-a-tool**: expose a file-based memory directory the agent itself reads/writes (CRUD tool compatible with Anthropic memory-tool semantics), so any model gets persistent agentic memory.
- **Sleep-time consolidation**: a background job (piggyback on the existing `Scheduler`) that periodically rewrites/merges memories — dedupe, resolve contradictions, distill. This composes two things agentu already has.

### 2.4 Structured outputs hardening

agentu has `output_schema` via OpenAI `response_format`. Add: Pydantic-model-in/Pydantic-instance-out (`infer(..., output_type=MyModel)`), validation-failure retry loop (reuse the guardrail self-correction machinery), and tool-based extraction fallback for providers without native json_schema.

---

## Tier 3 — Interop surfaces (don't invent wire formats)

### 3.1 A2A protocol support

**Trend.** A2A v1.0 (Linux Foundation, 150+ orgs, stable early 2026) is the agent⇄agent standard: Agent Cards at `/.well-known/agent-card.json`, task lifecycle (submitted → working → input-required → completed), JSON-RPC/HTTP+JSON transports. The consensus stack: **MCP = agent⇄tools, A2A = agent⇄agent, AG-UI = agent⇄user.**

**agentu today.** `serve()` exposes a bespoke REST API (`/execute`, `/process`).

**Enhancement.** `serve(agent, a2a=True)`: publish an Agent Card generated from the agent's name/tools/skills, and map A2A task lifecycle onto the existing session + observer machinery (via the official `a2a-sdk`). This makes every agentu agent discoverable/callable by Microsoft Agent Framework, Bedrock AgentCore, ADK, etc. — outsized interop payoff for modest effort.

### 3.2 AG-UI event stream

**Trend.** AG-UI (~16 typed SSE events: lifecycle, text deltas, tool calls, state deltas) is the de facto agent⇄frontend protocol, supported by LangGraph, CrewAI, Pydantic AI, ADK, Bedrock AgentCore.

**agentu today.** Custom SSE/WebSocket streaming in `runtime/serve.py`, custom event types in the observer.

**Enhancement.** An AG-UI adapter that maps agentu's `EventType` stream + `stream()` chunks onto AG-UI events (the `ag-ui-protocol` PyPI SDK does the encoding). Existing observability events make this mostly a translation layer, and it would let the dashboard be replaced/augmented by any AG-UI-compatible frontend.

### 3.3 Anthropic-native provider (and provider abstraction)

**agentu today.** OpenAI-compatible `/chat/completions` only. That covers Ollama/vLLM/OpenAI, but misses Anthropic-native capabilities that matter for agents: context editing (server-side compaction), the memory tool, prompt caching controls, and strict tool use. A thin provider seam (`LLMProvider` protocol with the current OpenAI-compat implementation as default, plus an Anthropic Messages implementation) keeps the "portability first" strategy while unlocking native features. The `AgentConfig` docstring already promises LiteLLM-style strings — either implement or remove the claim.

---

## Tier 4 — Harness engineering depth

### 4.1 Hooks

**Trend.** Lifecycle hooks (PreToolUse *blocking*, PostToolUse, Stop) are how mandatory policy is enforced in the harness rather than the prompt — a defining Claude Code/Codex feature.

**agentu today.** Middleware wraps LLM calls only; guardrails check input/output text. There's no way to intercept/deny/rewrite a *tool call* before it runs.

**Enhancement.** `agent.with_hooks(pre_tool=..., post_tool=..., on_stop=...)` where `pre_tool` can **block or modify** a call (return deny + reason fed back to the model). Sync callables, async callables, and shell commands (JSON on stdin, Claude Code–style) as hook targets. This also becomes the enforcement point for permission modes (4.2) and trifecta rules (5.1).

### 4.2 Permission modes and human-in-the-loop gates

**Trend.** Layered permissions (allow/deny/ask rules + modes like plan/acceptEdits/bypass + runtime callback) and durable HITL interrupts (LangGraph `interrupt()`/`Command(resume=...)`, A2A `input-required`) are standard.

**agentu today.** Three static permission tiers; DANGEROUS is blocked unless `allow_dangerous=True`. No "ask" path — no way for a human to approve a single call.

**Enhancements:**
- `can_use_tool` async callback: given (tool, args, context) → allow / deny / **ask**.
- An "ask" implementation that works headless: persist the pending call (SQLite, like the findings inbox), notify via the existing `NotifyMiddleware` (Slack/Discord), and resume when approved — i.e., interrupts that survive restarts. This composes three existing subsystems and fits the scheduled-agents story perfectly.
- Permission modes on the agent: `mode="plan"` (no write tools), `mode="auto"` (current), `mode="ask-writes"` (WRITE tools require approval).

### 4.3 Sandbox hardening

**agentu today.** Subprocess isolation with timeout kill; self-described "not security-grade"; **`max_memory_mb` is defined but never enforced** (no ulimit applied in `SubprocessSandbox.execute`).

**Enhancements:**
- Actually apply resource limits (`resource.setrlimit` in a preexec hook on POSIX).
- **Default-deny egress** option: the 2026 consensus is that network allowlists are *the* practical prompt-injection defense (kills the exfiltration leg of the lethal trifecta). Even env-var proxy enforcement (`HTTP(S)_PROXY` to a filtering proxy) beats nothing.
- Pluggable OS-level backend that shells out to Anthropic's open-source `sandbox-runtime` (Seatbelt on macOS, bubblewrap on Linux) when available; container/remote drivers (Docker, E2B) behind the same `SandboxBackend` protocol.

### 4.4 Agent-level checkpoint/resume

Workflows and ralph have checkpoint/resume; plain `infer()` sessions don't. Add session snapshot/restore (`session.checkpoint()` / `SessionManager.resume(id)`) with fork support, journaling each step to SQLite. This is the "durable execution lite" layer — and leaves a documented seam for Temporal-class engines for users who need infrastructure-grade durability.

---

## Tier 5 — Safety architecture

### 5.1 Lethal-trifecta accounting

**Trend.** Simon Willison's "lethal trifecta" (private-data access + untrusted content + external communication) and OWASP's Agentic Top 10 (2026) frame injection defense as *architectural*. Microsoft "spotlighting" (delimiting/marking untrusted content) is baseline hygiene; CaMeL-style quarantine is the reference architecture.

**agentu today.** Permission tiers exist, but nothing models data provenance.

**Enhancements:**
- Tag tools with capability flags: `reads_private`, `ingests_untrusted`, `communicates_externally`. Warn (or require explicit opt-in) when one agent composition has all three.
- **Spotlight tool outputs by default**: wrap results from `ingests_untrusted` tools in clear delimiters with an instruction that the content is data, not instructions.
- Guardrail hooks at all four points (user input, model output, **tool input, tool output**) — today only the first two exist; tool-input guardrails catch exfiltration attempts (e.g., a URL containing private data) before execution.
- Fix the README/PKG-INFO drift: `NoPII()`/`NoHallucination()` are documented but don't exist (actual classes: `PII`, `ContentFilter`, `MaxLength`, `JSONSchema`).

---

## Tier 6 — Observability & evals

### 6.1 OpenTelemetry GenAI spans

**Trend.** OTel GenAI semantic conventions (client LLM spans stable early 2026; `invoke_agent` → `chat` → `execute_tool` hierarchy) are what Langfuse, Phoenix, Logfire, LangSmith all consume.

**agentu today.** A custom `Observer` with custom event types and a bespoke dashboard — good, but an island.

**Enhancement.** An OTLP exporter for the observer: map existing events onto `gen_ai.*` spans (pin a semconv version; prompt-content capture opt-in). Keep the console/dashboard outputs; add interop. `pip install agentu[otel]`.

### 6.2 Trajectory evals + trace→eval export

`evaluate()` checks final answers. Once 2.1 lands (real multi-turn loops), add trajectory assertions: expected tool sequence, tool-argument schema validity, redundant-call detection. Add `observer.to_eval_case()` — converting a recorded production trace into a regression test case is the converged industry workflow.

### 6.3 Judge panels for maker-checker

`delegate()` uses a single checker. Add `judges=N` panel-consensus and a `best_of(n)` fan-out (N makers in parallel worktrees — the isolation already exists — one judge selects/merges). Add per-subagent token budgets and circuit breakers: multi-agent ≈ 15× single-agent tokens (Anthropic's research-system numbers), so cost governance is mandatory.

---

## Quick fixes (independent of trends)

| Fix | Location |
|---|---|
| STDIO MCP transport (`NotImplementedError`) | `src/agentu/mcp/transport.py:534` |
| Worktree branch cleanup is a `pass` stub | `src/agentu/workflow/worktree.py:138-141` |
| `with_worktree()` documented for `infer()` but only applied in `delegate()` | `_core/agent.py` |
| Sandbox `max_memory_mb` never enforced | `src/agentu/runtime/sandbox.py` |
| README documents nonexistent `NoPII()`/`NoHallucination()` guardrails | `README.md`, PKG-INFO |
| `AgentConfig` docstring claims LiteLLM support that doesn't exist | `_core/config.py` |
| Stale example paths in README (`examples/basic.py` vs `examples/01_basics/`) | `README.md` |
| `memory.db` committed to repo root | `.gitignore` |

---

## Suggested sequencing

1. **Now (correctness + table stakes):** quick fixes; multi-turn `infer()` loop (2.1); MCP STDIO (1.1); semantic memory recall (2.3, reuses existing embeddings).
2. **Next (standards):** SKILL.md format + `.agents/skills` discovery (1.2); pre-tool hooks (4.1); `can_use_tool` + ask-mode approvals via notifier (4.2); context compaction (2.2).
3. **Then (interop):** A2A serve mode (3.1); AG-UI adapter (3.2); OTel exporter (6.1); provider seam + Anthropic-native (3.3).
4. **Later (depth):** trifecta tagging + spotlighting (5.1); sandbox egress control (4.3); memory-as-a-tool + sleep-time consolidation (2.3); judge panels + `best_of` (6.3); trajectory evals (6.2).

The theme across all of it: agentu made the right early bets (code mode, loops, skills, tool search, sandbox). 2026's shift is from proprietary implementations of good ideas to **shared standards for the same ideas** — MCP v2, SKILL.md, A2A, AG-UI, OTel GenAI. Adopting those wire formats while keeping agentu's fluent, batteries-included API is the highest-leverage direction.

---

### Sources (primary)

- MCP: [2025-11-25 changelog](https://modelcontextprotocol.io/specification/2025-11-25/changelog) · [2026-07-28 RC](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/) · [MCP joins AAIF](https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/)
- A2A: [v1.0 announcement](https://a2a-protocol.org/latest/announcing-1.0/) · [a2a-sdk](https://pypi.org/project/a2a-sdk/)
- Context engineering: [Anthropic — effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [context management](https://claude.com/blog/context-management) · [advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) · [code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- Skills & harness: [Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) · [Codex skills](https://developers.openai.com/codex/skills) · [hooks](https://code.claude.com/docs/en/hooks) · [sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime)
- Memory: [memory tool cookbook](https://platform.claude.com/cookbook/tool-use-memory-cookbook) · [Letta sleep-time compute](https://www.letta.com/blog/sleep-time-compute/)
- Multi-agent: [Anthropic multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) · [OpenAI Agents SDK orchestration](https://openai.github.io/openai-agents-python/multi_agent/)
- Observability: [OTel GenAI](https://opentelemetry.io/blog/2026/genai-observability/) · [semconv repo](https://github.com/open-telemetry/semantic-conventions-genai)
- Streaming UX: [AG-UI](https://docs.ag-ui.com/introduction)
- Safety: [lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) · [CaMeL](https://arxiv.org/abs/2503.18813) · [OpenAI guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- Long-running agents: [long-running Claude](https://www.anthropic.com/research/long-running-Claude) · [ambient agents](https://www.langchain.com/blog/introducing-ambient-agents)
