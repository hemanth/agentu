# agentu

A harness-engineered AI agent runtime. Build agents with tool isolation, self-correction, and permission scoping out of the box.

```bash
pip install agentu
```

## Quick start

```python
from agentu import Agent

def search_products(query: str) -> list:
    return db.products.search(query)

agent = Agent("sales").with_tools([search_products])

# Call a tool directly
result = await agent.call("search_products", {"query": "laptop"})

# Or let the LLM figure it out
result = await agent.infer("Find me laptops under $1500")
```

`call()` runs a tool. `infer()` lets the LLM pick the tool and fill in the parameters from natural language.

## Sandboxed tool execution

Tools run in isolated subprocesses with timeouts and permission scoping. Separate what the agent can read from what it can write:

```python
agent = Agent("assistant").with_sandbox(
    read_tools=[search, get_weather],
    write_tools=[save_file, send_email],
    timeout=10,
)

result = await agent.infer("Find the weather and save it to a file")
```

- `read_tools` get READONLY permission, no side effects
- `write_tools` get WRITE permission, side effects allowed
- Every tool runs in a subprocess, not in your agent's process
- If a tool hangs past `timeout`, subprocess is killed, agent stays alive
- Sandbox exit codes, stderr, and timeouts are captured in the observer

## Code Mode

Instead of making individual JSON tool calls, the LLM writes Python code that calls your tools directly. Inspired by [Cloudflare's Code Mode](https://blog.cloudflare.com/code-mode/) — LLMs are better at writing code than making tool calls because they've seen millions of lines of real code, but only synthetic tool-call training data.

```python
# Code Mode: LLM writes Python that chains the calls
agent = Agent("bot", codemode=True).with_tools([search, get_weather, save_file])
await agent.infer("Search for weather in SF and save it")
# LLM writes:
#   results = tools.search("weather SF")
#   weather = tools.get_weather(location="SF")
#   tools.save_file("weather.txt", weather)
# ONE round trip, ONE code execution
```

How it works:
1. Your tools are converted to typed Python stubs and placed in the system prompt
2. The LLM writes Python code using `tools.search(query="...")` syntax
3. Safe stdlib imports allowed (math, json, re) — dangerous ones blocked (os, sys)
4. Auto-retry: if code fails, error feeds back to LLM for self-correction

## Guardrails with self-correction

When output guardrails fail, the agent retries automatically by feeding the violation back to the LLM:

```python
agent = Agent("assistant").with_guardrails(
    output_guardrails=[NoPII(), NoHallucination()],
    max_corrections=2,
)

result = await agent.infer("Summarize the customer data")
# If the LLM leaks PII, it retries up to 2 times with the violation as feedback
```

## Rule files

Prepend project-level rules to every LLM call:

```python
agent = Agent("assistant").with_rules("AGENTS.md")
```

The contents of `AGENTS.md` get prepended to the system prompt. Works with declarative config too:

```yaml
name: "support-agent"
model: "openai/gpt-4o"
rules: "AGENTS.md"
```

## Tool permissions

Three permission levels control what tools can do:

```python
from agentu import Agent, Tool, ToolPermission

agent = Agent("bot").with_tools([
    Tool(search, permission=ToolPermission.READONLY),     # always allowed
    Tool(save_file, permission=ToolPermission.WRITE),     # allowed, logged
    Tool(delete_all, permission=ToolPermission.DANGEROUS), # blocked by default
])

# Explicitly allow DANGEROUS tools
agent.with_permissions(allow_dangerous=True)
```

## Declarative configuration

Deploy agents from YAML or JSON with zero code:

**1. Create a `bot.yaml` (or `.json`)**
```yaml
name: "support-agent"
model: "openai/gpt-4o"
system_prompt: "You are an expert IT agent."
rules: "AGENTS.md"
notify:
  - "discord://webhook/id"
cache:
  preset: "distributed"
```

**2. Load dynamically**
```python
from agentu import Agent
import asyncio

async def main():
    agent = await Agent.from_config("bot.yaml")
    
    # Append local Python rules/tools if desired, then infer!
    agent.with_tools([resolve_ticket])
    await agent.infer("Help me reset my router")

asyncio.run(main())
```

*(Requires `pip install agentu[yaml]` to load `.yaml` files. JSON loads natively without extra dependencies).*

## Workflows

Chain agents with `>>` (sequential) and `&` (parallel):

```python
# One after another
workflow = researcher("Find AI trends") >> analyst("Analyze") >> writer("Summarize")

# All at once
workflow = search("AI") & search("ML") & search("Crypto")

# Parallel first, then merge
workflow = (search("AI") & search("ML")) >> analyst("Compare findings")

result = await workflow.run()
```

You can also pass data between steps with lambdas:

```python
workflow = (
    researcher("Find companies")
    >> analyst(lambda prev: f"Extract top 5 from: {prev['result']}")
    >> writer(lambda prev: f"Write report about: {prev['companies']}")
)
```

Interrupted workflows can resume from the last successful step:

```python
from agentu import resume_workflow

result = await workflow.run(checkpoint="./checkpoints", workflow_id="my-report")

# After a crash, pick up where you left off
await resume_workflow(result["checkpoint_path"])
```

## Caching

Cache LLM responses to skip redundant API calls. Works with both plain strings and full conversations.

```python
# Basic: memory + SQLite, 1-hour TTL
agent = Agent("assistant").with_cache()

# Same prompt, same response — no API call
await agent.infer("What is Python?")  # hits the LLM
await agent.infer("What is Python?")  # instant, from cache
```

### Presets

```python
# Exact match only (memory + SQLite)
agent.with_cache(preset="basic")

# Semantic matching — "vegan food" hits cache for "plant-based meals"
agent.with_cache(preset="smart", similarity_threshold=0.9)

# Offline-friendly with filesystem backup and background sync
agent.with_cache(preset="offline")

# Redis-backed for distributed setups
agent.with_cache(preset="distributed", redis_url="redis://localhost:6379")
```

### Conversation caching

Full conversation lists cache the same way strings do -- deterministic serialization, same hash, same hit:

```python
conversation = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What's the weather?"},
]
cache.set(conversation, "my-bot", "Looks sunny today.")
cache.get(conversation, "my-bot")  # → "Looks sunny today."
```

The second parameter is a `namespace` -- any string that scopes the cache. Usually the model name, but it can be anything.

### How matching works

| Strategy | How | When |
|---|---|---|
| Exact | SHA-256 hash of prompt + namespace + temperature | Default, always runs first |
| Semantic | Cosine similarity of embedding vectors | `preset="smart"` or higher, runs on exact miss |

Semantic matching uses an embedding model (local `all-MiniLM-L6-v2` or API-based `nomic-embed-text`) and only returns a hit when similarity exceeds the threshold (default 0.95).

## Memory

```python
agent.remember("Customer prefers email", importance=0.9)
memories = agent.recall(query="communication preferences")
```

SQLite-backed, searchable, persistent across sessions.

### Rationale Recording (ADRs)

Agents can explicitly record architectural decisions and the reasoning behind their actions, creating an automated audit trail.

```python
agent = Agent("architect", enable_memory=True, enable_rationale_recording=True)

# The agent will automatically evaluate trade-offs and use the `record_rationale` tool
await agent.infer("Should we use threading or asyncio? Record your reasoning.")

# Retrieve the decision later
memories = agent.recall(query="asyncio")
```

Rationale events are simultaneously saved to memory (with `memory_type="rationale"`) and emitted to the observability pipeline.
## Skills

Load domain expertise on demand, either from local paths or GitHub:

```python
from agentu import Agent, Skill

# From GitHub (cached locally at ~/.agentu/skills/)
agent = Agent("assistant").with_skills([
    "hemanth/agentu-skills/pdf-processor",
    "openai/skills/code-review@v1.0",
])

# From local
agent = Agent("assistant").with_skills(["./skills/my-skill"])

# Or define inline
pdf_skill = Skill(
    name="pdf-processing",
    description="Extract text and tables from PDF files",
    instructions="skills/pdf/SKILL.md",
    resources={"forms": "skills/pdf/FORMS.md"}
)
agent = Agent("assistant").with_skills([pdf_skill])
```

Skills load progressively: metadata first (100 chars), then instructions (1500 chars), then resources only when needed.

## Sessions

Stateful conversations with automatic context:

```python
from agentu import SessionManager

manager = SessionManager()
session = manager.create_session(agent)

await session.send("What's the weather in SF?")
await session.send("What about tomorrow?")  # knows you mean SF
```

Multi-user isolation, SQLite persistence, session timeout handling.

## Evaluation

Test your agents with simple assertions:

```python
from agentu import evaluate

test_cases = [
    {"ask": "What's 5 + 3?", "expect": 8},
    {"ask": "Weather in SF?", "expect": "sunny"}
]

results = await evaluate(agent, test_cases)
print(f"Accuracy: {results.accuracy}%")
print(results.to_json())  # export for CI/CD
```

Matching strategies: exact, substring, LLM-as-judge, or custom validators.

## Observability

All LLM calls, tool executions, self-corrections, and sandbox events are tracked automatically:

```python
from agentu import Agent, observe

observe.configure(output="console")  # or "json" or "silent"

agent = Agent("assistant").with_sandbox(
    read_tools=[search],
    write_tools=[save],
    timeout=10,
)
await agent.infer("Find me laptops")

metrics = agent.observer.get_metrics()
# {"tool_calls": 3, "total_duration_ms": 1240, "errors": 0}
```

Events captured: `tool_call`, `tool_blocked`, `self_correction`, `llm_request`, `inference_start`, `inference_end`, `error`, `session_create`, `session_end`.

Sandbox events include `sandbox_exit_code`, `sandbox_stderr`, and `sandbox_timed_out` for post-mortem debugging.

### Dashboard

```python
from agentu import serve

serve(agent, port=8000)
# http://localhost:8000/dashboard — live metrics
# http://localhost:8000/docs — auto-generated API docs
```

## Notifications

Send low-latency, non-blocking alerts to Slack, Discord, Email, or SMS when an agent finishes its task.

```bash
pip install agentu[notify]
```

```python
from agentu import Agent

# Attach notification middleware via the builder pattern
agent = Agent("my-bot").with_notifier([
    "slack://bot-token/channel-id",
    "discord://webhook_id/webhook_token"
])

# The agent executes without blocking, and posts a rich summary containing tokens and elapsed ms.
await agent.infer("Audit the database schema")
```

### Custom Formatting & Failure Alerts
Notifications trigger natively on Agent crashes too (e.g. rate limits). If you want to format exactly how the alert looks for successes or failures, provide a custom formatter:

```python
from agentu.middleware import NotifyMiddleware

def custom_format(context, response, error) -> str:
    if error:
        return f"🚨 AGENT CRASH 🚨\n{error}"
    return f"✅ Agent {context.namespace} finished in {context.elapsed_ms}ms"

# Fall back to base use() method to pass the custom formatter
agent.use(NotifyMiddleware(
    targets=["slack://bot-token/channel-id"], 
    formatter=custom_format
))
```

## Ralph mode

Run agents in autonomous loops with progress tracking:

```python
result = await agent.ralph(
    prompt_file="PROMPT.md",
    max_iterations=50,
    timeout_minutes=30,
    on_iteration=lambda i, data: print(f"[{i}] {data['result'][:50]}...")
)
```

The agent loops until all checkpoints in `PROMPT.md` are complete or limits are reached.

## Loop Engineering

Design the system that prompts your agents. Three primitives for building autonomous loops:

### Scheduled Automations

Run agents on a cadence with findings persisted to SQLite:

```python
# Every 30 minutes
agent = Agent("triage").with_schedule(every=30, prompt="Review open issues")

# Cron expression (daily at 9am)
agent = Agent("ops").with_schedule(cron="0 9 * * *", prompt_file="TRIAGE.md")

# Start the scheduler
await agent.start()

# Check findings
findings = agent.findings()  # pending findings
agent.stop()                 # graceful shutdown
```

### Sub-agents (maker-checker)

Split the maker from the checker. Define roles inline or from `.agents/` directory:

```python
agent = Agent("lead").with_subagents([
    {"name": "coder", "instructions": "Write clean code.", "role": "maker"},
    {"name": "reviewer", "instructions": "Review for bugs.", "role": "checker"},
])

# Or load from .agents/ directory
agent = Agent("lead").with_subagents(".agents/")

result = await agent.delegate("Refactor the auth module")
# {"result": "...", "review": "APPROVED: ...", "approved": True, "corrections": 0}
```

Sub-agents inherit the parent's model, tools, and API config unless overridden.

### Worktree Isolation

Isolate parallel agents with git worktrees:

```python
agent = Agent("builder").with_worktree()
result = await agent.infer("Refactor auth module")
# Runs in an isolated git worktree, auto-cleaned after
```

Combine all three for a full loop:

```python
agent = (
    Agent("ops")
    .with_tools([scan_ci, check_issues])
    .with_subagents(".agents/")
    .with_worktree()
    .with_schedule(every=60, prompt="Triage CI failures")
)
await agent.start()
```

## Tool search

When you have hundreds of tools, you don't want them all in context. Deferred tools are discovered on-demand:

```python
agent = Agent("payments").with_tools(defer=[charge_card, send_receipt, refund_payment])

# Agent calls search_tools("charge card") → finds charge_card → executes it
result = await agent.infer("charge $50 to card_123")
```

A `search_tools` function is auto-added. The agent searches, activates, and calls -- all internally.

## MCP

Connect to Model Context Protocol servers:

```python
agent = await Agent("bot").with_mcp(["http://localhost:3000"])
agent = await Agent("bot").with_mcp([
    {"url": "https://api.com/mcp", "headers": {"Auth": "Bearer xyz"}}
])
```

## LLM support

Works with any OpenAI-compatible API. Auto-detects available models from Ollama:

```python
Agent("assistant")                                        # first available Ollama model
Agent("assistant", model="qwen3")                         # specific model
Agent("assistant", model="gpt-4", api_key="sk-...")       # OpenAI
Agent("assistant", model="mistral", api_base="http://localhost:8000/v1")  # vLLM, LM Studio, etc.
```

## REST API

```python
from agentu import serve

serve(agent, port=8000, enable_cors=True)
```

Endpoints: `/execute`, `/process`, `/tools`, `/memory/remember`, `/memory/recall`, `/docs`

## API reference

```python
# Agent
agent = Agent(name)                       # auto-detect model
agent = Agent(name, model="qwen3")        # explicit model
agent = Agent(name, max_turns=5)          # limit multi-turn cycles
agent.with_tools([func1, func2])          # active tools
agent.with_tools(defer=[many_funcs])      # searchable tools
agent.with_cache(preset="smart")          # caching
agent.with_skills(["github/repo/skill"])  # skills
agent.with_rules("AGENTS.md")            # project-level rules
agent.with_notifier(["slack://bot-token"])       # notifications
agent.with_permissions(allow_dangerous=True)     # permission control
await agent.with_mcp([url])              # MCP servers

# Loop Engineering
agent.with_schedule(every=30, prompt="...")       # interval schedule
agent.with_schedule(cron="0 9 * * *", prompt="...") # cron schedule
agent.with_subagents([{...}])                    # inline sub-agents
agent.with_subagents(".agents/")                 # from directory
agent.with_worktree()                            # git isolation
await agent.start()                              # start schedules
agent.stop()                                     # stop schedules
agent.findings()                                 # get findings
await agent.delegate("task")                     # maker-checker

# Sandbox
agent.with_sandbox(                       # tool isolation
    read_tools=[search, get_weather],
    write_tools=[save_file, send_email],
    timeout=10,
)

# Guardrails
agent.with_guardrails(                    # self-correction
    output_guardrails=[NoPII()],
    max_corrections=2,
)

await agent.call("tool", params)          # direct tool execution
await agent.infer("natural language")     # LLM-routed execution

agent.remember(content, importance=0.8)   # store memory
agent.recall(query)                       # search memory

# Sessions
manager = SessionManager()
session = manager.create_session(agent)
await session.send("message")
session.get_history(limit=10)
session.clear_history()

# Evaluation
results = await evaluate(agent, test_cases)
results.accuracy     # 95.0
results.to_json()    # export

# Workflows
step1 >> step2          # sequential
step1 & step2           # parallel
await workflow.run()    # execute
```

## Examples

```bash
git clone https://github.com/hemanth/agentu && cd agentu

python examples/basic.py                # simple agent
python examples/workflow.py             # workflows
python examples/memory.py               # memory system
python examples/example_sessions.py     # stateful sessions
python examples/example_eval.py         # agent evaluation
python examples/example_observe.py      # observability
python examples/api.py                  # REST API
```

## Testing

```bash
pytest
pytest --cov=agentu
```

## License

MIT
