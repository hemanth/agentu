# agentu

Build AI agents that actually do things.

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

Full conversation lists cache the same way strings do — deterministic serialization, same hash, same hit:

```python
conversation = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What's the weather?"},
]
cache.set(conversation, "my-bot", "Looks sunny today.")
cache.get(conversation, "my-bot")  # → "Looks sunny today."
```

The second parameter is a `namespace` — any string that scopes the cache. Usually the model name, but it can be anything.

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

All LLM calls and tool executions are tracked automatically:

```python
from agentu import Agent, observe

observe.configure(output="console")  # or "json" or "silent"

agent = Agent("assistant").with_tools([...])
await agent.infer("Find me laptops")

metrics = agent.observer.get_metrics()
# {"tool_calls": 3, "total_duration_ms": 1240, "errors": 0}
```

### Dashboard

```python
from agentu import serve

serve(agent, port=8000)
# http://localhost:8000/dashboard — live metrics
# http://localhost:8000/docs — auto-generated API docs
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

## Tool search

When you have hundreds of tools, you don't want them all in context. Deferred tools are discovered on-demand:

```python
agent = Agent("payments").with_tools(defer=[charge_card, send_receipt, refund_payment])

# Agent calls search_tools("charge card") → finds charge_card → executes it
result = await agent.infer("charge $50 to card_123")
```

A `search_tools` function is auto-added. The agent searches, activates, and calls — all internally.

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
await agent.with_mcp([url])              # MCP servers

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
