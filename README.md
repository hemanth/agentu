# agentu

Run AI agents in Python with zero dependencies, subprocess tool isolation, AST-compiled contracts, and self-correction.

```bash
pip install agentu
```

## Quick start

```python
from agentu import Agent

def search_products(query: str, max_price: int = 1500) -> list[dict]:
    """Search catalog for matching products."""
    return [
        {"name": "ThinkPad X1 Carbon", "price": 1399},
        {"name": "MacBook Air M4", "price": 1199},
    ]

agent = Agent("sales", model="gemini-2.5-flash").with_tools([search_products])

# Call a tool directly
raw = await agent.call("search_products", {"query": "laptop"})

# Or let the LLM run the multi-turn loop
res = await agent.infer("Find me laptops under $1500")
print(res.text)
```

`call()` invokes a tool directly. `infer()` runs the async multi-turn agentic loop and returns an `InferResult`.

## Zero-config providers

```python
# Picks up GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY
gemini = Agent("analyst", model="gemini-2.5-flash")
claude = Agent("reviewer", model="claude-sonnet-4-20250514")
openai = Agent("coder", model="openai/gpt-4o")
groq   = Agent("fast", model="groq/llama-3.3-70b-versatile")
local  = Agent("offline", model="ollama/qwen3:latest")
```

All four protocol drivers (`GeminiDriver`, `ClaudeDriver`, `OpenAIDriver`, `OllamaDriver`) speak native REST/SSE over Python's stdlib (`asyncio` + `ssl`) out of the box and auto-upgrade to pooled `aiohttp` when installed — zero vendor SDK bloat.

## Sandboxed tool execution

```python
agent = Agent("ops", model="gpt-4o").with_sandbox(
    read_tools=[search_products, read_logs],
    write_tools=[save_report, send_email],
    timeout=10,
    max_memory_mb=256,
    allow_network=False,
)
```

- `read_tools` run as `READONLY`; `write_tools` run as `WRITE`; `DANGEROUS` tools stay blocked unless enabled via `.with_permissions(allow_dangerous=True)`
- Every sandboxed tool runs in an isolated subprocess — hung tools are killed at `timeout` without crashing your process
- Defer large tool catalogs with `.with_tools([core_tool], defer=[...100_extra_tools])` — `search_tools` ranks them via Fielded Okapi BM25F + character 3-gram cosine (`96.7%` Recall@1 / `100%` Recall@3 in `107 µs` across 50 tools vs `53.3%` naive word overlap)

## Code Mode

```python
agent = Agent("bot", model="claude-sonnet-4-20250514", codemode=True).with_tools([
    search_products, check_inventory, save_report
])
res = await agent.infer("Find in-stock laptops and save a summary")
```

Instead of N sequential JSON tool round-trips, the LLM writes typed Python against `tools.*` in a restricted sandbox (`math`, `json`, `re` allowed; `os`, `sys`, `subprocess`, `socket` blocked) and self-corrects on runtime errors.

## Structured outputs + tools

```python
from agentu import Agent, InferResult

class DealSummary:
    top_pick: str
    price: int
    rationale: str

    def check(self):
        assert 0 < self.price <= 1500
        assert len(self.rationale) >= 12

res: InferResult[DealSummary] = await agent.infer(
    "Pick the best laptop deal",
    output_type=DealSummary,
)
print(res.data.top_pick, res.data.price)

# Or pass an inline dict shape with zero classes
quick = await agent.infer("Best budget pick", output_type={"pick": str, "price": int})
print(quick.data.pick)
```

Plain classes need zero imports: `agentu` lifts `check()` assertions via `ast` into JSON Schema (`minimum`, `maximum`, `minLength`, `enum`, `pattern`) before the first LLM call, heals malformed/truncated LLM JSON in a single O(n) pushdown pass (`30.5 µs` p50), and reports all violated assertions with live field values on retry. `@dataclass`, `TypedDict`, and Pydantic `BaseModel` work via duck-typing too.

## Guardrails & hooks

```python
from agentu import Agent, PII, ContentFilter, HookAction, HookResult

async def block_prod_deletes(tool_name, params, ctx):
    if tool_name == "drop_table":
        return HookResult(action=HookAction.DENY, reason="Production drops blocked")
    return HookResult(action=HookAction.ALLOW)

agent = (
    Agent("support", model="gpt-4o")
    .with_guardrails(output_guardrails=[PII(), ContentFilter()], max_corrections=2)
    .with_hooks(pre_tool=block_prod_deletes)
)
```

Guardrail violations feed back into the model for automatic self-correction up to `max_corrections` attempts. Hooks intercept `pre_tool`, `post_tool`, and `on_stop` with `ALLOW`, `DENY`, or `MODIFY`.

## Multimodal & image generation

```python
res = await agent.infer(
    "Cross-check the keynote demo against the architecture PDF:",
    media=[
        {"type": "video", "url": "https://youtu.be/7Z5Vy9JBANs", "processing": "agentic"},
        "./latency_chart.png",
        "./whitepaper.pdf",
    ],
)

image = await agent.generate_image(
    "Minimalist isometric server rack on warm paper",
    model="gpt-image-2",
    output_path="renders/rack.png",
)
```

When targeting text-only models (e.g. `deepseek-chat` or local Ollama), install `pip install "agentu[media]"` and `agentu` automatically degrades videos and documents into Markdown transcripts.

## Skills, rules & MCP

```python
agent = await (
    Agent("engineer", model="claude-sonnet-4-20250514", auto_discover_rules=True)
    .with_skills(["./skills/security-audit", "github:hemanth/skills/python-perf"])
    .with_mcp(["http://localhost:3000", {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}])
)
```

Auto-discovers `AGENTS.md` / `CLAUDE.md` from your repo root, loads `SKILL.md` progressive instructions only when relevant, and connects to HTTP, Streamable HTTP, or STDIO MCP servers.

## Memory, context & caching

```python
agent = (
    Agent("researcher", model="gemini-2.5-flash", cache=True)
    .with_context(max_tokens=100_000, compaction="auto", keep_recent=5)
)
agent.remember("Customer prefers EU regions and PostgreSQL 17", importance=0.9)
```

Short-term working memory + SQLite long-term recall come built in. Context compaction triggers at 80% token budget (truncate old tool outputs → summarize → keep recent turns).

## Workflows, sub-agents & Ralph

```python
from agentu import Agent, ralph, run_maker_checker

pipeline = (researcher("Gather benchmarks") & analyst("Check pricing")) >> writer("Draft memo")
out = await pipeline.run()

reviewed = await run_maker_checker(maker=coder, checker=reviewer, task="Write rate limiter", max_rounds=3)
loop_stats = await ralph(coder.with_worktree(), "PROMPT.md", max_iterations=25, timeout_minutes=20)
```

Compose steps with `>>` (sequential) and `&` (parallel), delegate across agents with `.with_agents([...])`, isolate git edits with `.with_worktree()`, or run autonomous `ralph` loops with checkpoint recovery.

## Observability, evals & CLI

```bash
# One-shot prompt from terminal
agentu run "Summarize recent git commits" --model gemini-2.5-flash

# Interactive dev REPL (preserves live HTTP & MCP sessions across turns)
agentu dev my_agent.py

# Autonomous Ralph loop or FastAPI + SSE + Dashboard server (pip install "agentu[serve]")
agentu ralph PROMPT.md --max 30
agentu serve my_agent.py --port 8000
```

Attach OpenTelemetry GenAI spans via `.with_otel()` (`agentu[otel]`), score datasets and tool trajectories with `await evaluate(agent, cases)`, or serve HTTP/SSE/WebSocket endpoints with `serve(agent)`.

## Extras

- `pip install "agentu[http]"` — Progressive `aiohttp` connection-pool acceleration
- `pip install "agentu[serve]"` — FastAPI HTTP/SSE/WebSocket server & live dashboard
- `pip install "agentu[media]"` — Automatic video/PDF-to-Markdown degradation
- `pip install "agentu[search]"` — Built-in `SearchAgent` & DuckDuckGo `search_tool`
- `pip install "agentu[semantic]"` / `"agentu[redis]"` / `"agentu[vectors]"` / `"agentu[otel]"` / `"agentu[all]"`

## Related

- [Interactive Codelab](https://hemanth.github.io/agentu-codelab/) — interactive Python playground in your browser

## License

MIT © [Hemanth.HM](https://h3manth.com)
