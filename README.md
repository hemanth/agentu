# AgentU

**The sleekest way to build AI agents.**

```bash
pip install agentu
```

## Why AgentU?

```python
# This is all you need:
from agentu import Agent

def search_products(query: str) -> list:
    return db.products.search(query)

agent = Agent("sales").with_tools([search_products])

# Direct execution
result = await agent.call("search_products", {"query": "laptop"})

# Natural language (LLM figures out the tool + params)
result = await agent.infer("Find me laptops under $1500")
```

## Workflows in 3 Lines

```python
# Sequential: researcher → analyst → writer
workflow = researcher("Find AI trends") >> analyst("Analyze") >> writer("Summarize")

# Parallel: run 3 searches concurrently
workflow = search("AI") & search("ML") & search("Crypto")

# Combined: parallel then merge
workflow = (search("AI") & search("ML") & search("Crypto")) >> analyst("Compare")

result = await workflow.run()
```

**`>>` chains steps. `&` runs in parallel.** That's the entire API.

## Memory

```python
agent.remember("Customer prefers email", importance=0.9)
memories = agent.recall(query="email")
```

Stored in SQLite. Searchable. Persistent.

## REST API

```python
from agentu import serve

serve(agent, port=8000)
# curl -X POST localhost:8000/execute -d '{"tool_name": "search_products", ...}'
```

Auto-generated Swagger docs at `/docs`.

## Real-World Example: Automated Code Review

```python
import asyncio
from agentu import Agent

def get_pr_diff(pr_number: int) -> str:
    """Fetch PR changes from GitHub."""
    # GitHub API integration
    return "diff --git a/src/auth.py... +added_line -removed_line"

def run_tests(branch: str) -> dict:
    """Run test suite."""
    return {"passed": 47, "failed": 2, "coverage": 94.2}

def post_comment(pr_number: int, comment: str) -> bool:
    """Post review comment to GitHub."""
    return True

async def main():
    # Setup agents
    reviewer = Agent("reviewer", model="gpt-4").with_tools([get_pr_diff])
    tester = Agent("tester").with_tools([run_tests])
    commenter = Agent("commenter").with_tools([post_comment])

    # Parallel: review code + run tests
    workflow = reviewer("Review PR #247") & tester("Run tests on PR #247")
    code_review, test_results = await workflow.run()

    # Natural language: synthesize findings
    summary = await commenter.infer(
        f"Create a review comment for PR #247. "
        f"Code review: {code_review}. Tests: {test_results}. "
        f"Be constructive and specific."
    )

    # Post to GitHub
    await commenter.call("post_comment", {"pr_number": 247, "comment": summary})
    print("✓ Review posted to PR #247")

asyncio.run(main())
```

**What this does:**
- Reviews code and runs tests **in parallel** (saves time)
- Uses `infer()` to write **human-quality review comments**
- Posts directly to GitHub
- **Zero manual work** - runs on every PR

## Advanced: Lambda Control

Need precise data flow? Use lambdas:

```python
workflow = (
    researcher("Find companies")
    >> analyst(lambda prev: f"Extract top 5 from: {prev['result']}")
    >> writer(lambda prev: f"Write report about: {prev['companies']}")
)
```

## Skills: 96% Less Context

**NEW in v1.2.1**: Domain expertise that loads on-demand.

```python
from agentu import Agent, Skill

pdf_skill = Skill(
    name="pdf-processing",
    description="Extract text and tables from PDF files",
    instructions="skills/pdf/SKILL.md",
    resources={"forms": "skills/pdf/FORMS.md"}
)

agent = Agent("assistant").with_skills([pdf_skill])

# Skills auto-activate on matching prompts
await agent.infer("Extract tables from report.pdf")
```

**Progressive loading:** Metadata (100 chars) → Instructions (1500 chars) → Resources (on-demand)

**Result:** 100+ skills, minimal context footprint.

## LLM Support

Works with any OpenAI-compatible API. **Auto-detects available models** from Ollama:

```python
# Auto-detect (uses first available Ollama model)
Agent("assistant")

# Explicit model
Agent("assistant", model="qwen3")

# OpenAI
Agent("assistant", model="gpt-4", api_key="sk-...")

# vLLM, LM Studio, etc.
Agent("assistant", model="mistral", api_base="http://localhost:8000/v1")
```

## Tool Search

Scale to hundreds of tools without context bloat. Deferred tools are discovered on-demand:

```python
def charge_card(amount: float, card_id: str) -> dict:
    """Charge a credit card."""
    return {"status": "success", "amount": amount}

def send_receipt(email: str, transaction_id: str) -> bool:
    """Send receipt via email."""
    return True

def refund_payment(transaction_id: str) -> dict:
    """Refund a payment transaction."""
    return {"refunded": True}

# 3 payment tools deferred, discovered when needed
agent = Agent("payments").with_tools(defer=[charge_card, send_receipt, refund_payment])

result = await agent.infer("charge $50 to card_123")
# Agent calls search_tools("charge card") → activates charge_card → executes it
```

When `defer` is used, a `search_tools` function is auto-added. The agent searches for relevant tools, activates them, then calls them. Multi-turn happens internally.

## MCP Integration

Connect to Model Context Protocol servers:

```python
agent.with_mcp(["http://localhost:3000"])
agent.with_mcp([{"url": "https://api.com/mcp", "headers": {"Auth": "token"}}])
```

## API Reference

### Agent
```python
agent = Agent(name)                      # Auto-detects model from Ollama
agent = Agent(name, model="qwen3")       # Or specify explicitly
agent = Agent(name, max_turns=5)         # Limit multi-turn inference
agent.with_tools([func1, func2])         # Add active tools
agent.with_tools(defer=[many_funcs])     # Add searchable tools
agent.with_tools([core], defer=[many])   # Both in one call
agent.with_mcp([url])                    # Connect MCP servers

await agent.call("tool", params)         # Direct execution
await agent.infer("natural language")    # LLM routing (multi-turn)

agent.remember(content, importance=0.8)  # Store
agent.recall(query)                      # Search
```

### Workflows
```python
agent("task")                            # Create step
step1 >> step2                           # Sequential
step1 & step2                            # Parallel
await workflow.run()                     # Execute
```

### serve()
```python
serve(agent, port=8000, enable_cors=True)
```

**Endpoints:** `/execute`, `/process`, `/tools`, `/memory/remember`, `/memory/recall`, `/docs`

## Examples

```bash
git clone https://github.com/hemanth/agentu
cd agentu

python examples/basic.py       # Simple agent
python examples/workflow.py    # Workflows
python examples/memory.py      # Memory system
python examples/api.py         # REST API
```

## Testing

```bash
pytest
pytest --cov=agentu
```

## License

MIT
