#!/usr/bin/env python3
"""
agentu CLI - Command line interface for agentu.

Usage:
    agentu run "Explain this repo" --stream
    agentu dev [agent.py]
    agentu ralph PROMPT.md --max 50
    agentu serve [agent.py] --port 8000
"""

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional


def _load_agent_from_file(target: Optional[str], default_name: str, model: Optional[str], api_base: Optional[str]):
    """Load an Agent instance from a Python file or construct a smart-routed default Agent."""
    from agentu import Agent, get_driver_for_model, is_auto_routed_model

    agent = None
    if target:
        path = Path(target)
        if not path.exists():
            print(f"❌ Error: Agent file not found: {target}", file=sys.stderr)
            sys.exit(1)
        spec = importlib.util.spec_from_file_location("agentu_user_mod", path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for val in vars(mod).values():
                if isinstance(val, Agent):
                    agent = val
                    break

    if agent is not None:
        if model:
            if is_auto_routed_model(model) and not api_base:
                agent.driver = get_driver_for_model(model=model, temperature=agent.temperature)
                agent.model = agent.driver.model or model
                agent.api_base = (agent.driver.api_base or agent.api_base).rstrip("/")
                agent.api_key = agent.driver.api_key
            else:
                agent.model = model
                if agent.driver is not None:
                    agent.driver.model = model
        if api_base:
            agent.api_base = api_base.rstrip("/")
            if agent.driver is not None:
                agent.driver.api_base = api_base.rstrip("/")
        return agent

    return Agent(name=default_name, model=model, api_base=api_base)


def cmd_run(args):
    """Execute a single prompt from the command line."""
    agent = _load_agent_from_file(args.agent, "cli-agent", args.model, args.api_base)

    async def _execute():
        async with agent:
            if args.stream and not agent.tools:
                async for chunk in agent.stream(args.prompt):
                    print(chunk, end="", flush=True)
                print()
                return

            res = await agent.infer(args.prompt)
            if args.json:
                print(json.dumps(dict(res), indent=2, default=str))
            else:
                print(res.text)

    try:
        asyncio.run(_execute())
    except Exception as exc:
        print(f"❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_ralph(args):
    """Run agent in Ralph mode (autonomous loop)."""
    from agentu import ralph

    prompt_file = args.prompt
    if not Path(prompt_file).exists():
        print(f"❌ Error: Prompt file not found: {prompt_file}", file=sys.stderr)
        sys.exit(1)

    agent = _load_agent_from_file(args.agent, "ralph-cli", args.model, args.api_base)

    def on_progress(iteration, data):
        result_preview = str(data.get("result", ""))[:60]
        print(f"🔄 [{iteration}/{args.max}] {result_preview}...")

    print(f"🚀 Starting Ralph mode with {prompt_file} [model: {agent.model}]")
    print(f"   Max iterations: {args.max}")
    print(f"   Timeout: {args.timeout} minutes\n")

    async def _run_ralph():
        async with agent:
            return await ralph(
                agent,
                prompt_file,
                max_iterations=args.max,
                timeout_minutes=args.timeout,
                on_iteration=on_progress,
            )

    result = asyncio.run(_run_ralph())

    print("\n📊 Summary:")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Stopped by: {result['stopped_by']}")
    print(f"   Errors: {len(result['errors'])}")

    if result["errors"]:
        print("\n❌ Errors:")
        for err in result["errors"]:
            print(f"   - {err}")


def cmd_serve(args):
    """Start the agentu server with a user or demo agent."""
    from agentu import Tool, serve

    agent = _load_agent_from_file(args.agent, "agentu-server", args.model, args.api_base)
    if not agent.tools and not args.agent:
        def echo(message: str) -> str:
            """Echo back a message."""
            return f"Echo: {message}"

        agent.with_tools([Tool(echo)])

    print(f"🚀 Starting agentu server on port {args.port} [agent: {agent.name}, model: {agent.model}]")
    print(f"📖 API Docs: http://localhost:{args.port}/docs")
    print(f"📊 Dashboard: http://localhost:{args.port}/dashboard")

    serve(agent, port=args.port)


async def _dev_repl(agent, stream_default: bool = False) -> None:
    """Single-event-loop interactive REPL preserving HTTP & MCP sessions across turns."""
    from agentu import get_driver_for_model, is_auto_routed_model

    stream_enabled = stream_default
    print(f"🤖 agentu dev console [agent: {agent.name}, model: {agent.model}]")
    print("   Commands: /tools, /model [name], /stream, /history, /memory, /clear, exit\n")

    async with agent:
        while True:
            try:
                prompt = (await asyncio.to_thread(input, "👤 > ")).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                break
            if prompt == "/tools":
                active = [f"{t.name} ({t.permission.value})" for t in agent.tools]
                deferred = [t.name for t in agent.deferred_tools]
                print(f"🛠️  Active tools ({len(active)}): {active}")
                if deferred:
                    print(f"📦 Deferred tools ({len(deferred)}): {deferred}")
                print()
                continue
            if prompt.startswith("/model"):
                parts = prompt.split(maxsplit=1)
                if len(parts) == 1:
                    driver_name = type(agent.driver).__name__ if agent.driver else "OpenAICompatible"
                    print(f"🧠 Current model: {agent.model} (driver: {driver_name}, base: {agent.api_base})\n")
                else:
                    new_model = parts[1].strip()
                    if agent.driver is not None:
                        await agent.driver.close()
                    if is_auto_routed_model(new_model):
                        agent.driver = get_driver_for_model(model=new_model, temperature=agent.temperature)
                        agent.model = agent.driver.model or new_model
                        agent.api_base = (agent.driver.api_base or agent.api_base).rstrip("/")
                        agent.api_key = agent.driver.api_key
                    else:
                        agent.driver = None
                        agent.model = new_model
                    print(f"✅ Switched model to: {agent.model}\n")
                continue
            if prompt == "/stream":
                stream_enabled = not stream_enabled
                print(f"⚡ Streaming {'enabled' if stream_enabled else 'disabled'}.\n")
                continue
            if prompt == "/history":
                print(f"📜 Turns recorded: {len(agent.conversation_history)}")
                for idx, entry in enumerate(agent.conversation_history[-5:], 1):
                    print(f"   {idx}. {entry.get('user_input', '')[:60]} ({entry.get('turns', 1)} turn(s))")
                print()
                continue
            if prompt == "/memory":
                if agent.memory_enabled and agent.memory:
                    recent = agent.memory.short_term.get_recent(5)
                    print(f"🧬 Recent memories ({len(recent)}):")
                    for m in recent:
                        print(f"   - [{m.memory_type}] {m.content[:72]}")
                else:
                    print("🧬 Memory is disabled on this agent.")
                print()
                continue
            if prompt == "/clear":
                agent.conversation_history.clear()
                print("🧹 Conversation history cleared.\n")
                continue

            try:
                if stream_enabled and not agent.tools:
                    print("\n🤖 ", end="", flush=True)
                    async for chunk in agent.stream(prompt):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    res = await agent.infer(prompt)
                    if res.tool_calls:
                        used = ", ".join(tc.get("tool_used", "?") for tc in res.tool_calls)
                        print(f"   🔧 tools: {used}")
                    out = res.error if res.error else res.text
                    print(f"\n🤖 {out}\n")
            except Exception as exc:
                print(f"\n❌ Error: {exc}\n")


def cmd_dev(args):
    """Start an interactive agent debugging and testing REPL."""
    agent = _load_agent_from_file(args.agent, "dev-agent", args.model, args.api_base)
    asyncio.run(_dev_repl(agent, stream_default=args.stream))


def cmd_version(args):
    """Print version info."""
    from agentu import __version__
    print(f"agentu v{__version__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentu",
        description="A harness-engineered AI agent runtime",
    )
    parser.add_argument("--version", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a one-shot prompt from the terminal")
    run_parser.add_argument("prompt", help="Natural language prompt to execute")
    run_parser.add_argument("--agent", default=None, help="Optional Python file defining an Agent")
    run_parser.add_argument("--model", default=None, help="Model identifier (auto-detected if omitted)")
    run_parser.add_argument("--api-base", default=None, help="Custom API base URL")
    run_parser.add_argument("--stream", action="store_true", help="Stream tokens as they arrive")
    run_parser.add_argument("--json", action="store_true", help="Print full InferResult as JSON")
    run_parser.set_defaults(func=cmd_run)

    # dev command
    dev_parser = subparsers.add_parser("dev", help="Start interactive development console")
    dev_parser.add_argument("agent", nargs="?", default=None, help="Optional Python file defining an Agent")
    dev_parser.add_argument("--model", default=None, help="Model identifier (auto-detected if omitted)")
    dev_parser.add_argument("--api-base", default=None, help="Custom API base URL")
    dev_parser.add_argument("--stream", action="store_true", help="Enable streaming responses by default")
    dev_parser.set_defaults(func=cmd_dev)

    # ralph command
    ralph_parser = subparsers.add_parser("ralph", help="Run autonomous loop mode")
    ralph_parser.add_argument("prompt", help="Path to PROMPT.md file")
    ralph_parser.add_argument("--agent", default=None, help="Optional Python file defining an Agent")
    ralph_parser.add_argument("--max", type=int, default=50, help="Max iterations (default: 50)")
    ralph_parser.add_argument("--timeout", type=int, default=30, help="Timeout in minutes (default: 30)")
    ralph_parser.add_argument("--model", default=None, help="Model identifier (auto-detected if omitted)")
    ralph_parser.add_argument("--api-base", default=None, help="Custom API base URL")
    ralph_parser.set_defaults(func=cmd_ralph)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("agent", nargs="?", default=None, help="Optional Python file defining an Agent")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--model", default=None, help="Model identifier (auto-detected if omitted)")
    serve_parser.add_argument("--api-base", default=None, help="Custom API base URL")
    serve_parser.set_defaults(func=cmd_serve)

    # version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        cmd_version(args)
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
