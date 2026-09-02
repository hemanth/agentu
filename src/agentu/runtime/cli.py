#!/usr/bin/env python3
"""
agentu CLI - Command line interface for agentu.

Usage:
    agentu ralph PROMPT.md --max 50
    agentu serve --port 8000
"""

import argparse
import asyncio
import sys
from pathlib import Path


def cmd_ralph(args):
    """Run agent in Ralph mode (autonomous loop)."""
    from agentu import Agent, ralph
    
    prompt_file = args.prompt
    if not Path(prompt_file).exists():
        print(f"❌ Error: Prompt file not found: {prompt_file}")
        sys.exit(1)
    
    # Create a basic agent (user can customize via config later)
    agent = Agent(
        name="ralph-cli",
        model=args.model,
        api_base=args.api_base
    )
    
    def on_progress(iteration, data):
        result_preview = str(data.get('result', ''))[:60]
        print(f"🔄 [{iteration}/{args.max}] {result_preview}...")
    
    print(f"🚀 Starting Ralph mode with {prompt_file}")
    print(f"   Max iterations: {args.max}")
    print(f"   Timeout: {args.timeout} minutes")
    print()
    
    result = asyncio.run(ralph(
        agent,
        prompt_file,
        max_iterations=args.max,
        timeout_minutes=args.timeout,
        on_iteration=on_progress
    ))
    
    print()
    print("📊 Summary:")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Stopped by: {result['stopped_by']}")
    print(f"   Errors: {len(result['errors'])}")
    
    if result['errors']:
        print("\n❌ Errors:")
        for err in result['errors']:
            print(f"   - {err}")


def cmd_serve(args):
    """Start the agentu server with a demo agent."""
    from agentu import Agent, serve, Tool
    
    def echo(message: str) -> str:
        """Echo back a message."""
        return f"Echo: {message}"
    
    agent = Agent(
        name="agentu-server",
        model=args.model,
        api_base=args.api_base
    ).with_tools([Tool(echo)])
    
    print(f"🚀 Starting agentu server on port {args.port}")
    print(f"📖 API Docs: http://localhost:{args.port}/docs")
    print(f"📊 Dashboard: http://localhost:{args.port}/dashboard")
    
    serve(agent, port=args.port)


def cmd_dev(args):
    """Start an interactive agent debugging and testing REPL."""
    import importlib.util
    from agentu import Agent

    agent = None
    target = args.agent
    if target and Path(target).exists():
        spec = importlib.util.spec_from_file_location("dev_agent_mod", target)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for val in vars(mod).values():
                if isinstance(val, Agent):
                    agent = val
                    break

    if not agent:
        agent = Agent(name="dev-agent", model=args.model, api_base=args.api_base)

    print(f"🤖 agentu dev console [agent: {agent.name}, model: {agent.model}]")
    print("   Commands: /tools (list tools), /clear (clear history), 'exit' to quit\n")

    while True:
        try:
            prompt = input("👤 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            break
        if prompt == "/tools":
            print(f"🛠️  Available tools ({len(agent.tools)}): {[t.name for t in agent.tools]}")
            continue
        if prompt == "/clear":
            agent.conversation_history.clear()
            print("🧹 Conversation history cleared.\n")
            continue

        try:
            res = asyncio.run(agent.infer(prompt))
            text = res.get("text_response") if isinstance(res, dict) or hasattr(res, "get") else str(res)
            print(f"\n🤖 {text}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def cmd_version(args):
    """Print version info."""
    from agentu import __version__
    print(f"agentu v{__version__}")


def main():
    parser = argparse.ArgumentParser(
        prog="agentu",
        description="The sleekest way to build AI agents"
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # dev command
    dev_parser = subparsers.add_parser("dev", help="Start interactive development console")
    dev_parser.add_argument("agent", nargs="?", default=None, help="Optional path to python file defining an Agent")
    dev_parser.add_argument("--model", default="qwen3:latest", help="Model to use")
    dev_parser.add_argument("--api-base", default="http://localhost:11434/v1", help="API base URL")
    dev_parser.set_defaults(func=cmd_dev)

    # ralph command
    ralph_parser = subparsers.add_parser("ralph", help="Run autonomous loop mode")
    ralph_parser.add_argument("prompt", help="Path to PROMPT.md file")
    ralph_parser.add_argument("--max", type=int, default=50, help="Max iterations (default: 50)")
    ralph_parser.add_argument("--timeout", type=int, default=30, help="Timeout in minutes (default: 30)")
    ralph_parser.add_argument("--model", default="qwen3:latest", help="Model to use")
    ralph_parser.add_argument("--api-base", default="http://localhost:11434/v1", help="API base URL")
    ralph_parser.set_defaults(func=cmd_ralph)
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--model", default="qwen3:latest", help="Model to use")
    serve_parser.add_argument("--api-base", default="http://localhost:11434/v1", help="API base URL")
    serve_parser.set_defaults(func=cmd_serve)
    
    # version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)
    
    args = parser.parse_args()
    
    if args.version:
        cmd_version(args)
    elif hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
