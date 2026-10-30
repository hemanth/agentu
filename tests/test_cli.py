"""Tests for the packaged command-line entry point."""

from importlib.metadata import version
from pathlib import Path

import agentu
from agentu.runtime.cli import main


def test_console_script_targets_runtime_cli():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"

    assert 'agentu = "agentu.runtime.cli:main"' in pyproject.read_text(
        encoding="utf-8"
    )
    assert callable(main)


def test_runtime_version_comes_from_distribution_metadata():
    assert agentu.__version__ == version("agentu")


def test_cli_run_and_dev_preserve_event_loop(tmp_path, monkeypatch, capsys):
    from unittest.mock import AsyncMock
    from agentu.runtime import cli

    agent_file = tmp_path / "custom_agent.py"
    agent_file.write_text(
        "from agentu import Agent\n"
        "my_agent = Agent('custom', model='gemini-2.5-flash', enable_memory=False, auto_discover_rules=False)\n",
        encoding="utf-8",
    )

    loaded = cli._load_agent_from_file(str(agent_file), "fallback", model=None, api_base=None)
    assert loaded.name == "custom"
    # Preserves configured model instead of clobbering with qwen3:latest!
    assert loaded.model == "gemini-2.5-flash"

    # Test `agentu run`
    parser = cli.build_parser()
    args = parser.parse_args(["run", "Hello CLI", "--agent", str(agent_file)])
    monkeypatch.setattr(
        "agentu._core.agent.Agent._call_llm",
        AsyncMock(return_value="CLI reply ok"),
    )
    cli.cmd_run(args)
    out = capsys.readouterr().out
    assert "CLI reply ok" in out

    # Test `agentu dev` single-event-loop REPL across multiple commands
    inputs = iter(["/tools", "/model", "First turn", "Second turn", "/history", "/clear", "exit"])
    loop_ids = []

    async def fake_infer(self, prompt, **kwargs):
        import asyncio
        from agentu import InferResult
        loop_ids.append(id(asyncio.get_running_loop()))
        res = InferResult(text=f"echo:{prompt}", turns=1)
        self.conversation_history.append({"user_input": prompt, "response": res, "turns": 1})
        return res

    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr("agentu._core.agent.Agent.infer", fake_infer)

    dev_args = parser.parse_args(["dev", str(agent_file)])
    cli.cmd_dev(dev_args)
    dev_out = capsys.readouterr().out
    assert "echo:First turn" in dev_out
    assert "echo:Second turn" in dev_out
    # Both turns executed inside the EXACT same running asyncio event loop!
    assert len(loop_ids) == 2
    assert loop_ids[0] == loop_ids[1]
