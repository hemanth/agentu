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
