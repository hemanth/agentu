"""Shared test configuration and fixtures."""

import socket
import pytest


def _ollama_available(host="localhost", port=11434, timeout=0.5) -> bool:
    """Check if Ollama is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running on localhost:11434"
)
