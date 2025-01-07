from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class Tool:
    """Class representing a tool that can be used by an agent."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
