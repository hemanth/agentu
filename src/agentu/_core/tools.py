"""Tool utilities for agentu."""

import inspect
from typing import Callable, Dict, Any, Optional


class Tool:
    """A tool that can be used by an agent.

    Auto-infers name, description, and parameters from the function.

    Examples:
        >>> def add(x: int, y: int) -> int:
        ...     '''Add two numbers.'''
        ...     return x + y
        >>>
        >>> tool = Tool(add)  # Everything auto-inferred!
        >>>
        >>> # Or customize as needed:
        >>> tool = Tool(add, "Custom description")
        >>> tool = Tool(add, "Description", "custom_name")
    """

    def __init__(
        self,
        function: Callable,
        description: Optional[str] = None,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Initialize a Tool.

        Args:
            function: The function to wrap
            description: Optional description (uses docstring if not provided)
            name: Optional name (uses function name if not provided)
            parameters: Optional parameters dict (auto-inferred if not provided)
        """
        self.function = function
        self.name = name or function.__name__
        self.description = description or self._extract_description(function)
        self.parameters = parameters or self._extract_parameters(function)

    @staticmethod
    def _extract_description(function: Callable) -> str:
        """Extract description from function docstring."""
        doc = function.__doc__
        if not doc:
            return f"Execute {function.__name__}"

        # Get first line of docstring
        lines = doc.strip().split('\n')
        return lines[0].strip()

    @staticmethod
    def _extract_parameters(function: Callable) -> Dict[str, str]:
        """Extract parameters from function signature using type hints."""
        sig = inspect.signature(function)
        params = {}

        for param_name, param in sig.parameters.items():
            # Get type annotation
            if param.annotation != inspect.Parameter.empty:
                type_name = getattr(param.annotation, '__name__', str(param.annotation))
            else:
                type_name = "any"

            # Get default value info
            if param.default != inspect.Parameter.empty:
                params[param_name] = f"{type_name}: (default: {param.default})"
            else:
                params[param_name] = f"{type_name}"

        return params
