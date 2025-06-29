"""
Toolflow: Universal tool calling for LLMs

A Python library that provides a unified interface for tool calling across different AI providers.
"""

from .core.tool_execution import (
    set_max_workers,
    set_executor,
)
from .providers.openai import from_openai, from_openai_async
from .providers.anthropic import from_anthropic, from_anthropic_async
from .decorators import tool

__version__ = "0.1.0"

__all__ = [
    # Core tool execution configuration
    "set_max_workers",
    "set_executor", 
    
    # Provider factory functions
    "from_openai",
    "from_openai_async",
    "from_anthropic", 
    "from_anthropic_async",
    
    # Utilities and decorators
    "tool",
]
