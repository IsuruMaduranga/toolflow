"""
Toolflow: Universal tool calling for LLMs

A Python library that provides a unified interface for tool calling across different AI providers.
"""

from .core.tool_execution import (
    set_max_workers,
    get_max_workers,
    set_executor,
)
from .core.execution_loops import (
    set_async_yield_frequency,
)
from .providers.openai import from_openai
from .providers.anthropic import from_anthropic
from .decorators import tool

__version__ = "0.1.0"

__all__ = [
    # Core tool execution configuration
    "set_max_workers", 
    "get_max_workers",
    "set_executor",
    
    # Advanced streaming configuration
    "set_async_yield_frequency",
    
    # Provider factory functions
    "from_openai",
    "from_anthropic",
    
    # Utilities and decorators
    "tool",
]
