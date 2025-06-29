"""
Toolflow: Universal tool calling for LLMs

A Python library that provides a unified interface for tool calling across different AI providers.
"""

from .decorators import tool
from .providers.openai import from_openai, from_openai_async
from .providers.anthropic import from_anthropic, from_anthropic_async
from .core.tool_execution import (
    # New sync/async specific methods
    set_max_workers_sync, set_global_executor_sync, cleanup_sync_executor, get_sync_executor,
    set_max_workers_async, set_global_executor_async, cleanup_async_executor, get_async_executor,
    cleanup_executors, get_default_sync_max_workers
)

__version__ = "0.1.0"

__all__ = [
    "tool",
    "from_openai", 
    "from_openai_async",
    "from_anthropic",
    "from_anthropic_async",
    # New sync/async specific methods
    "set_max_workers_sync",
    "set_global_executor_sync", 
    "cleanup_sync_executor",
    "get_sync_executor",
    "set_max_workers_async",
    "set_global_executor_async",
    "cleanup_async_executor",
    "get_async_executor",
    "cleanup_executors",
    "get_default_sync_max_workers",
]
