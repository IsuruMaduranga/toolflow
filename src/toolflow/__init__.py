"""
Toolflow: Universal tool calling for LLMs

A Python library that provides a unified interface for tool calling across different AI providers.
"""

from .decorators import tool
from .providers.openai import from_openai, from_openai_async
from .providers.anthropic import from_anthropic, from_anthropic_async

__version__ = "0.1.0"

__all__ = [
    "tool",
    "from_openai", 
    "from_openai_async",
    "from_anthropic",
    "from_anthropic_async"
]
