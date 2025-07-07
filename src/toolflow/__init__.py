"""
Toolflow: Universal tool calling for LLMs

A Python library that provides a unified interface for tool calling across different AI providers.
"""

from toolflow.core import (
    set_max_workers,
    get_max_workers,
    set_executor,
    MaxToolCallsError,
    MissingAnnotationError,
    UndescribableTypeError,
    ResponseFormatError,
    MaxTokensError,
)
from toolflow.core import set_async_yield_frequency
from toolflow.providers import from_openai, from_anthropic, from_gemini
from toolflow.core import tool
from . import errors

__version__ = "0.3.2"

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
    "from_gemini",
    
    # Utilities and decorators
    "tool",
    
    # Exceptions
    "errors",
]
