"""
Toolflow: Universal tool calling for LLMs

A Python library that provides a unified interface for tool calling across different AI providers.
"""

from .decorators import tool
from .providers.openai import from_openai, from_openai_async

# Optional imports for other providers
try:
    from .providers.anthropic import from_anthropic, from_anthropic_async
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    # Anthropic provider not available, which is fine
    _ANTHROPIC_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    "tool",
    "from_openai", 
    "from_openai_async",
]

# Add Anthropic functions to __all__ if available
if _ANTHROPIC_AVAILABLE:
    __all__.extend(["from_anthropic", "from_anthropic_async"])       
