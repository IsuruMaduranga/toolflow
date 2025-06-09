"""
toolflow: A Python library that makes LLM tool calling as simple as decorating a function.

Just wrap your OpenAI client and pass decorated functions directly to the tools parameter.
"""

from .decorators import tool
from .open_ai import from_openai, from_openai_async

__version__ = "0.1.0"
__all__ = [
    "tool", 
    "from_openai",  
    "from_openai_async",
]       
