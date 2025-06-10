"""
Decorators for tool registration.
"""

from typing import Callable, Optional, TypeVar, Union
from functools import wraps
from .utils import get_tool_schema

F = TypeVar('F', bound=Callable)

def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    internal: bool = False
) -> Union[Callable[[F], F], F]:
    """
    Decorator to mark a function as a tool for LLM calling.
    
    Can be used as @tool or @tool(name="...", description="...")
    
    Args:
        func: The function being decorated (when used as @tool)
        name: Optional custom name for the tool (defaults to function name)
        description: Optional description (defaults to function docstring)
        internal: Whether this is an internal tool (defaults to False)
    
    Example:
        @tool
        def get_weather(city: str) -> str:
            \"\"\"Get the current weather for a city.\"\"\"
            return f"Weather in {city}: Sunny, 72°F"
        
        @tool(name="calculator", description="Add two numbers together")
        def add(a: int, b: int) -> int:
            return a + b
    """
    
    def decorator(func: F) -> F:
        import asyncio
        
        # Add metadata to the function for direct usage
        func._tool_metadata = get_tool_schema(func, name, description)
        func._tool_metadata_strict = get_tool_schema(func, name, description, strict=True)
        if not internal and func._tool_metadata['function']['name'] == "final_response_tool_internal":
            raise ValueError("final_response_tool_internal is an internally used tool by toolflow and cannot be used as a custom tool name")
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Copy metadata to wrapper
            async_wrapper._tool_metadata = func._tool_metadata
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Copy metadata to wrapper
            sync_wrapper._tool_metadata = func._tool_metadata
            return sync_wrapper
    
    # If used as @tool (without parentheses) AND no keyword arguments were provided
    if func is not None:
        return decorator(func)
    
    # If used as @tool(...) (with parentheses) OR with keyword arguments
    return decorator
