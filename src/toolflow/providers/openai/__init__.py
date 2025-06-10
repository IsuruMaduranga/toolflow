"""
OpenAI provider for toolflow.

This module provides factory functions to create toolflow wrappers around OpenAI clients.
"""
from .wrappers import OpenAIWrapper, OpenAIAsyncWrapper

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def from_openai(client) -> "OpenAIWrapper":
    """
    Create a toolflow wrapper around an existing OpenAI client.
    
    Args:
        client: An existing OpenAI client instance
    
    Returns:
        OpenAIWrapper that supports tool-py decorated functions
    
    Example:
        import openai
        import toolflow
        
        client = toolflow.from_openai(openai.OpenAI())
        
        @toolflow.tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        response = client.chat.completions.create(
            model="gpt-4",
            tools=[get_weather],
            messages=[{"role": "user", "content": "What's the weather in NYC?"}]
        )
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    return OpenAIWrapper(client)


def from_openai_async(client) -> "OpenAIAsyncWrapper":
    """
    Create a toolflow wrapper around an existing OpenAI async client.
    
    Args:
        client: An existing OpenAI async client instance
    
    Returns:
        OpenAIAsyncWrapper that supports tool-py decorated functions
    
    Example:
        import openai
        import toolflow
        
        client = toolflow.from_openai_async(openai.AsyncOpenAI())
        
        @toolflow.tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        response = await client.chat.completions.create(
            model="gpt-4",
            tools=[get_weather],
            messages=[{"role": "user", "content": "What's the weather in NYC?"}]
        )
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    return OpenAIAsyncWrapper(client)
