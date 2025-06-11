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


def from_openai(client, full_response: bool = False) -> "OpenAIWrapper":
    """
    Create a toolflow wrapper around an existing OpenAI client.
    
    Args:
        client: An existing OpenAI client instance
        full_response: If True, return the full OpenAI response object. 
                      If False (default), return only the content or parsed data.
    
    Returns:
        OpenAIWrapper that supports tool-py decorated functions
    
    Example:
        import openai
        import toolflow
        
        # Full response mode (default behavior)
        client = toolflow.from_openai(openai.OpenAI(), full_response=True)
        response = client.chat.completions.create(...)
        content = response.choices[0].message.content
        
        # Simplified response mode (new behavior)
        client = toolflow.from_openai(openai.OpenAI(), full_response=False)
        content = client.chat.completions.create(...)  # Returns only content
        
        # For structured outputs with simplified mode
        parsed_data = client.chat.completions.parse(...)  # Returns only parsed data
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    return OpenAIWrapper(client, full_response)


def from_openai_async(client, full_response: bool = False) -> "OpenAIAsyncWrapper":
    """
    Create a toolflow wrapper around an existing OpenAI async client.
    
    Args:
        client: An existing OpenAI async client instance
        full_response: If True, return the full OpenAI response object.
                      If False (default), return only the content or parsed data.
    
    Returns:
        OpenAIAsyncWrapper that supports tool-py decorated functions
    
    Example:
        import openai
        import toolflow
        
        # Full response mode
        client = toolflow.from_openai_async(openai.AsyncOpenAI(), full_response=True)
        response = await client.chat.completions.create(...)
        content = response.choices[0].message.content
        
        # Simplified response mode (default)
        client = toolflow.from_openai_async(openai.AsyncOpenAI(), full_response=False)
        content = await client.chat.completions.create(...)  # Returns only content
        
        # For structured outputs with simplified mode
        parsed_data = await client.chat.completions.parse(...)  # Returns only parsed data
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    return OpenAIAsyncWrapper(client, full_response)
