"""
OpenAI provider for toolflow.

This module provides factory functions to create toolflow wrappers around OpenAI clients.
"""
from typing import Union, overload
from .wrappers import OpenAIWrapper, AsyncOpenAIWrapper

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
@overload
def from_openai(client: openai.OpenAI, full_response: bool = False) -> OpenAIWrapper:
    ...

@overload
def from_openai(client: openai.AsyncOpenAI, full_response: bool = False) -> AsyncOpenAIWrapper:
    ...

def from_openai(client, full_response: bool = False):
    """
    Create a toolflow wrapper around an existing OpenAI client.
    
    Automatically detects whether the client is synchronous (OpenAI) or 
    asynchronous (AsyncOpenAI) and returns the appropriate wrapper.
    
    Args:
        client: An existing OpenAI client instance (OpenAI or AsyncOpenAI)
        full_response: If True, return the full OpenAI response object. 
                      If False (default), return only the content or parsed data.
    
    Returns:
        OpenAIWrapper or AsyncOpenAIWrapper that supports tool-py decorated functions
    
    Example:
        import openai
        import toolflow
        
        # Synchronous client
        sync_client = toolflow.from_openai(openai.OpenAI())
        content = sync_client.chat.completions.create(...)
        
        # Asynchronous client
        async_client = toolflow.from_openai(openai.AsyncOpenAI())
        content = await async_client.chat.completions.create(...)
        
        # Full response mode
        client = toolflow.from_openai(openai.OpenAI(), full_response=True)
        response = client.chat.completions.create(...)
        content = response.choices[0].message.content
        
        # Async context management
        async with toolflow.from_openai(openai.AsyncOpenAI()) as client:
            content = await client.chat.completions.create(...)
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    # Allow Mock objects for testing
    if hasattr(client, '_mock_name') or client.__class__.__name__ == 'Mock':
        # This is a mock object, assume sync wrapper for testing
        return OpenAIWrapper(client, full_response)
    
    # Detect client type and return appropriate wrapper
    if isinstance(client, openai.AsyncOpenAI):
        return AsyncOpenAIWrapper(client, full_response)
    elif isinstance(client, openai.OpenAI):
        return OpenAIWrapper(client, full_response)
    else:
        # Provide helpful error message
        if hasattr(client, '__class__'):
            client_type = client.__class__.__name__
            raise TypeError(
                f"Expected openai.OpenAI or openai.AsyncOpenAI client, got {client_type}. "
                f"Please pass a valid OpenAI() or AsyncOpenAI() client instance."
            )
        else:
            raise TypeError(
                f"Expected openai.OpenAI or openai.AsyncOpenAI client, got {type(client)}. "
                f"Please pass a valid OpenAI() or AsyncOpenAI() client instance."
            )
