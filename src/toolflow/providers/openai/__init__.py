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


def from_openai(client: "openai.OpenAI", full_response: bool = False) -> "OpenAIWrapper":
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
    
    # Validate client type
    # Allow Mock objects for testing
    if hasattr(client, '_mock_name') or client.__class__.__name__ == 'Mock':
        # This is a mock object, allow it for testing
        pass
    elif not isinstance(client, openai.OpenAI):
        if hasattr(client, '__class__'):
            client_type = client.__class__.__name__
            if client_type == "AsyncOpenAI":
                raise TypeError(
                    f"Expected synchronous OpenAI client, got AsyncOpenAI. "
                    f"Use toolflow.from_openai_async() for AsyncOpenAI clients."
                )
            else:
                raise TypeError(
                    f"Expected openai.OpenAI client, got {client_type}. "
                    f"Please pass a valid OpenAI() client instance."
                )
        else:
            raise TypeError(
                f"Expected openai.OpenAI client, got {type(client)}. "
                f"Please pass a valid OpenAI() client instance."
            )
    
    return OpenAIWrapper(client, full_response)


def from_openai_async(client: "openai.AsyncOpenAI", full_response: bool = False) -> "OpenAIAsyncWrapper":
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
    
    # Validate client type
    # Allow Mock objects for testing
    if hasattr(client, '_mock_name') or client.__class__.__name__ == 'Mock':
        # This is a mock object, allow it for testing
        pass
    elif not isinstance(client, openai.AsyncOpenAI):
        if hasattr(client, '__class__'):
            client_type = client.__class__.__name__
            if client_type == "OpenAI":
                raise TypeError(
                    f"Expected asynchronous AsyncOpenAI client, got OpenAI. "
                    f"Use toolflow.from_openai() for synchronous OpenAI clients."
                )
            else:
                raise TypeError(
                    f"Expected openai.AsyncOpenAI client, got {client_type}. "
                    f"Please pass a valid AsyncOpenAI() client instance."
                )
        else:
            raise TypeError(
                f"Expected openai.AsyncOpenAI client, got {type(client)}. "
                f"Please pass a valid AsyncOpenAI() client instance."
            )
    
    return OpenAIAsyncWrapper(client, full_response)
