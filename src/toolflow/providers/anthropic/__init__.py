"""
Anthropic provider for toolflow.

This module provides factory functions to create toolflow wrappers around Anthropic clients.
"""
from .wrappers import AnthropicWrapper, AnthropicAsyncWrapper

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def from_anthropic(client: "anthropic.Anthropic", full_response: bool = False) -> "AnthropicWrapper":
    """
    Create a toolflow wrapper around an existing Anthropic client.
    
    Args:
        client: An existing Anthropic client instance
        full_response: If True, return the full Anthropic response object. 
                      If False (default), return only the content or parsed data.
    
    Returns:
        AnthropicWrapper that supports tool-py decorated functions
    
    Example:
        import anthropic
        import toolflow
        
        # Full response mode
        client = toolflow.from_anthropic(anthropic.Anthropic(), full_response=True)
        response = client.messages.create(...)
        content = response.content[0].text
        
        # Simplified response mode (default)
        client = toolflow.from_anthropic(anthropic.Anthropic(), full_response=False)
        content = client.messages.create(...)  # Returns only content string
        
        @toolflow.tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            tools=[get_weather],
            messages=[{"role": "user", "content": "What's the weather in NYC?"}]
        )
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
    
    # Validate client type
    if not isinstance(client, anthropic.Anthropic):
        if hasattr(client, '__class__'):
            client_type = client.__class__.__name__
            if client_type == "AsyncAnthropic":
                raise TypeError(
                    f"Expected synchronous Anthropic client, got AsyncAnthropic. "
                    f"Use toolflow.from_anthropic_async() for AsyncAnthropic clients."
                )
            else:
                raise TypeError(
                    f"Expected anthropic.Anthropic client, got {client_type}. "
                    f"Please pass a valid Anthropic() client instance."
                )
        else:
            raise TypeError(
                f"Expected anthropic.Anthropic client, got {type(client)}. "
                f"Please pass a valid Anthropic() client instance."
            )
    
    return AnthropicWrapper(client, full_response)


def from_anthropic_async(client: "anthropic.AsyncAnthropic", full_response: bool = False) -> "AnthropicAsyncWrapper":
    """
    Create a toolflow wrapper around an existing Anthropic async client.
    
    Args:
        client: An existing Anthropic async client instance
        full_response: If True, return the full Anthropic response object.
                      If False (default), return only the content or parsed data.
    
    Returns:
        AnthropicAsyncWrapper that supports tool-py decorated functions
    
    Example:
        import anthropic
        import toolflow
        
        # Full response mode
        client = toolflow.from_anthropic_async(anthropic.AsyncAnthropic(), full_response=True)
        response = await client.messages.create(...)
        content = response.content[0].text
        
        # Simplified response mode (default)
        client = toolflow.from_anthropic_async(anthropic.AsyncAnthropic(), full_response=False)
        content = await client.messages.create(...)  # Returns only content string
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
    
    # Validate client type
    if not isinstance(client, anthropic.AsyncAnthropic):
        if hasattr(client, '__class__'):
            client_type = client.__class__.__name__
            if client_type == "Anthropic":
                raise TypeError(
                    f"Expected asynchronous AsyncAnthropic client, got Anthropic. "
                    f"Use toolflow.from_anthropic() for synchronous Anthropic clients."
                )
            else:
                raise TypeError(
                    f"Expected anthropic.AsyncAnthropic client, got {client_type}. "
                    f"Please pass a valid AsyncAnthropic() client instance."
                )
        else:
            raise TypeError(
                f"Expected anthropic.AsyncAnthropic client, got {type(client)}. "
                f"Please pass a valid AsyncAnthropic() client instance."
            )
    
    return AnthropicAsyncWrapper(client, full_response)
