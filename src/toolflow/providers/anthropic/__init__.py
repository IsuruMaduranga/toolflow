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


def from_anthropic(client, full_response: bool = False):
    """
    Create a toolflow wrapper around an existing Anthropic client.
    
    Automatically detects whether the client is synchronous (Anthropic) or 
    asynchronous (AsyncAnthropic) and returns the appropriate wrapper.
    
    Args:
        client: An existing Anthropic client instance (Anthropic or AsyncAnthropic)
        full_response: If True, return the full Anthropic response object. 
                      If False (default), return only the content or parsed data.
    
    Returns:
        AnthropicWrapper or AnthropicAsyncWrapper that supports tool-py decorated functions
    
    Example:
        import anthropic
        import toolflow
        
        # Synchronous client
        sync_client = toolflow.from_anthropic(anthropic.Anthropic())
        content = sync_client.messages.create(...)
        
        # Asynchronous client
        async_client = toolflow.from_anthropic(anthropic.AsyncAnthropic())
        content = await async_client.messages.create(...)
        
        # Full response mode
        client = toolflow.from_anthropic(anthropic.Anthropic(), full_response=True)
        response = client.messages.create(...)
        content = response.content[0].text
        
        # With tools
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
    
    # Allow Mock objects for testing
    if hasattr(client, '_mock_name') or client.__class__.__name__ == 'Mock':
        # This is a mock object, assume sync wrapper for testing
        return AnthropicWrapper(client, full_response)
    
    # Detect client type and return appropriate wrapper
    if isinstance(client, anthropic.AsyncAnthropic):
        return AnthropicAsyncWrapper(client, full_response)
    elif isinstance(client, anthropic.Anthropic):
        return AnthropicWrapper(client, full_response)
    else:
        # Provide helpful error message
        if hasattr(client, '__class__'):
            client_type = client.__class__.__name__
            raise TypeError(
                f"Expected anthropic.Anthropic or anthropic.AsyncAnthropic client, got {client_type}. "
                f"Please pass a valid Anthropic() or AsyncAnthropic() client instance."
            )
        else:
            raise TypeError(
                f"Expected anthropic.Anthropic or anthropic.AsyncAnthropic client, got {type(client)}. "
                f"Please pass a valid Anthropic() or AsyncAnthropic() client instance."
            )
