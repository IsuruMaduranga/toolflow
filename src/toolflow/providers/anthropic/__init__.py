"""
Anthropic provider for toolflow (placeholder).

This is a skeleton implementation showing how to add support for other AI providers
using the provider-specific folder structure.
"""
from .wrapper import AnthropicWrapper, AnthropicAsyncWrapper

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def from_anthropic(client) -> "AnthropicWrapper":
    """
    Create a toolflow wrapper around an existing Anthropic client.
    
    Args:
        client: An existing Anthropic client instance
    
    Returns:
        AnthropicWrapper that supports tool-py decorated functions
    
    Example:
        import anthropic
        import toolflow
        
        client = toolflow.from_anthropic(anthropic.Anthropic())
        
        @toolflow.tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            tools=[get_weather],
            messages=[{"role": "user", "content": "What's the weather in NYC?"}]
        )
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
    
    return AnthropicWrapper(client)


def from_anthropic_async(client) -> "AnthropicAsyncWrapper":
    """
    Create a toolflow wrapper around an existing Anthropic async client.
    
    Args:
        client: An existing Anthropic async client instance
    
    Returns:
        AnthropicAsyncWrapper that supports tool-py decorated functions
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
    
    return AnthropicAsyncWrapper(client) 