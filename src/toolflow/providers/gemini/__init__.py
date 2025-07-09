"""
Gemini provider for toolflow.

This module provides factory functions to create toolflow wrappers around Google Gemini clients.
"""
from .wrappers import GeminiWrapper

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def from_gemini(client, full_response: bool = False):
    """
    Create a toolflow wrapper around an existing Google Gemini client.
    
    Args:
        client: An existing Gemini client instance (GenerativeModel)
        full_response: If True, return the full Gemini response object. 
                      If False (default), return only the content or parsed data.
    
    Returns:
        GeminiWrapper that supports tool-py decorated functions
    
    Example:
        import google.generativeai as genai
        import toolflow
        
        # Configure API key
        genai.configure(api_key="your-api-key")
        
        # Create model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Wrap with toolflow
        client = toolflow.from_gemini(model)
        content = client.generate_content(...)
        
        # Full response mode
        client = toolflow.from_gemini(model, full_response=True)
        response = client.generate_content(...)
        content = response.text
        
        # With tools
        @toolflow.tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        response = client.generate_content(
            "What's the weather in NYC?",
            tools=[get_weather]
        )
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
    
    # Allow Mock objects for testing
    if hasattr(client, '_mock_name') or client.__class__.__name__ == 'Mock':
        # This is a mock object, assume wrapper for testing
        return GeminiWrapper(client, full_response)
    
    # Check if it's a GenerativeModel
    if hasattr(client, 'generate_content'):
        return GeminiWrapper(client, full_response)
    else:
        # Provide helpful error message
        if hasattr(client, '__class__'):
            client_type = client.__class__.__name__
            raise TypeError(
                f"Expected google.generativeai.GenerativeModel client, got {client_type}. "
                f"Please pass a valid GenerativeModel() client instance."
            )
        else:
            raise TypeError(
                f"Expected google.generativeai.GenerativeModel client, got {type(client)}. "
                f"Please pass a valid GenerativeModel() client instance."
            )
