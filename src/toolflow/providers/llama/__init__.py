"""
Llama provider for toolflow.

This module provides integration with Llama models through OpenRouter or similar services
that offer OpenAI-compatible APIs for Llama models.
"""

try:
    import openai
    from .wrappers import LlamaWrapper
    from .handler import LlamaHandler
    
    def from_llama(client, **kwargs):
        """
        Create a toolflow-enhanced Llama client.
        
        Args:
            client: OpenAI client configured for Llama models (e.g., through OpenRouter)
            **kwargs: Additional configuration options
            
        Returns:
            LlamaWrapper: Enhanced client with toolflow capabilities
            
        Example:
            import openai
            import toolflow
            
            # Configure for OpenRouter with Llama models
            client = openai.OpenAI(
                api_key="your-openrouter-key",
                base_url="https://openrouter.ai/api/v1"
            )
            
            enhanced_client = toolflow.from_llama(client)
        """
        if not _is_valid_openai_client(client):
            raise ValueError("Invalid client. Expected OpenAI client configured for Llama models.")
        
        return LlamaWrapper(client, **kwargs)
    
    def _is_valid_openai_client(client):
        """Check if the client is a valid OpenAI client."""
        return hasattr(client, 'chat') and hasattr(client.chat, 'completions')
    
    __all__ = ['from_llama', 'LlamaWrapper', 'LlamaHandler']

except ImportError as e:
    def from_llama(*args, **kwargs):
        raise ImportError(
            "The 'openai' package is required to use the Llama provider. "
            "Install it with: pip install openai"
        ) from e
    
    LlamaWrapper = None
    LlamaHandler = None
    __all__ = ['from_llama']
