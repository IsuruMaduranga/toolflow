# src/toolflow/core/handlers.py
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, List, Dict, Callable
from .constants import RESPONSE_FORMAT_TOOL_NAME

class AbstractProviderHandler(ABC):
    """
    An abstract base class for provider-specific handlers.

    This class defines the interface that provider-specific handlers must implement
    to be compatible with the core execution loops.
    """

    @abstractmethod
    def call_api(self, **kwargs) -> Any:
        """Call the provider's synchronous API."""
        pass

    @abstractmethod
    async def call_api_async(self, **kwargs) -> Any:
        """Call the provider's asynchronous API."""
        pass

    @abstractmethod
    def handle_response(self, response: Any) -> tuple[str | None, List[Dict], Any]:
        """Handle the response from the provider's API."""
        pass

    @abstractmethod
    def handle_streaming_response(self, response: Any) -> Generator[tuple[str | None, List[Dict] | None, Any], None, None]:
        """Handle a streaming response from the provider's API."""
        pass

    @abstractmethod
    async def handle_streaming_response_async(self, response: Any) -> AsyncGenerator[tuple[str | None, List[Dict] | None, Any], None]:
        """Handle an async streaming response from the provider's API."""
        pass
    
    @abstractmethod
    def create_tool_result_message(self, tool_results: List[Dict]) -> Dict:
        """Create a tool result message for the provider."""
        pass

    @abstractmethod
    def get_structured_output_tool(self, pydantic_model: Any) -> Dict:
        """Get the tool definition for structured output."""
        pass

    @abstractmethod
    def get_structured_output_tool(self, pydantic_model: Any) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": RESPONSE_FORMAT_TOOL_NAME,
                "description": pydantic_model.model_config.get("description", "Extract information and present it in a structured format."),
                "parameters": pydantic_model.model_json_schema(),
            },
        }
