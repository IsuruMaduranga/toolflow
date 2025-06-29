# src/toolflow/core/handlers.py
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, List, Dict, Callable
from .constants import RESPONSE_FORMAT_TOOL_NAME
from .utils import get_structured_output_tool, get_tool_schema

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
    def create_assistant_message(self, text: str | None, tool_calls: List[Dict]) -> Dict:
        """Create an assistant message with tool calls for the conversation."""
        pass
    
    @abstractmethod
    def create_tool_result_messages(self, tool_results: List[Dict]) -> List[Dict]:
        """Create tool result messages for the conversation."""
        pass

    # Override following method for provider specific handling
    def parse_structured_output(self, tool_call: Dict, response_format: Any) -> Any:
        """Handle the structured output from the tool call."""
        tool_arguments = tool_call["function"]["arguments"]
        response_data = tool_arguments.get('response', tool_arguments)
        return response_format.model_validate(response_data)

    # Override following method for provider specific handling
    def get_response_format_tool(self,response_format: Any) -> Dict:
        # check if response_format is a Pydantic model
        if not response_format:
            return None
        if isinstance(response_format, type) and hasattr(response_format, 'model_json_schema'):
            return get_structured_output_tool(response_format)
        raise ValueError(f"Response format {response_format} is not a Pydantic model")
    
    # Override following method for provider specific handling
    def get_tool_schema(self, tool: Any) -> Dict:
        """Get the tool schema for the tool."""
        return get_tool_schema(tool)
    
    # Override following method for provider specific handling
    def prepare_tool_schemas(self, tools: List[Any]) -> tuple[List[Dict], Dict]: 
        tool_schemas = []
        tool_map = {}

        if tools:
            response_format_tool_count = 0
            for tool in tools:
                # check is tool is a function else error
                if callable(tool):
                    schema = tool._tool_metadata if hasattr(tool, "_tool_metadata") else self.get_tool_schema(tool)
                    if schema["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME:
                        response_format_tool_count += 1
                        if response_format_tool_count > 1:
                            raise ValueError(f"You cannot use the {RESPONSE_FORMAT_TOOL_NAME} and response_format at the same time.")
                    tool_schemas.append(schema)
                    tool_map[schema["function"]["name"]] = tool
                    continue
                else:
                    raise ValueError(f"Tool {tool} is not a function")
        return tool_schemas, tool_map
