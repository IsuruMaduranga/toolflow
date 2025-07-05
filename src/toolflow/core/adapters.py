from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, List, Dict, Tuple, Optional
from .utils import get_structured_output_tool
from .constants import RESPONSE_FORMAT_TOOL_NAME

class TransportAdapter(ABC):
    """
    Protocol for handling API calls and streaming transport.
    
    This adapter is responsible for:
    - Making synchronous and asynchronous API calls
    - Handling raw streaming responses
    """

    @abstractmethod
    def call_api(self, **kwargs: Any) -> Any:
        """Call the provider's synchronous API and return raw response.
        Handle all API errors and raise a ValueError with a helpful message.
        """
        pass

    @abstractmethod
    async def call_api_async(self, **kwargs: Any) -> Any:
        """Call the provider's asynchronous API and return raw response.
        Handle all API errors and raise a ValueError with a helpful message.
        """
        pass

    @abstractmethod
    def stream_response(self, response: Any) -> Generator[Any, None, None]:
        """Handle a streaming response and yield raw chunks."""
        pass

    @abstractmethod
    async def stream_response_async(self, response: Any) -> AsyncGenerator[Any, None]:
        """Handle an async streaming response and yield raw chunks."""
        pass

    @abstractmethod
    def accumulate_streaming_response(self, response: Any) -> Generator[Tuple[Optional[str], Optional[List[Dict[str, Any]]], Any], None, None]:
        """
        Handle streaming response with tool call accumulation.
        
        This is a higher-level method that handles the complexity of accumulating
        tool calls across multiple chunks. Default implementation uses simple
        chunk-by-chunk parsing, but providers can override for complex accumulation.
        """
        pass

    @abstractmethod
    async def accumulate_streaming_response_async(self, response: Any) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict[str, Any]]], Any], None]:
        """
        Handle async streaming response with tool call accumulation.
        
        This is a higher-level method that handles the complexity of accumulating
        tool calls across multiple chunks. Default implementation uses simple
        chunk-by-chunk parsing, but providers can override for complex accumulation.
        """
        pass

    @abstractmethod
    def check_max_tokens_reached(self, response: Any) -> bool:
        """Check if max tokens was reached and return True if so."""
        pass


class MessageAdapter(ABC):
    """
    Protocol for handling message processing and parsing.
    
    This adapter is responsible for:
    - Parsing responses into standardized format
    - Parsing streaming chunks
    - Building messages for conversation
    - Handling tool schemas and structured output
    """

    @abstractmethod
    def parse_response(self, response: Any) -> Tuple[Optional[str], List[Dict[str, Any]], Any]:
        """Parse a complete response into (text, tool_calls, raw_response)."""
        pass

    @abstractmethod
    def parse_stream_chunk(self, chunk: Any) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Any]:
        """Parse a streaming chunk into (text, tool_calls, raw_chunk)."""
        pass

    @abstractmethod
    def build_assistant_message(self, text: Optional[str], tool_calls: List[Dict[str, Any]], original_response: Any = None) -> Dict[str, Any]:
        """Build an assistant message with tool calls for the conversation."""
        pass

    @abstractmethod
    def build_tool_result_messages(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build tool result messages for the conversation."""
        pass

    def get_tool_schema(self, tool: Any) -> Dict[str, Any]:
        """Get the tool schema for the tool."""
        from .utils import get_tool_schema
        return get_tool_schema(tool)
    
    def build_response_format_retry_message(self) -> Dict[str, Any]:
        """Build a response format retry message for Anthropic format."""
        return {
            "role": "user",
            "content": f"Call the {RESPONSE_FORMAT_TOOL_NAME} to provide the final response."
        }

    def prepare_tool_schemas(self, tools: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prepare tool schemas and tool map."""
        import inspect
        from .constants import RESPONSE_FORMAT_TOOL_NAME
        
        tool_schemas = []
        tool_map = {}

        if tools:
            for tool in tools:
                # check is tool is a function else error
                if inspect.isbuiltin(tool):
                    raise ValueError(f"Tool {tool} is a builtin function. You cannot use it as a tool.")
                if callable(tool):
                    # Check for existing metadata (from decorator OR previous caching)
                    # Use try/except pattern for thread-safety to avoid race conditions
                    try:
                        schema = tool._tool_metadata
                    except AttributeError:
                        schema = self.get_tool_schema(tool)
                        # Cache for future use, but only for non-built-in functions
                        tool._tool_metadata = schema

                    # check if the tool is the response format tool and it is not an internal tool
                    if schema["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME and not hasattr(tool, "__internal_tool__"):
                        raise ValueError(f"You cannot use the {RESPONSE_FORMAT_TOOL_NAME} as a tool. It is used internally to format the response.")
                        
                    tool_schemas.append(schema)
                    tool_map[schema["function"]["name"]] = tool
                    continue
                else:
                    raise ValueError(f"Tool {tool} is not a function")
        return tool_schemas, tool_map

class ResponseFormatAdapter(ABC):
    """
    Protocol for handling response format.
    """
    def prepare_response_format_tool(self, tools: List[Any], response_format: Any) -> Tuple[List[Any], bool]:
        """Get the response format tool schema."""
        if not response_format:
            return tools, False
        if isinstance(response_format, type) and hasattr(response_format, 'model_json_schema'):
            return tools + [get_structured_output_tool(response_format)], True
        raise ValueError(f"Response format {response_format} is not a Pydantic model")
