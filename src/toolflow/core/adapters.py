import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, List, Dict, Tuple, Optional
from .utils import get_structured_output_tool, extract_toolkit_methods, _get_cached_toolkit_schema, _cache_toolkit_schema, get_tool_schema
from .constants import RESPONSE_FORMAT_TOOL_NAME
from .protocols import BaseToolKit, BaseAsyncToolKit
from .logging_utils import get_toolflow_logger, log_toolkit_connection_error

# custom logger
logger = get_toolflow_logger("adapters")

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
        return get_tool_schema(tool)
    
    def build_response_format_retry_message(self) -> Dict[str, Any]:
        """Build a response format retry message for Anthropic format."""
        return {
            "role": "user",
            "content": f"Call the {RESPONSE_FORMAT_TOOL_NAME} to provide the final response."
        }

        def prepare_tool_schemas(self, tools: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prepare tool schemas and tool map for sync toolflow execution."""
        tool_schemas, tool_map = [], {}
        for tool in tools:
            if isinstance(tool, BaseAsyncToolKit):
                raise RuntimeError("Some tools require async execution (e.g., MCP ToolKit). Use an async client.")
            elif asyncio.iscoroutinefunction(tool):
                raise RuntimeError("Async functions are not supported in sync toolflow execution")
            else:
                tool_schemas, tool_map = self._prepare_tool_schema(tool, tool_schemas, tool_map)
        return tool_schemas, tool_map
    
    async def prepare_tool_schemas_async(self, tools: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prepare tool schemas and tool map for async toolflow execution."""
        tool_schemas, tool_map = [], {}
        for tool in tools:
            if isinstance(tool, BaseAsyncToolKit):
                tool_schemas, tool_map = await self._prepare_async_toolkit(tool, tool_schemas, tool_map)
            else:
                tool_schemas, tool_map = self._prepare_tool_schema(tool, tool_schemas, tool_map)
        return tool_schemas, tool_map
    
    def _prepare_tool_schema(self, tool: Any, tool_schemas: List[Dict[str, Any]], tool_map: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Sync implementation of tool schema preparation."""

        # Check if tool implements BaseToolKit protocol (sync toolkits)
        if isinstance(tool, BaseToolKit):
            # Handle sync BaseToolKit protocol
            for schema in tool.list_tools():
                tool_name = schema["function"]["name"]
                
                # check if the tool is the response format tool and it is not an internal tool
                if tool_name == RESPONSE_FORMAT_TOOL_NAME:
                    raise ValueError(f"You cannot use the {RESPONSE_FORMAT_TOOL_NAME} as a tool. It is used internally to format the response.")
                
                tool_schemas.append(schema)
                # Create a lambda that captures the tool and tool name
                tool_map[tool_name] = lambda args, t=tool, n=tool_name: t.call_tool(n, args)
            return tool_schemas, tool_map

        # Check if tool is a ToolKit instance (class instance with methods)
        # Must not be a built-in type, class, function, or builtin
        if (hasattr(tool, '__class__') and 
            not inspect.isclass(tool) and 
            not callable(tool) and 
            not inspect.isbuiltin(tool) and
            type(tool).__module__ != 'builtins'):
            
            # Extract methods from ToolKit instance
            try:
                methods = extract_toolkit_methods(tool)
                for method in methods:
                    # Process each method as a tool
                    # Use cache for ToolKit methods since schema generation is expensive
                    method_name = method.__name__
                    cached_schema = _get_cached_toolkit_schema(tool, method_name)
                    
                    if cached_schema:
                        schema = cached_schema
                    else:
                        # Generate schema and cache it
                        schema = self.get_tool_schema(method)
                        _cache_toolkit_schema(tool, method_name, schema)

                    # check if the tool is the response format tool and it is not an internal tool
                    if schema["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME and not hasattr(method, "__internal_tool__"):
                        raise ValueError(f"You cannot use the {RESPONSE_FORMAT_TOOL_NAME} as a tool. It is used internally to format the response.")
                        
                    tool_schemas.append(schema)
                    tool_map[schema["function"]["name"]] = method
                return tool_schemas, tool_map
            except ValueError as e:
                # If extraction fails, treat as invalid tool
                raise ValueError(f"Invalid ToolKit: {e}")
        
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
            return tool_schemas, tool_map
        else:
            raise ValueError(f"Tool {tool} is not a function or ToolKit instance")
    
    async def _prepare_async_toolkit(self, tool: Any, tool_schemas: List[Dict[str, Any]], tool_map: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Async implementation of tool schema preparation."""

        # Check if tool implements BaseAsyncToolKit protocol (async toolkits)
        if isinstance(tool, BaseAsyncToolKit):
            # Handle async BaseAsyncToolKit protocol
            schemas = []
            try:
                schemas = await tool.list_tools()
            except Exception as e:
                log_toolkit_connection_error(logger, tool.__class__.__name__, e)
                return tool_schemas, tool_map  # Return current state instead of empty list
            
            for schema in schemas:
                tool_name = schema["function"]["name"]
                
                # check if the tool is the response format tool and it is not an internal tool
                if tool_name == RESPONSE_FORMAT_TOOL_NAME:
                    raise ValueError(f"You cannot use the {RESPONSE_FORMAT_TOOL_NAME} as a tool. It is used internally to format the response.")
                
                tool_schemas.append(schema)
                # Create async function that captures the tool and tool name
                async def call_tool_async(name, args):
                    return await tool.call_tool(name, arguments=args)
                call_tool_async.__is_toolflow_dynamic_tool__ = True
                tool_map[tool_name] = call_tool_async
            return tool_schemas, tool_map
        else:
            raise ValueError(f"Tool {tool} is not a BaseAsyncToolKit instance")

class ResponseFormatAdapter(ABC):
    """
    Protocol for handling response format.
    """
    def prepare_response_format_tool(self, tools: List[Any], response_format: Any) -> Tuple[List[Any], bool]:
        """Get the response format tool schema."""
        if not response_format:
            return tools, False
        return tools + [get_structured_output_tool(response_format)], True
