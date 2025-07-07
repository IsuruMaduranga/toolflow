# src/toolflow/providers/openai/handlers.py
from __future__ import annotations
import json
from typing import Any, List, Dict, Generator, AsyncGenerator, Union, Optional, Tuple
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from toolflow.core import TransportAdapter, MessageAdapter, ResponseFormatAdapter

class OpenAIHandler(TransportAdapter, MessageAdapter, ResponseFormatAdapter):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI], original_create):
        self.client = client
        self.original_create = original_create

    def call_api(self, **kwargs) -> Any:
        try:
            return self.original_create(**kwargs)
        except Exception as e:
            # Extract tools from kwargs to help with error detection
            tools = kwargs.get('tools', [])
            self._handle_api_error(e, tools)

    async def call_api_async(self, **kwargs) -> Any:
        try:
            return await self.original_create(**kwargs)
        except Exception as e:
            # Extract tools from kwargs to help with error detection  
            tools = kwargs.get('tools', [])
            self._handle_api_error(e, tools)

    def _handle_api_error(self, error: Exception, tools: List[Any]) -> None:
        """Handle API errors and provide better messages for tool schema issues."""
        error_message = str(error).lower()
        
        # OpenAI-specific error patterns
        openai_schema_patterns = [
            'invalid schema',
            'function parameters',
            'extra required key',
            'is not valid under any of the given schemas',
            'prefixitems',
            'additional properties',
            'not of type',
            'schema missing items',
            'invalid request',
            'function_call is invalid',
            'required is required to be supplied',
            'additionalproperties'
        ]
        
        # Check if this is an OpenAI schema error
        is_openai_error = any(pattern in error_message for pattern in openai_schema_patterns)
        
        if is_openai_error and tools:
            raise ValueError(
                f"Tool schema compatibility error with OpenAI:\n\n"
                f"Original error: {error}\n\n"
                f"Common fixes:\n"
                f"  • Replace NamedTuple with List[List[float]] for coordinates\n"
                f"  • Use Dict[str, Any] instead of Dict[Enum, Any]\n"
                f"  • Make complex fields Optional with default values\n"
                f"  • Simplify nested structures\n"
            ) from error
        
        # If not a schema error or no tools, re-raise original
        raise error

    def stream_response(self, response: Generator[ChatCompletionChunk, None, None]) -> Generator[ChatCompletionChunk, None, None]:
        """Handle a streaming response and yield raw chunks."""
        for chunk in response:
            yield chunk

    async def stream_response_async(self, response: AsyncGenerator[ChatCompletionChunk, None]) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Handle an async streaming response and yield raw chunks."""
        async for chunk in response:
            yield chunk

    def parse_response(self, response: ChatCompletion) -> Tuple[Optional[str], List[Dict], Any]:
        """Parse a complete response into (text, tool_calls, raw_response)."""
        message = response.choices[0].message
        text_content = message.content
        tool_calls = []
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                    }
                })
        
        return text_content, tool_calls, response

    def check_max_tokens_reached(self, response: ChatCompletion) -> bool:
        """Check if max tokens was reached and return True if so."""
        if response.choices[0].finish_reason == "length":
            return True
        return False

    def parse_stream_chunk(self, chunk: ChatCompletionChunk) -> Tuple[Optional[str], Optional[List[Dict]], Any]:
        """Parse a streaming chunk into (text, tool_calls, raw_chunk)."""
        text = None
        tool_calls = None
        
        choice = chunk.choices[0] if chunk.choices else None
        if choice and choice.delta:
            # Handle text content
            if choice.delta.content:
                text = choice.delta.content
            
            # Handle tool calls (OpenAI streams them differently)
            if choice.delta.tool_calls:
                # Note: OpenAI tool calls in streaming need accumulation
                # This is simplified - full implementation would accumulate
                tool_calls = None  # Defer to accumulation logic
        
        return text, tool_calls, chunk

    def accumulate_streaming_response(self, response: Generator[ChatCompletionChunk, None, None]) -> Generator[Tuple[Optional[str], Optional[List[Dict]], Any], None, None]:
        """Handle streaming response with tool call accumulation for OpenAI."""
        accumulated_tool_calls = {}
        
        for chunk in self.stream_response(response):
            text = None
            tool_calls = None
            
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta:
                # Handle text content
                if choice.delta.content:
                    text = choice.delta.content
                
                # Handle tool calls accumulation
                if choice.delta.tool_calls:
                    for tool_call_delta in choice.delta.tool_calls:
                        index = tool_call_delta.index
                        
                        if index not in accumulated_tool_calls:
                            accumulated_tool_calls[index] = {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }
                        
                        # Accumulate tool call parts
                        if tool_call_delta.id:
                            accumulated_tool_calls[index]["id"] = tool_call_delta.id
                        
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                accumulated_tool_calls[index]["function"]["name"] = tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                accumulated_tool_calls[index]["function"]["arguments"] += tool_call_delta.function.arguments
            
            # Check if we have complete tool calls (separate from delta processing)
            if choice and choice.finish_reason == "tool_calls":
                # Tool calls are complete, parse arguments
                tool_calls = []
                for tool_call in accumulated_tool_calls.values():
                    try:
                        # Parse JSON arguments
                        tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
                        tool_calls.append(tool_call)
                    except json.JSONDecodeError:
                        # Handle malformed JSON gracefully
                        tool_call["function"]["arguments"] = {"error": "Invalid JSON"}
                        tool_calls.append(tool_call)
            
            yield text, tool_calls, chunk

    async def accumulate_streaming_response_async(self, response: AsyncGenerator[ChatCompletionChunk, None]) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict]], Any], None]:
        """Handle async streaming response with tool call accumulation for OpenAI."""
        accumulated_tool_calls = {}
        
        async for chunk in self.stream_response_async(response):
            text = None
            tool_calls = None
            
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta:
                # Handle text content
                if choice.delta.content:
                    text = choice.delta.content
                
                # Handle tool calls accumulation
                if choice.delta.tool_calls:
                    for tool_call_delta in choice.delta.tool_calls:
                        index = tool_call_delta.index
                        
                        if index not in accumulated_tool_calls:
                            accumulated_tool_calls[index] = {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }
                        
                        # Accumulate tool call parts
                        if tool_call_delta.id:
                            accumulated_tool_calls[index]["id"] = tool_call_delta.id
                        
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                accumulated_tool_calls[index]["function"]["name"] = tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                accumulated_tool_calls[index]["function"]["arguments"] += tool_call_delta.function.arguments
            
            # Check if we have complete tool calls (separate from delta processing)
            if choice and choice.finish_reason == "tool_calls":
                # Tool calls are complete, parse arguments
                tool_calls = []
                for tool_call in accumulated_tool_calls.values():
                    try:
                        # Parse JSON arguments
                        tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
                        tool_calls.append(tool_call)
                    except json.JSONDecodeError:
                        # Handle malformed JSON gracefully
                        tool_call["function"]["arguments"] = {"error": "Invalid JSON"}
                        tool_calls.append(tool_call)
            
            yield text, tool_calls, chunk

    def build_assistant_message(self, text: Optional[str], tool_calls: List[Dict], original_response: ChatCompletion = None) -> Dict:
        """Build an assistant message with tool calls for OpenAI format."""
        message = {
            "role": "assistant",
        }
        
        if text:
            message["content"] = text
        
        if tool_calls:
            openai_tool_calls = []
            for tool_call in tool_calls:
                openai_tool_calls.append({
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": json.dumps(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], dict) else tool_call["function"]["arguments"]
                    }
                })
            message["tool_calls"] = openai_tool_calls
        
        return message

    def build_tool_result_messages(self, tool_results: List[Dict]) -> List[Dict]:
        """Build tool result messages for OpenAI format."""
        messages = []
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": str(result["output"])
            })
        return messages
    
    def prepare_tool_schemas(self, tools: List[Any]) -> Tuple[List[Dict], Dict]:
        """Prepare tool schemas in OpenAI format."""
        # Get OpenAI-format schemas from parent
        tool_schemas, tool_map = super().prepare_tool_schemas(tools)
        
        # OpenAI uses the schema as-is
        return tool_schemas, tool_map

class OpenAICreateHandler(OpenAIHandler, ResponseFormatAdapter):
    pass
