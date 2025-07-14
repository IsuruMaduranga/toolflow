# src/toolflow/providers/anthropic/handler.py
from __future__ import annotations
import json
from typing import Any, List, Dict, Generator, AsyncGenerator, Union, Optional, Tuple
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, RawMessageStreamEvent

from toolflow.core import TransportAdapter, MessageAdapter, ResponseFormatAdapter

class AnthropicHandler(TransportAdapter, MessageAdapter, ResponseFormatAdapter):
    def __init__(self, client: Union[Anthropic, AsyncAnthropic], original_create):
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
        
        # Anthropic-specific error patterns
        anthropic_schema_patterns = [
            'invalid request',
            'tool schema',
            'input_schema',
            'invalid tool',
            'validation error',
            'json schema',
            'required field',
            'additional properties'
        ]
        
        # Check if this is an Anthropic schema error
        is_anthropic_error = any(pattern in error_message for pattern in anthropic_schema_patterns)
        
        if is_anthropic_error and tools:
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

    def stream_response(self, response: Generator[RawMessageStreamEvent, None, None]) -> Generator[RawMessageStreamEvent, None, None]:
        """Handle a streaming response and yield raw events."""
        for event in response:
            yield event

    async def stream_response_async(self, response: AsyncGenerator[RawMessageStreamEvent, None]) -> AsyncGenerator[RawMessageStreamEvent, None]:
        """Handle an async streaming response and yield raw events."""
        async for event in response:
            yield event

    def parse_response(self, response: Message) -> Tuple[Optional[str], List[Dict], Any]:
        """Parse a complete response into (text, tool_calls, raw_response)."""
        text_content = ""
        tool_calls = []
        
        for content_block in response.content:
            if hasattr(content_block, 'type'):
                if content_block.type == 'text':
                    text_content += content_block.text
                elif content_block.type == 'thinking':
                    text_content += f"\n<THINKING>\n{content_block.thinking}\n</THINKING>\n\n"
                elif content_block.type == 'tool_use':
                    tool_calls.append(self._format_tool_call(content_block))
        
        return text_content, tool_calls, response

    def check_max_tokens_reached(self, response: Message) -> bool:
        """Check if max tokens was reached and raise exception if so."""
        if response.stop_reason == "max_tokens":
            return True
        return False

    def parse_stream_chunk(self, event: RawMessageStreamEvent) -> Tuple[Optional[str], Optional[List[Dict]], Any]:
        """Parse a streaming event into (text, tool_calls, raw_event)."""
        text = None
        tool_calls = None
        
        if event.type == 'content_block_start':
            if hasattr(event.content_block, 'type'):
                if event.content_block.type == 'text':
                    text = ""  # Start text block
                elif event.content_block.type == 'tool_use':
                    # Tool use started, but not complete yet
                    tool_calls = None
        
        elif event.type == 'content_block_delta':
            if hasattr(event.delta, 'type'):
                if event.delta.type == 'text_delta':
                    text = event.delta.text
                elif event.delta.type == 'input_json_delta':
                    # Tool arguments being streamed, not complete yet
                    tool_calls = None
        
        elif event.type == 'content_block_stop':
            # Content block finished, might be a complete tool call
            # Note: Tool call completion needs to be handled by accumulation logic
            # in the execution loop since Anthropic streams tool calls in parts
            tool_calls = None
        
        return text, tool_calls, event

    def _create_tool_call_object(self, tool_id: str, tool_name: str, tool_input: Dict[str, Any]):
        """Create a tool call object that matches Anthropic's format for execution."""
        return type('ToolCall', (), {
            'id': tool_id,
            'name': tool_name,
            'input': tool_input
        })()

    def _accumulate_anthropic_streaming_content(
        self,
        chunk,
        message_content: List[Dict[str, Any]],
        accumulated_tool_calls: List[Any],
        accumulated_json_strings: Dict[int, str],
        graceful_error_handling: bool = True
    ) -> bool:
        """
        Accumulate Anthropic streaming content and detect tool calls.
        
        Returns:
            bool: True if tool calls were detected and completed, False otherwise
        """
        tool_calls_completed = False
        
        if chunk.type == "content_block_start":
            content_block = chunk.content_block
            
            if content_block.type == "text":
                message_content.append({
                    "type": "text",
                    "text": ""
                })

            elif content_block.type == "thinking":
                message_content.append({
                    "type": "thinking",
                    "thinking": "",
                    "signature": ""  # Will be filled when signature_delta arrives
                })
            elif content_block.type == "tool_use":
                message_content.append({
                    "type": "tool_use",
                    "id": content_block.id,
                    "name": content_block.name,
                    "input": {}
                })
                accumulated_json_strings[chunk.index] = ""
        
        elif chunk.type == "content_block_delta":
            delta = chunk.delta
            content_index = chunk.index
            
            if delta.type == "text_delta":
                # Update text content
                if (content_index < len(message_content) and 
                    message_content[content_index]["type"] == "text"):
                    message_content[content_index]["text"] += str(delta.text)
            
            elif delta.type == "thinking_delta":
                # Update thinking content
                if (content_index < len(message_content) and 
                    message_content[content_index]["type"] == "thinking"):
                    message_content[content_index]["thinking"] += str(delta.thinking)
            
            elif delta.type == "signature_delta":
                # Update thinking signature
                if (content_index < len(message_content) and 
                    message_content[content_index]["type"] == "thinking"):
                    message_content[content_index]["signature"] += str(delta.signature)
            
            elif delta.type == "input_json_delta":
                # Accumulate JSON for tool inputs
                if content_index in accumulated_json_strings:
                    accumulated_json_strings[content_index] += str(delta.partial_json)
        
        elif chunk.type == "content_block_stop":
            content_index = chunk.index
            
            # If this was a tool_use block, parse the accumulated JSON
            if (content_index in accumulated_json_strings and
                content_index < len(message_content) and
                message_content[content_index]["type"] == "tool_use"):
                
                try:
                    tool_input = json.loads(accumulated_json_strings[content_index])
                    message_content[content_index]["input"] = tool_input
                    
                    # Create a tool call object for execution
                    tool_call = self._create_tool_call_object(
                        message_content[content_index]["id"],
                        message_content[content_index]["name"],
                        tool_input
                    )
                    accumulated_tool_calls.append(tool_call)
                    
                except json.JSONDecodeError:
                    # Handle malformed JSON gracefully
                    if graceful_error_handling:
                        message_content[content_index]["input"] = {"error": "Invalid JSON"}
                    else:
                        raise Exception(f"Invalid JSON in tool input: {accumulated_json_strings[content_index]}")
                
                # Clean up accumulated JSON
                del accumulated_json_strings[content_index]
        
        elif chunk.type == "message_stop":
            # Message is complete, check if we have any tool calls
            if accumulated_tool_calls:
                tool_calls_completed = True
        
        return tool_calls_completed

    def _should_yield_chunk(self, chunk, return_full_response: bool, block_types: dict = None):
        """
        Determine if a chunk should be yielded and extract its content.
        
        Args:
            chunk: The streaming chunk from Anthropic
            return_full_response: Whether we're returning full response objects
            block_types: Dict to track block types by index (optional)
            
        Returns:
            Tuple of (should_yield: bool, content: str)
        """
        if return_full_response:
            return True, chunk
        
        # Initialize block_types if not provided
        if block_types is None:
            block_types = {}
        
        # Handle different types of Anthropic streaming events
        if hasattr(chunk, 'type'):
            if chunk.type == 'content_block_delta':
                if hasattr(chunk, 'delta'):
                    # Handle thinking content
                    if hasattr(chunk.delta, 'type') and chunk.delta.type == 'thinking_delta':
                        if hasattr(chunk.delta, 'thinking'):
                            return True, str(chunk.delta.thinking)
                    
                    # Handle text content
                    elif hasattr(chunk.delta, 'type') and chunk.delta.type == 'text_delta':
                        if hasattr(chunk.delta, 'text'):
                            return True, str(chunk.delta.text)
            
            elif chunk.type == 'content_block_start':
                if hasattr(chunk, 'content_block') and hasattr(chunk, 'index'):
                    # Track block type by index
                    block_types[chunk.index] = chunk.content_block.type
                    
                    # Handle thinking block start
                    if chunk.content_block.type == 'thinking':
                        return True, "\n<THINKING>\n"
                    
                    # Handle text block start  
                    elif chunk.content_block.type == 'text':
                        return True, ""  # No indicator for text start
            
            elif chunk.type == 'content_block_stop':
                if hasattr(chunk, 'index'):
                    # Check if this was a thinking block that's ending
                    block_type = block_types.get(chunk.index)
                    if block_type == 'thinking':
                        # Clean up the tracking
                        block_types.pop(chunk.index, None)
                        return True, "\n</THINKING>\n\n"
                    elif block_type == 'text':
                        # Clean up the tracking
                        block_types.pop(chunk.index, None)
                        return True, "\n"  # Just add newline for text blocks
        
        # Fallback for other chunk types
        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
            return True, str(chunk.delta.text)
        
        return False, "" 

    def accumulate_streaming_response(self, response: Generator[RawMessageStreamEvent, None, None]) -> Generator[Tuple[Optional[str], Optional[List[Dict]], Any], None, None]:
        """Handle streaming response with comprehensive tool call and thinking accumulation."""
        # Use the sophisticated accumulation logic from the old working code
        message_content = []
        accumulated_tool_calls = []
        accumulated_json_strings = {}
        block_types = {}  # Track block types for proper thinking tag handling
        
        for event in self.stream_response(response):
            # Accumulate content and detect tool calls using the old working logic
            tool_calls_completed = self._accumulate_anthropic_streaming_content(
                chunk=event,
                message_content=message_content,
                accumulated_tool_calls=accumulated_tool_calls,
                accumulated_json_strings=accumulated_json_strings,
                graceful_error_handling=True
            )
            
            # Determine what to yield using the old working logic
            should_yield, content = self._should_yield_chunk(event, False, block_types)
            
            # Yield content if available
            text = content if should_yield and content else None
            
            # Yield tool calls when they are completed
            tool_calls = None
            if tool_calls_completed and accumulated_tool_calls:
                # Convert tool calls to the expected format
                tool_calls = []
                for tool_call in accumulated_tool_calls:
                    formatted_call = {
                        "id": tool_call.id,
                        "type": "function", 
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.input
                        }
                    }
                    tool_calls.append(formatted_call)
            
            yield text, tool_calls, event

    async def accumulate_streaming_response_async(self, response: AsyncGenerator[RawMessageStreamEvent, None]) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict]], Any], None]:
        """Handle async streaming response with comprehensive tool call and thinking accumulation."""
        # Use the sophisticated accumulation logic from the old working code
        message_content = []
        accumulated_tool_calls = []
        accumulated_json_strings = {}
        block_types = {}  # Track block types for proper thinking tag handling
        
        async for event in self.stream_response_async(response):
            # Accumulate content and detect tool calls using the old working logic
            tool_calls_completed = self._accumulate_anthropic_streaming_content(
                chunk=event,
                message_content=message_content,
                accumulated_tool_calls=accumulated_tool_calls,
                accumulated_json_strings=accumulated_json_strings,
                graceful_error_handling=True
            )
            
            # Determine what to yield using the old working logic
            should_yield, content = self._should_yield_chunk(event, False, block_types)
            
            # Yield content if available
            text = content if should_yield and content else None
            
            # Yield tool calls when they are completed
            tool_calls = None
            if tool_calls_completed and accumulated_tool_calls:
                # Convert tool calls to the expected format
                tool_calls = []
                for tool_call in accumulated_tool_calls:
                    formatted_call = {
                        "id": tool_call.id,
                        "type": "function", 
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.input
                        }
                    }
                    tool_calls.append(formatted_call)
            
            yield text, tool_calls, event

    def build_assistant_message(self, text: Optional[str], tool_calls: List[Dict], original_response: Message = None) -> Dict:
        """Build an assistant message with tool calls for Anthropic format."""
        # If we have an original response with thinking blocks, use it directly to preserve signatures
        if original_response and self._has_thinking_blocks(original_response):
            return {
                "role": "assistant", 
                "content": original_response.content
            }
        
        # Standard message building for non-thinking responses
        content = []
        
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        # Add tool calls
        for tool_call in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": tool_call["function"]["arguments"]
            })
        
        return {
            "role": "assistant",
            "content": content
        }

    def _has_thinking_blocks(self, response: Message) -> bool:
        """Check if a response contains thinking or redacted_thinking blocks."""
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type in ('thinking', 'redacted_thinking'):
                return True
        return False

    def build_tool_result_messages(self, tool_results: List[Dict]) -> List[Dict]:
        """Build tool result messages for Anthropic format."""
        content = []
        for result in tool_results:
            content.append({
                "type": "tool_result",
                "tool_use_id": result["tool_call_id"],
                "content": str(result["output"])
            })
        
        return [{
            "role": "user",
            "content": content
        }]
    
    #@Override
    def prepare_tool_schemas(self, tools: List[Any]) -> Tuple[List[Dict], Dict]:
        """Prepare tool schemas in Anthropic format."""
        
        # Get OpenAI-format schemas from parent
        openai_tool_schemas, tool_map = super().prepare_tool_schemas(tools)
        
        # Convert OpenAI format to Anthropic format
        anthropic_tool_schemas = []
        for openai_schema in openai_tool_schemas:
            anthropic_schema = {
                "name": openai_schema['function']['name'],
                "description": openai_schema['function']['description'],
                "input_schema": openai_schema['function']['parameters']
            }
            anthropic_tool_schemas.append(anthropic_schema)
        
        return anthropic_tool_schemas, tool_map

    def _format_tool_call(self, tool_use_block) -> Dict:
        """Format Anthropic tool_use block to standard format."""
        return {
            "id": tool_use_block.id,
            "type": "function",
            "function": {
                "name": tool_use_block.name,
                "arguments": tool_use_block.input,
            },
        }
