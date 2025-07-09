# src/toolflow/providers/llama/handler.py
from __future__ import annotations
import json
from typing import Any, List, Dict, Generator, AsyncGenerator, Union, Optional, Tuple
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from toolflow.core import TransportAdapter, MessageAdapter, ResponseFormatAdapter

class LlamaHandler(TransportAdapter, MessageAdapter, ResponseFormatAdapter):
    """
    Handler for Llama models accessed through OpenAI-compatible APIs (like OpenRouter).
    
    Since Llama models are accessed through OpenAI-compatible interfaces, this handler
    leverages the existing OpenAI patterns while providing Llama-specific optimizations.
    """
    
    def __init__(self, client: Union[OpenAI, AsyncOpenAI], original_create):
        self.client = client
        self.original_create = original_create

    def call_api(self, **kwargs) -> Any:
        """Make API call to Llama model through OpenAI-compatible interface."""
        try:
            # Preprocess kwargs for Llama-specific optimizations
            kwargs = self._preprocess_llama_kwargs(kwargs)
            return self.original_create(**kwargs)
        except Exception as e:
            tools = kwargs.get('tools', [])
            self._handle_api_error(e, tools)

    async def call_api_async(self, **kwargs) -> Any:
        """Make async API call to Llama model through OpenAI-compatible interface."""
        try:
            kwargs = self._preprocess_llama_kwargs(kwargs)
            return await self.original_create(**kwargs)
        except Exception as e:
            tools = kwargs.get('tools', [])
            self._handle_api_error(e, tools)

    def call_api_streaming(self, **kwargs) -> Generator[Any, None, None]:
        """Make streaming API call to Llama model."""
        try:
            kwargs = self._preprocess_llama_kwargs(kwargs)
            kwargs['stream'] = True
            return self.original_create(**kwargs)
        except Exception as e:
            tools = kwargs.get('tools', [])
            self._handle_api_error(e, tools)

    async def call_api_streaming_async(self, **kwargs) -> AsyncGenerator[Any, None]:
        """Make async streaming API call to Llama model."""
        try:
            kwargs = self._preprocess_llama_kwargs(kwargs)
            kwargs['stream'] = True
            return await self.original_create(**kwargs)
        except Exception as e:
            tools = kwargs.get('tools', [])
            self._handle_api_error(e, tools)

    def _preprocess_llama_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess kwargs for Llama-specific optimizations.
        
        Llama models often work better with certain parameter adjustments
        when accessed through OpenAI-compatible APIs.
        """
        # Make a copy to avoid modifying the original
        processed_kwargs = kwargs.copy()
        
        # Llama models often benefit from slightly higher temperature for creativity
        if 'temperature' not in processed_kwargs:
            processed_kwargs['temperature'] = 0.7
        
        # Ensure reasonable max_tokens for Llama models
        if 'max_tokens' not in processed_kwargs:
            processed_kwargs['max_tokens'] = 2048
        
        # Some Llama model endpoints require specific model names
        if 'model' in processed_kwargs:
            processed_kwargs['model'] = self._normalize_llama_model_name(processed_kwargs['model'])
        
        return processed_kwargs

    def _normalize_llama_model_name(self, model_name: str) -> str:
        """
        Normalize Llama model names for different providers.
        
        Different services may use different naming conventions for Llama models.
        """
        # Common Llama model name patterns
        llama_mappings = {
            'llama2': 'meta-llama/llama-2-70b-chat',
            'llama-2': 'meta-llama/llama-2-70b-chat', 
            'llama3': 'meta-llama/llama-3-70b-instruct',
            'llama-3': 'meta-llama/llama-3-70b-instruct',
            'llama3.1': 'meta-llama/llama-3.1-70b-instruct',
            'llama-3.1': 'meta-llama/llama-3.1-70b-instruct',
            'llama3.2': 'meta-llama/llama-3.2-90b-instruct',
            'llama-3.2': 'meta-llama/llama-3.2-90b-instruct',
        }
        
        # If it's a shorthand, expand it; otherwise, keep as is
        return llama_mappings.get(model_name.lower(), model_name)

    def _handle_api_error(self, error: Exception, tools: List[Any]) -> None:
        """Handle API errors with Llama-specific context."""
        error_message = str(error).lower()
        
        # Llama/OpenRouter-specific error patterns
        llama_error_patterns = [
            'model not found',
            'model not available',
            'rate limit exceeded',
            'insufficient credits',
            'invalid model',
            'context length exceeded',
            'openrouter',
        ]
        
        # OpenAI-compatible error patterns (inherited behavior)
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
        
        # Check for Llama-specific errors
        is_llama_error = any(pattern in error_message for pattern in llama_error_patterns)
        is_schema_error = any(pattern in error_message for pattern in openai_schema_patterns)
        
        if is_llama_error:
            if 'model not found' in error_message or 'model not available' in error_message:
                raise ValueError(
                    f"Llama model error: {error}. "
                    "Check that your model name is correct and available through your provider. "
                    "Common Llama models: meta-llama/llama-3.1-70b-instruct, meta-llama/llama-2-70b-chat"
                ) from error
            elif 'rate limit' in error_message or 'insufficient credits' in error_message:
                raise ValueError(
                    f"Llama API quota/rate limit error: {error}. "
                    "Check your API quota and rate limits with your provider."
                ) from error
            elif 'context length exceeded' in error_message:
                raise ValueError(
                    f"Llama context length error: {error}. "
                    "Try reducing the input length or using a model with larger context window."
                ) from error
        
        if is_schema_error and tools:
            raise ValueError(
                f"Llama tool schema error: {error}. "
                "Llama models accessed through OpenAI-compatible APIs should support standard tool schemas. "
                "Check your tool function definitions and parameter types."
            ) from error
        
        # Re-raise the original error if not handled
        raise error

    def parse_response(self, response: Any) -> Tuple[str, List[Dict[str, Any]], Any]:
        """
        Parse response from Llama model (OpenAI-compatible format).
        
        Since Llama is accessed through OpenAI-compatible APIs, the response
        format should be the same as OpenAI responses.
        """
        if isinstance(response, (ChatCompletion, ChatCompletionChunk)):
            # Handle streaming chunks
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                
                # Extract text content
                content = ""
                if hasattr(choice, 'delta') and choice.delta:
                    # Streaming response
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        content = choice.delta.content
                elif hasattr(choice, 'message') and choice.message:
                    # Complete response
                    if hasattr(choice.message, 'content') and choice.message.content:
                        content = choice.message.content
                
                # Extract tool calls (same format as OpenAI)
                tool_calls = []
                if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'tool_calls'):
                    if choice.message.tool_calls:
                        for tool_call in choice.message.tool_calls:
                            tool_calls.append({
                                'id': tool_call.id,
                                'type': 'function',
                                'function': {
                                    'name': tool_call.function.name,
                                    'arguments': tool_call.function.arguments
                                }
                            })
                elif hasattr(choice, 'delta') and choice.delta and hasattr(choice.delta, 'tool_calls'):
                    # Streaming tool calls
                    if choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            if tool_call.function:
                                tool_calls.append({
                                    'id': getattr(tool_call, 'id', 'streaming'),
                                    'type': 'function', 
                                    'function': {
                                        'name': tool_call.function.name or '',
                                        'arguments': tool_call.function.arguments or ''
                                    }
                                })
                
                return content, tool_calls, response
        
        # Fallback for unexpected response formats
        return str(response), [], response

    def build_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build messages for Llama model (OpenAI-compatible format).
        
        Llama models accessed through OpenAI-compatible APIs use the same
        message format as OpenAI, so minimal processing is needed.
        """
        processed_messages = []
        
        for message in messages:
            # Ensure message has required fields
            if 'role' not in message:
                continue
            
            processed_message = {
                'role': message['role'],
                'content': message.get('content', '')
            }
            
            # Handle tool calls and tool responses (same as OpenAI format)
            if 'tool_calls' in message:
                processed_message['tool_calls'] = message['tool_calls']
            
            if 'tool_call_id' in message:
                processed_message['tool_call_id'] = message['tool_call_id']
            
            processed_messages.append(processed_message)
        
        return processed_messages

    def create_response_format_tool(self, response_format: Any) -> Optional[Dict[str, Any]]:
        """
        Create response format tool for structured outputs with Llama models.
        
        Since Llama is accessed through OpenAI-compatible APIs, we can leverage
        the same structured output patterns as OpenAI.
        """
        try:
            # Import here to avoid circular imports
            from toolflow.core.utils import create_response_format_tool
            return create_response_format_tool(response_format)
        except Exception:
            # Fallback: return None to disable structured outputs
            return None

    def stream_response(self, response: Generator[Any, None, None]) -> Generator[Any, None, None]:
        """Handle a streaming response and yield raw chunks."""
        for chunk in response:
            yield chunk

    async def stream_response_async(self, response: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
        """Handle an async streaming response and yield raw chunks."""
        async for chunk in response:
            yield chunk

    def check_max_tokens_reached(self, response: Any) -> bool:
        """Check if max tokens was reached and return True if so."""
        if hasattr(response, 'choices') and response.choices:
            if response.choices[0].finish_reason == "length":
                return True
        return False

    def parse_stream_chunk(self, chunk: Any) -> Tuple[Optional[str], Optional[List[Dict]], Any]:
        """Parse a streaming chunk into (text, tool_calls, raw_chunk)."""
        text = None
        tool_calls = None
        
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta') and choice.delta:
                # Handle text content
                if hasattr(choice.delta, 'content') and choice.delta.content:
                    text = choice.delta.content
                
                # Handle tool calls (defer to accumulation logic)
                if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                    tool_calls = None  # Defer to accumulation logic
        
        return text, tool_calls, chunk

    def accumulate_streaming_response(self, response: Generator[Any, None, None]) -> Generator[Tuple[Optional[str], Optional[List[Dict]], Any], None, None]:
        """Handle streaming response with tool call accumulation for Llama models."""
        accumulated_tool_calls = {}
        
        for chunk in self.stream_response(response):
            text = None
            tool_calls = None
            
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and choice.delta:
                    # Handle text content
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        text = choice.delta.content
                    
                    # Handle tool calls accumulation
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
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
                            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                                accumulated_tool_calls[index]["id"] = tool_call_delta.id
                            
                            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                    accumulated_tool_calls[index]["function"]["name"] = tool_call_delta.function.name
                                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                                    accumulated_tool_calls[index]["function"]["arguments"] += tool_call_delta.function.arguments
                
                # Check if we have complete tool calls
                if hasattr(choice, 'finish_reason') and choice.finish_reason == "tool_calls":
                    # Tool calls are complete, parse arguments
                    tool_calls = []
                    for tool_call in accumulated_tool_calls.values():
                        try:
                            parsed_args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
                            tool_calls.append({
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["function"]["name"],
                                    "arguments": parsed_args
                                }
                            })
                        except json.JSONDecodeError:
                            # Keep original if parsing fails
                            tool_calls.append(tool_call)
            
            yield text, tool_calls, chunk

    async def accumulate_streaming_response_async(self, response: AsyncGenerator[Any, None]) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict]], Any], None]:
        """Handle async streaming response with tool call accumulation for Llama models."""
        accumulated_tool_calls = {}
        
        async for chunk in self.stream_response_async(response):
            text = None
            tool_calls = None
            
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and choice.delta:
                    # Handle text content
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        text = choice.delta.content
                    
                    # Handle tool calls accumulation (same logic as sync)
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
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
                            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                                accumulated_tool_calls[index]["id"] = tool_call_delta.id
                            
                            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                    accumulated_tool_calls[index]["function"]["name"] = tool_call_delta.function.name
                                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                                    accumulated_tool_calls[index]["function"]["arguments"] += tool_call_delta.function.arguments
                
                # Check if we have complete tool calls
                if hasattr(choice, 'finish_reason') and choice.finish_reason == "tool_calls":
                    # Tool calls are complete, parse arguments
                    tool_calls = []
                    for tool_call in accumulated_tool_calls.values():
                        try:
                            parsed_args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
                            tool_calls.append({
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["function"]["name"],
                                    "arguments": parsed_args
                                }
                            })
                        except json.JSONDecodeError:
                            # Keep original if parsing fails
                            tool_calls.append(tool_call)
            
            yield text, tool_calls, chunk

    def build_assistant_message(self, text: Optional[str], tool_calls: List[Dict], original_response: Any = None) -> Dict:
        """Build an assistant message with tool calls for Llama models (OpenAI-compatible format)."""
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
        """Build tool result messages for Llama models (OpenAI-compatible format)."""
        messages = []
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": str(result["output"])
            })
        return messages
