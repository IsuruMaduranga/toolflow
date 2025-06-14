"""
Anthropic provider wrapper implementation.

This module provides the working implementation for Anthropic tool calling support.
"""
from typing import Any, Dict, List, Callable, Union, Iterator

from ...tool_execution import (
    validate_and_prepare_anthropic_tools,
    execute_anthropic_tools_sync,
    format_anthropic_tool_calls_for_messages
)
from ...streaming import accumulate_anthropic_streaming_content
from ...structured_output import (
    create_anthropic_response_tool,
    handle_anthropic_structured_response,
    validate_response_format
)


class AnthropicWrapper:
    """Wrapper around Anthropic client that supports tool-py functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.messages = MessagesWrapper(client, full_response)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client, name)


class MessagesWrapper:
    """Wrapper around Anthropic messages that processes toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._original_messages = client.messages
        self._full_response = full_response

    def _extract_response_content(self, response, full_response: bool, is_structured: bool = False):
        """Extract content from response based on full_response flag."""
        if full_response:
            return response
        
        if is_structured:
            return response.parsed
        
        text_content = ""
        for content_block in response.content:
            if hasattr(content_block, 'text'):
                text_content += content_block.text
        return text_content

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Callable] = None,
        stream: bool = False,
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 10,
        max_workers: int = 10,
        graceful_error_handling: bool = True,
        full_response: bool = None,
        response_format = None,
        **kwargs
    ) -> Any:
        """
        Create a message completion with tool support.
        
        Args:
            model: The Anthropic model to use
            messages: List of message dictionaries
            tools: List of toolflow decorated functions
            stream: Whether to stream the response
            parallel_tool_execution: Whether to execute tools in parallel
            max_tool_calls: Maximum number of tool calls allowed
            max_workers: Maximum number of workers for parallel execution
            graceful_error_handling: Whether to handle tool errors gracefully
            full_response: Whether to return full response (overrides client setting)
            response_format: Pydantic model for structured output
            **kwargs: Additional Anthropic API parameters
        
        Returns:
            Anthropic Message response or simplified content
        """
        all_kwargs = kwargs.copy()
        # Determine response format
        return_full_response = full_response if full_response is not None else self._full_response
        
        if response_format:
            if stream:
                raise ValueError("response_format is not supported for streaming")
            
            validate_response_format(response_format)
            # Create a dynamic response tool
            response_tool = create_anthropic_response_tool(response_format)

            tools = [] if not tools else list(tools)  # Make a copy to avoid modifying the original
            tools.append(response_tool)
 
         # Tools provided, handle tool execution
        if stream:
            return self._create_streaming(
                model=model,
                messages=messages,
                tools=tools,
                parallel_tool_execution=parallel_tool_execution,
                max_tool_calls=max_tool_calls,
                max_workers=max_workers,
                graceful_error_handling=graceful_error_handling,
                full_response=return_full_response,
                **all_kwargs
            )
        
        
        if tools is None:
            # No tools, direct API call
            response = self._original_messages.create(
                model=model,
                messages=messages,
                **all_kwargs
            )
            return self._extract_response_content(response, return_full_response)
        
       
        # If tools are provided, handle tool execution
        tool_call_count = 0
        tool_functions, tool_schemas = validate_and_prepare_anthropic_tools(tools)
        current_messages = messages.copy()
        
        while tool_call_count < max_tool_calls:
            
            # Make API call 
            response = self._original_messages.create(
                model=model,
                messages=current_messages,
                tools=tool_schemas,
                **all_kwargs
            )

            if response.stop_reason == "max_tokens":
                raise Exception("Max tokens reached without finding a solution")
            
            # Check for tool calls in response
            structured_tool_call = None
            tool_calls = []
            if hasattr(response, 'content'):
                for content_block in response.content:
                    if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        if (response_format and 
                            hasattr(content_block, 'name') and 
                            content_block.name == "final_response_tool_internal"):
                            structured_tool_call = content_block
                        else:
                            tool_calls.append(content_block)
            
            # Handle structured response if found
            if structured_tool_call:
                structured_response = handle_anthropic_structured_response(response, response_format)
                if structured_response:
                    return self._extract_response_content(structured_response, return_full_response, is_structured=True)
            
            if not tool_calls:
                # No tool calls, return final response
                return self._extract_response_content(response, return_full_response)
            
            # Execute tools
            tool_results = execute_anthropic_tools_sync(
                tool_functions=tool_functions,
                tool_calls=tool_calls,
                parallel_tool_execution=parallel_tool_execution,
                max_workers=max_workers,
                graceful_error_handling=graceful_error_handling
            )
            
            tool_call_count += len(tool_results)
            # Add assistant message with tool calls
            assistant_message = {
                "role": "assistant",
                "content": response.content
            }
            current_messages.append(assistant_message)
            
            # Add tool results
            tool_result_message = format_anthropic_tool_calls_for_messages(tool_results)
            current_messages.append(tool_result_message)

        raise Exception(f"Max tool calls reached ({max_tool_calls})")

    def _create_streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Callable] = None,
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 10,
        max_workers: int = 10,
        graceful_error_handling: bool = True,
        full_response: bool = None,
        **kwargs
    ) -> Iterator[Any]:
        """
        Create a streaming message completion with tool support.
        
        This method handles streaming responses while still supporting tool calls.
        When tool calls are detected in the stream, they are accumulated, executed,
        and then the conversation continues with a new streaming call.
        """
        return_full_response = full_response if full_response is not None else self._full_response
        
        def streaming_generator():
            current_messages = messages.copy()
            remaining_tool_calls = max_tool_calls
            
            while True:
                if remaining_tool_calls <= 0:
                    raise Exception(f"Max tool calls reached ({max_tool_calls})")
                
                if tools:
                    tool_functions, tool_schemas = validate_and_prepare_anthropic_tools(tools)
                    
                    # Make streaming API call with tools
                    stream = self._original_messages.create(
                        model=model,
                        messages=current_messages,
                        tools=tool_schemas,
                        stream=True,
                        **kwargs
                    )
                else:
                    # Make streaming API call without tools
                    stream = self._original_messages.create(
                        model=model,
                        messages=current_messages,
                        stream=True,
                        **kwargs
                    )
                
                # Accumulate streaming content
                message_content = []
                accumulated_tool_calls = []
                accumulated_json_strings = {}
                
                for chunk in stream:
                    if return_full_response:
                        yield chunk
                    
                    # Accumulate content and detect tool calls
                    has_tool_calls = accumulate_anthropic_streaming_content(
                        chunk=chunk,
                        message_content=message_content,
                        accumulated_tool_calls=accumulated_tool_calls,
                        accumulated_json_strings=accumulated_json_strings,
                        graceful_error_handling=graceful_error_handling
                    )
                    
                    # Yield text content if not full response
                    if not return_full_response:
                        if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                            if (hasattr(chunk, 'delta') and 
                                hasattr(chunk.delta, 'type') and 
                                chunk.delta.type == 'text_delta' and
                                hasattr(chunk.delta, 'text')):
                                yield chunk.delta.text
                
                # Check if we have tool calls to execute
                if accumulated_tool_calls and tools:
                    # Execute tools
                    tool_results = execute_anthropic_tools_sync(
                        tool_functions=tool_functions,
                        tool_calls=accumulated_tool_calls,
                        parallel_tool_execution=parallel_tool_execution,
                        max_workers=max_workers,
                        graceful_error_handling=graceful_error_handling
                    )
                    
                    remaining_tool_calls -= len(tool_results)
                    
                    # Add assistant message with tool calls
                    assistant_message = {
                        "role": "assistant",
                        "content": message_content
                    }
                    current_messages.append(assistant_message)
                    
                    # Add tool results
                    tool_result_message = format_anthropic_tool_calls_for_messages(tool_results)
                    current_messages.append(tool_result_message)
                    
                    # Continue the loop to get the next response
                    continue
                
                # No tool calls, we're done
                break
        
        return streaming_generator()

    def __getattr__(self, name):
        """Delegate all other attributes to the original messages."""
        return getattr(self._original_messages, name)
