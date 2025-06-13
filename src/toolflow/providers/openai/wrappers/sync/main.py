"""
Main synchronous OpenAI wrapper classes.

This module contains the core synchronous wrapper classes for OpenAI clients.
"""
from typing import Any, Dict, List, Callable, Iterator, Union

from ...tool_execution import (
    validate_and_prepare_openai_tools,
    execute_openai_tools_sync,
)
from ...streaming import (
    accumulate_openai_streaming_tool_calls,
    convert_accumulated_openai_tool_calls,
    format_openai_tool_calls_for_messages
)
from ...structured_output import (
    create_openai_response_tool,
    handle_openai_structured_response,
    validate_response_format
)
class OpenAIWrapper:
    """Wrapper around OpenAI client that supports tool-py functions."""
    
    def __init__(self, client, full_response: bool = False):
        from .beta import BetaWrapper
        self._client = client
        self._full_response = full_response
        self.chat = ChatWrapper(client, full_response)
        self.beta = BetaWrapper(client, full_response)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client, name)

class ChatWrapper:
    """Wrapper around OpenAI chat that handles toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.completions = CompletionsWrapper(client, full_response)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original chat."""
        return getattr(self._client.chat, name)


class CompletionsWrapper:
    """Wrapper around OpenAI completions that processes toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._original_completions = client.chat.completions
        self._full_response = full_response

    def _extract_response_content(self, response, full_response: bool, is_structured: bool = False):
        """Extract content from response based on full_response flag."""
        if full_response:
            return response
        
        if is_structured:
            return response.choices[0].message.parsed
        
        return response.choices[0].message.content

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
    ) -> Union[Any, Iterator[Any]]:
        """
        Create a chat completion with tool support.
        
        Args:
            tools: List of toolflow decorated functions or OpenAI tool dicts
            parallel_tool_execution: Whether to execute multiple tool calls in parallel (default: False)
            max_tool_calls: Maximum number of tool calls to execute
            max_workers: Maximum number of worker threads to use for parallel execution of sync tools
            graceful_error_handling: Whether to handle tool execution errors gracefully (default: True)
            stream: Whether to stream the response (default: False)
            **kwargs: All other OpenAI chat completion parameters
        
        Returns:
            OpenAI ChatCompletion response, potentially with tool results, or Iterator if stream=True
        """
        all_kwargs = kwargs.copy()
        # Use method-level full_response if provided, otherwise use client-level setting
        full_response = full_response if full_response is not None else self._full_response

        if response_format:
            if stream:
                raise ValueError("response_format is not supported for streaming")
            
            validate_response_format(response_format)
            # Create a dynamic response tool
            response_tool = create_openai_response_tool(response_format)

            tools = [] if not tools else list(tools)  # Make a copy to avoid modifying the original
            tools.append(response_tool)

        # Handle streaming
        if stream:
            return self._create_streaming(
                model=model,
                messages=messages,
                tools=tools,
                parallel_tool_execution=parallel_tool_execution,
                max_tool_calls=max_tool_calls,
                max_workers=max_workers,
                graceful_error_handling=graceful_error_handling,
                full_response=full_response,
                **all_kwargs
            )
        
        if tools is None:
            response = self._original_completions.create(
                model=model,
                messages=messages,
                **all_kwargs
            )
            return self._extract_response_content(response, full_response)
        
        # If tools are provided, handle tool execution
        tool_call_count = 0
        tool_functions, tool_schemas = validate_and_prepare_openai_tools(tools)
        current_messages = messages.copy()
        
        # Tool execution loop
        while tool_call_count < max_tool_calls:
            # Make the API call
            response = self._original_completions.create(
                model=model,
                messages=current_messages,
                tools=tool_schemas,
                **all_kwargs
            )

            if response.choices[0].finish_reason == "length":
                raise Exception("Max tokens reached without finding a solution")

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                # No tool calls, we're done
                return self._extract_response_content(response, full_response)
            
            # Handle structured response
            structured_response = handle_openai_structured_response(response, response_format)
            if structured_response:
                return self._extract_response_content(structured_response, full_response, is_structured=True)
            
            # Else we execute rest of the tools
            current_messages.append(response.choices[0].message)
            tool_results = execute_openai_tools_sync(
                tool_functions,
                tool_calls,
                parallel_tool_execution, 
                max_workers=max_workers,
                graceful_error_handling=graceful_error_handling
            )
            tool_call_count += len(tool_results)  
            current_messages.extend(tool_results)

        raise Exception("Max tool calls reached without finding a solution")

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
        Create a streaming chat completion with tool support.
        
        This method handles streaming responses while still supporting tool calls.
        When tool calls are detected in the stream, they are accumulated, executed,
        and then the conversation continues with a new streaming call.
        """
        if full_response is None:
            full_response = self._full_response
            
        def streaming_generator():
            current_messages = messages.copy()
            remaining_tool_calls = max_tool_calls
            
            while True:
                if remaining_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")
                
                if tools:
                    tool_functions, tool_schemas = validate_and_prepare_openai_tools(tools)
                    
                    # Make streaming API call with tools
                    stream = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        tools=tool_schemas,
                        stream=True,
                        **kwargs
                    )
                else:
                    # Make streaming API call without tools
                    stream = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        stream=True,
                        **kwargs
                    )
                
                # Accumulate the streamed response and detect tool calls
                accumulated_content = ""
                accumulated_tool_calls = []
                message_dict = {"role": "assistant", "content": ""}
                
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        
                        # Yield based on full_response flag
                        if full_response:
                            yield chunk
                        else:
                            # Only yield content if available
                            if delta.content:
                                yield delta.content
                        
                        # Accumulate content
                        if delta.content:
                            accumulated_content += delta.content
                            message_dict["content"] += delta.content
                        
                        # Accumulate tool calls
                        if delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                accumulate_openai_streaming_tool_calls(accumulated_tool_calls, tool_call_delta)
                
                # Check if we have tool calls to execute
                if accumulated_tool_calls and tools:
                    tool_calls = convert_accumulated_openai_tool_calls(accumulated_tool_calls)
                    
                    if tool_calls:
                        # Add assistant message with tool calls
                        message_dict["tool_calls"] = format_openai_tool_calls_for_messages(tool_calls)
                        current_messages.append(message_dict)
                        
                        # Execute tools
                        execution_response = execute_openai_tools_sync(
                            tool_functions,
                            tool_calls,
                            parallel_tool_execution,
                            max_workers=max_workers,
                            graceful_error_handling=graceful_error_handling
                        )
                        remaining_tool_calls -= len(execution_response)
                        current_messages.extend(execution_response)
                        
                        # Continue the loop to get the next response
                        continue
                
                # No tool calls, we're done
                break
        
        return streaming_generator()
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original completions."""
        return getattr(self._original_completions, name)

    def _handle_structured_response_tool_calls(self, response, tool_calls, response_format, full_response):
        """
        Handle tool calls when structured response format is specified.
        
        Returns:
            - The structured response if final_response_tool_internal is called alone
            - None if regular tool execution should continue
            
        Raises:
            Exception: If model calls final_response_tool_internal with other tools
        """
        final_response_tool_calls = [
            tc for tc in tool_calls 
            if tc.function.name == "final_response_tool_internal"
        ]
        
        if not final_response_tool_calls:
            # No final response tool, continue with regular tool execution
            return None
            
        if len(final_response_tool_calls) > 1:
            raise Exception("Model called final_response_tool_internal multiple times")
            
        if len(tool_calls) > 1:
            raise Exception(
                "Model called final_response_tool_internal along with other tools. "
                f"Expected only final_response_tool_internal, but got {len(tool_calls)} tools: "
                f"{[tc.function.name for tc in tool_calls]}"
            )
        
        # Single final_response_tool_internal call - handle structured response
        response = handle_openai_structured_response(response, response_format)
        return self._extract_response_content(response, full_response, is_structured=True)
