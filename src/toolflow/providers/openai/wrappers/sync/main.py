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
    
    def __init__(self, client):
        from .beta import BetaWrapper
        self._client = client
        self.chat = ChatWrapper(client)
        self.beta = BetaWrapper(client)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client, name)


class ChatWrapper:
    """Wrapper around OpenAI chat that handles toolflow functions."""
    
    def __init__(self, client):
        self._client = client
        self.completions = CompletionsWrapper(client)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original chat."""
        return getattr(self._client.chat, name)


class CompletionsWrapper:
    """Wrapper around OpenAI completions that processes toolflow functions."""
    
    def __init__(self, client):
        self._client = client
        self._original_completions = client.chat.completions

    def parse(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Union[Any, Iterator[Any]]:
        """Create a completion with structured output parsing."""
        tools = kwargs.get('tools', None)
        response_format = kwargs.get('response_format', None)
        stream = kwargs.get('stream', False)

        if stream and response_format:
            raise ValueError("response_format is not supported for streaming")
            
        # Dynamically add response_format to kwargs if it's a Pydantic model
        if response_format:
            validate_response_format(response_format)
            
            # Create a dynamic response tool
            response_tool = create_openai_response_tool(response_format)
            
            # Add the response tool to the tools list
            if tools is None:
                tools = []
            else:
                tools = list(tools)  # Make a copy to avoid modifying the original
            tools.append(response_tool)
        
        kwargs['tools'] = tools
        kwargs['handle_structured_response_internal'] = True
        return self.create(model, messages, **kwargs)

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Union[Any, Iterator[Any]]:
        """
        Create a chat completion with tool support.
        
        Args:
            tools: List of toolflow decorated functions or OpenAI tool dicts
            parallel_tool_execution: Whether to execute multiple tool calls in parallel (default: False)
            max_tool_calls: Maximum number of tool calls to execute
            max_workers: Maximum number of worker threads to use for parallel execution of sync tools
            stream: Whether to stream the response (default: False)
            **kwargs: All other OpenAI chat completion parameters
        
        Returns:
            OpenAI ChatCompletion response, potentially with tool results, or Iterator if stream=True
        """
        tools = kwargs.get('tools', None)
        parallel_tool_execution = kwargs.get('parallel_tool_execution', False)
        max_tool_calls = kwargs.get('max_tool_calls', 10)
        max_workers = kwargs.get('max_workers', 10)
        response_format = kwargs.get('response_format', None)
        stream = kwargs.get('stream', False)       

        # Handle streaming
        if stream:
            return self._create_streaming(
                model=model,
                messages=messages,
                tools=tools,
                parallel_tool_execution=parallel_tool_execution,
                max_tool_calls=max_tool_calls,
                max_workers=max_workers,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers']}
            )
        
        response = None
        if tools:
            tool_functions, tool_schemas = validate_and_prepare_openai_tools(tools)
            current_messages = messages.copy()
            
            # Tool execution loop
            while True:
                if max_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")
                
                if kwargs.get('handle_structured_response_internal', False):
                    response = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        tools=tool_schemas,
                        **{k: v for k, v in kwargs.items() if k not in [
                            'tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers', 'response_format', 'handle_structured_response_internal'
                        ]}
                    )
                    # Handle structured response
                    if response_format:
                        structured_response = handle_openai_structured_response(response, response_format)
                        if structured_response:
                            return structured_response
                else:
                    # Make the API call
                    response = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        tools=tool_schemas,
                        **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers']}
                    )

                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    current_messages.append(response.choices[0].message)
                    execution_response = execute_openai_tools_sync(
                        tool_functions, 
                        tool_calls, 
                        parallel_tool_execution,
                        max_workers=max_workers
                    )
                    max_tool_calls -= len(execution_response)
                    current_messages.extend(execution_response)
                else:
                    return response

        else: # No tools, just make the API call
            response = self._original_completions.create(
                model=model,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers']}
            )
        
        return response

    def _create_streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Callable] = None,
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 10,
        max_workers: int = 10,
        **kwargs
    ) -> Iterator[Any]:
        """
        Create a streaming chat completion with tool support.
        
        This method handles streaming responses while still supporting tool calls.
        When tool calls are detected in the stream, they are accumulated, executed,
        and then the conversation continues with a new streaming call.
        """
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
                        **kwargs
                    )
                else:
                    # Make streaming API call without tools
                    stream = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        **kwargs
                    )
                
                # Accumulate the streamed response and detect tool calls
                accumulated_content = ""
                accumulated_tool_calls = []
                message_dict = {"role": "assistant", "content": ""}
                
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        
                        # Yield the chunk for streaming output
                        yield chunk
                        
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
                            max_workers=max_workers
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
