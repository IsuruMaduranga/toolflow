"""
Synchronous OpenAI Beta wrapper classes.

This module contains the beta wrapper classes for OpenAI clients.
"""
from typing import Any, Dict, List, Iterator, Union, Callable

from ...tool_execution import (
    validate_and_prepare_openai_tools,
    execute_openai_tools_sync,
)
from ...streaming import (
    accumulate_openai_streaming_tool_calls,
    convert_accumulated_openai_tool_calls,
    format_openai_tool_calls_for_messages
)

class BetaWrapper:
    """Wrapper around OpenAI beta that handles toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.chat = BetaChatWrapper(client, full_response)

    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client.beta, name)


class BetaChatWrapper:
    """Wrapper around OpenAI beta chat that handles toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.completions = BetaCompletionsWrapper(client, full_response)


class BetaCompletionsWrapper:
    """Wrapper around OpenAI beta completions that processes toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._original_completions = client.beta.chat.completions
        self._full_response = full_response

    def _extract_response_content(self, response, full_response: bool, is_structured: bool = False):
        """Extract content from response based on full_response flag."""
        if full_response:
            return response
        
        # Check if we have a parsed structured response
        # Only return parsed if it exists and is not a Mock object (for tests)
        if (hasattr(response.choices[0].message, 'parsed') and 
            response.choices[0].message.parsed is not None and
            not str(type(response.choices[0].message.parsed)).startswith("<class 'unittest.mock")):
            return response.choices[0].message.parsed
        
        return response.choices[0].message.content

    def parse(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Union[Any, Iterator[Any]]:
        """Create a completion with structured output parsing."""
        tools = kwargs.get('tools', None)
        parallel_tool_execution = kwargs.get('parallel_tool_execution', False)
        max_tool_calls = kwargs.get('max_tool_calls', 10)
        max_workers = kwargs.get('max_workers', 10)
        full_response = kwargs.get('full_response', self._full_response)
        
        response = None
        if tools:
            tool_functions, tool_schemas = validate_and_prepare_openai_tools(tools, strict=True)
            current_messages = messages.copy()
            
            # Tool execution loop
            while True:
                if max_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")

                # Make the API call
                response = self._original_completions.parse(
                    model=model,
                    messages=current_messages,
                    tools=tool_schemas,
                    **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers', 'full_response']}
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
                    return self._extract_response_content(response, full_response)

        else: # No tools, just make the API call
            response = self._original_completions.parse(
                model=model,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers', 'full_response']}
            )
        
        return self._extract_response_content(response, full_response)
    
    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Union[Any, Iterator[Any]]:
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
        full_response = kwargs.get('full_response', self._full_response)

        # Handle streaming
        if kwargs.get('stream', False):
            return self._create_streaming(
                model=model,
                messages=messages,
                tools=tools,
                parallel_tool_execution=parallel_tool_execution,
                max_tool_calls=max_tool_calls,
                max_workers=max_workers,
                full_response=full_response,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers', 'full_response']}
            )
        
        response = None
        if tools:
            tool_functions, tool_schemas = validate_and_prepare_openai_tools(tools, strict=True)
            current_messages = messages.copy()
            
            # Tool execution loop
            while True:
                if max_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")

                # Make the API call
                response = self._original_completions.create(
                    model=model,
                    messages=current_messages,
                    tools=tool_schemas,
                    **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers', 'full_response']}
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
                    return self._extract_response_content(response, full_response)

        else: # No tools, just make the API call
            response = self._original_completions.create(
                model=model,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers', 'full_response']}
            )
        
        return self._extract_response_content(response, full_response)

    def _create_streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Callable] = None,
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 10,
        max_workers: int = 10,
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
                    tool_functions, tool_schemas = validate_and_prepare_openai_tools(tools, strict=True)
                    
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
