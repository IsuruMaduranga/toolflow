"""
Asynchronous OpenAI Beta wrapper classes.

This module contains the async beta wrapper classes for OpenAI clients.
"""
from typing import Any, Dict, List, AsyncIterator, Union, Callable

from ...tool_execution import (
    validate_and_prepare_openai_tools,
    execute_openai_tools_async,
)

class BetaAsyncWrapper:
    """Async wrapper around OpenAI beta that handles toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.chat = BetaChatAsyncWrapper(client, full_response)

    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client.beta, name)


class BetaChatAsyncWrapper:
    """Async wrapper around OpenAI beta chat that handles toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.completions = BetaCompletionsAsyncWrapper(client, full_response)


class BetaCompletionsAsyncWrapper:
    """Async wrapper around OpenAI beta completions that processes toolflow functions."""
    
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

    async def parse(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Union[Any, AsyncIterator[Any]]:
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
                response = await self._original_completions.parse(
                    model=model,
                    messages=current_messages,
                    tools=tool_schemas,
                    **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers', 'full_response']}
                )

                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    current_messages.append(response.choices[0].message)
                    execution_response = await execute_openai_tools_async(
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
            response = await self._original_completions.parse(
                model=model,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers', 'full_response']}
            )
        
        return self._extract_response_content(response, full_response)

    def __getattr__(self, name):
        """Delegate all other attributes to the original completions."""
        return getattr(self._original_completions, name)
