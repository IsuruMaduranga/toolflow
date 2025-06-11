"""
Anthropic provider wrapper implementation.

This module provides the working implementation for Anthropic tool calling support.
"""
from typing import Any, Dict, List, Callable, Union

from .tool_execution import (
    validate_and_prepare_anthropic_tools,
    execute_anthropic_tools_sync,
    convert_openai_messages_to_anthropic,
    extract_system_message,
    format_anthropic_tool_calls_for_messages
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

    def _extract_response_content(self, response, full_response: bool):
        """Extract content from response based on full_response flag."""
        if full_response:
            return response
        
        # For Anthropic, extract the text content from the first content block
        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    return content_block.text
                    
        return response

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Any:
        """
        Create a message completion with tool support.
        
        Args:
            tools: List of toolflow decorated functions
            max_tool_calls: Maximum number of tool calls to execute (default: 10)
            graceful_error_handling: Whether to handle tool execution errors gracefully (default: True)
            **kwargs: All other Anthropic message parameters
        
        Returns:
            Anthropic Message response, potentially with tool results
        """
        tools = kwargs.get('tools', None)
        max_tool_calls = kwargs.get('max_tool_calls', 10)
        graceful_error_handling = kwargs.get('graceful_error_handling', True)
        full_response = kwargs.get('full_response', self._full_response)

        response = None
        if tools:
            tool_functions, tool_schemas = validate_and_prepare_anthropic_tools(tools)
            
            # Convert OpenAI-style messages to Anthropic format
            anthropic_messages = convert_openai_messages_to_anthropic(messages)
            system_message = extract_system_message(messages)
            
            # Tool execution loop
            while True:
                if max_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")
                
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "messages": anthropic_messages,
                    "tools": tool_schemas,
                    **{k: v for k, v in kwargs.items() if k not in ['tools', 'max_tool_calls', 'graceful_error_handling', 'full_response']}
                }
                
                # Add system message if present
                if system_message:
                    request_params["system"] = system_message

                # Make the API call
                response = self._original_messages.create(**request_params)

                # Check for tool use in response
                tool_calls = []
                assistant_content = []
                
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_calls.append(content_block)
                    else:
                        assistant_content.append(content_block)

                if tool_calls:
                    # Add assistant's response with tool calls to conversation
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    # Execute tool calls
                    tool_results = execute_anthropic_tools_sync(
                        tool_functions,
                        tool_calls,
                        graceful_error_handling=graceful_error_handling
                    )
                    
                    # Add tool results to conversation
                    anthropic_messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                    
                    max_tool_calls -= len(tool_calls)
                else:
                    # No tool calls, we're done
                    return self._extract_response_content(response, full_response)

        else: # No tools, just make the API call
            # Convert OpenAI-style messages to Anthropic format
            anthropic_messages = convert_openai_messages_to_anthropic(messages)
            system_message = extract_system_message(messages)
            
            request_params = {
                "model": model,
                "messages": anthropic_messages,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'max_tool_calls', 'graceful_error_handling', 'full_response']}
            }
            
            # Add system message if present
            if system_message:
                request_params["system"] = system_message
                
            response = self._original_messages.create(**request_params)
        
        return self._extract_response_content(response, full_response)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original messages."""
        return getattr(self._original_messages, name)


class AnthropicAsyncWrapper:
    """Async wrapper around Anthropic client that supports tool-py functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.messages = MessagesAsyncWrapper(client, full_response)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client, name)


class MessagesAsyncWrapper:
    """Async wrapper around Anthropic messages that processes toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._original_messages = client.messages
        self._full_response = full_response

    async def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Any:
        """
        Create a message completion with tool support (async).
        
        This is a placeholder implementation for async Anthropic support.
        
        Args:
            tools: List of toolflow decorated functions
            **kwargs: All other Anthropic message parameters
        
        Returns:
            Anthropic Message response, potentially with tool results
        """
        tools = kwargs.get('tools', None)
        
        if tools:
            raise NotImplementedError(
                "Async Anthropic provider with tools not yet implemented. "
                "This would require async Anthropic-specific tool conversion and execution logic."
            )
        
        # For now, just pass through to original client without tools
        # Convert OpenAI-style messages to Anthropic format
        anthropic_messages = convert_openai_messages_to_anthropic(messages)
        system_message = extract_system_message(messages)
        
        request_params = {
            "model": model,
            "messages": anthropic_messages,
            **{k: v for k, v in kwargs.items() if k not in ['tools']}
        }
        
        # Add system message if present
        if system_message:
            request_params["system"] = system_message
            
        return await self._original_messages.create(**request_params)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original messages."""
        return getattr(self._original_messages, name)
