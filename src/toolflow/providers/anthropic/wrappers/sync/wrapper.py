"""
Anthropic provider wrapper implementation.

This module provides the working implementation for Anthropic tool calling support.
"""
from typing import Any, Dict, List, Callable, Union, Iterator, Optional, Iterable, Literal

# Import Anthropic types for proper parameter typing
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolUnionParam,
    ToolChoiceParam,
    MetadataParam,
    ThinkingConfigParam,
    ModelParam,
    Message,
    RawMessageStreamEvent
)
from anthropic._types import NOT_GIVEN, NotGiven, Headers, Query, Body
import httpx

from ...tool_execution import (
    validate_and_prepare_anthropic_tools,
    execute_anthropic_tools_sync,
    format_anthropic_tool_calls_for_messages
)
from ...streaming import accumulate_anthropic_streaming_content, should_yield_chunk
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
            if hasattr(content_block, 'type'):
                if content_block.type == 'text':
                    text_content += content_block.text
                elif content_block.type == 'thinking':
                    text_content += f"\n<THINKING>\n{content_block.thinking}\n</THINKING>\n\n"
        return text_content

    def create(
        self,
        *,
        # Required parameters
        messages: Iterable[MessageParam],
        model: ModelParam,
        
        # max_tokens with default for backward compatibility
        max_tokens: int = 1024,
        
        # Anthropic API parameters (in alphabetical order for consistency)
        metadata: MetadataParam | NotGiven = NOT_GIVEN,
        stop_sequences: List[str] | NotGiven = NOT_GIVEN,
        stream: bool | NotGiven = NOT_GIVEN,
        system: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        thinking: ThinkingConfigParam | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoiceParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ToolUnionParam] | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        
        # Toolflow-specific parameters
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 10,
        max_workers: int = 10,
        graceful_error_handling: bool = True,
        full_response: Optional[bool] = None,
        response_format: Optional[Any] = None,
    ) -> Union[Message, Iterator[RawMessageStreamEvent], Any]:
        """
        Send a structured list of input messages with text and/or image content, and the
        model will generate the next message in the conversation.

        The Messages API can be used for either single queries or stateless multi-turn
        conversations.

        Args:
            max_tokens: The maximum number of tokens to generate before stopping.
                Note that our models may stop before reaching this maximum. This parameter
                only specifies the absolute maximum number of tokens to generate.
            messages: Input messages.
                Our models are trained to operate on alternating `user` and `assistant`
                conversational turns. When creating a new `Message`, you specify the prior
                conversational turns with the `messages` parameter, and the model then generates
                the next `Message` in the conversation.
            model: The model that will complete your prompt.
                See models for additional details and options.
            metadata: An object describing metadata about the request.
            stop_sequences: Custom text sequences that will cause the model to stop generating.
                Our models will normally stop when they have naturally completed their turn,
                which will result in a response `stop_reason` of `"end_turn"`.
            stream: Whether to incrementally stream the response using server-sent events.
            system: System prompt.
                A system prompt is a way of providing context and instructions to Claude, such
                as specifying a particular goal or role.
            temperature: Amount of randomness injected into the response.
                Defaults to `1.0`. Ranges from `0.0` to `1.0`. Use `temperature` closer to `0.0`
                for analytical / multiple choice, and closer to `1.0` for creative and
                generative tasks.
            thinking: Configuration for enabling Claude's extended thinking.
                When enabled, responses include `thinking` content blocks showing Claude's
                thinking process before the final answer.
            tool_choice: How the model should use the provided tools. The model can use a specific tool,
                any available tool, decide by itself, or not use tools at all.
            tools: Definitions of tools that the model may use.
                If you include `tools` in your API request, the model may return `tool_use`
                content blocks that represent the model's use of those tools.
            top_k: Only sample from the top K options for each subsequent token.
            top_p: Use nucleus sampling.
            extra_headers: Send extra headers
            extra_query: Add additional query parameters to the request
            extra_body: Add additional JSON properties to the request
            timeout: Override the client-level default timeout for this request
            parallel_tool_execution: Whether to execute tools in parallel
            max_tool_calls: Maximum number of tool calls allowed
            max_workers: Maximum number of workers for parallel execution
            graceful_error_handling: Whether to handle tool errors gracefully
            full_response: Whether to return full response (overrides client setting)
            response_format: Pydantic model for structured output
        
        Returns:
            Anthropic Message response or Iterator for streaming, potentially with tool results
        """
        # Build kwargs dict for Anthropic API call, excluding toolflow-specific params
        anthropic_kwargs = {}
        
        # Check for interleaved thinking beta header
        if extra_headers and 'anthropic-beta' in extra_headers:
            # Check if interleaved thinking is requested
            beta_header = extra_headers['anthropic-beta']
            if isinstance(beta_header, str) and 'interleaved-thinking' in beta_header:
                # Interleaved thinking is supported, pass through the header
                pass
        
        # Add all Anthropic parameters that are not NOT_GIVEN
        if metadata is not NOT_GIVEN:
            anthropic_kwargs['metadata'] = metadata
        if stop_sequences is not NOT_GIVEN:
            anthropic_kwargs['stop_sequences'] = stop_sequences
        if stream is not NOT_GIVEN:
            anthropic_kwargs['stream'] = stream
        if system is not NOT_GIVEN:
            anthropic_kwargs['system'] = system
        if temperature is not NOT_GIVEN:
            anthropic_kwargs['temperature'] = temperature
        if thinking is not NOT_GIVEN:
            anthropic_kwargs['thinking'] = thinking
        if tool_choice is not NOT_GIVEN:
            anthropic_kwargs['tool_choice'] = tool_choice
        if top_k is not NOT_GIVEN:
            anthropic_kwargs['top_k'] = top_k
        if top_p is not NOT_GIVEN:
            anthropic_kwargs['top_p'] = top_p
        if extra_headers is not None:
            anthropic_kwargs['extra_headers'] = extra_headers
        if extra_query is not None:
            anthropic_kwargs['extra_query'] = extra_query
        if extra_body is not None:
            anthropic_kwargs['extra_body'] = extra_body
        if timeout is not NOT_GIVEN:
            anthropic_kwargs['timeout'] = timeout

        # Use method-level full_response if provided, otherwise use client-level setting
        effective_full_response = full_response if full_response is not None else self._full_response
        
        # Determine if streaming is enabled
        is_streaming = stream is not NOT_GIVEN and stream
        
        # Handle toolflow tools
        toolflow_tools = None
        if tools is not NOT_GIVEN:
            toolflow_tools = list(tools)

        if response_format:
            if is_streaming:
                raise ValueError("response_format is not supported for streaming")
            
            validate_response_format(response_format)
            # Create a dynamic response tool
            response_tool = create_anthropic_response_tool(response_format)

            toolflow_tools = toolflow_tools or []
            toolflow_tools.append(response_tool)
 
        # Handle streaming
        if is_streaming:
            return self._create_streaming(
                max_tokens=max_tokens,
                model=model,
                messages=messages,
                tools=toolflow_tools,
                parallel_tool_execution=parallel_tool_execution,
                max_tool_calls=max_tool_calls,
                max_workers=max_workers,
                graceful_error_handling=graceful_error_handling,
                full_response=effective_full_response,
                **anthropic_kwargs
            )
        
        if toolflow_tools is None:
            # No tools, direct API call
            response = self._original_messages.create(
                max_tokens=max_tokens,
                model=model,
                messages=messages,
                **anthropic_kwargs
            )
            return self._extract_response_content(response, effective_full_response)
        
       
        # If tools are provided, handle tool execution
        tool_call_count = 0
        tool_functions, tool_schemas = validate_and_prepare_anthropic_tools(toolflow_tools)
        current_messages = list(messages)
        
        final_response = ""
        while tool_call_count < max_tool_calls:
            
            # Make API call 
            response = self._original_messages.create(
                max_tokens=max_tokens,
                model=model,
                messages=current_messages,
                tools=tool_schemas,
                **anthropic_kwargs
            )

            # Handle response extraction based on full_response setting
            if effective_full_response:
                # For full_response mode, return the response directly when tools complete
                extracted_content = response
            else:
                # For non-full response mode, accumulate text content
                extracted_content = self._extract_response_content(response, effective_full_response)
                final_response = final_response + extracted_content

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
                    return self._extract_response_content(structured_response, effective_full_response, is_structured=True)
            
            if not tool_calls:
                # No tool calls, return appropriate response
                if effective_full_response:
                    return extracted_content
                else:
                    return final_response
            
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
        max_tokens: int,
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
            
            # Remove stream from kwargs to avoid duplicate parameter
            streaming_kwargs = {k: v for k, v in kwargs.items() if k != 'stream'}
            
            while True:
                if remaining_tool_calls <= 0:
                    raise Exception(f"Max tool calls reached ({max_tool_calls})")
                
                if tools:
                    tool_functions, tool_schemas = validate_and_prepare_anthropic_tools(tools)
                    
                    # Make streaming API call with tools
                    stream = self._original_messages.create(
                        max_tokens=max_tokens,
                        model=model,
                        messages=current_messages,
                        tools=tool_schemas,
                        stream=True,
                        **streaming_kwargs
                    )
                else:
                    # Make streaming API call without tools
                    stream = self._original_messages.create(
                        max_tokens=max_tokens,
                        model=model,
                        messages=current_messages,
                        stream=True,
                        **streaming_kwargs
                    )
                
                # Accumulate streaming content
                message_content = []
                accumulated_tool_calls = []
                accumulated_json_strings = {}
                block_types = {}  # Track block types for proper thinking tag handling
                
                for chunk in stream:
                    if return_full_response:
                        yield chunk
                    else:
                        # Accumulate content and detect tool calls
                        accumulate_anthropic_streaming_content(
                            chunk=chunk,
                            message_content=message_content,
                            accumulated_tool_calls=accumulated_tool_calls,
                            accumulated_json_strings=accumulated_json_strings,  
                            graceful_error_handling=graceful_error_handling
                        )
                        
                        # Yield text content if not full response
                        should_yield, content = should_yield_chunk(chunk, return_full_response, block_types)
                        if should_yield and content:
                            yield content
                
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
