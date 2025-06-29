# src/toolflow/core/execution_loops.py
from typing import List, Dict, Any, Generator, AsyncGenerator
from .handlers import AbstractProviderHandler
from .tool_execution import execute_tools, execute_tools_async
from .utils import filter_toolflow_params

def sync_execution_loop(
    handler: AbstractProviderHandler,
    **kwargs: Any,
) -> Any:
    """Synchronous execution loop for tool calling."""

    (kwargs,
     max_tool_calls,
     parallel_tool_execution,
     response_format,
     full_response,
     graceful_error_handling) = filter_toolflow_params(**kwargs)
    
    tools = kwargs.get("tools", [])
    messages = kwargs.get("messages", [])
    
    if not tools:
        response = handler.call_api(messages=messages, **kwargs)
        text, _, raw_response = handler.handle_response(response)
        return raw_response if full_response else text

    tool_schemas, tool_map = handler.prepare_tool_schemas(tools)
    response_format_tool = handler.prepare_response_format(response_format)
    if response_format_tool:
        tool_schemas.append(response_format_tool)
    
    kwargs["tools"] = tool_schemas
    for _ in range(max_tool_calls):
        response = handler.call_api(**kwargs)
        text, tool_calls, raw_response = handler.handle_response(response)

        if not tool_calls:
            return raw_response if full_response else text
        
        if response_format_tool:
            for tool_call in tool_calls:
                if tool_call["function"]["name"] == response_format_tool["function"]["name"]:
                    # Extract the structured data from the tool call arguments
                    parsed = handler.parse_structured_output(tool_call, response_format)
                    return raw_response if full_response else parsed

        # Add assistant message with tool calls to conversation
        messages.append(handler.create_assistant_message(text, tool_calls))
        
        tool_results = execute_tools(tool_calls, tool_map, parallel_tool_execution, graceful_error_handling)
        messages.extend(handler.create_tool_result_messages(tool_results))

    raise Exception("Max tool calls reached without a valid response")

async def async_execution_loop(
    handler: AbstractProviderHandler,
    **kwargs: Any,
) -> Any:
    """Asynchronous execution loop for tool calling."""

    (kwargs,
     max_tool_calls,
     parallel_tool_execution,
     response_format,
     full_response,
     graceful_error_handling) = filter_toolflow_params(**kwargs)
    
    tools = kwargs.get("tools", [])
    if not tools:
        response = await handler.call_api_async(**kwargs)
        text, _, raw_response = handler.handle_response(response)
        return raw_response if full_response else text

    tool_schemas, tool_map = handler.prepare_tool_schemas(tools)
    response_format_tool = handler.prepare_response_format(response_format)
    if response_format_tool:
        tool_schemas.append(response_format_tool)
    
    kwargs["tools"] = tool_schemas
    for _ in range(max_tool_calls):
        response = await handler.call_api_async(**kwargs)
        text, tool_calls, raw_response = handler.handle_response(response)

        if not tool_calls:
            return raw_response if full_response else text
        
        if response_format_tool:
            for tool_call in tool_calls:
                if tool_call["function"]["name"] == response_format_tool["function"]["name"]:
                    # Extract the structured data from the tool call arguments
                    parsed = handler.parse_structured_output(tool_call, response_format)
                    return raw_response if full_response else parsed

        # Add assistant message with tool calls to conversation
        kwargs["messages"].append(handler.create_assistant_message(text, tool_calls))
        
        tool_results = await execute_tools_async(tool_calls, tool_map, graceful_error_handling)
        kwargs["messages"].extend(handler.create_tool_result_messages(tool_results))

    raise Exception("Max tool calls reached without a valid response")

def sync_streaming_execution_loop(
    handler: AbstractProviderHandler,
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Synchronous streaming execution loop with tool calling support."""
    (kwargs,
     max_tool_calls,
     parallel_tool_execution,
     response_format,
     full_response,
     graceful_error_handling) = filter_toolflow_params(**kwargs)
    
    tools = kwargs.get("tools", [])
    messages = kwargs.get("messages", [])
    
    # If no tools, just do simple streaming
    if not tools:
        response = handler.call_api(**kwargs)
        for text, _, chunk in handler.handle_streaming_response(response):
            if full_response:
                yield chunk
            elif text:
                yield text
        return
    
    # Prepare tools for tool calling
    tool_schemas, tool_map = handler.prepare_tool_schemas(tools)
    response_format_tool = handler.prepare_response_format(response_format)
    if response_format_tool:
        tool_schemas.append(response_format_tool)
    
    kwargs["tools"] = tool_schemas
    remaining_tool_calls = max_tool_calls
    
    while remaining_tool_calls > 0:
        # Stream the response
        response = handler.call_api(**kwargs)
        
        accumulated_content = ""
        accumulated_tool_calls = []
        
        # Process the streaming response
        for text, partial_tool_calls, chunk in handler.handle_streaming_response(response):
            # Accumulate content for tool calls
            if text:
                accumulated_content += text
            
            # Yield based on full_response setting
            if full_response:
                yield chunk
            elif text:
                yield text
            
            # Accumulate tool calls (they come at the end of streaming)
            if partial_tool_calls:
                accumulated_tool_calls.extend(partial_tool_calls)
        
        # Check if we have tool calls to execute
        if accumulated_tool_calls:
            if response_format_tool:
                # Handle structured output
                for tool_call in accumulated_tool_calls:
                    if tool_call["function"]["name"] == response_format_tool["function"]["name"]:
                        parsed = handler.parse_structured_output(tool_call, response_format)
                        # For structured output, we don't continue streaming
                        return parsed if not full_response else {"parsed": parsed}
            
            # Add assistant message with tool calls to conversation
            messages.append(handler.create_assistant_message(accumulated_content, accumulated_tool_calls))
            
            # Execute tools
            tool_results = execute_tools(accumulated_tool_calls, tool_map, parallel_tool_execution, graceful_error_handling)
            messages.extend(handler.create_tool_result_messages(tool_results))
            
            # Update kwargs with new messages for next iteration
            kwargs["messages"] = messages
            remaining_tool_calls -= 1
            
            # Continue the loop for next streaming response
            continue
        else:
            # No tool calls, streaming is complete
            break
    
    if remaining_tool_calls <= 0:
        raise Exception("Max tool calls reached without a valid response")

def async_streaming_execution_loop(
    handler: AbstractProviderHandler,
    **kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """Asynchronous streaming execution loop with tool calling support."""
    (kwargs,
    max_tool_calls,
    parallel_tool_execution, # We ignore this in async case
    response_format,
    full_response,
    graceful_error_handling) = filter_toolflow_params(**kwargs)

    async def internal_generator():
        
        tools = kwargs.get("tools", [])
        messages = kwargs.get("messages", [])
        
        # If no tools, just do simple streaming
        if not tools:
            response = await handler.call_api_async(**kwargs)
            async for text, _, chunk in handler.handle_streaming_response_async(response):
                yield chunk if full_response else text
            return
        
        # Prepare tools for tool calling
        tool_schemas, tool_map = handler.prepare_tool_schemas(tools)
        response_format_tool = handler.prepare_response_format(response_format)
        if response_format_tool:
            tool_schemas.append(response_format_tool)
        
        kwargs["tools"] = tool_schemas
        remaining_tool_calls = max_tool_calls
        
        while remaining_tool_calls > 0:
            # Stream the response
            response = await handler.call_api_async(**kwargs)
            
            accumulated_content = ""
            accumulated_tool_calls = []
            
            # Process the streaming response
            async for text, partial_tool_calls, chunk in handler.handle_streaming_response_async(response):
                # Accumulate content for tool calls
                if text:
                    accumulated_content += text
                
                # Yield based on full_response setting
                if full_response:
                    yield chunk
                elif text:
                    yield text
                
                # Accumulate tool calls (they come at the end of streaming)
                if partial_tool_calls:
                    accumulated_tool_calls.extend(partial_tool_calls)
            
            # Check if we have tool calls to execute
            if accumulated_tool_calls:
                if response_format_tool:
                    # Handle structured output
                    for tool_call in accumulated_tool_calls:
                        if tool_call["function"]["name"] == response_format_tool["function"]["name"]:
                            parsed = handler.parse_structured_output(tool_call, response_format)
                            # For structured output, we don't continue streaming
                            yield parsed if not full_response else {"parsed": parsed}
                            return
                
                # Add assistant message with tool calls to conversation
                messages.append(handler.create_assistant_message(accumulated_content, accumulated_tool_calls))
                
                # Execute tools
                tool_results = await execute_tools_async(accumulated_tool_calls, tool_map, graceful_error_handling)
                messages.extend(handler.create_tool_result_messages(tool_results))
                
                # Update kwargs with new messages for next iteration
                kwargs["messages"] = messages
                remaining_tool_calls -= 1
                
                # Continue the loop for next streaming response
                continue
            else:
                # No tool calls, streaming is complete
                break
        
        if remaining_tool_calls <= 0:
            raise Exception("Max tool calls reached without a valid response")
    
    return internal_generator()
