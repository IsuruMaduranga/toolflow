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
     max_workers,
     response_format,
     full_response) = filter_toolflow_params(**kwargs)
    
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
        
        tool_results = execute_tools(tool_calls, tool_map, max_workers if parallel_tool_execution else 1)
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
     max_workers,
     response_format,
     full_response) = filter_toolflow_params(**kwargs)
    
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
        
        tool_results = await execute_tools_async(tool_calls, tool_map, max_workers if parallel_tool_execution else 1)
        kwargs["messages"].extend(handler.create_tool_result_messages(tool_results))

    raise Exception("Max tool calls reached without a valid response")

def sync_streaming_execution_loop(
    handler: AbstractProviderHandler,
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Synchronous streaming execution loop."""
    (kwargs,
     max_tool_calls,
     parallel_tool_execution,   
     max_workers,
     response_format,
     full_response) = filter_toolflow_params(**kwargs)
    
    response = handler.call_api(stream=True, **kwargs)
    
    text_so_far = ""
    for text, tool_calls, chunk in handler.handle_streaming_response(response):
        if text:
            text_so_far += text
        if tool_calls:
            # Streaming with tool calls is not yet supported in the core loop
            # This part can be enhanced later
            pass
        
        yield chunk if kwargs.get("full_response") else text

def async_streaming_execution_loop(
    handler: AbstractProviderHandler,
    **kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """Asynchronous streaming execution loop."""
    async def internal_generator():
        (kwargs,
         max_tool_calls,
         parallel_tool_execution,   
         max_workers,
         response_format,
         full_response) = filter_toolflow_params(**kwargs)
        
        response = await handler.call_api_async(stream=True, **kwargs)
        
        text_so_far = ""
        async for text, tool_calls, chunk in handler.handle_streaming_response_async(response):
            if text:
                text_so_far += text
            if tool_calls:
                 # Streaming with tool calls is not yet supported in the core loop
                pass

            yield chunk if kwargs.get("full_response") else text
                
    return internal_generator()
