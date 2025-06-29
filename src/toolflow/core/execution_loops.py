# src/toolflow/core/execution_loops.py
from typing import List, Dict, Any, Generator, AsyncGenerator
from .handlers import AbstractProviderHandler
from .tool_execution import execute_tools, execute_tools_async
from .utils import filter_toolflow_params, prepare_tool_schemas, prepare_response_format

def sync_execution_loop(
    handler: AbstractProviderHandler,
    max_tool_calls: int = 5,
    parallel_tool_execution: bool = False,
    max_workers: int | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous execution loop for tool calling."""
    tools = kwargs.get("tools", [])
    messages = kwargs.get("messages", [])
    response_format = kwargs.get("response_format", None)
    tool_schemas, tool_map, pydantic_model = prepare_tool_schemas(tools, handler)
    
    for _ in range(max_tool_calls):
        response = handler.call_api(messages=messages, tools=tool_schemas, **kwargs)
        text, tool_calls, raw_response = handler.handle_response(response)

        if not tool_calls:
            if pydantic_model and text:
                return pydantic_model.model_validate_json(text)
            return raw_response if kwargs.get("full_response") else text

        tool_results = execute_tools(tool_calls, tool_map, max_workers if parallel_tool_execution else 1)
        messages.append(handler.create_tool_result_message(tool_results))

    # Final call after max tool calls
    response = handler.call_api(messages=messages, tools=tool_schemas, **kwargs)
    text, _, raw_response = handler.handle_response(response)
    if pydantic_model and text:
        return pydantic_model.model_validate_json(text)
    return raw_response if kwargs.get("full_response") else text

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

    tool_schemas, tool_map = prepare_tool_schemas(tools, handler)
    response_format_tool = prepare_response_format(response_format, handler)
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
                    # Need to implement a retry mechanism here if the response is not valid
                    return response_format.model_validate_json(text)

        tool_results = await execute_tools_async(tool_calls, tool_map, max_workers if parallel_tool_execution else 1)
        kwargs["messages"].append(handler.create_tool_result_message(tool_results))

def sync_streaming_execution_loop(
    handler: AbstractProviderHandler,
    messages: List[Dict],
    tools: List[Any],
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Synchronous streaming execution loop."""
    tool_schemas, tool_map, pydantic_model = prepare_tool_schemas(tools, handler)
    
    response = handler.call_api(messages=messages, tools=tool_schemas, stream=True, **kwargs)
    
    text_so_far = ""
    for text, tool_calls, chunk in handler.handle_streaming_response(response):
        if text:
            text_so_far += text
        if tool_calls:
            # Streaming with tool calls is not yet supported in the core loop
            # This part can be enhanced later
            pass
        
        if pydantic_model:
            yield chunk # Yield raw chunks for pydantic model streaming
        else:
             yield chunk if kwargs.get("full_response") else text

def async_streaming_execution_loop(
    handler: AbstractProviderHandler,
    messages: List[Dict],
    tools: List[Any],
    **kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """Asynchronous streaming execution loop."""
    async def internal_generator():
        tool_schemas, tool_map, pydantic_model = prepare_tool_schemas(tools, handler)
        
        response = await handler.call_api_async(messages=messages, tools=tool_schemas, stream=True, **kwargs)
        
        text_so_far = ""
        async for text, tool_calls, chunk in handler.handle_streaming_response_async(response):
            if text:
                text_so_far += text
            if tool_calls:
                 # Streaming with tool calls is not yet supported in the core loop
                pass

            if pydantic_model:
                yield chunk
            else:
                yield chunk if kwargs.get("full_response") else text
                
    return internal_generator()
