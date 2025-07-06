from __future__ import annotations
from typing import Any, Generator, AsyncGenerator, Union, List, Dict
import asyncio
from pydantic import ValidationError
from .adapters import TransportAdapter, MessageAdapter, ResponseFormatAdapter
from .utils import filter_toolflow_params, process_response_format
from .constants import RESPONSE_FORMAT_TOOL_NAME
from .tool_execution import execute_tools_sync, execute_tools_async
from .exceptions import MaxToolCallsError, MaxTokensError, ResponseFormatError

Handler = Union[TransportAdapter, MessageAdapter, ResponseFormatAdapter]

_ASYNC_YIELD_FREQUENCY = 0

def set_async_yield_frequency(frequency: int) -> None:
    global _ASYNC_YIELD_FREQUENCY
    if frequency < 0:
        raise ValueError("async_yield_frequency must be >= 0")
    _ASYNC_YIELD_FREQUENCY = frequency

def get_async_yield_frequency() -> int:
    return _ASYNC_YIELD_FREQUENCY

def _check_max_tokens(handler: Handler, response: Any):
    """Check if max tokens was reached, only if handler supports it."""
    if hasattr(handler, 'check_max_tokens_reached'):
        if handler.check_max_tokens_reached(response):
            raise MaxTokensError("Max tokens reached without finding a solution")

def _initialize_execution_context(handler: Handler, **kwargs: Any):
    kwargs, max_tool_call_rounds, max_response_format_retries, parallel_tool_execution, full_response, graceful_error_handling = filter_toolflow_params(**kwargs)
    tools = kwargs.get("tools", [])
    messages = kwargs.get("messages", [])
    response_format_tool_call_required = False

    if isinstance(handler, ResponseFormatAdapter):
        response_format = kwargs.pop("response_format", None)
        tools, response_format_tool_call_required = handler.prepare_response_format_tool(tools, response_format)

    # Only set tools in kwargs if there are actual tools
    if tools:
        kwargs["tools"] = tools

    return kwargs, tools, messages, max_tool_call_rounds, max_response_format_retries, parallel_tool_execution, full_response, graceful_error_handling, response_format_tool_call_required

def _create_error_message(max_response_format_retries: int, error_message: str) -> str:
    return f"""
    Failed to parse structured output after {max_response_format_retries} retries.

    {f"{error_message}" if error_message else ""}

    HINT:
    - If this is a data format issue, consider clearly documenting the response format parameters.
    - Or use properly parsable types like Pydantic models, dataclasses, or TypedDicts.
    
    """

def sync_execution_loop(handler: Handler, **kwargs: Any) -> Any:
    (
        kwargs, tools, messages,
        max_tool_call_rounds, max_response_format_retries,
        parallel_tool_execution, full_response,
        graceful_error_handling, response_format_tool_call_required
    ) = _initialize_execution_context(handler, **kwargs)

    if not tools:
        response = handler.call_api(**kwargs)
        text, _, raw_response = handler.parse_response(response)
        _check_max_tokens(handler, response)
        return raw_response if full_response else text
    
    # Test whether all tools are sync
    for tool in tools:
        if asyncio.iscoroutinefunction(tool):
            raise RuntimeError("Async tools are not supported in sync toolflow execution")

    tool_schemas, tool_map = handler.prepare_tool_schemas(tools)
    kwargs["tools"] = tool_schemas
    remaining_tool_calls = max_tool_call_rounds
    remaining_retry_count = max_response_format_retries

    while remaining_tool_calls > 0:
        response = handler.call_api(**kwargs)
        text, tool_calls, raw_response = handler.parse_response(response)
        _check_max_tokens(handler, response)

        if response_format_tool_call_required:
            result = process_response_format(handler, raw_response, text, tool_calls, tool_map, messages,)
            if result is not None:
                structured_response, should_continue = result
                if should_continue:
                    remaining_retry_count -= 1
                    if remaining_retry_count < 0:
                        raise ResponseFormatError(_create_error_message(max_response_format_retries, result[0]))
                    continue
                return raw_response if full_response else structured_response

        if not tool_calls:
            return raw_response if full_response else text

        messages.append(handler.build_assistant_message(text, tool_calls, raw_response))
        tool_results = execute_tools_sync(tool_calls, tool_map, parallel_tool_execution, graceful_error_handling)
        messages.extend(handler.build_tool_result_messages(tool_results))
        remaining_tool_calls -= 1

    raise MaxToolCallsError("Max tool calls reached before completing the response.", max_tool_call_rounds)

async def async_execution_loop(handler: Handler, **kwargs: Any) -> Any:
    (
        kwargs, tools, messages,
        max_tool_call_rounds, max_response_format_retries,
        parallel_tool_execution, full_response,
        graceful_error_handling, response_format_tool_call_required
    ) = _initialize_execution_context(handler, **kwargs)

    if not tools:
        response = await handler.call_api_async(**kwargs)
        text, _, raw_response = handler.parse_response(response)
        _check_max_tokens(handler, response)
        return raw_response if full_response else text

    tool_schemas, tool_map = handler.prepare_tool_schemas(tools)
    kwargs["tools"] = tool_schemas
    remaining_tool_calls = max_tool_call_rounds
    remaining_retry_count = max_response_format_retries

    while remaining_tool_calls > 0:
        response = await handler.call_api_async(**kwargs)
        text, tool_calls, raw_response = handler.parse_response(response)
        _check_max_tokens(handler, response)

        if response_format_tool_call_required:
            result = process_response_format(handler, raw_response, text, tool_calls, tool_map, messages)
            if result is not None:
                structured_response, should_continue = result
                if should_continue:
                    remaining_retry_count -= 1
                    if remaining_retry_count < 0:
                        raise ResponseFormatError(_create_error_message(max_response_format_retries, result[0]))
                    continue
                return raw_response if full_response else structured_response

        if not tool_calls:
            return raw_response if full_response else text

        messages.append(handler.build_assistant_message(text, tool_calls, raw_response))
        tool_results = await execute_tools_async(tool_calls, tool_map, graceful_error_handling, parallel_tool_execution)
        messages.extend(handler.build_tool_result_messages(tool_results))
        remaining_tool_calls -= 1

    raise MaxToolCallsError("Max tool calls reached before completing the response.", max_tool_call_rounds)

def sync_streaming_execution_loop(handler: Handler, **kwargs: Any) -> Generator[Any, None, None]:
    (
        kwargs, tools, messages,
        max_tool_call_rounds, _,
        parallel_tool_execution, full_response,
        graceful_error_handling, _
    ) = _initialize_execution_context(handler, **kwargs)

    if not tools:
        response = handler.call_api(**kwargs)
        for text, _, raw_chunk in handler.accumulate_streaming_response(response):
            if full_response:
                yield raw_chunk
            elif text:
                yield text
        return
    
    for tool in tools:
        if asyncio.iscoroutinefunction(tool):
            raise RuntimeError("Async tools are not supported in sync streaming execution")

    tool_schemas, tool_map = handler.prepare_tool_schemas(tools)
    kwargs["tools"] = tool_schemas
    remaining_tool_calls = max_tool_call_rounds

    while remaining_tool_calls > 0:
        response = handler.call_api(**kwargs)
        accumulated_content = ""
        accumulated_tool_calls = []

        for text, partial_tool_calls, raw_chunk in handler.accumulate_streaming_response(response):
            if text:
                accumulated_content += text
            if full_response:
                yield raw_chunk
            elif text:
                yield text
            if partial_tool_calls:
                accumulated_tool_calls.extend(partial_tool_calls)

        if accumulated_tool_calls:
            content = accumulated_content if accumulated_content else None
            messages.append(handler.build_assistant_message(content, accumulated_tool_calls, None))
            tool_results = execute_tools_sync(accumulated_tool_calls, tool_map, parallel_tool_execution, graceful_error_handling)
            messages.extend(handler.build_tool_result_messages(tool_results))
            kwargs["messages"] = messages
            remaining_tool_calls -= 1
            continue
        else:
            break

    if remaining_tool_calls <= 0:
        raise MaxToolCallsError("Max tool calls reached without a valid response", max_tool_call_rounds)

async def async_streaming_execution_loop(handler: Handler, **kwargs: Any) -> AsyncGenerator[Any, None]:
    (
        kwargs, tools, messages,
        max_tool_call_rounds, _,
        parallel_tool_execution, full_response,
        graceful_error_handling, _
    ) = _initialize_execution_context(handler, **kwargs)

    if not tools:
        chunk_count = 0
        response = await handler.call_api_async(**kwargs)
        async for text, _, raw_chunk in handler.accumulate_streaming_response_async(response):
            yield raw_chunk if full_response else text
            yield_freq = get_async_yield_frequency()
            if yield_freq > 0:
                chunk_count += 1
                if chunk_count % yield_freq == 0:
                    await asyncio.sleep(0)
        return

    tool_schemas, tool_map = handler.prepare_tool_schemas(tools)
    kwargs["tools"] = tool_schemas
    remaining_tool_calls = max_tool_call_rounds

    while remaining_tool_calls > 0:
        response = await handler.call_api_async(**kwargs)
        accumulated_content = ""
        accumulated_tool_calls = []
        chunk_count = 0

        async for text, partial_tool_calls, raw_chunk in handler.accumulate_streaming_response_async(response):
            if text:
                accumulated_content += text
            if full_response:
                yield raw_chunk
            elif text:
                yield text
            if partial_tool_calls:
                accumulated_tool_calls.extend(partial_tool_calls)

            yield_freq = get_async_yield_frequency()
            if yield_freq > 0:
                chunk_count += 1
                if chunk_count % yield_freq == 0:
                    await asyncio.sleep(0)

        if accumulated_tool_calls:
            content = accumulated_content if accumulated_content else None
            messages.append(handler.build_assistant_message(content, accumulated_tool_calls, None))
            tool_results = await execute_tools_async(accumulated_tool_calls, tool_map, graceful_error_handling, parallel_tool_execution)
            messages.extend(handler.build_tool_result_messages(tool_results))
            kwargs["messages"] = messages
            remaining_tool_calls -= 1
            continue
        else:
            break

    if remaining_tool_calls <= 0:
        raise MaxToolCallsError("Max tool calls reached without a valid response", max_tool_call_rounds)
