# src/toolflow/core/tool_execution.py
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Dict, Callable, Any, Coroutine, Optional, Union, get_origin, get_args, Tuple, get_type_hints
import inspect
from pydantic import TypeAdapter
from .constants import RESPONSE_FORMAT_TOOL_NAME

# ===== GLOBAL EXECUTOR (SHARED BY SYNC AND ASYNC) =====

_custom_executor: Optional[ThreadPoolExecutor] = None
_global_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_MAX_WORKERS = 4

def set_max_workers(max_workers: int) -> None:
    """Set the number of worker threads for the global executor."""
    global _global_executor
    global _MAX_WORKERS
    _MAX_WORKERS = max_workers
    with _executor_lock:
        if _global_executor:
            _global_executor.shutdown(wait=True)
            _global_executor = None

def get_max_workers() -> int:
    """Get the number of worker threads for the global executor."""
    return _MAX_WORKERS if _MAX_WORKERS else int(os.getenv("TOOLFLOW_SYNC_MAX_WORKERS", 4))

def set_executor(executor: ThreadPoolExecutor) -> None:
    """Set a custom global executor (used by both sync and async)."""
    global _global_executor
    global _custom_executor
    with _executor_lock:
        if _global_executor:
            _global_executor.shutdown(wait=True) 
        if _custom_executor:
            _custom_executor.shutdown(wait=True)
        _custom_executor = executor

def _get_sync_executor() -> ThreadPoolExecutor:
    """Get the executor for sync tool execution.
    Returns the custom executor if set, otherwise the global executor.
    """
    global _global_executor
    global _custom_executor
    
    with _executor_lock:
        if _global_executor is None and _custom_executor is None:
            max_workers = get_max_workers()
            _global_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="toolflow-"
            )
        result = _custom_executor if _custom_executor else _global_executor
        assert result is not None  # Should never be None due to logic above
        return result

def _get_async_executor() -> Optional[ThreadPoolExecutor]:
    """
    Get the executor for async tool execution.
    Returns the custom executor if set, otherwise None (uses asyncio's default).
    """
    with _executor_lock:
        return _custom_executor

# ===== TOOL EXECUTION FUNCTIONS =====
async def execute_tools_async(
    tool_calls: List[Dict[str, Any]],
    tool_map: Dict[str, Callable[..., Any]],
    graceful_error_handling: bool = True,
    parallel: bool = False
) -> List[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    results = []
    unknown_tool_results = []

    if parallel and len(tool_calls) > 1:
        all_tasks = []
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            if tool_name == RESPONSE_FORMAT_TOOL_NAME:
                continue
            tool_func = tool_map.get(tool_name)
            if not tool_func:
                if graceful_error_handling:
                    unknown_tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "output": f"Error: Unknown tool '{tool_name}' - tool not found in available tools",
                        "is_error": True,
                    })
                    continue
                else:
                    raise KeyError(f"Unknown tool: {tool_name}")

            if asyncio.iscoroutinefunction(tool_func):
                coro = run_async_tool(tool_call, tool_func, graceful_error_handling)
            else:
                coro = loop.run_in_executor(
                    _get_async_executor(), run_sync_tool, tool_call, tool_func, graceful_error_handling
                )
            all_tasks.append(coro)

        gathered = await asyncio.gather(*all_tasks)
        return list(gathered) + unknown_tool_results

    else:
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            if tool_name == RESPONSE_FORMAT_TOOL_NAME:
                continue
            tool_func = tool_map.get(tool_name)
            if not tool_func:
                if graceful_error_handling:
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": f"Error: Unknown tool '{tool_name}' - tool not found in available tools",
                        "is_error": True,
                    })
                    continue
                else:
                    raise KeyError(f"Unknown tool: {tool_name}")

            if asyncio.iscoroutinefunction(tool_func):
                result = await run_async_tool(tool_call, tool_func, graceful_error_handling)
            else:
                result = await loop.run_in_executor(
                    _get_async_executor(), run_sync_tool, tool_call, tool_func, graceful_error_handling
                )
            results.append(result)

        return results

def execute_tools_sync(
    tool_calls: List[Dict[str, Any]],
    tool_map: Dict[str, Callable[..., Any]],
    parallel: bool = False,
    graceful_error_handling: bool = True
) -> List[Dict[str, Any]]:
    if not tool_calls:
        return []

    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        if tool_name == RESPONSE_FORMAT_TOOL_NAME:
            continue
        tool_func = tool_map.get(tool_name)
        if tool_func and asyncio.iscoroutinefunction(tool_func):
            raise RuntimeError("Async tools are not supported in sync toolflow execution")

    results = []

    if parallel and len(tool_calls) > 1:
        executor = _get_sync_executor()
        futures = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            if tool_name == RESPONSE_FORMAT_TOOL_NAME:
                continue
            tool_func = tool_map.get(tool_name)
            if not tool_func:
                if graceful_error_handling:
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": f"Error: Unknown tool '{tool_name}' - tool not found in available tools",
                        "is_error": True,
                    })
                    continue
                else:
                    raise KeyError(f"Unknown tool: {tool_name}")

            futures.append(executor.submit(run_sync_tool, tool_call, tool_func, graceful_error_handling))

        done, _ = wait(futures)
        for future in done:
            results.append(future.result())

    else:
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            if tool_name == RESPONSE_FORMAT_TOOL_NAME:
                continue
            tool_func = tool_map.get(tool_name)
            if not tool_func:
                if graceful_error_handling:
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": f"Error: Unknown tool '{tool_name}' - tool not found in available tools",
                        "is_error": True,
                    })
                    continue
                else:
                    raise KeyError(f"Unknown tool: {tool_name}")

            result = run_sync_tool(tool_call, tool_func, graceful_error_handling)
            results.append(result)

    return results

def _prepare_tool_arguments(tool_func: Callable[..., Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare function arguments by converting JSON-like values to proper Python types.
    
    Uses Pydantic v2 TypeAdapter to validate and coerce arguments into the expected types.
    
    Args:
        tool_func: The tool function to inspect
        arguments: Dictionary of arguments from the tool call
        
    Returns:
        Prepared arguments with proper Python types instantiated
        
    Raises:
        Exception: If argument conversion fails (validation errors are not suppressed)
    """
    sig = inspect.signature(tool_func)
    type_hints = get_type_hints(tool_func, include_extras=True)  # include Annotated[] info
    prepared_args = {}

    for param_name, param in sig.parameters.items():
        if param_name in arguments:
            param_annotation = type_hints.get(param_name, param.annotation)
            arg_value = arguments[param_name]

            converted_value = _convert_argument_type(param_annotation, arg_value)
            prepared_args[param_name] = converted_value

        elif param.default is not inspect.Parameter.empty:
            # Optional: include defaults for missing args
            prepared_args[param_name] = param.default

        # else: skip missing required args – let function raise error

    return prepared_args


def _convert_argument_type(annotation: Any, value: Any) -> Any:
    """
    Convert JSON-like `value` into `annotation` using Pydantic v2's TypeAdapter.
    
    Args:
        annotation: The type hint to validate against
        value: The value to convert
        
    Returns:
        Converted and validated value
        
    Raises:
        ValueError: If conversion fails
    """
    if annotation is Any:
        return value  # Any type → pass through

    try:
        adapter = TypeAdapter(annotation)
        return adapter.validate_python(value)
    except Exception as e:
        raise ValueError(
            f"Failed to convert value {value!r} to type {annotation}: {e}"
        ) from e

def run_sync_tool(tool_call: Dict[str, Any], tool_func: Callable[..., Any], graceful_error_handling: bool = True) -> Dict[str, Any]:
    try:
        # Prepare arguments by converting dictionaries to Pydantic models when needed
        prepared_args = _prepare_tool_arguments(tool_func, tool_call["function"]["arguments"])
        result = tool_func(**prepared_args)
        return {"tool_call_id": tool_call["id"], "output": result}
    except Exception as e:
        if graceful_error_handling:
            return {
                "tool_call_id": tool_call["id"],
                "output": f"Error executing tool {tool_call['function']['name']}: {e}",
                "is_error": True,
            }
        else:
            error_msg = f"""Error executing tool {tool_call['function']['name']}: {e}
            TIP: If this is a data format issue, consider clearly documenting the tool's parameters.
            """
            raise type(e)(error_msg) from e

async def run_async_tool(tool_call: Dict[str, Any], tool_func: Callable[..., Coroutine[Any, Any, Any]], graceful_error_handling: bool = True) -> Dict[str, Any]:
    try:
        # Prepare arguments by converting dictionaries to Pydantic models when needed
        prepared_args = _prepare_tool_arguments(tool_func, tool_call["function"]["arguments"])
        result = await tool_func(**prepared_args)
        return {"tool_call_id": tool_call["id"], "output": result}
    except Exception as e:
        if graceful_error_handling:
            return {
                "tool_call_id": tool_call["id"],
                "output": f"Error executing tool {tool_call['function']['name']}: {e}",
                "is_error": True,
            }
        else:
            error_msg = f"""Error executing tool {tool_call['function']['name']}: {e}
            TIP: If this is a data format issue, consider clearly documenting the tool's parameters.
            """
            raise type(e)(error_msg) from e
