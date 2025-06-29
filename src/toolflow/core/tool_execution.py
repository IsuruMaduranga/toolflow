# src/toolflow/core/tool_execution.py
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable, Any, Coroutine, Optional

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
        return _custom_executor if _custom_executor else _global_executor

def _get_async_executor() -> Optional[ThreadPoolExecutor]:
    """
    Get the executor for async tool execution.
    Returns the custom executor if set, otherwise None (uses asyncio's default).
    """
    with _executor_lock:
        return _custom_executor

# ===== TOOL EXECUTION FUNCTIONS =====

def execute_tools(
    tool_calls: List[Dict],
    tool_map: Dict[str, Callable],
    parallel: bool = False,
    graceful_error_handling: bool = True
) -> List[Dict]:
    """
    Executes tool calls synchronously.
    
    Args:
        tool_calls: List of tool calls to execute
        tool_map: Mapping of tool names to functions
        parallel: If True, use global thread pool; if False, execute sequentially
        graceful_error_handling: If True, return error messages; if False, raise exceptions
    """
    if not tool_calls:
        return []
    
    if not parallel:
        # Sequential execution (default for playground use)
        return [_run_sync_tool(tool_call, tool_map[tool_call["function"]["name"]], graceful_error_handling) for tool_call in tool_calls]
    
    # Parallel execution using global thread pool
    executor = _get_sync_executor()
    future_to_tool_call = {
        executor.submit(
            _run_sync_tool,
            tool_call,
            tool_map[tool_call["function"]["name"]],
            graceful_error_handling
        ): tool_call
        for tool_call in tool_calls
    }
    
    tool_results = []
    for future in future_to_tool_call:
        tool_results.append(future.result())
    
    return tool_results


async def execute_tools_async(
    tool_calls: List[Dict],
    tool_map: Dict[str, Callable[..., Any]],
    graceful_error_handling: bool = True
) -> List[Dict]:
    """
    Executes tool calls asynchronously, handling both sync and async tools.
    
    - Async tools run concurrently using asyncio.gather()
    - Sync tools run in thread pool:
        * Uses global executor if set via set_global_executor()
        * Otherwise uses asyncio's default thread pool
    - Always executes tools concurrently for optimal async performance
    
    Args:
        tool_calls: List of tool calls to execute
        tool_map: Mapping of tool names to functions
        graceful_error_handling: If True, return error messages; if False, raise exceptions
    """
    sync_tool_calls = []
    async_tool_tasks: List[Coroutine] = []

    for tool_call in tool_calls:
        tool_func = tool_map.get(tool_call["function"]["name"])
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                async_tool_tasks.append(
                    _run_async_tool(tool_call, tool_func, graceful_error_handling)
                )
            else:
                sync_tool_calls.append(tool_call)
    
    # Run sync tools using custom executor or asyncio's default
    sync_results = []
    if sync_tool_calls:
        loop = asyncio.get_running_loop()
        executor = _get_async_executor()  # Custom executor or None (asyncio default)
        
        futures = [
            loop.run_in_executor(
                executor,
                _run_sync_tool,
                call,
                tool_map[call["function"]["name"]],
                graceful_error_handling
            )
            for call in sync_tool_calls
        ]
        sync_results = await asyncio.gather(*futures)

    # Run async tools concurrently
    async_results = await asyncio.gather(*async_tool_tasks)

    return sync_results + async_results


def _run_sync_tool(tool_call: Dict, tool_func: Callable, graceful_error_handling: bool = True) -> Dict:
    try:
        result = tool_func(**tool_call["function"]["arguments"])
        return {"tool_call_id": tool_call["id"], "output": result}
    except Exception as e:
        if graceful_error_handling:
            return {
                "tool_call_id": tool_call["id"],
                "output": f"Error executing tool {tool_call['function']['name']}: {e}",
                "is_error": True,
            }
        else:
            raise

async def _run_async_tool(tool_call: Dict, tool_func: Callable[..., Coroutine], graceful_error_handling: bool = True) -> Dict:
    try:
        result = await tool_func(**tool_call["function"]["arguments"])
        return {"tool_call_id": tool_call["id"], "output": result}
    except Exception as e:
        if graceful_error_handling:
            return {
                "tool_call_id": tool_call["id"],
                "output": f"Error executing tool {tool_call['function']['name']}: {e}",
                "is_error": True,
            }
        else:
            raise
