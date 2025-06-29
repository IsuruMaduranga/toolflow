# src/toolflow/core/tool_execution.py
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable, Any, Coroutine, Optional

# ===== SYNC TOOL EXECUTION CONFIGURATION =====

# Global configuration for sync tool execution
_DEFAULT_SYNC_MAX_WORKERS = 4
_sync_global_executor: Optional[ThreadPoolExecutor] = None
_sync_executor_lock = threading.Lock()

def get_default_sync_max_workers() -> int:
    """Get the default max workers for sync tool execution."""
    return int(os.getenv("TOOLFLOW_SYNC_MAX_WORKERS", _DEFAULT_SYNC_MAX_WORKERS))

def set_max_workers_sync(max_workers: int) -> None:
    """Set the number of worker threads for sync tool execution."""
    global _sync_global_executor
    with _sync_executor_lock:
        if _sync_global_executor:
            _sync_global_executor.shutdown(wait=True)
        _sync_global_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="toolflow-sync-"
        )

def set_global_executor_sync(executor: ThreadPoolExecutor) -> None:
    """Set a custom global executor for sync tool execution."""
    global _sync_global_executor
    with _sync_executor_lock:
        if _sync_global_executor:
            _sync_global_executor.shutdown(wait=True)
        _sync_global_executor = executor

def get_sync_executor() -> ThreadPoolExecutor:
    """Get the executor for sync tool execution."""
    global _sync_global_executor
    if _sync_global_executor is None:
        with _sync_executor_lock:
            if _sync_global_executor is None:
                max_workers = get_default_sync_max_workers()
                _sync_global_executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="toolflow-sync-"
                )
    return _sync_global_executor

def cleanup_sync_executor():
    """Shutdown and cleanup the sync global executor."""
    global _sync_global_executor
    with _sync_executor_lock:
        if _sync_global_executor:
            _sync_global_executor.shutdown(wait=True)
            _sync_global_executor = None

# ===== ASYNC TOOL EXECUTION CONFIGURATION =====

# Global configuration for async tool execution
_async_global_executor: Optional[ThreadPoolExecutor] = None
_async_executor_lock = threading.Lock()

def set_max_workers_async(max_workers: int) -> None:
    """
    Set the number of worker threads for async tool execution.
    Note: This is ignored if using asyncio's default executor.
    Only has effect if you've set a custom async executor.
    """
    global _async_global_executor
    with _async_executor_lock:
        if _async_global_executor:
            # Only update if we have a custom executor, don't create one
            # if user is using asyncio's default
            _async_global_executor.shutdown(wait=True)
            _async_global_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="toolflow-async-"
            )

def set_global_executor_async(executor: ThreadPoolExecutor) -> None:
    """Set a custom global executor for async tool execution."""
    global _async_global_executor
    with _async_executor_lock:
        if _async_global_executor:
            _async_global_executor.shutdown(wait=True)
        _async_global_executor = executor

def get_async_executor() -> Optional[ThreadPoolExecutor]:
    """
    Get the executor for async tool execution.
    Returns None if using asyncio's default executor.
    """
    return _async_global_executor

def cleanup_async_executor():
    """Shutdown and cleanup the async global executor."""
    global _async_global_executor
    with _async_executor_lock:
        if _async_global_executor:
            _async_global_executor.shutdown(wait=True)
            _async_global_executor = None

# ===== UNIFIED CLEANUP =====

def cleanup_executors():
    """Shutdown and cleanup all global executors."""
    cleanup_sync_executor()
    cleanup_async_executor()

# ===== TOOL EXECUTION FUNCTIONS =====

def execute_tools(
    tool_calls: List[Dict],
    tool_map: Dict[str, Callable],
    parallel: bool = False,  # Changed default to False for playground-friendliness
    graceful_error_handling: bool = True
) -> List[Dict]:
    """
    Executes tool calls synchronously.
    
    Args:
        tool_calls: List of tool calls to execute
        tool_map: Mapping of tool names to functions
        parallel: If True, use global sync thread pool; if False, execute sequentially
        graceful_error_handling: If True, return error messages; if False, raise exceptions
    
    The global sync thread pool defaults to 4 threads and can be configured via:
    - TOOLFLOW_SYNC_MAX_WORKERS environment variable
    - set_max_workers_sync() function
    - set_global_executor_sync() function
    """
    if not tool_calls:
        return []
    
    if not parallel:
        # Sequential execution (default for playground use)
        return [_run_sync_tool(tool_call, tool_map[tool_call["function"]["name"]], graceful_error_handling) for tool_call in tool_calls]
    
    # Parallel execution using global sync thread pool
    executor = get_sync_executor()
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
        * Uses custom async executor if set via set_global_executor_async()
        * Otherwise uses asyncio's default thread pool (recommended)
    
    Always executes tools concurrently for optimal async performance.
    
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
    
    # Run sync tools using custom async executor or asyncio's default
    sync_results = []
    if sync_tool_calls:
        loop = asyncio.get_running_loop()
        
        # Use custom async executor if set, otherwise use asyncio's default (None)
        executor = get_async_executor()
        
        futures = [
            loop.run_in_executor(
                executor,  # Custom async executor or None (asyncio default)
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
