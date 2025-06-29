# src/toolflow/core/tool_execution.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable, Any, Coroutine

def execute_tools(
    tool_calls: List[Dict],
    tool_map: Dict[str, Callable],
    max_workers: int | None
) -> List[Dict]:
    """
    Executes tool calls synchronously, using a thread pool for parallelism.
    """
    tool_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tool_call = {
            executor.submit(
                tool_map[tool_call["function"]["name"]],
                **tool_call["function"]["arguments"]
            ): tool_call
            for tool_call in tool_calls
        }
        for future in future_to_tool_call:
            tool_call = future_to_tool_call[future]
            try:
                result = future.result()
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "output": result,
                    }
                )
            except Exception as e:
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "output": f"Error executing tool {tool_call['function']['name']}: {e}",
                        "is_error": True,
                    }
                )
    return tool_results


async def execute_tools_async(
    tool_calls: List[Dict],
    tool_map: Dict[str, Callable[..., Any]],
    max_workers: int | None
) -> List[Dict]:
    """
    Executes tool calls asynchronously, handling both sync and async tools.
    """
    sync_tool_calls = []
    async_tool_tasks: List[Coroutine] = []
    tool_call_map = {tool_call['id']: tool_call for tool_call in tool_calls}

    for tool_call in tool_calls:
        tool_func = tool_map.get(tool_call["function"]["name"])
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                async_tool_tasks.append(
                    _run_async_tool(tool_call, tool_func)
                )
            else:
                sync_tool_calls.append(tool_call)
    
    # Run sync tools in a thread pool
    sync_results = []
    if sync_tool_calls:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    _run_sync_tool,
                    call,
                    tool_map[call["function"]["name"]]
                )
                for call in sync_tool_calls
            ]
            sync_results = await asyncio.gather(*futures)

    # Run async tools concurrently
    async_results = await asyncio.gather(*async_tool_tasks)

    return sync_results + async_results


def _run_sync_tool(tool_call: Dict, tool_func: Callable) -> Dict:
    try:
        result = tool_func(**tool_call["function"]["arguments"])
        return {"tool_call_id": tool_call["id"], "output": result}
    except Exception as e:
        return {
            "tool_call_id": tool_call["id"],
            "output": f"Error executing tool {tool_call['function']['name']}: {e}",
            "is_error": True,
        }

async def _run_async_tool(tool_call: Dict, tool_func: Callable[..., Coroutine]) -> Dict:
    try:
        result = await tool_func(**tool_call["function"]["arguments"])
        return {"tool_call_id": tool_call["id"], "output": result}
    except Exception as e:
        return {
            "tool_call_id": tool_call["id"],
            "output": f"Error executing tool {tool_call['function']['name']}: {e}",
            "is_error": True,
        }
