"""
OpenAI-specific tool execution utilities.

This module handles the specifics of how OpenAI formats and executes tool calls.
"""
import json
import asyncio
import concurrent.futures
from typing import Any, Dict, List, Callable


def validate_and_prepare_openai_tools(tools: List[Callable], strict: bool = False) -> tuple[Dict[str, Callable], List[Dict]]:
    """Validate tools and prepare OpenAI-specific schemas and function mappings."""
    tool_functions = {}
    tool_schemas = []
    
    for tool in tools:
        if isinstance(tool, Callable) and hasattr(tool, '_tool_metadata'):
            tool_schemas.append(tool._tool_metadata_strict if strict else tool._tool_metadata)
            tool_functions[tool._tool_metadata['function']['name']] = tool
        else:
            raise ValueError(f"Only decorated functions via @tool are supported. Got {type(tool)}")
    
    return tool_functions, tool_schemas


def execute_openai_tools_sync(
    tool_functions: Dict[str, Callable],
    tool_calls: List[Any],
    parallel_tool_execution: bool = False,
    max_workers: int = 10,
    graceful_error_handling: bool = False
) -> List[Dict[str, Any]]:
    """Execute OpenAI tool calls synchronously."""
    
    def execute_single_tool(tool_call):
        """Execute a single OpenAI tool call."""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        tool_function = tool_functions.get(tool_name, None)
        if not tool_function:
            raise ValueError(f"Tool {tool_name} not found")
        
        try:
            # Parse JSON arguments and call the function directly
            parsed_args = json.loads(tool_args) if tool_args else {}
            result = tool_function(**parsed_args)
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps(result) if not isinstance(result, str) else result
            }
        except Exception as e:
            if graceful_error_handling:
                return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": f"Error executing tool {tool_name}: {e}"
                }
            else:
                raise Exception(f"Error executing tool {tool_name}: {e}")
  
    # Sequential execution
    if not parallel_tool_execution or len(tool_calls) == 1:
        return [execute_single_tool(tool_call) for tool_call in tool_calls]
    
    # Parallel execution using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tool_calls), max_workers)) as executor:
        future_to_tool_call = {
            executor.submit(execute_single_tool, tool_call): tool_call 
            for tool_call in tool_calls
        }
        
        execution_response = []
        for future in concurrent.futures.as_completed(future_to_tool_call):
            tool_call = future_to_tool_call[future]
            try:
                result = future.result()
                execution_response.append(result)
            except Exception as e:
                # Re-raise with tool context
                raise Exception(f"Error in parallel tool execution: {e}")
        
        # Sort results to maintain order of tool_calls
        call_id_to_result = {result["tool_call_id"]: result for result in execution_response}
        ordered_results = [call_id_to_result[tool_call.id] for tool_call in tool_calls]
        
        return ordered_results


async def execute_openai_tools_async(
    tool_functions: Dict[str, Callable],
    tool_calls: List[Any],
    parallel_tool_execution: bool = False,
    max_workers: int = 10,
    graceful_error_handling: bool = False
) -> List[Dict[str, Any]]:
    """Execute OpenAI tool calls asynchronously, separating sync and async tools."""
    
    async def execute_single_tool(tool_call):
        """Execute a single tool call (async)."""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        tool_function = tool_functions.get(tool_name, None)
        if not tool_function:
            raise ValueError(f"Tool {tool_name} not found")
        
        try:
            # Parse JSON arguments and call the function
            parsed_args = json.loads(tool_args) if tool_args else {}
            
            # Check if the tool function is async
            if asyncio.iscoroutinefunction(tool_function):
                result = await tool_function(**parsed_args)
            else:
                # Run sync functions in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool_function(**parsed_args))
            
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps(result) if not isinstance(result, str) else result
            }
        except Exception as e:
            if graceful_error_handling:
                return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": f"Error executing tool {tool_name}: {e}"
                }
            else:
                raise Exception(f"Error executing tool {tool_name}: {e}")
    
    # Sequential execution
    if not parallel_tool_execution or len(tool_calls) == 1:
        return [await execute_single_tool(tool_call) for tool_call in tool_calls]
    
    # Parallel execution: separate sync and async tools for optimal performance
    sync_tools = []
    async_tools = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_function = tool_functions.get(tool_name)
        if tool_function and asyncio.iscoroutinefunction(tool_function):
            async_tools.append(tool_call)
        else:
            sync_tools.append(tool_call)
    
    # Execute sync and async tools in parallel
    tasks = []
    
    # Run sync tools in thread pool
    if sync_tools:
        def execute_sync_tools():
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sync_tools), max_workers)) as executor:
                futures = []
                for tool_call in sync_tools:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    tool_function = tool_functions[tool_name]
                    
                    def execute_sync_tool(tc=tool_call, tf=tool_function):
                        try:
                            parsed_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            result = tf(**parsed_args)
                            return {
                                "tool_call_id": tc.id,
                                "role": "tool",
                                "content": json.dumps(result) if not isinstance(result, str) else result
                            }
                        except Exception as e:
                            raise Exception(f"Error executing tool {tc.function.name}: {e}")
                    
                    futures.append(executor.submit(execute_sync_tool))
                
                return [future.result() for future in futures]
        
        # Run sync tools in a separate thread pool
        loop = asyncio.get_event_loop()
        sync_task = loop.run_in_executor(None, execute_sync_tools)
        tasks.append(sync_task)
    
    # Run async tools with asyncio.gather
    if async_tools:
        async def execute_async_tools():
            async_tasks = []
            for tool_call in async_tools:
                tool_name = tool_call.function.name
                tool_function = tool_functions[tool_name]
                
                async def execute_async_tool(tc=tool_call, tf=tool_function):
                    try:
                        parsed_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        result = await tf(**parsed_args)
                        return {
                            "tool_call_id": tc.id,
                            "role": "tool",
                            "content": json.dumps(result) if not isinstance(result, str) else result
                        }
                    except Exception as e:
                        raise Exception(f"Error executing tool {tc.function.name}: {e}")
                
                async_tasks.append(execute_async_tool())
            
            return await asyncio.gather(*async_tasks)
        
        tasks.append(execute_async_tools())
    
    # Wait for both sync and async tools to complete
    if tasks:
        try:
            results = await asyncio.gather(*tasks)
            
            # Flatten results from sync and async execution
            all_results = []
            for result_group in results:
                if isinstance(result_group, list):
                    all_results.extend(result_group)
                else:
                    all_results.append(result_group)
            
            # Sort results to maintain order of tool_calls
            call_id_to_result = {result["tool_call_id"]: result for result in all_results}
            ordered_results = [call_id_to_result[tool_call.id] for tool_call in tool_calls]
            
            return ordered_results
        except Exception as e:
            raise Exception(f"Error in parallel tool execution: {e}")
    
    return []
