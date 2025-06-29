#!/usr/bin/env python3
"""
Examples of configuring sync and async thread pool execution in toolflow.

This demonstrates the simplified API with a single shared global executor.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import toolflow
from openai import OpenAI, AsyncOpenAI

@toolflow.tool
def cpu_bound_task(n: int) -> str:
    """Simulate a CPU-bound task."""
    total = sum(i * i for i in range(n))
    return f"Computed sum of squares up to {n}: {total}"

@toolflow.tool
async def async_task(message: str) -> str:
    """An async task."""
    await asyncio.sleep(0.1)  # Simulate async work
    return f"Async processed: {message}"

def example_1_sync_sequential():
    """Example 1: Sync with sequential execution (default behavior)."""
    print("=== Example 1: Sync Sequential (Default) ===")
    
    client = toolflow.from_openai(OpenAI())
    
    # Default: parallel_execution=False, uses sequential execution
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Run some CPU tasks sequentially"}],
        tools=[cpu_bound_task],
        parallel_tool_execution=False,  # Default
        max_tool_calls=3
    )
    print("Sequential execution completed")
    print()

def example_2_sync_parallel():
    """Example 2: Sync with parallel execution using global thread pool."""
    print("=== Example 2: Sync Parallel ===")
    
    # Configure global thread pool
    toolflow.set_max_workers(8)  # Set to 8 threads
    
    client = toolflow.from_openai(OpenAI())
    
    # Enable parallel execution
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Run some CPU tasks in parallel"}],
        tools=[cpu_bound_task],
        parallel_tool_execution=True,  # Enable parallel execution
        max_tool_calls=3
    )
    print("Parallel execution completed")
    print()

def example_3_custom_executor():
    """Example 3: Custom thread pool executor (shared by sync and async)."""
    print("=== Example 3: Custom Global Executor ===")
    
    # Create custom executor
    custom_executor = ThreadPoolExecutor(
        max_workers=6,
        thread_name_prefix="my-custom-pool-"
    )
    
    # Set custom executor (used by both sync and async)
    toolflow.set_executor(custom_executor)
    
    client = toolflow.from_openai(OpenAI())
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Use custom executor"}],
        tools=[cpu_bound_task],
        parallel_tool_execution=True,
        max_tool_calls=3
    )
    print("Custom executor execution completed")
    print()

async def example_4_async_default():
    """Example 4: Async with default asyncio thread pool."""
    print("=== Example 4: Async Default (asyncio thread pool) ===")
    
    client = toolflow.from_openai_async(AsyncOpenAI())
    
    # Async always runs tools concurrently
    # Uses asyncio's default thread pool for sync tools (since no global executor set)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Run mixed sync/async tasks"}],
        tools=[cpu_bound_task, async_task],
        max_tool_calls=3
    )
    print("Async execution completed (using asyncio default)")
    print()

async def example_5_async_with_global_executor():
    """Example 5: Async using the same global executor as sync."""
    print("=== Example 5: Async with Global Executor ===")
    
    # Set a global executor (this will be used by both sync and async)
    custom_executor = ThreadPoolExecutor(
        max_workers=10,
        thread_name_prefix="shared-executor-"
    )
    toolflow.set_executor(custom_executor)
    
    client = toolflow.from_openai_async(AsyncOpenAI())
    
    # Now async will use the same global executor for sync tools
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Use shared global executor"}],
        tools=[cpu_bound_task, async_task],
        max_tool_calls=3
    )
    print("Async execution completed (using shared global executor)")
    print()

def example_6_environment_variable():
    """Example 6: Configure thread pool via environment variable."""
    print("=== Example 6: Environment Variable Configuration ===")
    
    import os
    # Set environment variable (in practice, set this before starting your app)
    os.environ["TOOLFLOW_SYNC_MAX_WORKERS"] = "12"
    
    # The global executor will use 12 threads when created
    client = toolflow.from_openai(OpenAI())
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Use environment configured threads"}],
        tools=[cpu_bound_task],
        parallel_tool_execution=True,
        max_tool_calls=3
    )
    print("Environment variable configuration completed")
    print()

def main():
    """Run all examples."""
    print("Toolflow Simplified Thread Pool Configuration Examples")
    print("=" * 55)
    print()
    print("Key Behavior:")
    print("- Sync: Sequential by default, can enable parallel execution")
    print("- Async: Always concurrent, uses global executor if set, otherwise asyncio default")
    print("- One shared global executor for both sync and async")
    print()
    
    # Sync examples
    example_1_sync_sequential()
    example_2_sync_parallel()
    example_3_custom_executor()
    
    # Async examples
    asyncio.run(example_4_async_default())
    asyncio.run(example_5_async_with_global_executor())
    
    # Environment configuration
    example_6_environment_variable()
    
    print("All examples completed!")

if __name__ == "__main__":
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. These examples demonstrate the API structure.")
        print("Set OPENAI_API_KEY to run with real API calls.")
        print()
    
    main() 