"""
Example demonstrating parallel tool execution performance benefits with async tools.

This example shows how toolflow automatically executes multiple async tools
in parallel, significantly improving performance compared to sequential execution.
"""

import asyncio
import time
import toolflow
from anthropic import AsyncAnthropic
import os

@toolflow.tool
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    print(f"Calculating Fibonacci number sync {n}...")
    time.sleep(1)
    fib_n = fibonacci(n)
    print(f"Fibonacci number sync {n} is {fib_n}")
    return fib_n

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@toolflow.tool
async def calculate_fibonacci_async(n: int) -> int:
    """Calculate the nth Fibonacci number asynchronously."""
    print(f"Starting async Fibonacci calculation for {n}...")
    await asyncio.sleep(2)  # Simulate async I/O operation
    fib_n = await fibonacci_async(n)
    print(f"Finished async Fibonacci calculation for {n}: {fib_n}")
    return fib_n


async def fibonacci_async(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return await fibonacci_async(n-1) + await fibonacci_async(n-2)


@toolflow.tool
async def fetch_api_data(endpoint: str) -> dict:
    """Simulate fetching data from an API."""
    print(f"Starting API fetch from {endpoint}...")
    await asyncio.sleep(0.8)  # Simulate network delay
    print(f"Finished API fetch from {endpoint}")
    return {"endpoint": endpoint, "data": f"Sample data from {endpoint}", "status": "success"}


@toolflow.tool
async def process_data(data_type: str) -> str:
    """Simulate data processing."""
    print(f"Starting data processing for {data_type}...")
    await asyncio.sleep(0.6)  # Simulate processing time
    print(f"Finished data processing for {data_type}")
    return f"Processed {data_type} data successfully"


async def main():
    # Initialize async client - default behavior: simplified API (returns content directly)
    async_client = toolflow.from_anthropic(AsyncAnthropic())

    print("=== Async Execution ===")
    start_time = time.time()
    content = await async_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Calculate 10th, 11th and 12th Fibonacci numbers using async tools."}],
        tools=[calculate_fibonacci_async]
        # parallel_tool_execution=False (default) - sequential execution
    )
    end_time = time.time()
    print(f"Sequential response: {content}")  # Direct string output
    print(f"Sequential execution time: {end_time - start_time:.2f} seconds")
    print()

    start_time = time.time()
    content = await async_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Calculate 10th, 11th and 12th Fibonacci numbers using async tools."}],
        tools=[calculate_fibonacci_async],
        parallel_tool_execution=True  # Enable parallel execution
    ) 
    end_time = time.time()
    print(f"Parallel response: {content}")  # Direct string output
    print(f"Parallel execution time: {end_time - start_time:.2f} seconds")
    print()

    print("=== Mixed Async Tools Demo ===")
    start_time = time.time()
    content = await async_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Fetch data from 'users' and 'products' APIs, and process 'analytics' data."}],
        tools=[fetch_api_data, process_data],
        parallel_tool_execution=True  # Enable parallel execution
    )
    end_time = time.time()
    print(f"Mixed async tools response: {content}")  # Direct string output
    print(f"Mixed parallel execution time: {end_time - start_time:.2f} seconds")
    print()

    print("=== Async Parallel Execution with Sync and Async Tools ===")
    start_time = time.time()
    content = await async_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Calculate 10th, 11th, 12th and 13th Fibonacci numbers and fetch data from 'users' and 'products' APIs."}],
        tools=[calculate_fibonacci, fetch_api_data],
        max_tool_call_rounds=10,
        parallel_tool_execution=True
    )
    end_time = time.time()
    print(f"Mixed async tools response: {content}")  # Direct string output
    print(f"Mixed parallel execution time: {end_time - start_time:.2f} seconds")
    print()
   
if __name__ == "__main__":
    asyncio.run(main())
