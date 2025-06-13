"""
Example demonstrating parallel tool execution performance benefits.

This example shows how toolflow automatically executes multiple tools
in parallel, significantly improving performance compared to sequential execution.
"""

import time
import toolflow
from anthropic import Anthropic

@toolflow.tool
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    print(f"Calculating Fibonacci number {n}...")
    time.sleep(1)
    fib_n = fibonacci(n)
    print(f"Fibonacci number {n} is {fib_n}")
    return fib_n


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


def main():
    # Default behavior: simplified API (returns content directly)
    client = toolflow.from_anthropic(Anthropic())

    # Sequential execution
    print("Sequential execution:")
    start_time = time.time()
    content = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[calculate_fibonacci]
    )
    end_time = time.time()
    print(content)  # Direct string output
    print(f"Time taken: {end_time - start_time} seconds")

    print()

    # Parallel execution
    print("Parallel execution:")
    start_time = time.time()
    content = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[calculate_fibonacci],
        parallel_tool_execution=True
    ) 
    end_time = time.time()
    print(content)  # Direct string output
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
