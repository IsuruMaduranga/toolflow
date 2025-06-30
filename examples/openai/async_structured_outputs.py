import asyncio
import toolflow
from openai import AsyncOpenAI
import os
from pydantic import BaseModel

async def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return await fibonacci(n-1) + await fibonacci(n-2)

class Fib(BaseModel):
    n: int
    value_of_n_th_fibonacci: int

class FibonacciResponse(BaseModel):
    fibonacci_numbers: list[Fib]

async def main():
    # Default behavior: simplified API (returns parsed data directly)
    client = toolflow.from_openai(AsyncOpenAI())

    # Toolflow enhanced API - returns parsed data directly
    parsed_data = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[fibonacci],
        response_format=FibonacciResponse
    )
    print(parsed_data)
    
if __name__ == "__main__":
    asyncio.run(main())
