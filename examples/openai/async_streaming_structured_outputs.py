import asyncio
import toolflow
import openai
import os
from typing import List
from pydantic import BaseModel

@toolflow.tool
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Fib(BaseModel):
    n: int
    value_of_n_th_fibonacci: int

class FibonacciResponse(BaseModel):
    fibonacci_numbers: List[Fib]

async def main():
    # Default behavior: simplified API (returns parsed data directly)
    client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    # Toolflow enhanced API - returns parsed data directly
    parsed_data = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "First explain me what is the Fibonacci sequence and then give me the 10th, 11th and 12th Fibonacci numbers."}],
        tools=[fibonacci],
        response_format=FibonacciResponse,
        stream=True
    )
    
    async for chunk in parsed_data:
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
        else:
            print("\nFibonacci numbers:")
            print(chunk)

if __name__ == "__main__":
    asyncio.run(main())
