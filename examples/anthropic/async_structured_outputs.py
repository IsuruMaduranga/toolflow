import asyncio
import toolflow
from anthropic import AsyncAnthropic
from typing import List
from pydantic import BaseModel

@toolflow.tool
async def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return await fibonacci(n-1) + await fibonacci(n-2)

class Fib(BaseModel):
    n: int
    value_of_n_th_fibonacci: int

class FibonacciResponse(BaseModel):
    fibonacci_numbers: List[Fib]

async def main():
    # Default behavior: simplified API (returns parsed data directly)
    client = toolflow.from_anthropic(AsyncAnthropic())        

    # Toolflow enhanced API - returns parsed data directly
    parsed_data = await client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[fibonacci],
        response_format=FibonacciResponse
    )
    
    print("Parsed data:", parsed_data) 
    print(isinstance(parsed_data, FibonacciResponse)) # Direct FibonacciResponse object
    
    # For full response access, use full_response=True
    full_client = toolflow.from_anthropic(AsyncAnthropic(), full_response=True)
    response = await full_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[fibonacci],
        response_format=FibonacciResponse
    )
    print("Full response access:")
    print(response.content)
    
if __name__ == "__main__":
    asyncio.run(main())
