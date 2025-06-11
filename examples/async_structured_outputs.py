import asyncio
import toolflow
from openai import AsyncOpenAI
import os
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
    fibonacci_numbers: list[Fib]

async def main():
    # Default behavior: simplified API (returns parsed data directly)
    client = toolflow.from_openai_async(AsyncOpenAI())

    # Toolflow enhanced API - returns parsed data directly
    parsed_data = await client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[fibonacci],
        response_format=FibonacciResponse
    )
    
    print("Parsed data:", parsed_data) 
    print(isinstance(parsed_data, FibonacciResponse)) # Direct FibonacciResponse object

    # Beta API - returns parsed data directly
    beta_parsed_data = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[fibonacci],
        response_format=FibonacciResponse
    )

    print("Beta parsed data:", beta_parsed_data) 
    print(isinstance(beta_parsed_data, FibonacciResponse)) # Direct FibonacciResponse object
    
    # # For full response access, use full_response=True
    # full_client = toolflow.from_openai_async(AsyncOpenAI(), full_response=True)
    # response = await full_client.chat.completions.parse(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
    #     tools=[fibonacci],
    #     response_format=FibonacciResponse
    # )
    # print("Full response access:")
    # print(response.choices[0].message.parsed)
    # print(response.choices[0].message.content)
    
if __name__ == "__main__":
    asyncio.run(main())
