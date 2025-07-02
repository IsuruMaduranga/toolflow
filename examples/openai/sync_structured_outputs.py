import time
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

def main():
    # Default behavior: simplified API (returns parsed data directly)
    client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    # Toolflow enhanced API - returns parsed data directly
    parsed_data = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[fibonacci],
        response_format=FibonacciResponse
    )
    
    print("Parsed data:", parsed_data)  # Direct FibonacciResponse object

if __name__ == "__main__":
    main()
