import time
import toolflow
import openai
import os
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
    fibonacci_numbers: list[Fib]

def main():
    client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
        tools=[fibonacci],
        response_format=FibonacciResponse
    )
    
    print(response.choices[0].message.parsed)
    print(response.choices[0].message.content)
    
if __name__ == "__main__":
    main()
