import openai
import toolflow
import os
import time
from pydantic import BaseModel

class operation(BaseModel):
    operation: str
    a: float
    b: float

def sync_calculator(op: operation) -> float:
    """Perform basic mathematical operations (sync version)."""
    if op.operation == "add":
        return op.a + op.b
    elif op.operation == "subtract":
        return op.a - op.b
    elif op.operation == "multiply":
        print(f"Tool: Multiplying {op.a} and {op.b}")
        return op.a * op.b
    elif op.operation == "divide":
        print(f"Tool: Dividing {op.a} by {op.b}")
        if op.b == 0:
            raise ValueError("Cannot divide by zero")
        return op.a / op.b
    else:
        raise ValueError(f"Unknown operation: {op}")

def main():
    """Main async function demonstrating the async client."""
    
    # Create async OpenAI client (you'll need to set your API key)
    # Default behavior: simplified API (returns content directly)
    client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    # Using async client with a sync tool
    content = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 3.145 divided by 2?"}],
        tools=[sync_calculator],
        max_tool_call_rounds=5,
    )

    print("Result:", content)

if __name__ == "__main__":
    main()
