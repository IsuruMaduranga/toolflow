import openai
import toolflow
import os

@toolflow.tool
def simple_calculator(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical operations."""
    print(f"Tool: {operation} {a} and {b}")
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def test_simple_streaming():
    """Test streaming without tools."""
    print("=== Simple Streaming Test ===")
    
    client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a short poem about programming"}],
        stream=True,
    )
    
    print("Streaming response:")
    for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)
    print("\n")


def test_streaming_with_tools():
    """Test streaming with tool calls."""
    print("=== Streaming with Tools Test ===")
    
    client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Calculate 15 * 7, then add 25 to the result"}],
        tools=[simple_calculator],
        stream=True,
        max_tool_call_rounds=5,
    )
    
    print("Streaming response with tools:")
    for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)
    print("\n")


def test_streaming_full_response():
    """Test streaming with full_response=True."""
    print("=== Streaming Full Response Test ===")
    
    client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True,
        full_response=True,
    )
    
    print("Full response chunks:")
    for chunk in stream:
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                print(f"Content: {delta.content}")
    print("\n")


def main():
    """Run all streaming tests."""
    print("Testing OpenAI Streaming Implementation\n")
    
    try:
        test_simple_streaming()
        test_streaming_with_tools()
        test_streaming_full_response()
        print("All streaming tests completed!")
    except Exception as e:
        print(f"Error during streaming tests: {e}")


if __name__ == "__main__":
    main() 