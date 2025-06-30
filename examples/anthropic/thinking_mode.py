#!/usr/bin/env python3
"""
Clean Anthropic Example with Toolflow

Demonstrates key features:
- Basic thinking mode
- Streaming with thinking
- Tool use with thinking
- Structured output
- Both sync and async

Set your ANTHROPIC_API_KEY environment variable before running.
"""

import os
import asyncio
import toolflow
from typing import List
from pydantic import BaseModel

import dotenv
dotenv.load_dotenv()


@toolflow.tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    print(f"TOOL_CALL: Calculating expression: {expression}")
    try:
        result = eval(expression)  # Note: eval is for demo only
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@toolflow.tool
def analyze_data(data: List[float], analysis_type: str = "statistical") -> str:
    """Analyze numerical data."""
    print(f"TOOL_CALL: Analyzing data: {data}")
    if not data:
        return "No data provided"
    
    if analysis_type == "statistical":
        mean = sum(data) / len(data)
        return f"Mean: {mean:.2f}, Max: {max(data)}, Min: {min(data)}, Count: {len(data)}"
    elif analysis_type == "trend":
        if len(data) < 2:
            return "Need at least 2 data points"
        trend = "increasing" if data[-1] > data[0] else "decreasing" if data[-1] < data[0] else "stable"
        return f"Trend: {trend} (from {data[0]} to {data[-1]})"
    
    return f"Unknown analysis type: {analysis_type}"


class MathResult(BaseModel):
    operation: str
    result: float
    explanation: str


def basic_thinking():
    """Basic thinking mode example."""
    print("Basic Thinking Mode")
    print("-" * 20)
    
    import anthropic
    client = toolflow.from_anthropic(anthropic.Anthropic())
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[{
            "role": "user",
            "content": "Solve: First think, then calculate. If a train travels 120 km in 2 hours, what's its speed in m/s?"
        }]
    )
    
    print(response)
    print()


def streaming_with_thinking():
    """Streaming with thinking mode."""
    print("Streaming with Thinking")
    print("-" * 20)
    
    import anthropic
    client = toolflow.from_anthropic(anthropic.Anthropic())
    
    stream = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
        stream=True,
        messages=[{
            "role": "user",
            "content": "Think step by step: What's 15 * 8 + 7?"
        }]
    )
    
    for chunk in stream:
        if isinstance(chunk, str):
            print(chunk, end='', flush=True)
    print("\n")


def tools_with_thinking():
    """Tool use with thinking mode."""
    print("Tools with Thinking")
    print("-" * 20)
    
    import anthropic
    client = toolflow.from_anthropic(anthropic.Anthropic())
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=3000,
        thinking={"type": "enabled", "budget_tokens": 1500},
        tools=[calculate, analyze_data],
        messages=[{
            "role": "user",
            "content": """
            Analyze this sales data: [100, 120, 110, 130, 125]
            
            1. First think
            2. Calculate the total sales
            3. Get statistical analysis
            4. Analyze the trend
            """
        }]
    )
    
    print(response)
    print()


def structured_output_with_thinking():
    """Structured output with thinking."""
    print("Structured Output with Thinking")
    print("-" * 30)
    
    import anthropic
    client = toolflow.from_anthropic(anthropic.Anthropic())
    
    result = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
        response_format=MathResult,
        messages=[{
            "role": "user",
            "content": "Calculate 25 * 4 + 10 and explain the steps"
        }]
    )

    print(isinstance(result, MathResult))
    print(result)


async def async_thinking():
    """Async thinking mode example."""
    print("Async Thinking")
    print("-" * 15)
    
    import anthropic
    client = toolflow.from_anthropic(anthropic.AsyncAnthropic())
    
    response = await client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
        tools=[calculate],
        messages=[{
            "role": "user",
            "content": "Calculate the area of a circle with radius 5. Use π ≈ 3.14159"
        }]
    )
    
    print(response)
    print()


def test_both_modes():
    """Test both full_response modes."""
    print("Testing Response Modes")
    print("-" * 20)
    
    import anthropic
    client = toolflow.from_anthropic(anthropic.Anthropic())
    
    # Test full_response=True (raw events)
    print("full_response=True (first 3 events):")
    stream = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
        stream=True,
        full_response=True,
        messages=[{"role": "user", "content": "Think: why is the sky blue?"}]
    )
    
    for i, chunk in enumerate(stream):
        print(chunk)
    
    # Test full_response=False (processed strings)
    print("\nfull_response=False (processed text):")
    stream = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
        stream=True,
        full_response=False,
        messages=[{"role": "user", "content": "Think: why is the sky blue?"}]
    )
    
    count = 0
    for chunk in stream:
        print(chunk)

    print()


def main():
    """Run all examples."""
    print("Anthropic Toolflow Examples")
    print("=" * 30)
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Please set your ANTHROPIC_API_KEY environment variable")
        return
    
    try:
        # Basic examples
        basic_thinking()
        streaming_with_thinking()
        tools_with_thinking()
        structured_output_with_thinking()
        
        # Async example
        asyncio.run(async_thinking())
        
        # Mode testing
        #test_both_modes()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
