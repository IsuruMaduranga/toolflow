#!/usr/bin/env python3
"""
OpenAI Reasoning Examples with Toolflow

Demonstrates key features with OpenAI's reasoning models:
- Basic reasoning mode
- Streaming with reasoning
- Tool use with reasoning
- Structured output
- Both sync and async

Set your OPENAI_API_KEY environment variable before running.
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
    reasoning_steps: List[str]


def basic_reasoning():
    """Basic reasoning mode example."""
    print("Basic Reasoning Mode")
    print("-" * 20)
    
    import openai
    client = toolflow.from_openai(openai.OpenAI())
    
    response = client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": "Solve step by step: If a train travels 120 km in 2 hours, what's its speed in m/s? Show your reasoning process."
            }
        ]
    )
    
    print(response)
    print()


def streaming_with_reasoning():
    """Streaming with reasoning mode."""
    print("Streaming with Reasoning")
    print("-" * 25)
    
    import openai
    client = toolflow.from_openai(openai.OpenAI())
    
    stream = client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=2000,
        stream=True,
        messages=[
            {
                "role": "user",
                "content": "Think through this step by step: What's 15 * 8 + 7? Show your reasoning."
            }
        ]
    )
    
    for chunk in stream:
        if isinstance(chunk, str):
            print(chunk, end='', flush=True)
    print("\n")


def tools_with_reasoning():
    """Tool use with reasoning mode."""
    print("Tools with Reasoning")
    print("-" * 20)
    
    import openai
    client = toolflow.from_openai(openai.OpenAI())
    
    response = client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=4000,
        tools=[calculate, analyze_data],
        messages=[
            {
                "role": "user",
                "content": """
                Analyze this sales data: [100, 120, 110, 130, 125]
                
                Please reason through this step by step:
                1. Calculate the total sales
                2. Get statistical analysis
                3. Analyze the trend
                4. Provide insights
                """
            }
        ]
    )
    
    print(response)
    print()


def structured_output_with_reasoning():
    """Structured output with reasoning."""
    print("Structured Output with Reasoning")
    print("-" * 35)
    
    import openai
    client = toolflow.from_openai(openai.OpenAI())
    
    result = client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=2000,
        response_format=MathResult,
        messages=[
            {
                "role": "user",
                "content": "Calculate 25 * 4 + 10. Think through each step and explain your reasoning process."
            }
        ]
    )

    print("Structured result:")
    print(f"Type: {type(result)}")
    print(f"Operation: {result.operation}")
    print(f"Result: {result.result}")
    print(f"Explanation: {result.explanation}")
    print(f"Reasoning steps: {result.reasoning_steps}")
    print()


async def async_reasoning():
    """Async reasoning mode example."""
    print("Async Reasoning")
    print("-" * 15)
    
    import openai
    client = toolflow.from_openai(openai.AsyncOpenAI())
    
    response = await client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=2000,
        tools=[calculate],
        messages=[
            {
                "role": "user",
                "content": "Calculate the area of a circle with radius 5. Use π ≈ 3.14159. Reason through the formula and calculation step by step."
            }
        ]
    )
    
    print(response)
    print()


def reasoning_with_system_prompt():
    """Reasoning with system prompt guidance."""
    print("Reasoning with System Prompt")
    print("-" * 30)
    
    import openai
    client = toolflow.from_openai(openai.OpenAI())
    
    response = client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=2000,
        messages=[
            {
                "role": "system",
                "content": "You are a careful mathematician. Always show your step-by-step reasoning process and double-check your work."
            },
            {
                "role": "user",
                "content": "If I have 3 boxes with 15 apples each, and I eat 7 apples, how many apples do I have left?"
            }
        ]
    )
    
    print(response)
    print()


def complex_reasoning_with_tools():
    """Complex multi-step reasoning with tools."""
    print("Complex Reasoning with Tools")
    print("-" * 30)
    
    import openai
    client = toolflow.from_openai(openai.OpenAI())
    
    response = client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=4000,
        tools=[calculate, analyze_data],
        messages=[
            {
                "role": "user",
                "content": """
                I need to analyze monthly revenue data and make a decision:
                
                Monthly revenue (in thousands): [85, 92, 78, 105, 98, 112, 89, 94, 103, 118, 95, 107]
                
                Please reason through this analysis:
                1. Calculate the total annual revenue
                2. Find the average monthly revenue
                3. Analyze the trend over the year
                4. Identify the best and worst performing months
                5. Calculate what a 15% increase would look like
                6. Recommend whether this shows healthy growth
                
                Show your reasoning at each step.
                """
            }
        ]
    )
    
    print(response)
    print()


def test_reasoning_modes():
    """Test different response modes with reasoning."""
    print("Testing Reasoning Response Modes")
    print("-" * 35)
    
    import openai
    client = toolflow.from_openai(openai.OpenAI())
    
    # Test full_response=True (raw response object)
    print("full_response=True (raw response):")
    response = client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=3000,
        full_response=True,
        messages=[
            {
                "role": "user",
                "content": "Reason through: Why is the sky blue? Explain the physics step by step."
            }
        ]
    )
    
    print(f"Response type: {type(response)}")
    print(f"Content: {response.choices[0].message.content[:200]}...")
    
    # Test full_response=False (processed text)
    print("\nfull_response=False (processed text):")
    response = client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=3000,
        full_response=False,
        messages=[
            {
                "role": "user",
                "content": "Reason through: Why is the sky blue? Explain the physics step by step."
            }
        ]
    )
    
    print(f"Response type: {type(response)}")
    print(f"Content: {response[:200]}...")
    print()


async def async_streaming_reasoning():
    """Async streaming with reasoning."""
    print("Async Streaming Reasoning")
    print("-" * 25)
    
    import openai
    client = toolflow.from_openai(openai.AsyncOpenAI())
    
    stream = await client.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        max_completion_tokens=2000,
        stream=True,
        messages=[
            {
                "role": "user",
                "content": "Think step by step: How would you design a simple calculator? Break down the reasoning process."
            }
        ]
    )
    
    async for chunk in stream:
        if isinstance(chunk, str):
            print(chunk, end='', flush=True)
    print("\n")


def main():
    """Run all examples."""
    print("OpenAI Reasoning Examples with Toolflow")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        return
    
    try:
        # Basic examples
        basic_reasoning()
        streaming_with_reasoning()
        tools_with_reasoning()
        structured_output_with_reasoning()
        
        # Advanced examples
        reasoning_with_system_prompt()
        complex_reasoning_with_tools()
        
        # Response mode testing
        test_reasoning_modes()
        
        # Async examples
        asyncio.run(async_reasoning())
        asyncio.run(async_streaming_reasoning())
        
        print("All reasoning examples completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
