#!/usr/bin/env python3
"""Simple test to verify Gemini tool calling works without infinite loops."""

import os
import toolflow
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
client = toolflow.from_gemini(model)

@toolflow.tool
def simple_weather(city: str) -> str:
    """Get weather for a city."""
    print(f"Getting weather for {city}")
    return f"The weather in {city} is sunny, 75°F"

@toolflow.tool 
def simple_math(expression: str) -> str:
    """Calculate a math expression."""
    print(f"Calculating: {expression}")
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

def test_basic_tool_calling():
    """Test basic tool calling without infinite loops."""
    print("=== Testing Basic Tool Calling ===")
    
    try:
        response = client.generate_content(
            "What's the weather in Paris and calculate 25 + 17",
            tools=[simple_weather, simple_math],
            max_tool_call_rounds=3  # Limit to 3 rounds to prevent infinite loops
        )
        print(f"Response: {response}")
        print("✅ Tool calling completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_single_tool():
    """Test single tool call."""
    print("\n=== Testing Single Tool ===")
    
    try:
        response = client.generate_content(
            "What's the weather in Tokyo?",
            tools=[simple_weather],
            max_tool_call_rounds=2
        )
        print(f"Response: {response}")
        print("✅ Single tool completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_basic_tool_calling()
    test_single_tool()
