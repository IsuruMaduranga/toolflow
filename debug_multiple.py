#!/usr/bin/env python3
"""Debug multiple tool calls."""

import os
import toolflow
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
client = toolflow.from_gemini(model)

@toolflow.tool
def simple_weather(city: str) -> str:
    """Get weather for a city."""
    print(f"[TOOL] Getting weather for {city}")
    return f"The weather in {city} is sunny, 75°F"

@toolflow.tool 
def simple_math(expression: str) -> str:
    """Calculate a math expression."""
    print(f"[TOOL] Calculating: {expression}")
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

# Debug patches
original_build_tool_result_messages = client.handler.build_tool_result_messages

def debug_build_tool_result_messages(tool_results):
    print(f"\n=== DEBUG: build_tool_result_messages ===")
    print(f"Tool results: {tool_results}")
    result = original_build_tool_result_messages(tool_results)
    print(f"Built messages: {json.dumps(result, indent=2, default=str)}")
    print("="*50)
    return result

client.handler.build_tool_result_messages = debug_build_tool_result_messages

def test_multiple_tools():
    print("=== Testing Multiple Tool Calls ===")
    
    try:
        response = client.generate_content(
            "Get weather for Paris and calculate 25 + 17",
            tools=[simple_weather, simple_math],
            max_tool_call_rounds=2
        )
        print(f"Final Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_multiple_tools()
