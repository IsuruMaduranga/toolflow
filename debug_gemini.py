#!/usr/bin/env python3
"""Debug test to see what messages are being sent to Gemini API."""

import os
import toolflow
import google.generativeai as genai
from dotenv import load_dotenv
import json

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
def simple_math(expression: str) -> str:
    """Calculate a mathematical expression. For example: '5 + 3', '10 * 2', '15 / 3'."""
    print(f"[TOOL] Calculating: {expression}")
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

@toolflow.tool
def simple_weather(city: str) -> str:
    """Get weather information for a city."""
    print(f"[TOOL] Getting weather for: {city}")
    return f"The weather in {city} is sunny, 75°F"

# Monkey patch to debug
original_call_api = client.handler.call_api
original_build_assistant_message = client.handler.build_assistant_message

def debug_build_assistant_message(text, tool_calls, original_response=None):
    print(f"\n=== DEBUG: build_assistant_message ===")
    print(f"Text: {repr(text)}")
    print(f"Tool calls: {tool_calls}")
    result = original_build_assistant_message(text, tool_calls, original_response)
    print(f"Result: {json.dumps(result, indent=2, default=str)}")
    print("="*50)
    return result

def debug_call_api(**kwargs):
    print("\n=== DEBUG: API Call ===")
    gemini_kwargs = client.handler._convert_to_gemini_format(**kwargs)
    print(f"Contents being sent to Gemini:")
    print(json.dumps(gemini_kwargs.get('contents'), indent=2, default=str))
    print("="*50)
    return original_call_api(**kwargs)

client.handler.call_api = debug_call_api
client.handler.build_assistant_message = debug_build_assistant_message

def test_debug():
    """Debug test to see message flow."""
    print("=== Debug Test ===")
    
    try:
        response = client.generate_content(
            "Please use the simple_math tool to calculate 5 + 3",
            tools=[simple_math, simple_weather],
            max_tool_call_rounds=2
        )
        print(f"Final Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug()
