"""
Example demonstrating Google Gemini integration with toolflow.

This example shows how to use Google Gemini with toolflow's enhanced
features including tool calling and parallel execution.
"""

import os
import toolflow
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
try:
    import google.generativeai as genai
    
    # Configure your API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY environment variable")
    
    genai.configure(api_key=api_key)
    
    # Create model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Wrap with toolflow
    client = toolflow.from_gemini(model)
    
except ImportError:
    print("Google Generative AI library not installed. Install with: pip install google-generativeai")
    exit(1)

# A data class for complex tool parameters
@dataclass
class MathOperation:
    operation: str
    a: float
    b: float

@toolflow.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulate weather API call
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C", 
        "Tokyo": "Rainy, 20°C",
        "San Francisco": "Foggy, 18°C",
        "Paris": "Clear, 20°C"
    }
    return weather_data.get(city, f"Weather data not available for {city}")

@toolflow.tool
def calculator(op: MathOperation) -> float:
    """Perform basic mathematical operations: add, subtract, multiply, divide."""
    print(f"Tool: Calculating {op.a} {op.operation} {op.b}")
    
    if op.operation == "add":
        return op.a + op.b
    elif op.operation == "subtract":
        return op.a - op.b
    elif op.operation == "multiply":
        return op.a * op.b
    elif op.operation == "divide":
        if op.b == 0:
            raise ValueError("Cannot divide by zero")
        return op.a / op.b
    else:
        raise ValueError(f"Unknown operation: {op.operation}")

@toolflow.tool
def simple_math(expression: str) -> str:
    """Safely evaluate a simple mathematical expression."""
    try:
        # Simple calculator - only allow basic operations
        allowed_chars = set('0123456789+-*/.()')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return "Error: Only basic mathematical operations are allowed"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def main():
    print("=== Gemini with Toolflow Sync Example ===")
    
    # Simple text generation
    print("\n1. Simple text generation:")
    response = client.generate_content("Tell me a short joke about programming")
    print(response)
    
    # Tool calling with simple tool
    print("\n2. Tool calling - Weather:")
    response = client.generate_content(
        "What's the weather like in New York and London?",
        tools=[get_weather]
    )
    print(response)
    
    # Tool calling with dataclass parameters
    print("\n3. Complex tool with dataclass parameters:")
    response = client.generate_content(
        "Calculate 15 multiplied by 8, then subtract 20 from the result",
        tools=[calculator]
    )
    print(response)
    
    # Multiple tools with parallel execution
    print("\n4. Multiple tools with parallel execution:")
    response = client.generate_content(
        "What's the weather in Tokyo and Paris? Also calculate 15 * 23 + 7",
        tools=[get_weather, simple_math],
        parallel_tool_execution=True
    )
    print(response)
    
    # Complex conversation with context
    print("\n5. Conversation with context and tools:")
    response = client.generate_content(
        "I'm planning a trip to San Francisco. Can you check the weather there and calculate the tip for a $85 restaurant bill at 18%?",
        tools=[get_weather, simple_math],
        max_tool_call_rounds=3
    )
    print(response)

if __name__ == "__main__":
    main()
