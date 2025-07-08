"""
Example demonstrating synchronous streaming with Google Gemini and toolflow.

This example shows how toolflow handles streaming responses while still
supporting tool calls with parallel execution.
"""

import os
import time
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
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    print(f"\n[Tool] Getting weather for {city}...")
    time.sleep(1)  # Simulate API delay
    
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C", 
        "Tokyo": "Rainy, 22°C",
        "San Francisco": "Foggy, 18°C",
        "Paris": "Clear, 20°C"
    }
    result = weather_data.get(city, f"Weather data not available for {city}")
    print(f"[Tool] Weather result: {result}")
    return result

@toolflow.tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression."""
    print(f"\n[Tool] Calculating: {expression}")
    time.sleep(0.5)  # Simulate computation
    try:
        result = eval(expression)
        print(f"[Tool] Math result: {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@toolflow.tool
def fetch_news_summary(topic: str) -> str:
    """Fetch a news summary for a given topic."""
    print(f"\n[Tool] Fetching news for {topic}...")
    time.sleep(1.2)  # Simulate API call delay
    
    # Mock news data
    news_data = {
        "technology": "Latest tech news: AI advances continue, new smartphone releases, and cloud computing growth.",
        "finance": "Financial markets update: Stock indices mixed, cryptocurrency volatile, interest rates stable.",
        "sports": "Sports highlights: Championship games, player transfers, and upcoming tournaments.",
        "science": "Science news: New research discoveries, space exploration updates, and medical breakthroughs.",
        "politics": "Political news: Policy updates, election coverage, and international relations."
    }
    
    result = news_data.get(topic.lower(), f"No news summary available for {topic}")
    print(f"[Tool] News result: {result[:50]}...")
    return result

def main():
    print("=== Gemini Streaming with Toolflow Example ===\n")
    
    # Example 1: Simple streaming without tools (default: content only)
    print("1. Simple streaming without tools:")
    print("Question: Write a short story about a robot learning to paint.")
    print("Response: ", end="", flush=True)
    
    stream = client.generate_content(
        "Write a short story about a robot learning to paint.",
        stream=True  # Default: yields content only
    )
    
    for content in stream:
        print(content, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 2: Streaming with full_response=True
    print("2. Streaming with full_response=True (full response objects):")
    print("Question: Explain quantum computing in simple terms.")
    print("Response: ", end="", flush=True)
    
    stream = client.generate_content(
        "Explain quantum computing in simple terms.",
        stream=True,
        full_response=True  # Returns full response objects
    )
    
    for chunk in stream:
        # Extract text from Gemini response chunk
        try:
            if hasattr(chunk, 'text') and chunk.text:
                print(chunk.text, end="", flush=True)
        except ValueError:
            # Handle cases where chunk has no text content
            pass
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 3: Streaming with tools (content only)
    print("3. Streaming with tools:")
    print("Question: Get weather for Paris and calculate 25*4+10")
    print("Response: ", end="", flush=True)
    
    stream = client.generate_content(
        "Get the weather for Paris and calculate 25*4+10",
        tools=[get_weather, calculate_math],
        stream=True  # Tools execute, then stream final response
    )
    
    for content in stream:
        print(content, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 4: Streaming with parallel tool execution
    print("4. Streaming with parallel tool execution:")
    print("Question: Get weather for London and Tokyo, calculate 15*8-20, and fetch tech news")
    print("Response: ", end="", flush=True)
    
    stream = client.generate_content(
        "Get weather for London and Tokyo, calculate 15*8-20, and get a technology news summary",
        tools=[get_weather, calculate_math, fetch_news_summary],
        parallel_tool_execution=True,
        stream=True
    )
    
    for content in stream:
        print(content, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 5: Client-level full_response=True
    print("5. Client with full_response=True (all responses are full objects):")
    full_client = toolflow.from_gemini(model, full_response=True)
    print("Question: Tell me about the benefits of renewable energy.")
    print("Response: ", end="", flush=True)
    
    stream = full_client.generate_content(
        "Tell me about the benefits of renewable energy.",
        stream=True
    )
    
    for chunk in stream:
        try:
            if hasattr(chunk, 'text') and chunk.text:
                print(chunk.text, end="", flush=True)
        except ValueError:
            pass
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 6: Streaming conversation with context
    print("6. Streaming conversation with context:")
    print("Question: Multi-turn conversation with tools")
    print("Response: ", end="", flush=True)
    
    stream = client.generate_content(
        "I'm planning a trip to San Francisco. Can you help me with weather and some quick calculations for my budget?",
        tools=[get_weather, calculate_math],
        stream=True
    )
    
    for content in stream:
        print(content, end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    main()
