"""
Example demonstrating synchronous streaming support with tool calls.

This example shows how toolflow handles sync streaming responses while still
supporting both sync and async tool calls with parallel execution.
"""

import os
import time
import toolflow
import openai

@toolflow.tool
def sync_math_tool(expression: str) -> str:
    """Calculate a mathematical expression (sync version)."""
    print(f"\n[Sync Tool] Calculating: {expression}")
    time.sleep(0.5)  # Simulate computation
    try:
        result = eval(expression)
        print(f"[Sync Tool] Math result: {result}")
        return str(result)
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@toolflow.tool
def sync_weather_tool(city: str) -> str:
    """Get weather for a city synchronously."""
    print(f"\n[Sync Tool] Getting weather for {city}...")
    time.sleep(1)  # Simulate API call delay
    
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C", 
        "Tokyo": "Rainy, 22°C",
        "San Francisco": "Foggy, 18°C",
        "Paris": "Clear, 20°C"
    }
    result = weather_data.get(city, f"Weather data not available for {city}")
    print(f"[Sync Tool] Weather result: {result}")
    return result

@toolflow.tool
def sync_fetch_url(url: str) -> str:
    """Fetch content from a URL synchronously."""
    print(f"\n[Sync Tool] Fetching URL: {url}")
    try:
        # Simulate fetching URL content
        time.sleep(0.5)
        
        # Mock response based on URL
        if "httpbin.org/json" in url:
            result = '{"message": "This is a test JSON response", "status": "success"}'
        elif "example.com" in url:
            result = "<html><body><h1>Example Domain</h1><p>This domain is for use in examples.</p></body></html>"
        else:
            result = f"Mock content from {url}"
            
        print(f"[Sync Tool] Fetched {len(result)} characters from {url}")
        return result[:200] + "..." if len(result) > 200 else result
        
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

def main():
    client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    print("=== Toolflow Sync Streaming Example ===\n")
    
    # Example 1: Simple sync streaming without tools (default: content only)
    print("1. Simple sync streaming without tools (default: content only):")
    print("Question: Write a haiku about programming.")
    print("Response: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a haiku about programming."}],
        stream=True  # Default: full_response=False, yields content only
    )
    
    for content in stream:
        print(content, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 2: Streaming with full_response=True (traditional behavior)
    print("2. Streaming with full_response=True (full chunk objects):")
    print("Question: Write a short poem about Python.")
    print("Response: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a short poem about Python."}],
        stream=True,
        full_response=True  # Returns full chunk objects
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 3: Sync streaming with tools (default: content only)
    print("3. Sync streaming with tools (default: content only):")
    print("Question: Get weather for Paris and calculate 15*8+12")
    print("Response: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Get weather for Paris and calculate 15*8+12"}],
        tools=[sync_math_tool, sync_weather_tool],
        stream=True  # Default: yields content only
    )
    
    for content in stream:
        print(content, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 4: Sync streaming with parallel tool execution
    print("4. Sync streaming with parallel tool execution:")
    print("Question: Get weather for New York, London, San Francisco, calculate 100*5-50 and 200/4, fetch from example.com")
    print("Response: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Get weather for New York, London, San Francisco, calculate 100*5-50 and 200/4, and fetch content from example.com"}],
        tools=[sync_math_tool, sync_weather_tool, sync_fetch_url],
        parallel_tool_execution=True,  # This will run sync tools in parallel using thread pools
        stream=True
    )
    
    for content in stream:
        print(content, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 5: Client-level full_response=True
    print("5. Client with full_response=True (all responses are full objects):")
    full_client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")), full_response=True)
    print("Question: Tell me a joke.")
    print("Response: ", end="", flush=True)
    
    stream = full_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    main()
