"""
Example demonstrating async streaming support with tool calls.

This example shows how toolflow handles async streaming responses while still
supporting both sync and async tool calls with parallel execution.
"""

import asyncio
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
async def async_weather_tool(city: str) -> str:
    """Get weather for a city asynchronously."""
    print(f"\n[Async Tool] Getting weather for {city}...")
    await asyncio.sleep(1)  # Simulate async API call
    
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C", 
        "Tokyo": "Rainy, 22°C",
        "San Francisco": "Foggy, 18°C",
        "Paris": "Clear, 20°C"
    }
    result = weather_data.get(city, f"Weather data not available for {city}")
    print(f"[Async Tool] Weather result: {result}")
    return result

@toolflow.tool
async def async_fetch_url(url: str) -> str:
    """Fetch content from a URL asynchronously."""
    print(f"\n[Async Tool] Fetching URL: {url}")
    try:
        # Simulate fetching URL content
        await asyncio.sleep(0.5)
        
        # Mock response based on URL
        if "httpbin.org/json" in url:
            result = '{"message": "This is a test JSON response", "status": "success"}'
        elif "example.com" in url:
            result = "<html><body><h1>Example Domain</h1><p>This domain is for use in examples.</p></body></html>"
        else:
            result = f"Mock content from {url}"
            
        print(f"[Async Tool] Fetched {len(result)} characters from {url}")
        return result[:200] + "..." if len(result) > 200 else result
        
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

async def main():
    client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    print("=== Toolflow Async Streaming Example ===\n")
    
    # Example 1: Simple async streaming without tools
    print("1. Simple async streaming without tools:")
    print("Question: Write a haiku about programming.")
    print("Response: ", end="", flush=True)
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a haiku about programming."}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 2: Async streaming with mixed sync/async tools
    print("2. Async streaming with mixed sync/async tools:")
    print("Question: Get weather for Paris and Tokyo, calculate 15*8+12, and fetch data from httpbin.org/json")
    print("Response: ", end="", flush=True)
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Get weather for Paris and Tokyo, calculate 15*8+12, and fetch data from httpbin.org/json"}],
        tools=[sync_math_tool, async_weather_tool, async_fetch_url],
        stream=True
    )
    
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")
    
    # Example 3: Async streaming with parallel tool execution
    print("3. Async streaming with parallel tool execution:")
    print("Question: Get weather for New York, London, San Francisco, calculate 100*5-50 and 200/4, fetch from example.com")
    print("Response: ", end="", flush=True)
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Get weather for New York, London, San Francisco, calculate 100*5-50 and 200/4, and fetch content from example.com"}],
        tools=[sync_math_tool, async_weather_tool, async_fetch_url],
        parallel_tool_execution=True,  # This will run sync and async tools in parallel
        stream=True
    )
    
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    asyncio.run(main()) 