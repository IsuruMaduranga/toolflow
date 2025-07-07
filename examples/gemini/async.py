"""
Example demonstrating asynchronous operations with Google Gemini and toolflow.

This example shows how to use async/await with Gemini for better performance
in applications that need to handle multiple concurrent requests.
"""

import os
import asyncio
import time
import toolflow
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Note: Gemini doesn't have native async support, but toolflow can still provide async interfaces
client = toolflow.from_gemini(model)

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
        "Paris": "Clear, 20°C",
        "Berlin": "Overcast, 12°C"
    }
    result = weather_data.get(city, f"Weather data not available for {city}")
    print(f"[Async Tool] Weather result for {city}: {result}")
    return result

@toolflow.tool
async def async_calculate_tool(expression: str) -> str:
    """Calculate a mathematical expression asynchronously."""
    print(f"\n[Async Tool] Calculating: {expression}")
    await asyncio.sleep(0.5)  # Simulate async computation
    try:
        result = eval(expression)
        print(f"[Async Tool] Math result: {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@toolflow.tool
async def async_translate_tool(text: str, target_language: str) -> str:
    """Translate text to a target language asynchronously."""
    print(f"\n[Async Tool] Translating '{text[:30]}...' to {target_language}")
    await asyncio.sleep(1.2)  # Simulate API call delay
    
    # Mock translation service
    translations = {
        "spanish": {
            "Hello, how are you?": "Hola, ¿cómo estás?",
            "Good morning": "Buenos días",
            "Thank you": "Gracias",
            "The weather is nice today": "El clima está agradable hoy"
        },
        "french": {
            "Hello, how are you?": "Bonjour, comment allez-vous?",
            "Good morning": "Bonjour",
            "Thank you": "Merci",
            "The weather is nice today": "Il fait beau aujourd'hui"
        },
        "german": {
            "Hello, how are you?": "Hallo, wie geht es dir?",
            "Good morning": "Guten Morgen",
            "Thank you": "Danke",
            "The weather is nice today": "Das Wetter ist heute schön"
        }
    }
    
    result = translations.get(target_language.lower(), {}).get(text, f"Translation not available for '{text}' to {target_language}")
    print(f"[Async Tool] Translation result: {result}")
    return result

@toolflow.tool
async def async_fetch_data(data_type: str) -> str:
    """Fetch data of a specific type asynchronously."""
    print(f"\n[Async Tool] Fetching {data_type} data...")
    await asyncio.sleep(1.5)  # Simulate API call
    
    data_sources = {
        "news": "Breaking: Tech stocks rise 5% amid AI breakthroughs",
        "stocks": "AAPL: $175.25 (+2.1%), GOOGL: $142.50 (+1.8%)",
        "crypto": "BTC: $45,230 (+3.2%), ETH: $2,890 (+2.7%)",
        "sports": "NBA Finals: Lakers lead 3-2 against Celtics",
        "science": "New study reveals breakthrough in quantum computing"
    }
    
    result = data_sources.get(data_type.lower(), f"No data available for {data_type}")
    print(f"[Async Tool] Data result: {result}")
    return result

async def example_basic_async():
    """Basic async example."""
    print("1. Basic async text generation:")
    
    # Since Gemini doesn't have native async, we use asyncio.to_thread for non-blocking execution
    response = await asyncio.to_thread(
        client.generate_content, 
        "Write a haiku about artificial intelligence"
    )
    print(f"Response: {response}")
    print("\n" + "="*60 + "\n")

async def example_async_tools():
    """Async tools example."""
    print("2. Async tools with concurrent execution:")
    
    start_time = time.time()
    response = await asyncio.to_thread(
        client.generate_content,
        "Get weather for New York and London, calculate 25*8+15, and translate 'Good morning' to Spanish",
        tools=[async_weather_tool, async_calculate_tool, async_translate_tool],
        parallel_tool_execution=True  # This enables true async parallel execution
    )
    execution_time = time.time() - start_time
    
    print(f"Response: {response}")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print("\n" + "="*60 + "\n")

async def example_concurrent_requests():
    """Multiple concurrent requests example."""
    print("3. Multiple concurrent requests:")
    
    # Create multiple tasks that run concurrently using asyncio.to_thread
    tasks = [
        asyncio.to_thread(client.generate_content, "What's the capital of France?"),
        asyncio.to_thread(client.generate_content, "Calculate the square root of 144"),
        asyncio.to_thread(client.generate_content, "Tell me a joke about programming"),
        asyncio.to_thread(client.generate_content, "What year was Python first released?")
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    execution_time = time.time() - start_time
    
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {result}")
    
    print(f"\nAll {len(tasks)} requests completed in {execution_time:.2f} seconds")
    print("\n" + "="*60 + "\n")

async def example_async_streaming():
    """Async streaming example."""
    print("4. Async streaming:")
    print("Question: Explain the benefits of async programming")
    print("Response: ", end="", flush=True)
    
    # Note: For streaming, we need to handle it differently since it's an iterator
    def generate_streaming():
        return client.generate_content(
            "Explain the benefits of async programming in Python",
            stream=True
        )
    
    # Get the streaming response in a thread
    stream_response = await asyncio.to_thread(generate_streaming)
    
    # Process the stream (this part is synchronous but fast)
    for content in stream_response:
        print(content, end="", flush=True)
    
    print("\n\n" + "="*60 + "\n")

async def example_complex_async_workflow():
    """Complex async workflow with multiple steps."""
    print("5. Complex async workflow:")
    
    # Step 1: Get weather data for multiple cities concurrently
    weather_tasks = [
        async_weather_tool("New York"),
        async_weather_tool("Tokyo"), 
        async_weather_tool("London")
    ]
    
    print("Getting weather data for multiple cities...")
    weather_results = await asyncio.gather(*weather_tasks)
    
    # Step 2: Fetch different types of data concurrently
    data_tasks = [
        async_fetch_data("news"),
        async_fetch_data("stocks"),
        async_calculate_tool("100 * 1.05 ** 10")  # Compound interest calculation
    ]
    
    print("Fetching various data types...")
    data_results = await asyncio.gather(*data_tasks)
    
    # Step 3: Use all the gathered information in a final AI request
    context = f"""
    Weather Data: {', '.join(weather_results)}
    Market Data: {data_results[1]}
    News: {data_results[0]}
    Investment Calculation: {data_results[2]}
    """
    
    print("Generating final analysis...")
    final_response = await asyncio.to_thread(
        client.generate_content,
        f"Based on this information, provide a brief summary and analysis: {context}"
    )
    
    print(f"Final Analysis: {final_response}")

async def main():
    """Main async function demonstrating various async patterns."""
    print("=== Gemini Async with Toolflow Example ===\n")
    
    await example_basic_async()
    await example_async_tools()
    await example_concurrent_requests()
    await example_async_streaming()
    await example_complex_async_workflow()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
