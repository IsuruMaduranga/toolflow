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

# Note: Since Gemini doesn't have native async support, we need to use sync tools
# but we can still demonstrate async patterns with asyncio.to_thread
# implement this temp part with real api currently i implement just dummy things

@toolflow.tool
def async_weather_tool(city: str) -> str:
    """Get weather for a city (sync implementation for Gemini compatibility)."""
    print(f"\n[Tool] Getting weather for {city}...")
    time.sleep(1)  # Simulate API call delay
    
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C", 
        "Tokyo": "Rainy, 22°C",
        "San Francisco": "Foggy, 18°C",
        "Paris": "Clear, 20°C",
        "Berlin": "Overcast, 12°C"
    }
    result = weather_data.get(city, f"Weather data not available for {city}")
    print(f"[Tool] Weather result for {city}: {result}")
    return result

@toolflow.tool
def async_calculate_tool(expression: str) -> str:
    """Calculate a mathematical expression (sync implementation for Gemini compatibility)."""
    print(f"\n[Tool] Calculating: {expression}")
    time.sleep(0.5)  # Simulate computation delay
    try:
        result = eval(expression)
        print(f"[Tool] Math result: {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@toolflow.tool
def async_translate_tool(text: str, target_language: str) -> str:
    """Translate text to a target language (sync implementation for Gemini compatibility)."""
    print(f"\n[Tool] Translating '{text[:30]}...' to {target_language}")
    time.sleep(1.2)  # Simulate API call delay
    
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
    print(f"[Tool] Translation result: {result}")
    return result

@toolflow.tool
def async_fetch_data(data_type: str) -> str:
    """Fetch data of a specific type (sync implementation for Gemini compatibility)."""
    print(f"\n[Tool] Fetching {data_type} data...")
    time.sleep(1.5)  # Simulate API call
    
    data_sources = {
        "news": "Breaking: Tech stocks rise 5% amid AI breakthroughs",
        "stocks": "AAPL: $175.25 (+2.1%), GOOGL: $142.50 (+1.8%)",
        "crypto": "BTC: $45,230 (+3.2%), ETH: $2,890 (+2.7%)",
        "sports": "NBA Finals: Lakers lead 3-2 against Celtics",
        "science": "New study reveals breakthrough in quantum computing"
    }
    
    result = data_sources.get(data_type.lower(), f"No data available for {data_type}")
    print(f"[Tool] Data result: {result}")
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
    
    # For Gemini streaming, we need to handle it differently since the stream returns objects
    def generate_streaming():
        stream = client.generate_content(
            "Explain the benefits of async programming in Python",
            stream=True
        )
        # Process the stream and collect text
        text_parts = []
        for chunk in stream:
            try:
                if hasattr(chunk, 'text') and chunk.text:
                    text_parts.append(chunk.text)
            except ValueError:
                # This happens when chunk has no valid text parts
                pass
        return "".join(text_parts)
    
    # Get the streaming response in a thread
    full_text = await asyncio.to_thread(generate_streaming)
    print(full_text)
    
    print("\n\n" + "="*60 + "\n")

async def example_complex_async_workflow():
    """Complex async workflow with multiple steps."""
    print("5. Complex async workflow:")
    
    # Step 1: Get weather data for multiple cities concurrently using threads
    weather_tasks = [
        asyncio.to_thread(async_weather_tool, "New York"),
        asyncio.to_thread(async_weather_tool, "Tokyo"), 
        asyncio.to_thread(async_weather_tool, "London")
    ]
    
    print("Getting weather data for multiple cities...")
    weather_results = await asyncio.gather(*weather_tasks)
    
    # Step 2: Fetch different types of data concurrently using threads
    data_tasks = [
        asyncio.to_thread(async_fetch_data, "news"),
        asyncio.to_thread(async_fetch_data, "stocks"),
        asyncio.to_thread(async_calculate_tool, "100 * 1.05 ** 10")  # Compound interest calculation
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
