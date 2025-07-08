"""
Example demonstrating asynchronous parallel tool execution with Google Gemini.

This example shows how toolflow handles async parallel tool execution for
maximum performance when using multiple async tools simultaneously.
"""

import os
import asyncio
import time
import toolflow
import google.generativeai as genai
from dataclasses import dataclass
from typing import List
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

@dataclass
class ApiCall:
    name: str
    delay: float
    result: str

@toolflow.tool
async def fetch_user_profile(user_id: str) -> str:
    """Fetch user profile data asynchronously."""
    print(f"\n[Async Tool] Fetching profile for user {user_id}...")
    await asyncio.sleep(1.2)  # Simulate database query
    
    profiles = {
        "user123": "John Doe, Software Engineer, San Francisco",
        "user456": "Jane Smith, Data Scientist, New York", 
        "user789": "Bob Johnson, Product Manager, Seattle"
    }
    
    result = profiles.get(user_id, f"Profile not found for {user_id}")
    print(f"[Async Tool] Profile result: {result}")
    return result

@toolflow.tool
async def fetch_weather_forecast(city: str, days: int = 3) -> str:
    """Fetch weather forecast for multiple days asynchronously."""
    print(f"\n[Async Tool] Fetching {days}-day forecast for {city}...")
    await asyncio.sleep(1.5)  # Simulate weather API call
    
    forecasts = {
        "New York": ["Sunny 75°F", "Cloudy 68°F", "Rain 62°F"],
        "London": ["Overcast 60°F", "Fog 58°F", "Clear 65°F"],
        "Tokyo": ["Rain 70°F", "Sunny 78°F", "Cloudy 72°F"],
        "Paris": ["Clear 68°F", "Sunny 75°F", "Rain 60°F"]
    }
    
    city_forecast = forecasts.get(city, ["Data unavailable"] * days)
    result = f"{city} {days}-day forecast: " + ", ".join(city_forecast[:days])
    print(f"[Async Tool] Forecast result: {result}")
    return result

@toolflow.tool
async def calculate_financial_metrics(principal: float, rate: float, years: int) -> str:
    """Calculate various financial metrics asynchronously."""
    print(f"\n[Async Tool] Calculating financial metrics for ${principal} at {rate}% for {years} years...")
    await asyncio.sleep(0.8)  # Simulate complex calculations
    
    # Compound interest calculation
    compound_amount = principal * (1 + rate/100) ** years
    simple_interest = principal * rate * years / 100
    total_simple = principal + simple_interest
    
    result = f"Principal: ${principal}, Compound: ${compound_amount:.2f}, Simple: ${total_simple:.2f}"
    print(f"[Async Tool] Financial result: {result}")
    return result

@toolflow.tool
async def fetch_stock_data(symbols: List[str]) -> str:
    """Fetch stock data for multiple symbols asynchronously."""
    print(f"\n[Async Tool] Fetching stock data for {', '.join(symbols)}...")
    await asyncio.sleep(1.3)  # Simulate stock API call
    
    stock_prices = {
        "AAPL": "$175.25 (+2.1%)",
        "GOOGL": "$142.50 (+1.8%)",
        "MSFT": "$378.90 (+0.9%)",
        "TSLA": "$242.15 (-1.2%)",
        "NVDA": "$821.30 (+4.5%)",
        "AMZN": "$155.75 (+1.1%)"
    }
    
    results = []
    for symbol in symbols:
        price = stock_prices.get(symbol.upper(), "Data unavailable")
        results.append(f"{symbol.upper()}: {price}")
    
    result = "Stock prices: " + ", ".join(results)
    print(f"[Async Tool] Stock result: {result}")
    return result

@toolflow.tool
async def fetch_news_headlines(category: str, count: int = 3) -> str:
    """Fetch news headlines for a category asynchronously."""
    print(f"\n[Async Tool] Fetching {count} {category} headlines...")
    await asyncio.sleep(1.1)  # Simulate news API call
    
    news_data = {
        "technology": [
            "AI Model Achieves Breakthrough in Natural Language Understanding",
            "Major Tech Company Announces New Quantum Computing Initiative", 
            "Cybersecurity Firm Reports 40% Increase in Enterprise Adoption"
        ],
        "finance": [
            "Federal Reserve Maintains Interest Rates Amid Economic Uncertainty",
            "Cryptocurrency Market Shows Signs of Recovery After Recent Volatility",
            "Global Stock Markets React to Latest Economic Indicators"
        ],
        "sports": [
            "Championship Series Reaches Game 7 in Thrilling Matchup",
            "Olympic Training Centers Report Record Athlete Performance",
            "Major League Announces New Sustainability Initiative"
        ]
    }
    
    headlines = news_data.get(category.lower(), ["No headlines available"])[:count]
    result = f"{category.title()} news: " + "; ".join(headlines)
    print(f"[Async Tool] News result: {result[:100]}...")
    return result

@toolflow.tool
async def translate_batch(texts: List[str], target_language: str) -> str:
    """Translate multiple texts to target language asynchronously."""
    print(f"\n[Async Tool] Translating {len(texts)} texts to {target_language}...")
    await asyncio.sleep(1.0)  # Simulate translation API call
    
    translations = {
        "spanish": {
            "Hello": "Hola",
            "Thank you": "Gracias", 
            "Good morning": "Buenos días",
            "How are you?": "¿Cómo estás?",
            "Goodbye": "Adiós"
        },
        "french": {
            "Hello": "Bonjour",
            "Thank you": "Merci",
            "Good morning": "Bonjour",
            "How are you?": "Comment allez-vous?",
            "Goodbye": "Au revoir"
        }
    }
    
    lang_dict = translations.get(target_language.lower(), {})
    results = []
    for text in texts:
        translated = lang_dict.get(text, f"[{text}]")
        results.append(f"{text} -> {translated}")
    
    result = f"Translations to {target_language}: " + "; ".join(results)
    print(f"[Async Tool] Translation result: {result}")
    return result

async def example_sequential_vs_parallel():
    """Compare sequential vs parallel execution."""
    print("1. Sequential vs Parallel Execution Comparison:")
    
    # Sequential execution
    print("\nSequential execution:")
    start_time = time.time()
    response_seq = await client.generate_content_async(
        "Get user profile for user123, weather forecast for New York, and calculate compound interest for $1000 at 5% for 10 years",
        tools=[fetch_user_profile, fetch_weather_forecast, calculate_financial_metrics],
        parallel_tool_execution=False,  # Sequential
        max_tool_call_rounds=5
    )
    sequential_time = time.time() - start_time
    print(f"Sequential result: {response_seq}")
    print(f"Sequential time: {sequential_time:.2f} seconds")
    
    # Parallel execution
    print("\nParallel execution:")
    start_time = time.time()
    response_par = await client.generate_content_async(
        "Get user profile for user456, weather forecast for London, and calculate compound interest for $2000 at 4% for 15 years",
        tools=[fetch_user_profile, fetch_weather_forecast, calculate_financial_metrics],
        parallel_tool_execution=True,  # Parallel
        max_tool_call_rounds=5
    )
    parallel_time = time.time() - start_time
    print(f"Parallel result: {response_par}")
    print(f"Parallel time: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.1f}x faster")
    print("\n" + "="*60 + "\n")

async def example_complex_parallel_workflow():
    """Complex workflow with many parallel tools."""
    print("2. Complex Parallel Workflow:")
    
    start_time = time.time()
    response = await client.generate_content_async(
        """Create a comprehensive report that includes:
        1. User profiles for user123 and user789
        2. 5-day weather forecasts for Tokyo and Paris
        3. Stock data for AAPL, GOOGL, and TSLA
        4. Technology and finance news headlines
        5. Financial calculations for $5000 at 6% for 20 years
        6. Translate 'Hello' and 'Thank you' to Spanish and French""",
        tools=[
            fetch_user_profile,
            fetch_weather_forecast, 
            fetch_stock_data,
            fetch_news_headlines,
            calculate_financial_metrics,
            translate_batch
        ],
        parallel_tool_execution=True,
        max_tool_call_rounds=10
    )
    execution_time = time.time() - start_time
    
    print(f"Comprehensive report: {response}")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print("\n" + "="*60 + "\n")

async def example_async_streaming_with_tools():
    """Async streaming with parallel tools."""
    print("3. Async Streaming with Parallel Tools:")
    
    response = await client.generate_content_async(
        "Get weather forecasts for New York, London, and Tokyo, plus fetch the latest technology news headlines",
        tools=[fetch_weather_forecast, fetch_news_headlines],
        parallel_tool_execution=True
    )
    print(f"Response: {response}")
    print("\n" + "="*60 + "\n")

async def example_error_handling():
    """Demonstrate error handling in async parallel execution."""
    print("4. Error Handling in Async Parallel Execution:")
    
    @toolflow.tool
    async def failing_tool(should_fail: bool) -> str:
        """A tool that may fail to demonstrate error handling."""
        await asyncio.sleep(0.5)
        if should_fail:
            raise Exception("This tool intentionally failed")
        return "This tool succeeded"
    
    try:
        response = await client.generate_content_async(
            "Try to use both a working tool and a failing tool",
            tools=[failing_tool, fetch_weather_forecast],
            parallel_tool_execution=True,
            graceful_error_handling=True,  # Handle tool errors gracefully
            max_tool_call_rounds=3
        )
        print(f"Response with error handling: {response}")
    except Exception as e:
        print(f"Error caught: {e}")
    
    print("\n" + "="*60 + "\n")

async def example_concurrent_clients():
    """Multiple concurrent client requests."""
    print("5. Multiple Concurrent Client Requests:")
    
    # Create multiple independent requests that run concurrently
    async def task1():
        return await client.generate_content_async(
            "Get weather for New York",
            tools=[fetch_weather_forecast],
            parallel_tool_execution=True
        )
    
    async def task2():
        return await client.generate_content_async(
            "Get stock data for AAPL",
            tools=[fetch_stock_data],
            parallel_tool_execution=True
        )
    
    start_time = time.time()
    results = await asyncio.gather(task1(), task2())
    execution_time = time.time() - start_time
    
    for i, result in enumerate(results, 1):
        print(f"Concurrent request {i}: {result[:100]}...")
    
    print(f"\nAll concurrent requests completed in {execution_time:.2f} seconds")

async def main():
    """Main async function demonstrating parallel execution patterns."""
    print("=== Gemini Async Parallel Execution Example ===\n")
    
    await example_sequential_vs_parallel()
    await example_complex_parallel_workflow()
    await example_async_streaming_with_tools()
    await example_error_handling()
    await example_concurrent_clients()

if __name__ == "__main__":
    asyncio.run(main())
