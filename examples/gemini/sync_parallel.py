"""
Example demonstrating synchronous tool calling with parallel execution using Gemini.

This example shows how toolflow handles parallel tool execution to improve
performance when multiple tools can be run simultaneously.
"""

import os
import time
import toolflow
import google.generativeai as genai
from dataclasses import dataclass
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
class WeatherRequest:
    city: str
    units: str = "metric"

@toolflow.tool
def get_weather(request: WeatherRequest) -> str:
    """Get weather information for a city."""
    print(f"\n[Tool] Getting weather for {request.city} in {request.units} units...")
    time.sleep(1)  # Simulate API delay
    
    weather_data = {
        "New York": {"metric": "Sunny, 22°C", "imperial": "Sunny, 72°F"},
        "London": {"metric": "Cloudy, 15°C", "imperial": "Cloudy, 59°F"},
        "Tokyo": {"metric": "Rainy, 20°C", "imperial": "Rainy, 68°F"},
        "Paris": {"metric": "Clear, 18°C", "imperial": "Clear, 64°F"},
        "Sydney": {"metric": "Windy, 25°C", "imperial": "Windy, 77°F"},
        "Berlin": {"metric": "Overcast, 12°C", "imperial": "Overcast, 54°F"}
    }
    
    result = weather_data.get(request.city, {}).get(request.units, f"Weather data not available for {request.city}")
    print(f"[Tool] Weather result: {result}")
    return result

@toolflow.tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression."""
    print(f"\n[Tool] Calculating: {expression}")
    time.sleep(0.5)  # Simulate computation
    try:
        result = eval(expression.replace("^", "**"))  # Allow ^ for exponentiation
        print(f"[Tool] Math result: {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@toolflow.tool
def fetch_stock_price(symbol: str) -> str:
    """Fetch stock price for a given symbol."""
    print(f"\n[Tool] Fetching stock price for {symbol}...")
    time.sleep(1.5)  # Simulate API delay
    
    # Mock stock data
    stock_data = {
        "AAPL": "$175.25",
        "GOOGL": "$142.50",
        "MSFT": "$378.90",
        "TSLA": "$242.15",
        "NVDA": "$821.30",
        "AMZN": "$155.75"
    }
    
    result = stock_data.get(symbol.upper(), f"Stock data not available for {symbol}")
    print(f"[Tool] Stock result: {result}")
    return f"{symbol.upper()}: {result}"

@toolflow.tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency from one type to another."""
    print(f"\n[Tool] Converting {amount} {from_currency} to {to_currency}...")
    time.sleep(0.8)  # Simulate API delay
    
    # Mock exchange rates (relative to USD)
    rates = {
        "USD": 1.0,
        "EUR": 0.85,
        "GBP": 0.73,
        "JPY": 110.0,
        "CAD": 1.25,
        "AUD": 1.35
    }
    
    if from_currency not in rates or to_currency not in rates:
        return f"Currency conversion not available for {from_currency} to {to_currency}"
    
    # Convert to USD first, then to target currency
    usd_amount = amount / rates[from_currency]
    converted_amount = usd_amount * rates[to_currency]
    
    result = f"{amount} {from_currency} = {converted_amount:.2f} {to_currency}"
    print(f"[Tool] Conversion result: {result}")
    return result

def main():
    print("=== Gemini Parallel Tool Execution Example ===\n")
    
    # Example 1: Sequential tool execution (default)
    print("1. Sequential tool execution (parallel_tool_execution=False):")
    print("Question: Get weather for London, calculate 15*8+25, and get AAPL stock price")
    
    start_time = time.time()
    response = client.generate_content(
        "Get the weather for London, calculate 15*8+25, and fetch the stock price for AAPL",
        tools=[get_weather, calculate_math, fetch_stock_price],
        parallel_tool_execution=False,  # Tools run one after another
        max_tool_call_rounds=5
    )
    sequential_time = time.time() - start_time
    
    print(f"Response: {response}")
    print(f"Time taken: {sequential_time:.2f} seconds\n")
    print("="*60 + "\n")
    
    # Example 2: Parallel tool execution
    print("2. Parallel tool execution (parallel_tool_execution=True):")
    print("Question: Get weather for Tokyo and Berlin, calculate 100/4+50, and get GOOGL stock price")
    
    start_time = time.time()
    response = client.generate_content(
        "Get the weather for Tokyo and Berlin, calculate 100/4+50, and fetch the stock price for GOOGL",
        tools=[get_weather, calculate_math, fetch_stock_price],
        parallel_tool_execution=True,  # Tools run in parallel when possible
        max_tool_call_rounds=5
    )
    parallel_time = time.time() - start_time
    
    print(f"Response: {response}")
    print(f"Time taken: {parallel_time:.2f} seconds\n")
    print("="*60 + "\n")
    
    # Example 3: Complex parallel execution with multiple tool types
    print("3. Complex parallel execution with currency conversion:")
    print("Question: Weather for NYC and Paris, convert $100 to EUR and GBP, and calculate 25^2")
    
    start_time = time.time()
    response = client.generate_content(
        "Get weather for New York and Paris, convert $100 USD to EUR and to GBP, calculate 25^2, and get TSLA stock price",
        tools=[get_weather, calculate_math, convert_currency, fetch_stock_price],
        parallel_tool_execution=True,
        max_tool_call_rounds=8
    )
    complex_time = time.time() - start_time
    
    print(f"Response: {response}")
    print(f"Time taken: {complex_time:.2f} seconds\n")
    print("="*60 + "\n")
    
    # Example 4: Using dataclass for complex parameters
    print("4. Using dataclass parameters with parallel execution:")
    print("Question: Get weather in imperial units for Sydney")
    
    response = client.generate_content(
        "Get the weather for Sydney in imperial units (Fahrenheit)",
        tools=[get_weather],
        max_tool_call_rounds=3
    )
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
