#!/usr/bin/env python3
"""
Basic Llama provider example using toolflow.

This example demonstrates how to use Llama models through OpenRouter or similar
OpenAI-compatible services with toolflow's enhanced capabilities.

Requirements:
- pip install openai toolflow
- Set LLAMA_API_KEY environment variable
- Configure for your Llama model provider (e.g., OpenRouter)
"""

import os
import asyncio
from typing import List
from dataclasses import dataclass
import openai
import toolflow

# Configuration for OpenRouter (adjust for your provider)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLAMA_MODEL = "meta-llama/llama-3.1-70b-instruct"

@dataclass
class WeatherInfo:
    location: str
    temperature: int
    condition: str
    humidity: int

def setup_llama_client() -> openai.OpenAI:
    """Setup OpenAI client configured for Llama models through OpenRouter."""
    api_key = os.getenv('LLAMA_API_KEY')
    if not api_key:
        raise ValueError(
            "LLAMA_API_KEY environment variable is required. "
            "Get your key from OpenRouter or your Llama model provider."
        )
    
    return openai.OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

@toolflow.tool
def get_weather(city: str) -> WeatherInfo:
    """Get current weather information for a city."""
    # Simulate weather data (replace with real API call)
    weather_data = {
        "new york": WeatherInfo("New York, NY", 72, "Partly Cloudy", 65),
        "london": WeatherInfo("London, UK", 58, "Rainy", 80),
        "tokyo": WeatherInfo("Tokyo, Japan", 75, "Sunny", 70),
        "sydney": WeatherInfo("Sydney, Australia", 68, "Windy", 55),
    }
    
    return weather_data.get(city.lower(), WeatherInfo(city, 70, "Unknown", 50))

@toolflow.tool
def calculate_temperature_difference(temp1: int, temp2: int) -> dict:
    """Calculate the temperature difference between two values."""
    difference = abs(temp1 - temp2)
    warmer_temp = max(temp1, temp2)
    cooler_temp = min(temp1, temp2)
    
    return {
        "difference": difference,
        "warmer_temperature": warmer_temp,
        "cooler_temperature": cooler_temp,
        "significant_difference": difference > 10
    }

@toolflow.tool
def get_city_facts(city: str) -> dict:
    """Get interesting facts about a city."""
    facts_db = {
        "new york": {
            "population": "8.3 million",
            "famous_for": "Statue of Liberty, Central Park, Broadway",
            "timezone": "Eastern Time",
            "nickname": "The Big Apple"
        },
        "london": {
            "population": "9 million", 
            "famous_for": "Big Ben, Tower Bridge, British Museum",
            "timezone": "Greenwich Mean Time",
            "nickname": "The Big Smoke"
        },
        "tokyo": {
            "population": "14 million",
            "famous_for": "Mount Fuji views, Sushi, Technology",
            "timezone": "Japan Standard Time", 
            "nickname": "The Eastern Capital"
        },
        "sydney": {
            "population": "5.3 million",
            "famous_for": "Opera House, Harbour Bridge, Beaches",
            "timezone": "Australian Eastern Time",
            "nickname": "Harbour City"
        }
    }
    
    return facts_db.get(city.lower(), {
        "population": "Unknown",
        "famous_for": "Various attractions",
        "timezone": "Local time",
        "nickname": "Beautiful city"
    })

def demonstrate_basic_llama():
    """Demonstrate basic Llama model usage with toolflow."""
    print("🦙 Basic Llama Model Demo")
    print("=" * 50)
    
    # Setup client
    client = setup_llama_client()
    enhanced_client = toolflow.from_llama(client)
    
    # Basic text generation
    print("\n1. Basic Text Generation:")
    response = enhanced_client.chat.completions.create(
        model=DEFAULT_LLAMA_MODEL,
        messages=[
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        max_tokens=200,
        temperature=0.7
    )
    print(f"Response: {response.choices[0].message.content}")

def demonstrate_tool_calling():
    """Demonstrate tool calling with Llama models."""
    print("\n🔧 Tool Calling Demo")
    print("=" * 50)
    
    client = setup_llama_client() 
    enhanced_client = toolflow.from_llama(client)
    
    # Single tool call
    print("\n1. Single Tool Call - Weather:")
    response = enhanced_client.chat.completions.create(
        model=DEFAULT_LLAMA_MODEL,
        messages=[
            {"role": "user", "content": "What's the weather like in Tokyo?"}
        ],
        tools=[get_weather],
        max_tokens=500
    )
    print(f"Response: {response.choices[0].message.content}")
    
    # Multiple tool calls
    print("\n2. Multiple Tool Calls - Weather Comparison:")
    response = enhanced_client.chat.completions.create(
        model=DEFAULT_LLAMA_MODEL,
        messages=[
            {"role": "user", "content": "Compare the weather between New York and London, and tell me some facts about both cities."}
        ],
        tools=[get_weather, calculate_temperature_difference, get_city_facts],
        max_tokens=800,
        parallel_tool_calls=True
    )
    print(f"Response: {response.choices[0].message.content}")

def demonstrate_parallel_execution():
    """Demonstrate parallel tool execution performance."""
    print("\n⚡ Parallel Execution Demo")
    print("=" * 50)
    
    client = setup_llama_client()
    enhanced_client = toolflow.from_llama(client)
    
    import time
    
    # Test parallel execution
    print("\nTesting parallel tool execution:")
    start_time = time.time()
    
    response = enhanced_client.chat.completions.create(
        model=DEFAULT_LLAMA_MODEL,
        messages=[
            {"role": "user", "content": "Get weather and facts for New York, London, Tokyo, and Sydney. Then compare temperatures between New York and Tokyo."}
        ],
        tools=[get_weather, get_city_facts, calculate_temperature_difference],
        parallel_tool_calls=True,
        max_tokens=1200
    )
    
    execution_time = time.time() - start_time
    print(f"Parallel execution completed in {execution_time:.2f} seconds")
    print(f"Response: {response.choices[0].message.content}")

async def demonstrate_async_operations():
    """Demonstrate async operations with Llama models."""
    print("\n🔄 Async Operations Demo")
    print("=" * 50)
    
    # Setup async client
    async_client = openai.AsyncOpenAI(
        api_key=os.getenv('LLAMA_API_KEY'),
        base_url=OPENROUTER_BASE_URL,
    )
    enhanced_async_client = toolflow.from_llama(async_client)
    
    print("\n1. Async Text Generation:")
    response = await enhanced_async_client.chat.completions.create(
        model=DEFAULT_LLAMA_MODEL,
        messages=[
            {"role": "user", "content": "What are the benefits of using Llama models?"}
        ],
        max_tokens=300
    )
    print(f"Async Response: {response.choices[0].message.content}")
    
    print("\n2. Async Tool Calling:")
    response = await enhanced_async_client.chat.completions.create(
        model=DEFAULT_LLAMA_MODEL,
        messages=[
            {"role": "user", "content": "Get weather for Tokyo and some interesting facts about the city."}
        ],
        tools=[get_weather, get_city_facts],
        max_tokens=600
    )
    print(f"Async Tool Response: {response.choices[0].message.content}")

def main():
    """Run all Llama provider demonstrations."""
    print("🦙 Toolflow Llama Provider Examples")
    print("=" * 60)
    print("Using OpenRouter for Llama model access")
    print("Model:", DEFAULT_LLAMA_MODEL)
    print("=" * 60)
    
    try:
        # Basic demonstrations
        demonstrate_basic_llama()
        demonstrate_tool_calling()
        demonstrate_parallel_execution()
        
        # Async demonstration
        print("\nRunning async examples...")
        asyncio.run(demonstrate_async_operations())
        
        print("\n✅ All Llama provider examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure LLAMA_API_KEY is set in your environment")
        print("2. Verify your API key is valid for OpenRouter or your provider")
        print("3. Check that the model name is correct for your provider")
        print("4. Ensure you have sufficient API credits/quota")

if __name__ == "__main__":
    main()
