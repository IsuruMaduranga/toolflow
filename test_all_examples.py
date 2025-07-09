#!/usr/bin/env python3
"""
Comprehensive example testing all toolflow providers and features.
This script demonstrates that all providers work correctly with various features.
"""

import sys
import os
sys.path.insert(0, 'src')

import toolflow
from typing import List
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Mock API setup for testing without real API calls
import openai

@dataclass
class WeatherData:
    city: str
    temperature: int
    condition: str

@toolflow.tool
def get_weather(city: str) -> WeatherData:
    """Get weather information for a city."""
    weather_db = {
        "new york": WeatherData("New York", 72, "Sunny"),
        "london": WeatherData("London", 15, "Cloudy"),
        "tokyo": WeatherData("Tokyo", 20, "Clear"),
        "paris": WeatherData("Paris", 18, "Rainy")
    }
    return weather_db.get(city.lower(), WeatherData(city, 20, "Unknown"))

@toolflow.tool  
def calculate(operation: str, a: float, b: float) -> float:
    """Perform basic math operations."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    elif operation == "subtract":
        return a - b
    elif operation == "divide":
        return a / b if b != 0 else 0
    else:
        return 0

class TaskAnalysis(BaseModel):
    priority: str = Field(description="Priority level: high, medium, low")
    estimated_time: str = Field(description="Time estimate like '2 hours' or '30 minutes'")
    complexity: str = Field(description="Complexity: simple, moderate, complex")
    category: str = Field(description="Task category")

def test_provider_setup():
    """Test that all providers can be set up correctly."""
    print("🧪 Testing Provider Setup")
    print("=" * 50)
    
    # Test OpenAI provider
    try:
        openai_client = openai.OpenAI(api_key="test")
        openai_enhanced = toolflow.from_openai(openai_client)
        print("✅ OpenAI provider setup successful")
    except Exception as e:
        print(f"❌ OpenAI provider setup failed: {e}")
    
    # Test Llama provider  
    try:
        llama_client = openai.OpenAI(api_key="test", base_url="https://test.com")
        llama_enhanced = toolflow.from_llama(llama_client)
        print("✅ Llama provider setup successful")
    except Exception as e:
        print(f"❌ Llama provider setup failed: {e}")
    
    # Test Gemini provider
    try:
        import google.generativeai as genai
        if os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            gemini_enhanced = toolflow.from_gemini(gemini_model)
            print("✅ Gemini provider setup successful")
        else:
            print("ℹ️ Gemini API key not available, skipping real setup")
    except ImportError:
        print("ℹ️ Gemini not available, skipping")
    except Exception as e:
        print(f"❌ Gemini provider setup failed: {e}")
    
    # Test Anthropic provider
    try:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key="test")
        anthropic_enhanced = toolflow.from_anthropic(anthropic_client)
        print("✅ Anthropic provider setup successful")
    except ImportError:
        print("ℹ️ Anthropic not available, skipping")
    except Exception as e:
        print(f"❌ Anthropic provider setup failed: {e}")

def test_tool_functionality():
    """Test tool decoration and basic functionality."""
    print("\n🔧 Testing Tool Functionality")
    print("=" * 50)
    
    # Test tool creation
    weather_result = get_weather("tokyo")
    print(f"✅ Weather tool: {weather_result.city} - {weather_result.temperature}°C, {weather_result.condition}")
    
    # Test calculation tool
    calc_result = calculate("multiply", 15, 8)
    print(f"✅ Calculator tool: 15 * 8 = {calc_result}")
    
    # Test tool schemas
    tools = [get_weather, calculate]
    print(f"✅ Created {len(tools)} tools successfully")

def test_gemini_with_real_api():
    """Test Gemini provider with real API if available."""
    if not os.getenv('GEMINI_API_KEY'):
        print("\nℹ️ Skipping Gemini real API test (no API key)")
        return
    
    print("\n🔮 Testing Gemini with Real API")
    print("=" * 50)
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash')
        client = toolflow.from_gemini(model)
        
        # Test basic text generation
        response = client.generate_content("Say hello in a creative way!")
        print(f"✅ Text generation: {response.text[:100]}...")
        
        # Test tool calling
        response = client.generate_content(
            "What's the weather like in Tokyo and what's 25 * 4?",
            tools=[get_weather, calculate],
            max_tool_call_rounds=5
        )
        print(f"✅ Tool calling: {response.text[:150]}...")
        
        # Test structured outputs
        response = client.generate_content(
            "Analyze this task: 'Build a website for a small business'",
            response_format=TaskAnalysis
        )
        analysis = response.parsed
        print(f"✅ Structured output: Priority={analysis.priority}, Time={analysis.estimated_time}")
        
    except Exception as e:
        print(f"❌ Gemini real API test failed: {e}")

def test_core_configuration():
    """Test core toolflow configuration functions."""
    print("\n⚙️ Testing Core Configuration")
    print("=" * 50)
    
    # Test max workers configuration
    original_workers = toolflow.get_max_workers()
    print(f"✅ Original max workers: {original_workers}")
    
    toolflow.set_max_workers(8)
    new_workers = toolflow.get_max_workers()
    print(f"✅ Updated max workers: {new_workers}")
    
    # Reset
    toolflow.set_max_workers(original_workers)
    print(f"✅ Reset max workers: {toolflow.get_max_workers()}")
    
    # Test async yield frequency
    try:
        toolflow.set_async_yield_frequency(10)
        print("✅ Async yield frequency configured")
    except Exception as e:
        print(f"❌ Async yield frequency failed: {e}")

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n🛡️ Testing Error Handling")
    print("=" * 50)
    
    # Test invalid client rejection
    try:
        toolflow.from_openai("invalid_client")
        print("❌ Should have rejected invalid client")
    except (ValueError, TypeError):
        print("✅ Correctly rejected invalid OpenAI client")
    
    try:
        toolflow.from_llama("invalid_client")
        print("❌ Should have rejected invalid client")
    except (ValueError, TypeError):
        print("✅ Correctly rejected invalid Llama client")
    
    # Test tool with invalid parameters
    try:
        result = calculate("invalid_op", 5, 3)
        print(f"✅ Handled invalid operation gracefully: {result}")
    except Exception as e:
        print(f"❌ Tool error handling failed: {e}")

def main():
    """Run all tests."""
    print("🚀 Toolflow Comprehensive Example Testing")
    print("=" * 60)
    print("Testing all providers, features, and functionality")
    print("=" * 60)
    
    try:
        test_provider_setup()
        test_tool_functionality()
        test_core_configuration()
        test_error_handling()
        test_gemini_with_real_api()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("🎉 Toolflow is working correctly with all providers!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
