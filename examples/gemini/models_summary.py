"""
Gemini Models Summary

Quick reference for the most important Gemini models to use with toolflow.
"""

def print_recommended_models():
    """Print recommended Gemini models with usage guidance."""
    
    print("🚀 RECOMMENDED GEMINI MODELS FOR TOOLFLOW\n")
    
    models = [
        {
            "name": "gemini-2.0-flash",
            "description": "Latest Gemini 2.0 - fastest with great capabilities",
            "context": "1M tokens",
            "output": "8K tokens", 
            "best_for": "Most applications, tool calling, production use",
            "priority": "⭐⭐⭐⭐⭐"
        },
        {
            "name": "gemini-1.5-flash",
            "description": "Proven, stable, fast model",
            "context": "1M tokens",
            "output": "8K tokens",
            "best_for": "Reliable production applications, tool calling",
            "priority": "⭐⭐⭐⭐"
        },
        {
            "name": "gemini-1.5-flash-8b",
            "description": "Smaller, cost-effective version",
            "context": "1M tokens", 
            "output": "8K tokens",
            "best_for": "Cost-sensitive applications, simple tasks",
            "priority": "⭐⭐⭐"
        },
        {
            "name": "gemini-1.5-pro",
            "description": "Most capable for complex reasoning",
            "context": "2M tokens",
            "output": "8K tokens",
            "best_for": "Complex analysis, multimodal tasks, reasoning",
            "priority": "⭐⭐⭐⭐"
        },
        {
            "name": "gemini-2.5-flash",
            "description": "Latest 2.5 generation - cutting edge",
            "context": "1M tokens",
            "output": "64K tokens",
            "best_for": "Latest features, large outputs, experimental",
            "priority": "⭐⭐⭐⭐⭐"
        }
    ]
    
    for model in models:
        print(f"{model['priority']} {model['name']}")
        print(f"   📝 {model['description']}")
        print(f"   📊 Context: {model['context']} | Output: {model['output']}")
        print(f"   🎯 Best for: {model['best_for']}")
        print()
    
    print("💡 QUICK START:")
    print("""
import os
import toolflow
import google.generativeai as genai

# Configure API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Recommended: Start with gemini-2.0-flash
model = genai.GenerativeModel('gemini-2.0-flash')
client = toolflow.from_gemini(model)

# Use with tools
@toolflow.tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72°F"

response = client.generate_content(
    "What's the weather in San Francisco?",
    tools=[get_weather]
)
print(response)
    """)

if __name__ == "__main__":
    print_recommended_models()
