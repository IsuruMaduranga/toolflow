"""
Simple script to list available Google Gemini models.

This script connects to the Gemini API and retrieves all available models
with their capabilities and details.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_gemini_models():
    """List all available Gemini models with their details."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ Google Generative AI library not installed.")
        print("Install with: pip install google-generativeai")
        return

    # Configure API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set.")
        print("Please set your API key in a .env file or environment variable.")
        return

    try:
        genai.configure(api_key=api_key)
        
        print("=== Available Gemini Models ===\n")
        
        # List all models
        models = genai.list_models()
        
        for model in models:
            print(f"📝 Model: {model.name}")
            print(f"   Display Name: {model.display_name}")
            print(f"   Description: {model.description}")
            
            # Check supported generation methods
            supported_methods = []
            if hasattr(model, 'supported_generation_methods'):
                supported_methods = model.supported_generation_methods
                print(f"   Supported Methods: {', '.join(supported_methods)}")
            
            # Check input/output token limits
            if hasattr(model, 'input_token_limit'):
                print(f"   Input Token Limit: {model.input_token_limit:,}")
            if hasattr(model, 'output_token_limit'):
                print(f"   Output Token Limit: {model.output_token_limit:,}")
            
            print()
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        print("Please check your API key and internet connection.")

def get_recommended_models():
    """Get recommended models for different use cases."""
    print("=== Recommended Models by Use Case ===\n")
    
    recommendations = {
        "🚀 General Use (Fast & Efficient)": [
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b"
        ],
        "🧠 Complex Reasoning": [
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ],
        "👁️ Multimodal (Text + Images)": [
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "💰 Cost-Effective": [
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash"
        ]
    }
    
    for use_case, models in recommendations.items():
        print(f"{use_case}:")
        for model in models:
            print(f"   • {model}")
        print()

def test_model_access():
    """Test access to a specific model."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ Google Generative AI library not installed.")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set.")
        return

    try:
        genai.configure(api_key=api_key)
        
        # Test with gemini-1.5-flash (most common model)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        print("=== Testing Model Access ===\n")
        print("✅ Successfully connected to Gemini API")
        print("✅ Model 'gemini-1.5-flash' is accessible")
        
        # Test a simple generation
        response = model.generate_content("Say 'Hello from Gemini!'")
        print(f"✅ Test generation successful: {response.text}")
        
    except Exception as e:
        print(f"❌ Error testing model access: {e}")

if __name__ == "__main__":
    print("🤖 Gemini Models Explorer\n")
    
    # Show recommendations first
    get_recommended_models()
    
    # Test basic access
    test_model_access()
    
    # List all available models
    list_gemini_models()
    
    print("=== Usage with Toolflow ===")
    print("""
To use any of these models with toolflow:

```python
import os
import toolflow
import google.generativeai as genini

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create model (replace with your preferred model)
model = genai.GenerativeModel('gemini-1.5-flash')

# Wrap with toolflow
client = toolflow.from_gemini(model)

# Use with tools
response = client.generate_content(
    "Your prompt here",
    tools=[your_tool_function]
)
```
    """)
