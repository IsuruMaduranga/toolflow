#!/usr/bin/env python3
"""
Quick Gemini Models Checker

A simple utility to check available Gemini models and test API connectivity.
Can be run standalone or imported as a module.

Usage:
    python gemini_models_check.py
    
Environment Variables:
    GEMINI_API_KEY - Your Google AI API key
"""

import os
import sys

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import google.generativeai as genai
        return True, genai
    except ImportError:
        return False, None

def quick_model_list():
    """Display a quick list of popular Gemini models."""
    models = {
        "gemini-1.5-flash": {
            "description": "Fast, efficient model for most tasks",
            "best_for": "General use, tool calling, fast responses",
            "context": "1M tokens"
        },
        "gemini-1.5-flash-8b": {
            "description": "Smaller, faster version of Flash",
            "best_for": "Cost-effective, simple tasks",
            "context": "1M tokens"
        },
        "gemini-1.5-pro": {
            "description": "Most capable model for complex reasoning",
            "best_for": "Complex analysis, multimodal tasks",
            "context": "2M tokens"
        },
        "gemini-1.0-pro": {
            "description": "Reliable model for production use",
            "best_for": "Stable production applications",
            "context": "32K tokens"
        }
    }
    
    print("🤖 Popular Gemini Models\n")
    for model_name, info in models.items():
        print(f"📝 {model_name}")
        print(f"   • {info['description']}")
        print(f"   • Best for: {info['best_for']}")
        print(f"   • Context: {info['context']}")
        print()

def check_api_key():
    """Check if API key is available."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY found")
        return api_key
    else:
        print("❌ GEMINI_API_KEY not found")
        print("   Set it with: export GEMINI_API_KEY='your-key-here'")
        print("   Or create a .env file with: GEMINI_API_KEY=your-key-here")
        return None

def test_api_connection(genai, api_key):
    """Test basic API connectivity."""
    try:
        genai.configure(api_key=api_key)
        
        # Try to list models (minimal API call)
        models = list(genai.list_models())
        
        print(f"✅ API connection successful")
        print(f"✅ Found {len(models)} available models")
        
        # Test with a simple generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello!")
        
        print("✅ Basic generation test successful")
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def list_all_models(genai, api_key):
    """List all available models with details."""
    try:
        genai.configure(api_key=api_key)
        
        print("\n🔍 All Available Models:\n")
        
        for model in genai.list_models():
            print(f"📋 {model.name}")
            
            if hasattr(model, 'display_name'):
                print(f"   Display Name: {model.display_name}")
            
            if hasattr(model, 'description'):
                print(f"   Description: {model.description}")
            
            # Token limits
            if hasattr(model, 'input_token_limit'):
                print(f"   Input Tokens: {model.input_token_limit:,}")
            if hasattr(model, 'output_token_limit'):
                print(f"   Output Tokens: {model.output_token_limit:,}")
            
            # Supported methods
            if hasattr(model, 'supported_generation_methods'):
                methods = ', '.join(model.supported_generation_methods)
                print(f"   Methods: {methods}")
            
            print()
            
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

def main():
    """Main function."""
    print("🚀 Gemini Models Checker\n")
    
    # Check dependencies
    deps_available, genai = check_dependencies()
    if not deps_available:
        print("❌ google-generativeai not installed")
        print("   Install with: pip install google-generativeai")
        return 1
    
    print("✅ Dependencies available")
    
    # Show quick model list first
    quick_model_list()
    
    # Check API key
    api_key = check_api_key()
    if not api_key:
        print("\n💡 To test API connectivity, set your GEMINI_API_KEY")
        return 0
    
    # Test API connection
    print("\n🔗 Testing API Connection...")
    if test_api_connection(genai, api_key):
        
        # Ask if user wants full model list
        try:
            choice = input("\n📋 Show detailed model list? (y/N): ").strip().lower()
            if choice in ['y', 'yes']:
                list_all_models(genai, api_key)
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return 0
    
    print("\n🎯 Usage with Toolflow:")
    print("""
import os
import toolflow
import google.generativeai as genai

# Configure
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create model (recommended: gemini-1.5-flash)
model = genai.GenerativeModel('gemini-1.5-flash')

# Use with toolflow
client = toolflow.from_gemini(model)
response = client.generate_content("Hello!")
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
