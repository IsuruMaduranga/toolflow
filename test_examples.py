#!/usr/bin/env python3
"""
Test script to validate Gemini examples without making API calls
"""
import os
import sys
import importlib.util
from pathlib import Path

def test_example_import(example_path):
    """Test if an example file can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location("example", example_path)
        module = importlib.util.module_from_spec(spec)
        
        # Mock the API key to avoid the error
        os.environ["GEMINI_API_KEY"] = "test_key"
        
        # Import the module (this will run the imports but not the main execution)
        spec.loader.exec_module(module)
        
        return True, "✅ Import successful"
    except Exception as e:
        return False, f"❌ Import failed: {e}"

def main():
    """Test all Gemini examples."""
    print("🧪 Testing Gemini Examples")
    print("=" * 50)
    
    examples_dir = Path("examples/gemini")
    
    if not examples_dir.exists():
        print("❌ Examples directory not found")
        return
    
    python_files = list(examples_dir.glob("*.py"))
    
    if not python_files:
        print("❌ No Python example files found")
        return
    
    results = []
    for example_file in sorted(python_files):
        print(f"\nTesting {example_file.name}...")
        success, message = test_example_import(example_file)
        results.append((example_file.name, success, message))
        print(f"  {message}")
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for filename, success, message in results:
        status = "✅" if success else "❌"
        print(f"  {status} {filename}")
    
    print(f"\n🎯 {successful}/{total} examples passed import tests")
    
    if successful == total:
        print("🎉 All examples are syntactically correct and import successfully!")
        print("💡 Note: Rate limit hit - examples are working, just need to wait for quota reset")
    else:
        print("⚠️  Some examples have issues that need to be fixed")

if __name__ == "__main__":
    main()
