#!/usr/bin/env python3
"""
Live Integration Test Runner for Toolflow

This script runs comprehensive integration tests using your actual OpenAI API key.
It will consume OpenAI credits, so use responsibly.

Usage:
    python run_live_tests.py
    
Set your OpenAI API key:
    export OPENAI_API_KEY='your-api-key-here'
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking requirements...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return False
    
    # Check if openai package is available
    try:
        import openai
        print("✅ OpenAI package available")
    except ImportError:
        print("❌ OpenAI package not available")
        print("Install it with: pip install openai")
        return False
    
    # Check if pydantic is available
    try:
        import pydantic
        print("✅ Pydantic package available")
    except ImportError:
        print("⚠️ Pydantic package not available - some tests will be skipped")
    
    # Check if toolflow is available
    try:
        import toolflow
        print("✅ Toolflow package available")
    except ImportError:
        print("❌ Toolflow package not available")
        print("Make sure you're in the correct directory and toolflow is installed")
        return False
    
    return True

def run_quick_test():
    """Run a quick test to verify the setup works."""
    print("\n🚀 Running quick verification test...")
    
    try:
        import openai
        import toolflow
        
        @toolflow.tool
        def test_add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2 + 3?"}],
            tools=[test_add],
            max_tool_calls=2
        )
        
        if response.choices[0].message.content and "5" in response.choices[0].message.content:
            print("✅ Quick test passed - toolflow is working correctly!")
            return True
        else:
            print("❌ Quick test failed - unexpected response")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed with error: {e}")
        return False

def run_tests(test_categories=None):
    """Run the live integration tests."""
    print("\n🧪 Running live integration tests...")
    print("Note: These tests will consume OpenAI API credits\n")
    
    # Base command
    cmd = ["python", "-m", "pytest", "tests/test_integration_live.py", "-v", "-s"]
    
    # Add specific test categories if requested
    if test_categories:
        for category in test_categories:
            cmd.append(f"-k {category}")
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main function."""
    print("🤖 Toolflow Live Integration Test Runner")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    # Ask user if they want to run quick test first
    print("\n" + "=" * 50)
    while True:
        choice = input("Run quick verification test first? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            if not run_quick_test():
                print("\n❌ Quick test failed. Check your setup.")
                sys.exit(1)
            break
        elif choice in ['n', 'no']:
            break
        else:
            print("Please enter 'y' or 'n'")
    
    # Show available test categories
    print("\n" + "=" * 50)
    print("Available test categories:")
    print("1. Basic tool calling")
    print("2. Structured output")
    print("3. Async functionality")
    print("4. Streaming")
    print("5. Error handling")
    print("6. Comprehensive workflows")
    print("7. All tests")
    
    # Ask which tests to run
    print("\n" + "=" * 50)
    while True:
        choice = input("Which tests would you like to run? (1-7 or 'all'): ").strip()
        
        test_categories = None
        if choice == "1":
            test_categories = ["TestBasicToolCalling"]
        elif choice == "2":
            test_categories = ["TestStructuredOutput"]
        elif choice == "3":
            test_categories = ["TestAsyncFunctionality"]
        elif choice == "4":
            test_categories = ["TestStreamingFunctionality"]
        elif choice == "5":
            test_categories = ["TestErrorHandling"]
        elif choice == "6":
            test_categories = ["TestComprehensiveWorkflow"]
        elif choice in ["7", "all"]:
            test_categories = None  # Run all tests
        else:
            print("Please enter a number 1-7 or 'all'")
            continue
        
        break
    
    # Run the tests
    print("\n" + "=" * 50)
    success = run_tests(test_categories)
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 