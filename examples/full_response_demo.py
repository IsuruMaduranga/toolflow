#!/usr/bin/env python3
"""
Demonstration of the full_response parameter in toolflow.

This script shows how to use the new full_response parameter to control
whether toolflow returns the complete OpenAI response object or just the
content/parsed data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import toolflow
from unittest.mock import Mock

# Mock OpenAI client for demonstration
class MockMessage:
    def __init__(self, content="Demo response", parsed=None):
        self.content = content
        self.parsed = parsed
        self.tool_calls = None

class MockChoice:
    def __init__(self, content="Demo response", parsed=None):
        self.message = MockMessage(content, parsed)

class MockResponse:
    def __init__(self, content="Demo response", parsed=None):
        self.choices = [MockChoice(content, parsed)]

class MockCompletions:
    def create(self, **kwargs):
        return MockResponse("Hello from toolflow with simplified response!")
    
    def parse(self, **kwargs):
        class ParsedData:
            def __init__(self):
                self.result = "structured data"
        return MockResponse("JSON content", ParsedData())

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockBeta:
    def __init__(self):
        self.chat = MockChat()

class MockOpenAI:
    def __init__(self):
        self.chat = MockChat()
        self.beta = MockBeta()

def main():
    """Demonstrate the full_response parameter functionality."""
    print("🚀 Toolflow full_response Parameter Demo")
    print("=" * 50)
    
    mock_client = MockOpenAI()
    
    # Demo 1: Traditional behavior (full_response=True)
    print("\n1️⃣ Traditional behavior (full_response=True):")
    client_full = toolflow.from_openai(mock_client, full_response=True)
    response_full = client_full.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(f"   Type: {type(response_full)}")
    print(f"   Content: {response_full.choices[0].message.content}")
    
    # Demo 2: Simplified behavior (full_response=False)
    print("\n2️⃣ Simplified behavior (full_response=False):")
    client_simple = toolflow.from_openai(mock_client, full_response=False)
    response_simple = client_simple.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(f"   Type: {type(response_simple)}")
    print(f"   Content: {response_simple}")
    
    # Demo 3: Method-level override
    print("\n3️⃣ Method-level override:")
    print("   Client default is full_response=False, but overriding to True at method level")
    response_override = client_simple.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        full_response=True  # Override the client setting
    )
    print(f"   Type: {type(response_override)}")
    print(f"   Content: {response_override.choices[0].message.content}")
    
    # Demo 4: Structured output with simplified response
    print("\n4️⃣ Structured output (parse) with full_response=False:")
    try:
        from pydantic import BaseModel
        
        class DemoModel(BaseModel):
            result: str
        
        parsed_response = client_simple.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            response_format=DemoModel
        )
        print(f"   Type: {type(parsed_response)}")
        print(f"   Result: {getattr(parsed_response, 'result', 'Mock parsed data')}")
    except ImportError:
        print("   Pydantic not available - skipping structured output demo")
    
    # Demo 5: Beta API with simplified response
    print("\n5️⃣ Beta API with full_response=False:")
    beta_response = client_simple.beta.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(f"   Type: {type(beta_response)}")
    print(f"   Content: {beta_response}")
    
    print("\n✨ Summary:")
    print("   • full_response=False: Get content directly (simpler API)")
    print("   • full_response=True: Get full OpenAI response object (traditional)")
    print("   • Can override at method level with full_response parameter")
    print("   • Works with regular calls, structured outputs, and beta API")
    print("   • Both sync and async clients support this parameter")

if __name__ == "__main__":
    main() 