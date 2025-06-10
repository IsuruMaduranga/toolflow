#!/usr/bin/env python3

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import toolflow

# Mock OpenAI client for testing
class MockOpenAIClient:
    class Chat:
        class Completions:
            def create(self, **kwargs):
                # Mock response that would trigger the response_tool
                class MockResponse:
                    class Choice:
                        class Message:
                            def __init__(self):
                                self.content = None
                                self.tool_calls = [
                                    type('MockToolCall', (), {
                                        'id': 'test-id',
                                        'function': type('MockFunction', (), {
                                            'name': 'response_tool',
                                            'arguments': '{"response": {"city": "San Francisco", "temperature": 72, "condition": "Sunny", "humidity": 65, "forecast": ["Sunny", "Partly Cloudy"]}}'
                                        })()
                                    })()
                                ]
                        
                        def __init__(self):
                            self.message = self.Message()
                    
                    def __init__(self):
                        self.choices = [self.Choice()]
                
                return MockResponse()
        
        def __init__(self):
            self.completions = self.Completions()
    
    def __init__(self):
        self.chat = self.Chat()

# Mock Pydantic BaseModel
class BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, data):
        return cls(**data)

# Mock response format
class WeatherResponse(BaseModel):
    city: str
    temperature: int
    condition: str
    humidity: int
    forecast: list

def test_response_format():
    print("Testing response_format feature...")
    
    # Mock client
    mock_client = MockOpenAIClient()
    
    # Wrap with toolflow
    try:
        client = toolflow.from_openai(mock_client)
        print("✓ Successfully created toolflow client")
    except Exception as e:
        print(f"✗ Failed to create toolflow client: {e}")
        return False
    
    # Test response_format parameter
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather?"}],
            response_format=WeatherResponse
        )
        print("✓ Successfully called create with response_format")
        
        # Check if we have structured output
        if hasattr(response, '_structured_output'):
            print("✓ Response has _structured_output attribute")
            print(f"  Structured output: {response._structured_output}")
        else:
            print("✗ Response missing _structured_output attribute")
            return False
            
    except Exception as e:
        print(f"✗ Failed to call create with response_format: {e}")
        return False
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_response_format()
    sys.exit(0 if success else 1)
