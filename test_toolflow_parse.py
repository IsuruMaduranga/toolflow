#!/usr/bin/env python3
"""
Test file for toolflow's enhanced parse() method with auto tool calling
"""

import os
from pydantic import BaseModel
from typing import List
import openai
import toolflow

# Set up clients
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
toolflow_client = toolflow.from_openai(openai_client, full_response=True)

# Test models
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]

class WeatherInfo(BaseModel):
    location: str
    temperature: int
    condition: str

@toolflow.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@toolflow.tool  
def search_calendar(query: str) -> str:
    """Search calendar events."""
    return f"Found 3 meetings matching '{query}'"

@toolflow.tool
def get_event_details(event_id: str) -> str:
    """Get detailed information about a calendar event."""
    return f"Event {event_id}: Team standup on Monday at 9 AM with Alice, Bob, Charlie"

def test_basic_parse():
    """Test basic parse method with structured output only"""
    print("=== Test 1: Basic Parse Method ===")
    
    response = toolflow_client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        response_format=CalendarEvent,
    )
    
    print(f"Response type: {type(response)}")
    print(f"Parsed event: {response.choices[0].message.parsed}")
    if response.choices[0].message.parsed:
        event = response.choices[0].message.parsed
        print(f"Event name: {event.name}")
        print(f"Event date: {event.date}")
        print(f"Participants: {event.participants}")
    print()

def test_parse_with_tools():
    """Test parse method with tool calling"""
    print("=== Test 2: Parse Method with Tool Calling ===")
    
    response = toolflow_client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": "Get weather for San Francisco and extract it as weather info"},
        ],
        tools=[get_weather],
        response_format=WeatherInfo,
    )
    
    print(f"Response type: {type(response)}")
    print(f"Message content: {response.choices[0].message.content}")
    print(f"Tool calls: {len(response.choices[0].message.tool_calls) if response.choices[0].message.tool_calls else 0}")
    print(f"Parsed response: {response.choices[0].message.parsed}")
    print()

def test_parse_multiple_tools():
    """Test parse method with multiple tools and structured output"""
    print("=== Test 3: Parse Method with Multiple Tools ===")
    
    response = toolflow_client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": "Search for 'standup' meetings and get details for event 123, then create a calendar event summary"},
        ],
        tools=[search_calendar, get_event_details],
        response_format=CalendarEvent,
        max_tool_calls=5,
    )
    
    print(f"Response type: {type(response)}")
    print(f"Final parsed result: {response.choices[0].message.parsed}")
    print()

def test_parse_vs_create():
    """Compare parse method vs create method for structured output"""
    print("=== Test 4: Parse vs Create Comparison ===")
    
    # Using parse method
    print("Using parse method:")
    parse_response = toolflow_client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": "Bob and Alice have a project meeting on Wednesday"},
        ],
        response_format=CalendarEvent,
    )
    print(f"Parse result: {parse_response.choices[0].message.parsed}")
    
    # Using create method with response_format
    print("\nUsing create method with response_format:")
    create_response = toolflow_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": "Bob and Alice have a project meeting on Wednesday"},
        ],
        response_format=CalendarEvent,
        full_response=True,
    )
    print(f"Create response type: {type(create_response)}")
    print(f"Create result: {create_response}")
    print()

def test_error_handling():
    """Test error handling in parse method"""
    print("=== Test 5: Error Handling ===")
    
    try:
        # Try streaming with parse (should fail)
        response = toolflow_client.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": "Test"}],
            response_format=CalendarEvent,
            stream=True,  # This should raise an error
        )
    except ValueError as e:
        print(f"Expected error caught: {e}")
    
    print()

if __name__ == "__main__":
    try:
        print("Testing toolflow's enhanced parse() method\n")
        
        test_basic_parse()
        test_parse_with_tools()
        test_parse_multiple_tools()
        test_parse_vs_create()
        test_error_handling()
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 