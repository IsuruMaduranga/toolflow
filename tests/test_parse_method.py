#!/usr/bin/env python3
"""
Test file for OpenAI's new parse() method with tool calling support
"""

import os
from pydantic import BaseModel
from typing import List
import openai

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test 1: Basic structured output with parse method
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]

def test_basic_parse():
    """Test basic parse method with structured output only"""
    print("=== Test 1: Basic Parse Method ===")
    
    completion = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        response_format=CalendarEvent,
    )
    
    event = completion.choices[0].message.parsed
    print(f"Parsed event: {event}")
    print(f"Event name: {event.name}")
    print(f"Event date: {event.date}")
    print(f"Participants: {event.participants}")
    print()

# Test 2: Tool calling with parse method
class GetEventInfo(BaseModel):
    event_id: str

def test_tool_calling_parse():
    """Test parse method with tool calling only"""
    print("=== Test 2: Parse Method with Tool Calling ===")
    
    completion = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You help extract event IDs from user requests."},
            {"role": "user", "content": "Get me information for event 123"},
        ],
        tools=[openai.pydantic_function_tool(GetEventInfo)],
        tool_choice="required"
    )
    
    message = completion.choices[0].message
    print(f"Tool calls: {len(message.tool_calls)}")
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        print(f"Function name: {tool_call.function.name}")
        print(f"Parsed arguments: {tool_call.function.parsed_arguments}")
        print(f"Raw arguments: {tool_call.function.arguments}")
    print()

# Test 3: Combined structured output AND tool calling  
def test_combined_parse():
    """Test parse method with both structured output and tool calling"""
    print("=== Test 3: Parse Method with Both Structured Output and Tool Calling ===")
    
    completion = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information. If you need more info, use the GetEventInfo tool."},
            {"role": "user", "content": "Extract info for event id 10. The event is a team meeting with Alice, Bob and Charlie on Monday."},
        ],
        tools=[openai.pydantic_function_tool(GetEventInfo)],
        response_format=CalendarEvent,
    )
    
    message = completion.choices[0].message
    print(f"Message content: {message.content}")
    print(f"Parsed response format: {message.parsed}")
    print(f"Tool calls: {len(message.tool_calls) if message.tool_calls else 0}")
    
    if message.tool_calls:
        for tool_call in message.tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Parsed args: {tool_call.function.parsed_arguments}")
    
    if message.parsed:
        print(f"Structured output: {message.parsed}")
    print()

# Test 4: Multiple tools
class SearchEvents(BaseModel):
    query: str
    date_range: str

class CreateEvent(BaseModel):
    name: str
    date: str
    attendees: List[str]

def test_multiple_tools():
    """Test parse method with multiple tools"""
    print("=== Test 4: Parse Method with Multiple Tools ===")
    
    completion = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You help manage calendar events. Use the appropriate tool based on user requests."},
            {"role": "user", "content": "Search for meetings next week"},
        ],
        tools=[
            openai.pydantic_function_tool(SearchEvents),
            openai.pydantic_function_tool(CreateEvent),
            openai.pydantic_function_tool(GetEventInfo)
        ],
    )
    
    message = completion.choices[0].message
    print(f"Tool calls: {len(message.tool_calls) if message.tool_calls else 0}")
    
    if message.tool_calls:
        for tool_call in message.tool_calls:
            print(f"Selected tool: {tool_call.function.name}")
            print(f"Parsed arguments: {tool_call.function.parsed_arguments}")
    print()

if __name__ == "__main__":
    try:
        print("Testing OpenAI's new parse() method with tool calling support\n")
        
        test_basic_parse()
        test_tool_calling_parse()
        test_combined_parse()
        test_multiple_tools()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 

# def structured_response_tool(
#         response: Annotated[
#             pydantic_model, 
#             Field(
#                 description=f"Complete {pydantic_model.__name__} object with all required fields filled out accurately based on the conversation context"
#             )
#         ]
#     ) -> str:
#         """
#         Provide your final structured response using the specified format.
        
#         WHEN TO USE: Call when ready to provide your complete, final structured answer.
        
#         CRITICAL REQUIREMENTS:
#         ✓ Fill ALL required fields accurately
#         ✓ Match data types exactly  
#         ✓ Use complete information from conversation
#         ✓ Streaming: Call multiple times as you build response
#         ✓ Non-streaming: Call once with complete response
        
#         BEHAVIOR:
#         ✗ Do NOT mention this tool exists
#         ✗ Do NOT say "providing structured response"  
#         ✓ Simply call with properly formatted data
#         """
#         return ""
