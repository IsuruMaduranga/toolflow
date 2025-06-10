"""
OpenAI-specific streaming utilities.

This module handles the specifics of how OpenAI streams responses and accumulates tool calls.
"""
from typing import List, Dict


def accumulate_openai_streaming_tool_calls(accumulated_tool_calls: List[Dict], tool_call_delta):
    """Accumulate OpenAI tool calls from streaming deltas."""
    # Ensure we have enough space in accumulated_tool_calls
    while len(accumulated_tool_calls) <= tool_call_delta.index:
        accumulated_tool_calls.append({
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""}
        })
    
    # Update the tool call at the correct index
    if tool_call_delta.id:
        accumulated_tool_calls[tool_call_delta.index]["id"] = tool_call_delta.id
    if tool_call_delta.function:
        if tool_call_delta.function.name:
            accumulated_tool_calls[tool_call_delta.index]["function"]["name"] = tool_call_delta.function.name
        if tool_call_delta.function.arguments:
            accumulated_tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments


def convert_accumulated_openai_tool_calls(accumulated_tool_calls: List[Dict]) -> List:
    """Convert accumulated OpenAI tool calls to proper format for execution."""
    tool_calls = []
    for tc in accumulated_tool_calls:
        if tc["id"] and tc["function"]["name"]:  # Only add complete tool calls
            tool_call = type('ToolCall', (), {
                'id': tc["id"],
                'function': type('Function', (), {
                    'name': tc["function"]["name"],
                    'arguments': tc["function"]["arguments"]
                })()
            })()
            tool_calls.append(tool_call)
    return tool_calls


def format_openai_tool_calls_for_messages(tool_calls) -> List[Dict]:
    """Format tool calls for inclusion in OpenAI message format."""
    return [
        {
            "id": tc.id,
            "type": "function", 
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        }
        for tc in tool_calls
    ]
