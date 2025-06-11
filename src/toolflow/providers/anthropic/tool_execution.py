"""
Anthropic-specific tool execution utilities.

This module handles the specifics of how Anthropic formats and executes tool calls.
"""
import json
from typing import Any, Dict, List, Callable


def validate_and_prepare_anthropic_tools(tools: List[Callable]) -> tuple[Dict[str, Callable], List[Dict]]:
    """Validate tools and prepare Anthropic-specific schemas and function mappings."""
    tool_functions = {}
    tool_schemas = []
    
    for tool in tools:
        if isinstance(tool, Callable) and hasattr(tool, '_tool_metadata'):
            # Convert OpenAI-style metadata to Anthropic format
            openai_metadata = tool._tool_metadata
            anthropic_schema = {
                "name": openai_metadata['function']['name'],
                "description": openai_metadata['function']['description'],
                "input_schema": openai_metadata['function']['parameters']
            }
            
            tool_schemas.append(anthropic_schema)
            tool_functions[openai_metadata['function']['name']] = tool
        else:
            raise ValueError(f"Only decorated functions via @tool are supported. Got {type(tool)}")
    
    return tool_functions, tool_schemas


def execute_anthropic_tools_sync(
    tool_functions: Dict[str, Callable],
    tool_calls: List[Any],
    graceful_error_handling: bool = True
) -> List[Dict[str, Any]]:
    """Execute Anthropic tool calls synchronously."""
    
    def execute_single_tool(tool_call):
        """Execute a single Anthropic tool call."""
        tool_name = tool_call.name
        tool_input = tool_call.input
        
        tool_function = tool_functions.get(tool_name, None)
        if not tool_function:
            raise ValueError(f"Tool {tool_name} not found")
        
        try:
            # Anthropic provides input as a dict already
            result = tool_function(**tool_input)
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps(result) if not isinstance(result, str) else result
            }
        except Exception as e:
            if graceful_error_handling:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": f"Error executing tool {tool_name}: {e}",
                    "is_error": True
                }
            else:
                raise Exception(f"Error executing tool {tool_name}: {e}")
    
    # Execute tools sequentially
    return [execute_single_tool(tool_call) for tool_call in tool_calls]


def convert_openai_messages_to_anthropic(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages to Anthropic format."""
    anthropic_messages = []
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            # System messages are handled separately in Anthropic
            continue
        elif role == "user":
            anthropic_messages.append({
                "role": "user",
                "content": content
            })
        elif role == "assistant":
            anthropic_messages.append({
                "role": "assistant", 
                "content": content
            })
        elif role == "tool":
            # Convert tool results to Anthropic format
            # This is handled by our tool execution
            continue
            
    return anthropic_messages


def extract_system_message(messages: List[Dict[str, Any]]) -> str:
    """Extract system message from OpenAI-style messages for Anthropic."""
    for message in messages:
        if message["role"] == "system":
            return message["content"]
    return None


def format_anthropic_tool_calls_for_messages(tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format tool results for Anthropic message format."""
    return {
        "role": "user",
        "content": tool_results
    }
