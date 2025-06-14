"""
Anthropic-specific streaming utilities.

This module handles the specifics of how Anthropic streams responses and accumulates tool calls.
"""
import json
from typing import List, Dict, Any


def create_tool_call_object(tool_id: str, tool_name: str, tool_input: Dict[str, Any]):
    """Create a tool call object that matches Anthropic's format for execution."""
    return type('ToolCall', (), {
        'id': tool_id,
        'name': tool_name,
        'input': tool_input
    })()


def accumulate_anthropic_streaming_content(
    chunk,
    message_content: List[Dict[str, Any]],
    accumulated_tool_calls: List[Any],
    accumulated_json_strings: Dict[int, str],
    graceful_error_handling: bool = True
) -> bool:
    """
    Accumulate Anthropic streaming content and detect tool calls.
    
    Returns:
        bool: True if tool calls were detected and completed, False otherwise
    """
    tool_calls_completed = False
    
    if chunk.type == "content_block_start":
        content_block = chunk.content_block
        
        if content_block.type == "text":
            message_content.append({
                "type": "text",
                "text": ""
            })

        elif content_block.type == "thinking":
            message_content.append({
                "type": "thinking",
                "thinking": "",
                "signature": ""  # Will be filled when signature_delta arrives
            })
        elif content_block.type == "tool_use":
            message_content.append({
                "type": "tool_use",
                "id": content_block.id,
                "name": content_block.name,
                "input": {}
            })
            accumulated_json_strings[chunk.index] = ""
    
    elif chunk.type == "content_block_delta":
        delta = chunk.delta
        content_index = chunk.index
        
        if delta.type == "text_delta":
            # Update text content
            if (content_index < len(message_content) and 
                message_content[content_index]["type"] == "text"):
                message_content[content_index]["text"] += str(delta.text)
        
        elif delta.type == "thinking_delta":
            # Update thinking content
            if (content_index < len(message_content) and 
                message_content[content_index]["type"] == "thinking"):
                message_content[content_index]["thinking"] += str(delta.thinking)
        
        elif delta.type == "signature_delta":
            # Update thinking signature
            if (content_index < len(message_content) and 
                message_content[content_index]["type"] == "thinking"):
                message_content[content_index]["signature"] += str(delta.signature)
        
        elif delta.type == "input_json_delta":
            # Accumulate JSON for tool inputs
            if content_index in accumulated_json_strings:
                accumulated_json_strings[content_index] += str(delta.partial_json)
    
    elif chunk.type == "content_block_stop":
        content_index = chunk.index
        
        # If this was a tool_use block, parse the accumulated JSON
        if (content_index in accumulated_json_strings and
            content_index < len(message_content) and
            message_content[content_index]["type"] == "tool_use"):
            
            try:
                tool_input = json.loads(accumulated_json_strings[content_index])
                message_content[content_index]["input"] = tool_input
                
                # Create a tool call object for execution
                tool_call = create_tool_call_object(
                    message_content[content_index]["id"],
                    message_content[content_index]["name"],
                    tool_input
                )
                accumulated_tool_calls.append(tool_call)
                
            except json.JSONDecodeError:
                # Handle malformed JSON gracefully
                if graceful_error_handling:
                    message_content[content_index]["input"] = {"error": "Invalid JSON"}
                else:
                    raise Exception(f"Invalid JSON in tool input: {accumulated_json_strings[content_index]}")
            
            # Clean up accumulated JSON
            del accumulated_json_strings[content_index]
    
    elif chunk.type == "message_stop":
        # Message is complete, check if we have any tool calls
        if accumulated_tool_calls:
            tool_calls_completed = True
    
    return tool_calls_completed


def should_yield_chunk(chunk, return_full_response: bool, block_types: dict = None):
    """
    Determine if a chunk should be yielded and extract its content.
    
    Args:
        chunk: The streaming chunk from Anthropic
        return_full_response: Whether we're returning full response objects
        block_types: Dict to track block types by index (optional)
        
    Returns:
        Tuple of (should_yield: bool, content: str)
    """
    if return_full_response:
        return True, chunk
    
    # Initialize block_types if not provided
    if block_types is None:
        block_types = {}
    
    # Handle different types of Anthropic streaming events
    if hasattr(chunk, 'type'):
        if chunk.type == 'content_block_delta':
            if hasattr(chunk, 'delta'):
                # Handle thinking content
                if hasattr(chunk.delta, 'type') and chunk.delta.type == 'thinking_delta':
                    if hasattr(chunk.delta, 'thinking'):
                        return True, str(chunk.delta.thinking)
                
                # Handle text content
                elif hasattr(chunk.delta, 'type') and chunk.delta.type == 'text_delta':
                    if hasattr(chunk.delta, 'text'):
                        return True, str(chunk.delta.text)
        
        elif chunk.type == 'content_block_start':
            if hasattr(chunk, 'content_block') and hasattr(chunk, 'index'):
                # Track block type by index
                block_types[chunk.index] = chunk.content_block.type
                
                # Handle thinking block start
                if chunk.content_block.type == 'thinking':
                    return True, "\n<THINKING>\n"
                
                # Handle text block start  
                elif chunk.content_block.type == 'text':
                    return True, ""  # No indicator for text start
        
        elif chunk.type == 'content_block_stop':
            if hasattr(chunk, 'index'):
                # Check if this was a thinking block that's ending
                block_type = block_types.get(chunk.index)
                if block_type == 'thinking':
                    # Clean up the tracking
                    block_types.pop(chunk.index, None)
                    return True, "\n</THINKING>\n\n"
                elif block_type == 'text':
                    # Clean up the tracking
                    block_types.pop(chunk.index, None)
                    return True, "\n"  # Just add newline for text blocks
    
    # Fallback for other chunk types
    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
        return True, str(chunk.delta.text)
    
    return False, "" 
