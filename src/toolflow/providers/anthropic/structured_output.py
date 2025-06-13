"""
Anthropic-specific structured output utilities.

This module handles structured output parsing for Anthropic responses using Pydantic models.
"""
import json
from typing import Callable
from toolflow.utils import get_tool_schema

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None


def create_anthropic_response_tool(response_format) -> Callable:
    func_description = f"You must call this tool to provide your final response. Because user expects the final response in `{response_format.__name__}` format. This tool must be your last tool call."
    def final_response_tool_internal(response: response_format) -> str:
        return "Response formatted successfully"
    
    final_response_tool_internal._tool_metadata = get_tool_schema(final_response_tool_internal, "final_response_tool_internal", func_description)
    return final_response_tool_internal


def handle_anthropic_structured_response(response, response_format):
    """Handle Anthropic structured response parsing."""

    if hasattr(response, 'content') and response.content:
        for content_block in response.content:
            if (hasattr(content_block, 'type') and 
                content_block.type == 'tool_use' and 
                hasattr(content_block, 'name') and 
                content_block.name == "final_response_tool_internal"):
                
                # Parse the tool input and extract the actual response data
                tool_input = content_block.input
                # The response data is nested under the 'response' key
                response_data = tool_input.get('response', tool_input)
                
                # Create the parsed Pydantic model
                parsed_model = response_format.model_validate(response_data)
                
                response.content = json.dumps(response_data)
                response.parsed = parsed_model
                
                return response
    return None


def validate_response_format(response_format):
    """Validate that response_format is a Pydantic model."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for structured output. Install with: pip install pydantic")
    
    if not (hasattr(response_format, "__annotations__") and hasattr(response_format, "model_validate")):
        raise ValueError("response_format must be a Pydantic model")
