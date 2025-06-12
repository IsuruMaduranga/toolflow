"""
OpenAI-specific structured output utilities.

This module handles structured output parsing for OpenAI responses using Pydantic models.
"""
import json
from typing import Callable

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None


def create_openai_response_tool(response_format) -> Callable:
    """Create a dynamic response tool for OpenAI structured output."""
    from toolflow.decorators import tool
    
    @tool(name="final_response_tool_internal", internal=True)
    def final_response_tool_internal(response: response_format) -> str:
        f"""
        You must call this tool to provide your final response.
        Because user expects the final response in `{response_format.__name__}` format.
        This tool must be your last tool call.
        """
        pass

    return final_response_tool_internal

def handle_openai_structured_response(response, response_format):
    """Handle OpenAI structured response parsing."""
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls and tool_calls[0].function.name == "final_response_tool_internal":
        # Parse the arguments and extract the actual response data
        args = json.loads(tool_calls[0].function.arguments)
        # The response data is nested under the 'response' key
        response_data = args.get('response', args)
        
        # Create the parsed Pydantic model
        parsed_model = response_format.model_validate(response_data)
        
        # Modify the response to include both content and parsed
        response.choices[0].message.content = json.dumps(response_data)
        response.choices[0].message.parsed = parsed_model
        
        return response
    return None


def validate_response_format(response_format):
    """Validate that response_format is a Pydantic model."""
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for structured output. Install with: pip install pydantic")
    
    if not (hasattr(response_format, "__annotations__") and hasattr(response_format, "model_validate")):
        raise ValueError("response_format must be a Pydantic model")
