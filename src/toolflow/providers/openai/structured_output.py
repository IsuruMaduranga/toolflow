"""
OpenAI-specific structured output utilities.

This module handles structured output parsing for OpenAI responses using Pydantic models.
"""
import json
from typing import Callable, Any

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

def handle_openai_structured_response(response, response_format) -> Any|None:
    """
    Handle OpenAI structured response parsing.
    Returns a tuple with the response and a boolean indicating if a structured response was found.
    """
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return None
        
    for tool_call in tool_calls:
        if tool_call.function.name == "final_response_tool_internal":
            args = json.loads(tool_call.function.arguments)
            response_data = args.get('response', args)
            parsed_model = response_format.model_validate(response_data)
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
