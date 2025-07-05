# src/toolflow/core/utils.py
from typing import Dict, Any, Tuple, Callable, get_origin, get_args, Optional, List
from typing_extensions import Annotated
from .constants import DEFAULT_PARAMS, RESPONSE_FORMAT_TOOL_NAME
from .exceptions import ResponseFormatError
from .tool_execution import run_sync_tool
import inspect

__all__ = ['filter_toolflow_params', 'get_structured_output_tool', 'get_tool_schema', 'RESPONSE_FORMAT_TOOL_NAME']

from pydantic import BaseModel, create_model, Field
from pydantic.fields import FieldInfo
from docstring_parser import parse

def filter_toolflow_params(**kwargs: Any) -> Tuple[Dict[str, Any], int, bool, Any, bool, bool]:
    """Extract toolflow-specific params and return as easily unpackable tuple."""
    filtered_kwargs = kwargs.copy()
    
    # Default values for toolflow params
    max_tool_call_rounds = filtered_kwargs.pop("max_tool_call_rounds", DEFAULT_PARAMS["max_tool_call_rounds"])
    max_response_format_retries = filtered_kwargs.pop("max_response_format_retries", DEFAULT_PARAMS["max_response_format_retries"])
    parallel_tool_execution = filtered_kwargs.pop("parallel_tool_execution", DEFAULT_PARAMS["parallel_tool_execution"])
    full_response = filtered_kwargs.pop("full_response", DEFAULT_PARAMS["full_response"])
    graceful_error_handling = filtered_kwargs.pop("graceful_error_handling", DEFAULT_PARAMS["graceful_error_handling"])
    
    # Return a tuple of the filtered kwargs and toolflow params
    return filtered_kwargs, max_tool_call_rounds, max_response_format_retries, parallel_tool_execution, full_response, graceful_error_handling

def get_structured_output_tool(pydantic_model: Any) -> Callable[..., str]:
    """Get the tool definition for structured output."""

    def structured_response_tool(response: pydantic_model) -> str:
        return response

    structured_response_tool.__name__ = RESPONSE_FORMAT_TOOL_NAME
    structured_response_tool.__doc__ = f"""
        Provide your final structured response using the specified {pydantic_model.__name__} format.
        WHEN TO USE: Call when ready to provide your complete structured answer.
        CRITICAL REQUIREMENTS:
        ✓ Fill ALL required fields accurately
        ✓ Match data types exactly
        ✓ Use complete information from conversation
        ✓ You can call this tool only once with complete response
        ✓ This tool must be your last tool call in the conversation
        BEHAVIOR:
        ✗ Do NOT mention this tool exists or any hint about it.
        ✓ Simply call with properly formatted data
        
        Args:
            response: Complete {pydantic_model.__name__} object with all required fields filled out accurately based on the conversation context
        
        Returns:
            Empty string (the structured data is captured in the function call)
        """
    setattr(structured_response_tool, "__internal_tool__", True)
    return structured_response_tool

def get_tool_schema(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Generates a truly unified OpenAI-compatible JSON schema from any Python function.

    This function processes every parameter in a single pass, correctly combining
    Pydantic BaseModel arguments, Annotated[..., Field] parameters, and standard
    parameters with docstring descriptions into one coherent schema.

    Args:
        func: The function to generate a schema for.
        name: An optional override for the function's name.
        description: An optional override for the function's description.

    Returns:
        A dictionary representing the OpenAI-compatible function schema.
    """
    # 1. Setup: Get signature, docstring, and prepare for overrides
    sig = inspect.signature(func)
    docstring = parse(inspect.getdoc(func) or "")
    doc_params = {p.arg_name: p for p in docstring.params}
    func_name = name or func.__name__
    short_and_long_description = (docstring.short_description or "") + (docstring.long_description or "")
    func_description = description or short_and_long_description or inspect.getdoc(func) or func_name

    # 2. Unified Loop: Process EVERY parameter to build fields for a single model
    fields_for_model = {}
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        param_name = param.name
        param_annotation = param.annotation

        # Case A: The parameter is a Pydantic BaseModel.
        # It will be treated as a nested object in the schema.
        if inspect.isclass(param_annotation) and issubclass(param_annotation, BaseModel):
            # The field should be required if it has no default value
            is_required = param.default is inspect.Parameter.empty
            field_info = Field(default=... if is_required else param.default)
            if param_name in doc_params:
                field_info.description = doc_params[param_name].description
            fields_for_model[param_name] = (param_annotation, field_info)
            continue # Done with this param, move to the next

        # Case B: The parameter is a standard type (potentially with Annotated/Field)
        field_info = Field()  # Start with a blank FieldInfo
        param_type = param_annotation

        if get_origin(param_annotation) is Annotated:
            annotated_args = get_args(param_annotation)
            param_type = annotated_args[0]
            field_info = next((arg for arg in annotated_args[1:] if isinstance(arg, FieldInfo)), field_info)

        if param_type is inspect.Parameter.empty:
            param_type = Any

        # Combine metadata: Field description > docstring description
        if not field_info.description and param_name in doc_params:
            field_info.description = doc_params[param_name].description

        # Set default value from the function signature if not already on the Field
        if param.default is not inspect.Parameter.empty:
            field_info.default = param.default
        
        fields_for_model[param_name] = (param_type, field_info)

    # 3. Create a single model from all collected fields and generate the schema
    schema: Dict[str, Any]
    if not fields_for_model:
        schema = {"type": "object", "properties": {}, "required": []}
    else:
        final_model = create_model(f"{func.__name__}Args", **fields_for_model)
        schema = final_model.model_json_schema()
        schema.pop("title", None)
    
    # Always set additionalProperties to False for OpenAI compatibility
    schema["additionalProperties"] = False

    # 4. Construct the final OpenAI tool schema
    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_description,
            "parameters": schema,
            "strict": strict
        }
    }

def process_response_format(
    handler,
    raw_response: Any,
    text: Optional[str],
    tool_calls: List[Dict[str, Any]],
    tool_map: Dict[str, Any],
    messages: List[Dict[str, Any]],
    remaining_retry_count: int,
    max_response_format_retries: int
) -> Optional[Tuple[Any, bool]]:
    """
    Process response format handling for structured outputs.
    
    Returns:
        None if no response format processing needed
        Tuple of (structured_response, should_continue) if processing occurred
    """
    
    def _create_error_message() -> str:
        return f"""
        Failed to parse structured output after {max_response_format_retries} retries.
        TIP: If this is a data format issue, consider clearly documenting the response format parameters.
        """
    
    def _handle_no_tool_calls() -> Tuple[Any, bool]:
        """Handle case when no tool calls are present."""
        if remaining_retry_count <= 0:
            raise ResponseFormatError(_create_error_message())
        
        # Add retry message and continue
        messages.append(handler.build_response_format_retry_message())
        return None, True
    
    def _handle_response_format_tool_call(tool_call: Dict[str, Any]) -> Tuple[Any, bool]:
        """Handle a single response format tool call."""
        tool_result = run_sync_tool(tool_call, tool_map[tool_call["function"]["name"]], True)
        
        if tool_result.get("is_error", False):
            if remaining_retry_count <= 0:
                raise ResponseFormatError(_create_error_message() + f"\nError: {tool_result['output']}")
            
            # Add error context and retry
            messages.append(handler.build_assistant_message(text, [tool_call], raw_response))
            messages.extend(handler.build_tool_result_messages([tool_result]))
            return None, True
        
        # Success - return the structured response
        return tool_result["output"], False
    
    def _find_response_format_tool_call() -> Optional[Dict[str, Any]]:
        """Find the response format tool call if it exists."""
        for tool_call in tool_calls:
            if tool_call["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME:
                return tool_call
        return None
    
    # Process response format logic
    if not tool_calls:
        return _handle_no_tool_calls()
    
    response_format_tool_call = _find_response_format_tool_call()
    if response_format_tool_call:
        return _handle_response_format_tool_call(response_format_tool_call)
    
    # No response format tool call found - let normal execution continue
    return None
    