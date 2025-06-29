# src/toolflow/core/utils.py
from typing import Dict, Any, List
from .handlers import AbstractProviderHandler
from .constants import DEFAULT_PARAMS, RESPONSE_FORMAT_TOOL_NAME

def filter_toolflow_params(**kwargs) -> tuple[Dict[str, Any], int, bool, int | None, Any, bool]:
    """Extract toolflow-specific params and return as easily unpackable tuple."""
    filtered_kwargs = kwargs.copy()
    
    # Default values for toolflow params
    max_tool_calls = filtered_kwargs.pop("max_tool_calls", DEFAULT_PARAMS["max_tool_calls"])
    parallel_tool_execution = filtered_kwargs.pop("parallel_tool_execution", DEFAULT_PARAMS["parallel_tool_execution"])
    max_workers = filtered_kwargs.pop("max_workers", DEFAULT_PARAMS["max_workers"])
    response_format = filtered_kwargs.pop("response_format", DEFAULT_PARAMS["response_format"])
    full_response = filtered_kwargs.pop("full_response", DEFAULT_PARAMS["full_response"])
    
    # Return a tuple of the filtered kwargs and toolflow params
    return filtered_kwargs, max_tool_calls, parallel_tool_execution, max_workers, response_format, full_response

def prepare_tool_schemas(tools: List[Any], handler: AbstractProviderHandler) -> tuple[List[Dict], Dict, Dict | None]:
    from toolflow.utils import get_tool_schema

    tool_schemas = []
    tool_map = {}

    if tools:
        for tool in tools:
            # check is tool is a function else error
            if callable(tool):
                schema = get_tool_schema(tool)
                if schema["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME:
                    raise ValueError(f"You cannot use the {RESPONSE_FORMAT_TOOL_NAME} tool as a tool. It is used internally to format the response.")
                tool_schemas.append(schema)
                tool_map[schema["function"]["name"]] = tool
                continue
            else:
                raise ValueError(f"Tool {tool} is not a function")
    return tool_schemas, tool_map

def prepare_response_format(response_format: Any, handler: AbstractProviderHandler) -> Dict:
    # check if response_format is a Pydantic model
    if not response_format:
        return None
    if isinstance(response_format, type) and hasattr(response_format, 'model_json_schema'):
        return handler.get_structured_output_tool(response_format)
    raise ValueError(f"Response format {response_format} is not a Pydantic model")
