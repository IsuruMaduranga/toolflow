import inspect
from typing import Any
from pydantic import create_model

def get_tool_schema(func, name, description):
    """
    Get the schema for a tool function.
    """
    # Get function signature
    sig = inspect.signature(func)
    
    # Filter out *args and **kwargs as Pydantic create_model doesn't directly handle them
    # when generating a schema for tool definitions without explicit typing.
    parameters_for_model = {}
    required_params = []
    
    for param in sig.parameters.values():
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # Handle missing type annotations by defaulting to Any
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            
            # Check if parameter has a default value
            if param.default == inspect.Parameter.empty:
                # No default value - required parameter
                parameters_for_model[param.name] = (annotation, ...)
                required_params.append(param.name)
            else:
                # Has default value - optional parameter
                parameters_for_model[param.name] = (annotation, param.default)

    model = create_model(
        f"{func.__name__}Model",
        **parameters_for_model
    )

    schema = model.model_json_schema()

    schema.pop('title', None)

    if 'properties' in schema:
        schema['additionalProperties'] = False
    
    # Override required fields to only include parameters without defaults
    if required_params:
        schema['required'] = required_params
    else:
        schema['required'] = []

    return {
    "type": "function",
    "function": {
        "name": name or func.__name__,
        "description": description or inspect.getdoc(func) or func.__name__, # Use function name as fallback
        "parameters": schema
    }
}
