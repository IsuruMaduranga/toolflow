import inspect
from pydantic import create_model

def get_tool_schema(func, name, description):
    """
    Get the schema for a tool function.
    """
    # Filter out *args and **kwargs as Pydantic create_model doesn't directly handle them
    # when generating a schema for tool definitions without explicit typing.
    parameters_for_model = {
        param.name: (param.annotation, ...)
        for param in inspect.signature(func).parameters.values()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }

    model = create_model(
        f"{func.__name__}Model",
        **parameters_for_model
    )

    schema = model.model_json_schema()

    schema.pop('title', None)

    if 'properties' in schema:
        schema['additionalProperties'] = False

    return {
    "type": "function",
    "function": {
        "name": name or func.__name__,
        "description": description or inspect.getdoc(func), # Use inspect.getdoc for the function's docstring
        "parameters": schema
    }
}
