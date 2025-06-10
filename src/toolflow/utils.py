import inspect
from typing import Callable, Annotated, get_origin, get_args, Any
from pydantic import BaseModel, create_model, Field
from pydantic.fields import FieldInfo
from docstring_parser import parse

def get_tool_schema(
    func: Callable,
    name: str = None,
    description: str = None,
    strict: bool = False
) -> dict:
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
    func_description = description or docstring.short_description or inspect.getdoc(func) or func_name

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
