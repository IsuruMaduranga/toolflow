# src/toolflow/core/utils.py
import dataclasses
from typing import Dict, Any, Tuple, Callable, get_origin, get_args, Optional, List, Union, Set
from typing_extensions import Annotated
from pydantic import TypeAdapter
from .constants import DEFAULT_PARAMS, RESPONSE_FORMAT_TOOL_NAME
from .exceptions import ResponseFormatError, MissingAnnotationError, UndescribableTypeError
from .tool_execution import run_sync_tool
import inspect
from pydantic import BaseModel, create_model, Field, ValidationError
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.fields import FieldInfo
from docstring_parser import parse

__all__ = ['filter_toolflow_params', 'get_structured_output_tool', 'get_tool_schema', 'RESPONSE_FORMAT_TOOL_NAME']



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

def _is_optional(tp: Any) -> bool:
    """Return *True* if ``tp`` is Optional/Union[..., None]."""
    origin = get_origin(tp)
    return origin is Union and type(None) in get_args(tp)

def _ensure_pydantic_dataclass(dc: type) -> type:
    """Wrap a *dataclass* with **pydantic.dataclasses.dataclass** if needed."""
    if hasattr(dc, "__pydantic_validator__"):
        # Already wrapped
        return dc
    try:
        from pydantic.dataclasses import dataclass as pydantic_dc

        return pydantic_dc(dc)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover – fallback, treat as Any
        return Any  # noqa: ANN401

def _schema_from_type(tp: Any) -> Dict[str, Any]:
    """Return a JSON‑schema fragment for *tp* using Pydantic *TypeAdapter*."""
    try:
        # Pydantic handles Enum, Literal, BaseModel, TypedDict, etc.
        ta = TypeAdapter(tp)
        return ta.json_schema()
    except Exception:  # pragma: no cover – fallback mapping
        origin = get_origin(tp)
        if origin in (list, List, set, Set, tuple, Tuple):
            return {"type": "array"}
        if origin in (dict, Dict):
            return {"type": "object"}
        # Primitive fallbacks
        if tp in (str,):
            return {"type": "string"}
        if tp in (int,):
            return {"type": "integer"}
        if tp in (float,):
            return {"type": "number"}
        if tp in (bool,):
            return {"type": "boolean"}
        # Last resort
        return {"type": "object"}
    
def _assert_describable(tp: Any, param_name: str, func_name: str) -> None:  # noqa: ANN401
    """Raise *UndescribableTypeError* if ``tp`` can't become JSON‑Schema."""

    try:
        TypeAdapter(tp).json_schema()
    except (PydanticSchemaGenerationError, ValidationError, TypeError, ValueError) as exc:  # noqa: E501
        raise UndescribableTypeError(
            f"Parameter '{param_name}' of function '{func_name}' uses type "
            f"{tp!r} which Pydantic cannot render as JSON Schema. "
            "Consider replacing it with a BaseModel, dataclass, TypedDict, "
            "primitive, or another describable container – or implement "
            "__get_pydantic_core_schema__."
        ) from exc

def get_tool_schema(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Generate an OpenAI‑compatible *function‑tool* schema for *func*.

    Raises
    ------
    MissingAnnotationError
        If any parameter lacks a type annotation.
    UndescribableTypeError
        If any annotation cannot be transformed into JSON‑Schema by Pydantic.
    """

    # -------------------------------------------------- signature & docstring
    sig = inspect.signature(func)
    if not hasattr(func, "_tf_doc"):
        func._tf_doc = parse(inspect.getdoc(func) or "")  # type: ignore[attr-defined]
    doc_params = {p.arg_name: p for p in func._tf_doc.params}

    func_name = name or func.__name__
    merged_doc_descr = (
        (func._tf_doc.short_description or "") + (func._tf_doc.long_description or "")
    ).strip()
    func_description = description or merged_doc_descr or func_name

    # -------------------------------------------------- collect field specs
    fields_for_model: Dict[str, tuple[Any, FieldInfo]] = {}

    for param in sig.parameters.values():
        # *args / **kwargs are not representable in OpenAI tool schemas
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        pname = param.name

        if param.annotation is inspect.Parameter.empty:
            raise MissingAnnotationError(
                f"Missing type annotation for parameter '{pname}' in function "
                f"'{func.__name__}'. All tool parameters must be type‑annotated."
            )

        ptype = param.annotation
        pdefault = param.default if param.default is not inspect.Parameter.empty else ...

        # Annotated[…, Field(...)] – split metadata out
        field_info = Field()
        if get_origin(ptype) is Annotated:  # type: ignore[attr-defined]
            root, *extras = get_args(ptype)
            ptype = root
            field_info = next(
                (meta for meta in extras if isinstance(meta, FieldInfo)), field_info
            )

        # Wrap plain dataclasses for Pydantic compatibility
        if dataclasses.is_dataclass(ptype) and not issubclass(ptype, BaseModel):
            ptype = _ensure_pydantic_dataclass(ptype)

        # Ensure describable (raises if not)
        _assert_describable(ptype, pname, func_name)

        # Use docstring description if Field.description absent
        if not field_info.description and pname in doc_params:
            field_info.description = doc_params[pname].description

        # Default → optional unless overridden in Field
        if pdefault is not ... and field_info.default is ...:
            field_info.default = pdefault

        fields_for_model[pname] = (ptype, field_info)

    # -------------------------------------------------- Build synthetic model
    if not fields_for_model:
        parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    else:
        DynamicModel = create_model(f"{func.__name__}Args", **fields_for_model)
        parameters_schema = DynamicModel.model_json_schema()  # type: ignore[attr-defined]
        parameters_schema.pop("title", None)

    # OpenAI guideline: forbid additional keys at top level
    parameters_schema["additionalProperties"] = False

    # -------------------------------------------------- Return final schema
    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_description.strip(),
            "parameters": parameters_schema,
            "strict": strict,
        },
    }

def process_response_format(
    handler,
    raw_response: Any,
    text: Optional[str],
    tool_calls: List[Dict[str, Any]],
    tool_map: Dict[str, Any],
    messages: List[Dict[str, Any]]
) -> Optional[Tuple[Any, bool]]:
    """
    Process response format handling for structured outputs.
    
    Returns:
        None if no response format processing needed
        Tuple of (structured_response, should_continue) if processing occurred
    """
    
    def _handle_no_tool_calls() -> Tuple[Any, bool]:
        """Handle case when no tool calls are present."""
        # Add retry message and continue
        messages.append(handler.build_response_format_retry_message())
        return None, True
    
    def _handle_response_format_tool_call(tool_call: Dict[str, Any]) -> Tuple[Any, bool]:
        """Handle a single response format tool call."""
        try:
            tool_result = run_sync_tool(tool_call, tool_map[tool_call["function"]["name"]], False)  # Don't use graceful error handling for structured output
            
            # Success - return the structured response
            return tool_result["output"], False
        except Exception as e:
            # Add error context and retry
            error_result = {
                "tool_call_id": tool_call["id"],
                "output": f"Error executing structured output tool: {e}",
                "is_error": True,
            }
            messages.append(handler.build_assistant_message(text, [tool_call], raw_response))
            messages.extend(handler.build_tool_result_messages([error_result]))
            return str(e), True
    
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
