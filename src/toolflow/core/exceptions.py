from typing import Optional

class MaxToolCallsError(Exception):
    """
    Raised when the maximum number of tool calls is reached without completion.
    
    This allows callers to catch this specific case and potentially increase
    the max_tool_call_rounds budget or handle the scenario appropriately.
    """
    def __init__(self, message: str, max_tool_call_rounds: Optional[int] = None):
        super().__init__(message)
        self.max_tool_call_rounds = max_tool_call_rounds

class MaxTokensError(Exception):
    """
    Raised when the maximum number of tokens is reached without completion.
    """
    def __init__(self, message: str):
        super().__init__(message)

class ResponseFormatError(Exception):
    """
    Raised when response format is specified but model did not return a structured output.
    """
    def __init__(self, message: str):
        super().__init__(message)

class MissingAnnotationError(TypeError):
    """Raised when a function parameter lacks a type annotation."""

class UndescribableTypeError(TypeError):
    """Raised when a parameter's annotation cannot be rendered as JSON‑Schema."""
