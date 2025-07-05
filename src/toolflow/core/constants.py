RESPONSE_FORMAT_TOOL_NAME = "provide_final_answer"

DEFAULT_PARAMS = {
    "max_tool_call_rounds": 10,
    "max_response_format_retries": 2,
    "parallel_tool_execution": False,  # Default to sequential for playground-friendliness
    "full_response": False,
    "graceful_error_handling": True
}
