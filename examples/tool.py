
import toolflow
import json
from pydantic import BaseModel, Field
from typing import Annotated
from toolflow.core.constants import RESPONSE_FORMAT_TOOL_NAME

class UserDetails(BaseModel):
    """Detailed information about the user."""
    name: str = Field(..., description="Full name of the user.")
    email: str = Field(..., description="A valid email address.")

# @toolflow.tool
# def create_new_user(
#     details: UserDetails,
#     send_welcome_email: Annotated[bool, Field(description="Set to true to send a welcome email.")] = True,
#     department: str = "general"
# ):
#     """
#     Creates a new user with detailed information and optional settings. hdj
#     jwldjkj
#     """
#     pass


def create_anthropic_response_tool(response_format):
    """Create a dynamic response tool for Anthropic structured output."""
    from toolflow.core.decorators import tool
    
    # 1. Define the base function WITHOUT the decorator syntax.
    def final_response_tool_internal(response: response_format) -> str:
        """This is a placeholder docstring and will be replaced."""
        return "Response formatted successfully"
    
    # 2. Manually create the dynamic docstring and assign it to the function.
    # This must be done BEFORE the decorator is applied.
    final_response_tool_internal.__doc__ = f"""You must call this tool to provide your final response. Because user expects the final response in `{response_format.__name__}` format. This tool must be your last tool call."""

    # 3. Manually apply the decorator to the function.
    # This is equivalent to using the @tool(...) syntax but gives us control over the timing.
    # The decorator's logic will now execute and see the correct, dynamic docstring.
    decorated_tool = tool(name=RESPONSE_FORMAT_TOOL_NAME, internal=True)(final_response_tool_internal)

    return decorated_tool

tool = create_anthropic_response_tool(UserDetails)

print(json.dumps(tool._tool_metadata, indent=2))

#print(json.dumps(create_new_user._tool_metadata, indent=2))
