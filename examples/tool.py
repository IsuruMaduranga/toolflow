
import toolflow
import json
from pydantic import BaseModel, Field
from typing import Annotated

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
#     Creates a new user with detailed information and optional settings.

#     Args:
#         department: The department to assign the new user to.
#     """
#     pass


@toolflow.tool
def no_params_tool():
    """Tool with no parameters."""
    return "no params"



print(json.dumps(no_params_tool._tool_metadata, indent=2))
