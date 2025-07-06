#!/usr/bin/env python3

from pydantic import BaseModel
from toolflow import tool
from toolflow.core.tool_execution import run_sync_tool

class UserData(BaseModel):
    name: str
    age: int
    email: str

@tool
def process_user_data(user: UserData) -> str:
    """Process user data with validation.
    
    Args:
        user: User data with name (string), age (integer), and email (string)
               Example: {"name": "John", "age": 25, "email": "john@example.com"}
    
    Returns:
        Processed user information string
    """
    return f"Processed user: {user.name}, age {user.age}, email {user.email}"

def test_enhanced_error_messages():
    """Test that enhanced error messages provide helpful guidance."""
    
    print("Testing enhanced error messages...\n")
    
    # Test 1: Validation error (wrong type)
    print("1. Testing validation error (age as string instead of int):")
    tool_call = {
        "id": "test_1",
        "function": {
            "name": "process_user_data",
            "arguments": {
                "user": {"name": "John", "age": "twenty-five", "email": "john@example.com"}
            }
        }
    }
    
    result = run_sync_tool(tool_call, process_user_data, graceful_error_handling=True)
    print(f"Result: {result['output']}\n")
    print("-" * 80 + "\n")
    
    # Test 2: Missing required field
    print("2. Testing missing required field (no email):")
    tool_call = {
        "id": "test_2", 
        "function": {
            "name": "process_user_data",
            "arguments": {
                "user": {"name": "Jane", "age": 30}
            }
        }
    }
    
    result = run_sync_tool(tool_call, process_user_data, graceful_error_handling=True)
    print(f"Result: {result['output']}\n")
    print("-" * 80 + "\n")
    
    # Test 3: Successful execution for comparison
    print("3. Testing successful execution:")
    tool_call = {
        "id": "test_3",
        "function": {
            "name": "process_user_data", 
            "arguments": {
                "user": {"name": "Alice", "age": 28, "email": "alice@example.com"}
            }
        }
    }
    
    result = run_sync_tool(tool_call, process_user_data, graceful_error_handling=True)
    print(f"Result: {result['output']}\n")

if __name__ == "__main__":
    test_enhanced_error_messages() 