import pytest
from toolflow import tool, from_openai
import datetime
import openai
import os
from unittest.mock import Mock, patch


@tool
def get_current_time():
    """Get the current time."""
    return str(datetime.datetime.now())


@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


def test_tool_decorator():
    """Test that the @tool decorator works properly."""
    # Test that the decorator doesn't break function behavior
    result = divide(10.0, 2.0)
    assert result == 5.0
    
    # Test that get_current_time returns a string
    time_result = get_current_time()
    assert isinstance(time_result, str)


def test_divide_function():
    """Test the divide function."""
    assert divide(10, 2) == 5.0
    assert divide(15, 3) == 5.0
    assert divide(7, 2) == 3.5
    
    # Test division by zero
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)


def test_get_current_time_function():
    """Test the get_current_time function."""
    result = get_current_time()
    assert isinstance(result, str)
    # Test that it looks like a datetime string
    assert len(result) > 10  # Basic sanity check


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_integration():
    """Test integration with OpenAI API (requires API key)."""
    client = from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    # This is a basic integration test - in a real test environment you'd mock this
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 3.145 divided by 2?"}],
        tools=[divide],
    )
    assert response.choices[0].message.content is not None


def test_import_functionality():
    """Test that the toolflow imports work correctly."""
    from toolflow import tool, from_openai
    
    # Basic smoke test - if we get here, imports worked
    assert tool is not None
    assert from_openai is not None
