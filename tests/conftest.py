"""
Common test fixtures and utilities for toolflow tests.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from toolflow import tool, from_openai, from_openai_async
import datetime
import time
import asyncio


# Common test tools
@tool
def simple_math_tool(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def divide_tool(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


@tool
def get_current_time_tool():
    """Get the current time."""
    return str(datetime.datetime.now())


@tool
async def async_math_tool(a: float, b: float) -> float:
    """Add two numbers asynchronously."""
    await asyncio.sleep(0.001)
    return a + b


@tool
def slow_tool(name: str, delay: float) -> str:
    """A tool that takes time to execute (for testing parallel execution)."""
    time.sleep(delay)
    return f"Result from {name} after {delay}s"


@tool
async def slow_async_tool(name: str, delay: float) -> str:
    """An async tool that takes time to execute."""
    await asyncio.sleep(delay)
    return f"Async result from {name} after {delay}s"


@tool
def failing_tool(should_fail: bool = True) -> str:
    """A tool that can fail for testing error handling."""
    if should_fail:
        raise ValueError("This tool failed intentionally")
    return "Success"


# Fixtures
@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock()
    client.chat.completions = Mock()
    return client


@pytest.fixture
def mock_async_openai_client():
    """Create a mock async OpenAI client."""
    client = Mock()
    client.chat.completions = AsyncMock()
    return client


@pytest.fixture
def sync_toolflow_client(mock_openai_client):
    """Create a sync toolflow client with mocked OpenAI client."""
    # Use full_response=True for backward compatibility with existing tests
    # The default behavior (full_response=False) is tested in specific test classes
    return from_openai(mock_openai_client, full_response=True)


@pytest.fixture
def async_toolflow_client(mock_async_openai_client):
    """Create an async toolflow client with mocked OpenAI client."""
    # Use full_response=True for backward compatibility with existing tests
    # The default behavior (full_response=False) is tested in specific test classes
    return from_openai_async(mock_async_openai_client, full_response=True)


def create_mock_tool_call(call_id: str, function_name: str, arguments: dict):
    """Helper to create a mock tool call."""
    import json
    mock_call = Mock()
    mock_call.id = call_id
    mock_call.function.name = function_name
    mock_call.function.arguments = json.dumps(arguments)
    return mock_call


def create_mock_response(tool_calls=None, content=None):
    """Helper to create a mock OpenAI response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.tool_calls = tool_calls
    response.choices[0].message.content = content
    return response


def create_mock_streaming_chunk(content=None, tool_calls=None):
    """Helper to create a mock streaming chunk."""
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.tool_calls = tool_calls
    return chunk
