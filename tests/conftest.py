"""
Common test fixtures and utilities for toolflow tests.
"""
import pytest
from unittest.mock import Mock, AsyncMock
import datetime
import time
import asyncio

# Import toolflow functions with graceful fallbacks
try:
    from toolflow import tool, from_openai, from_openai_async
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from toolflow import from_anthropic, from_anthropic_async
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Common test tools (provider-agnostic)
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


@tool
def weather_tool(city: str) -> str:
    """Get weather for a city (mock tool)."""
    return f"Weather in {city}: Sunny, 72°F"


@tool
def calculator_tool(operation: str, a: float, b: float) -> float:
    """Perform mathematical operations."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


# OpenAI-specific fixtures
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
    if not OPENAI_AVAILABLE:
        pytest.skip("OpenAI not available")
    return from_openai(mock_openai_client, full_response=True)


@pytest.fixture
def async_toolflow_client(mock_async_openai_client):
    """Create an async toolflow client with mocked OpenAI client."""
    if not OPENAI_AVAILABLE:
        pytest.skip("OpenAI not available")
    return from_openai_async(mock_async_openai_client, full_response=True)


# Anthropic-specific fixtures
@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = Mock()
    client.messages = Mock()
    return client


@pytest.fixture
def mock_async_anthropic_client():
    """Create a mock async Anthropic client."""
    client = Mock()
    client.messages = AsyncMock()
    return client


@pytest.fixture
def sync_anthropic_client(mock_anthropic_client):
    """Create a sync toolflow Anthropic client with mocked client."""
    if not ANTHROPIC_AVAILABLE:
        pytest.skip("Anthropic not available")
    return from_anthropic(mock_anthropic_client, full_response=True)


@pytest.fixture
def async_anthropic_client(mock_async_anthropic_client):
    """Create an async toolflow Anthropic client with mocked client."""
    if not ANTHROPIC_AVAILABLE:
        pytest.skip("Anthropic not available")
    return from_anthropic_async(mock_async_anthropic_client, full_response=True)


# OpenAI helper functions
def create_mock_openai_tool_call(call_id: str, function_name: str, arguments: dict):
    """Helper to create a mock OpenAI tool call."""
    import json
    mock_call = Mock()
    mock_call.id = call_id
    mock_call.function.name = function_name
    mock_call.function.arguments = json.dumps(arguments)
    return mock_call


def create_mock_openai_response(tool_calls=None, content=None):
    """Helper to create a mock OpenAI response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.tool_calls = tool_calls
    response.choices[0].message.content = content
    return response


def create_mock_openai_streaming_chunk(content=None, tool_calls=None):
    """Helper to create a mock OpenAI streaming chunk."""
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.tool_calls = tool_calls
    return chunk


# Anthropic helper functions
def create_mock_anthropic_tool_call(call_id: str, tool_name: str, tool_input: dict):
    """Helper to create a mock Anthropic tool call."""
    mock_call = Mock()
    mock_call.id = call_id
    mock_call.name = tool_name
    mock_call.input = tool_input
    mock_call.type = "tool_use"
    return mock_call


def create_mock_anthropic_response(tool_calls=None, content=None):
    """Helper to create a mock Anthropic response."""
    response = Mock()
    
    if content is None and tool_calls is None:
        content = "Test response"
    
    # Build content array
    content_blocks = []
    
    if content:
        content_blocks.append(Mock(type="text", text=content))
    
    if tool_calls:
        for tool_call in tool_calls:
            content_blocks.append(tool_call)
    
    response.content = content_blocks
    return response


def create_mock_anthropic_streaming_chunk(chunk_type: str, **kwargs):
    """Helper to create a mock Anthropic streaming chunk."""
    chunk = Mock()
    chunk.type = chunk_type
    
    if chunk_type == "content_block_start":
        chunk.index = kwargs.get("index", 0)
        chunk.content_block = kwargs.get("content_block")
    elif chunk_type == "content_block_delta":
        chunk.index = kwargs.get("index", 0)
        chunk.delta = kwargs.get("delta")
    elif chunk_type == "content_block_stop":
        chunk.index = kwargs.get("index", 0)
    elif chunk_type == "message_start":
        chunk.message = kwargs.get("message", Mock())
    elif chunk_type == "message_delta":
        chunk.delta = kwargs.get("delta", Mock())
    elif chunk_type == "message_stop":
        pass
    
    return chunk


# Backward compatibility aliases
create_mock_tool_call = create_mock_openai_tool_call
create_mock_response = create_mock_openai_response
create_mock_streaming_chunk = create_mock_openai_streaming_chunk
