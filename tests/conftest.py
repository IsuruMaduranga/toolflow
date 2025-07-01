"""
Shared test fixtures and utilities for the toolflow test suite.
"""
import pytest
import asyncio
import time
import datetime
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any
from pydantic import BaseModel

# Import toolflow components
try:
    import toolflow
    from toolflow import tool, from_openai, from_anthropic
    TOOLFLOW_AVAILABLE = True
except ImportError:
    TOOLFLOW_AVAILABLE = False

# Pydantic models for structured output testing
class Person(BaseModel):
    name: str
    age: int
    skills: List[str]

class TeamAnalysis(BaseModel):
    people: List[Person]
    average_age: float
    top_skills: List[str]

class WeatherInfo(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int

# Common test tools
@tool
def simple_math_tool(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def divide_tool(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def multiply_tool(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def weather_tool(city: str) -> str:
    """Get weather information for a city (mock)."""
    return f"Weather in {city}: Sunny, 72°F, Humidity: 45%"

@tool
def population_tool(city: str) -> str:
    """Get population information for a city (mock)."""
    populations = {
        "new york": "8.3 million",
        "london": "9.0 million", 
        "tokyo": "14.0 million",
        "paris": "2.1 million"
    }
    return f"Population of {city}: {populations.get(city.lower(), 'Unknown')}"

@tool
def slow_tool(name: str, delay: float = 0.1) -> str:
    """A tool that takes time to execute (for parallel execution testing)."""
    time.sleep(delay)
    return f"Completed {name} after {delay}s delay"

@tool
def failing_tool(should_fail: bool = True, error_message: str = "Tool failed") -> str:
    """A tool that can fail for error handling testing."""
    if should_fail:
        raise ValueError(error_message)
    return "Tool succeeded"

@tool
def get_time_tool() -> str:
    """Get current timestamp."""
    return datetime.datetime.now().isoformat()

@tool
async def async_math_tool(a: float, b: float) -> float:
    """Add two numbers asynchronously."""
    await asyncio.sleep(0.01)  # Small delay to simulate async work
    return a + b

@tool
async def slow_async_tool(name: str, delay: float = 0.1) -> str:
    """An async tool that takes time to execute."""
    await asyncio.sleep(delay)
    return f"Async completed {name} after {delay}s delay"

@tool
async def failing_async_tool(should_fail: bool = True) -> str:
    """An async tool that can fail."""
    await asyncio.sleep(0.01)
    if should_fail:
        raise ValueError("Async tool failed")
    return "Async tool succeeded"

# Tool collections for testing
BASIC_TOOLS = [simple_math_tool, divide_tool, multiply_tool]
MIXED_TOOLS = [weather_tool, population_tool, simple_math_tool]
SLOW_TOOLS = [slow_tool]
ASYNC_TOOLS = [async_math_tool, slow_async_tool]
ERROR_TOOLS = [failing_tool, failing_async_tool]

# OpenAI Mock Helpers
def create_openai_tool_call(call_id: str, function_name: str, arguments: dict):
    """Create a mock OpenAI tool call."""
    import json
    mock_call = Mock()
    mock_call.id = call_id
    mock_call.function = Mock()
    mock_call.function.name = function_name
    mock_call.function.arguments = json.dumps(arguments)
    mock_call.type = "function"
    return mock_call

def create_openai_response(content: str = None, tool_calls: List = None, usage: dict = None):
    """Create a mock OpenAI response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls or []
    response.choices[0].finish_reason = "stop" if not tool_calls else "tool_calls"
    
    # Add usage information
    response.usage = Mock()
    if usage:
        for key, value in usage.items():
            setattr(response.usage, key, value)
    else:
        response.usage.total_tokens = 100
        response.usage.prompt_tokens = 50
        response.usage.completion_tokens = 50
    
    return response

def create_openai_structured_response(json_data: dict, usage: dict = None):
    """Create a mock OpenAI response with a structured output tool call."""
    tool_call = create_openai_tool_call(
        "call_structured_output", 
        "final_response_tool_internal", 
        {"response": json_data}
    )
    return create_openai_response(content=None, tool_calls=[tool_call], usage=usage)

def create_openai_stream_chunk(content: str = None, tool_calls: List = None, finish_reason: str = None):
    """Create a mock OpenAI streaming chunk."""
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.tool_calls = tool_calls
    chunk.choices[0].finish_reason = finish_reason
    return chunk

# Anthropic Mock Helpers  
def create_anthropic_tool_call(call_id: str, tool_name: str, tool_input: dict):
    """Create a mock Anthropic tool call."""
    mock_call = Mock()
    mock_call.id = call_id
    mock_call.name = tool_name
    mock_call.input = tool_input
    mock_call.type = "tool_use"
    return mock_call

def create_anthropic_response(content: str = None, tool_calls: List = None, usage: dict = None):
    """Create a mock Anthropic response."""
    response = Mock()
    
    # Create content blocks
    content_blocks = []
    if content:
        text_block = Mock()
        text_block.type = "text"
        text_block.text = content
        content_blocks.append(text_block)
    
    if tool_calls:
        content_blocks.extend(tool_calls)
    
    response.content = content_blocks
    response.stop_reason = "end_turn" if not tool_calls else "tool_use"
    
    # Add usage information
    response.usage = Mock()
    if usage:
        for key, value in usage.items():
            setattr(response.usage, key, value)
    else:
        response.usage.input_tokens = 50
        response.usage.output_tokens = 50
    
    return response

def create_anthropic_structured_response(json_data: dict, usage: dict = None):
    """Create a mock Anthropic response with a structured output tool call."""
    tool_call = create_anthropic_tool_call(
        "toolu_structured_output", 
        "final_response_tool_internal", 
        {"response": json_data}
    )
    return create_anthropic_response(content=None, tool_calls=[tool_call], usage=usage)

def create_anthropic_stream_chunk(chunk_type: str, **kwargs):
    """Create a mock Anthropic streaming chunk."""
    chunk = Mock()
    chunk.type = chunk_type
    
    if chunk_type == "content_block_delta":
        chunk.delta = Mock()
        chunk.delta.type = kwargs.get("delta_type", "text_delta")
        if "text" in kwargs:
            chunk.delta.text = kwargs["text"]
    elif chunk_type == "content_block_start":
        chunk.content_block = Mock()
        chunk.content_block.type = kwargs.get("block_type", "text")
    
    return chunk

# Test fixtures
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    return client

@pytest.fixture  
def mock_async_openai_client():
    """Mock async OpenAI client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = AsyncMock()
    return client

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    client = Mock()
    client.messages = Mock()
    return client

@pytest.fixture
def mock_async_anthropic_client():
    """Mock async Anthropic client."""
    client = Mock()
    client.messages = AsyncMock()
    return client

@pytest.fixture
def toolflow_openai_client(mock_openai_client):
    """Toolflow-wrapped OpenAI client."""
    if not TOOLFLOW_AVAILABLE:
        pytest.skip("Toolflow not available")
    return from_openai(mock_openai_client)

@pytest.fixture
def toolflow_anthropic_client(mock_anthropic_client):
    """Toolflow-wrapped Anthropic client."""
    if not TOOLFLOW_AVAILABLE:
        pytest.skip("Toolflow not available")
    return from_anthropic(mock_anthropic_client)

@pytest.fixture(autouse=True)
def reset_toolflow_config():
    """Reset toolflow configuration before each test."""
    if TOOLFLOW_AVAILABLE:
        # Reset to defaults
        toolflow.set_max_workers(4)
        toolflow.set_async_yield_frequency(0)

@pytest.fixture
def sample_messages():
    """Sample message list for testing."""
    return [
        {"role": "user", "content": "What's 10 + 5 and what's the weather in NYC?"}
    ]

# Pytest marks for test organization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.live = pytest.mark.live
pytest.mark.slow = pytest.mark.slow 