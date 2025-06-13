"""
Test async functionality of the toolflow library with Anthropic.

This module tests:
- Async client with sync tools
- Async client with async tools  
- Mixed sync/async tool execution
- Parallel execution with async client
- Error handling in async contexts
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock
from typing import List
from pydantic import BaseModel

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False

import toolflow
from ..conftest import (
    simple_math_tool,
    async_math_tool,
    slow_tool,
    slow_async_tool,
    failing_tool,
    calculator_tool,
    create_mock_anthropic_tool_call,
    create_mock_anthropic_response
)


# Test Pydantic models for structured output
class WeatherReport(BaseModel):
    location: str
    temperature: int
    description: str


class CalculationResult(BaseModel):
    operation: str
    operands: list[int]
    result: int


@toolflow.tool
def sync_add(a: int, b: int) -> int:
    """Synchronously add two numbers."""
    return a + b


@toolflow.tool
async def async_multiply(a: int, b: int) -> int:
    """Asynchronously multiply two numbers."""
    await asyncio.sleep(0.01)  # Simulate async work
    return a * b


@toolflow.tool
async def async_get_weather(city: str) -> str:
    """Asynchronously get weather for a city."""
    await asyncio.sleep(0.01)  # Simulate async API call
    return f"Weather in {city}: 75°F, partly cloudy"


def create_mock_anthropic_async_client():
    """Create a mock async Anthropic client."""
    mock_client = Mock()
    mock_messages = Mock()
    mock_client.messages = mock_messages
    return mock_client, mock_messages


def create_mock_anthropic_text_block(text):
    """Create a mock text content block."""
    mock_block = Mock()
    mock_block.type = "text"
    mock_block.text = text
    return mock_block


def create_mock_anthropic_tool_use_block(tool_id, name, input_data):
    """Create a mock tool use content block."""
    mock_block = Mock()
    mock_block.type = "tool_use"
    mock_block.id = tool_id
    mock_block.name = name
    mock_block.input = input_data
    return mock_block


class TestAnthropicAsyncBasics:
    """Test basic async functionality with Anthropic."""
    
    @pytest.fixture
    def async_anthropic_client(self, mock_async_anthropic_client):
        """Create async toolflow Anthropic client."""
        return toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=True)
    
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_sync_tools(self, async_anthropic_client, mock_async_anthropic_client):
        """Test async client with synchronous tools."""
        # Mock tool call and responses
        tool_call = create_mock_anthropic_tool_call("call_sync", "simple_math_tool", {"a": 15.0, "b": 25.0})
        mock_response_1 = create_mock_anthropic_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_anthropic_response(content="The sum is 40.0")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 15 + 25?"}],
            tools=[simple_math_tool]
        )
        
        assert response.content[0].text == "The sum is 40.0"
        assert mock_async_anthropic_client.messages.create.call_count == 2

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_async_tools(self, async_anthropic_client, mock_async_anthropic_client):
        """Test async client with async tools."""
        tool_call = create_mock_anthropic_tool_call("call_async", "async_math_tool", {"a": 8.0, "b": 7.0})
        mock_response_1 = create_mock_anthropic_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_anthropic_response(content="The async result is 15.0")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate 8 + 7 asynchronously"}],
            tools=[async_math_tool]
        )
        
        assert response.content[0].text == "The async result is 15.0"
        assert mock_async_anthropic_client.messages.create.call_count == 2

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_mixed_sync_async_tools(self, async_anthropic_client, mock_async_anthropic_client):
        """Test async client with both sync and async tools."""
        # Multiple tool calls - one sync, one async
        tool_call_1 = create_mock_anthropic_tool_call("call_sync", "simple_math_tool", {"a": 10.0, "b": 5.0})
        tool_call_2 = create_mock_anthropic_tool_call("call_async", "async_math_tool", {"a": 3.0, "b": 4.0})
        
        # Create mock response with both tool calls
        text_block = Mock(type="text", text="I'll calculate both for you:")
        mock_response_1 = Mock()
        mock_response_1.content = [text_block, tool_call_1, tool_call_2]
        
        mock_response_2 = create_mock_anthropic_response(content="Sync result: 15.0, Async result: 7.0")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate 10+5 and 3+4 (one sync, one async)"}],
            tools=[simple_math_tool, async_math_tool],
            parallel_tool_execution=True
        )
        
        assert response.content[0].text == "Sync result: 15.0, Async result: 7.0"
        assert mock_async_anthropic_client.messages.create.call_count == 2


class TestAnthropicAsyncParallelExecution:
    """Test parallel execution with async Anthropic client."""
    
    @pytest.fixture
    def async_anthropic_client(self, mock_async_anthropic_client):
        """Create async toolflow Anthropic client."""
        return toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=True)
    
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_parallel_sync_tools(self, async_anthropic_client, mock_async_anthropic_client):
        """Test parallel execution of multiple sync tools."""
        # Multiple tool calls that can be executed in parallel
        tool_call_1 = create_mock_anthropic_tool_call("call_1", "calculator_tool", {"operation": "add", "a": 10.0, "b": 20.0})
        tool_call_2 = create_mock_anthropic_tool_call("call_2", "calculator_tool", {"operation": "multiply", "a": 5.0, "b": 6.0})
        tool_call_3 = create_mock_anthropic_tool_call("call_3", "calculator_tool", {"operation": "subtract", "a": 50.0, "b": 15.0})
        
        text_block = Mock(type="text", text="Calculating multiple operations:")
        mock_response_1 = Mock()
        mock_response_1.content = [text_block, tool_call_1, tool_call_2, tool_call_3]
        
        mock_response_2 = create_mock_anthropic_response(content="Results: 30.0, 30.0, 35.0")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate 10+20, 5*6, and 50-15"}],
            tools=[calculator_tool],
            parallel_tool_execution=True,
            max_workers=3
        )
        execution_time = time.time() - start_time
        
        assert response.content[0].text == "Results: 30.0, 30.0, 35.0"
        # Parallel execution should be relatively fast
        assert execution_time < 1.0  # Should complete quickly with mocked tools

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_parallel_async_tools(self, async_anthropic_client, mock_async_anthropic_client):
        """Test parallel execution of multiple async tools."""
        tool_call_1 = create_mock_anthropic_tool_call("call_1", "slow_async_tool", {"name": "task1", "delay": 0.1})
        tool_call_2 = create_mock_anthropic_tool_call("call_2", "slow_async_tool", {"name": "task2", "delay": 0.1})
        
        text_block = Mock(type="text", text="Running async tasks:")
        mock_response_1 = Mock()
        mock_response_1.content = [text_block, tool_call_1, tool_call_2]
        
        mock_response_2 = create_mock_anthropic_response(content="Both async tasks completed")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Run two async tasks in parallel"}],
            tools=[slow_async_tool],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        assert response.content[0].text == "Both async tasks completed"
        # Should be faster than sequential execution (0.2s vs 0.1s for parallel)
        assert execution_time < 0.15  # Some overhead but should be much faster than 0.2s

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_sequential_vs_parallel_performance(self, async_anthropic_client, mock_async_anthropic_client):
        """Compare sequential vs parallel execution performance."""
        # Test with slow tools to see the difference
        tool_call_1 = create_mock_anthropic_tool_call("call_1", "slow_tool", {"name": "seq1", "delay": 0.05})
        tool_call_2 = create_mock_anthropic_tool_call("call_2", "slow_tool", {"name": "seq2", "delay": 0.05})
        
        text_block = Mock(type="text", text="Processing tasks:")
        mock_response_1 = Mock()
        mock_response_1.content = [text_block, tool_call_1, tool_call_2]
        
        mock_response_2 = create_mock_anthropic_response(content="Tasks completed")
        
        # Test sequential execution
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Run tasks sequentially"}],
            tools=[slow_tool],
            parallel_tool_execution=False
        )
        sequential_time = time.time() - start_time
        
        # Reset mocks for parallel test
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # Test parallel execution
        start_time = time.time()
        await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Run tasks in parallel"}],
            tools=[slow_tool],
            parallel_tool_execution=True
        )
        parallel_time = time.time() - start_time
        
        # Parallel should be faster (though with mocking, the difference might be minimal)
        print(f"Sequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s")
        assert parallel_time <= sequential_time * 1.1  # Allow some variance


class TestAnthropicAsyncErrorHandling:
    """Test error handling in async contexts."""
    
    @pytest.fixture
    def async_anthropic_client(self, mock_async_anthropic_client):
        """Create async toolflow Anthropic client."""
        return toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=True)
    
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_tool_error_handling(self, async_anthropic_client, mock_async_anthropic_client):
        """Test graceful error handling with async tools."""
        tool_call = create_mock_anthropic_tool_call("call_error", "failing_tool", {"should_fail": True})
        mock_response_1 = create_mock_anthropic_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_anthropic_response(content="I encountered an error but handled it gracefully.")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Try to use the failing tool"}],
            tools=[failing_tool],
            graceful_error_handling=True
        )
        
        assert response.content[0].text == "I encountered an error but handled it gracefully."
        assert mock_async_anthropic_client.messages.create.call_count == 2

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_max_tool_calls_limit(self, async_anthropic_client, mock_async_anthropic_client):
        """Test max tool calls limit with async client."""
        # Create a tool call that would cause infinite recursion
        tool_call = create_mock_anthropic_tool_call("call_loop", "simple_math_tool", {"a": 1.0, "b": 1.0})
        mock_response = create_mock_anthropic_response(tool_calls=[tool_call])
        mock_async_anthropic_client.messages.create.return_value = mock_response
        
        with pytest.raises(Exception, match="Max tool calls reached"):
            await async_anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Keep calculating"}],
                tools=[simple_math_tool],
                max_tool_calls=3  # Low limit
            )

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_timeout_handling(self, async_anthropic_client, mock_async_anthropic_client):
        """Test timeout handling in async context."""
        # Mock a slow response
        async def slow_response():
            await asyncio.sleep(0.1)  # Simulate slow response
            return create_mock_anthropic_response(content="Slow response")
        
        mock_async_anthropic_client.messages.create.return_value = slow_response()
        
        # This should complete successfully (no actual timeout set in our implementation)
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test timeout"}]
        )
        
        assert response.content[0].text == "Slow response"

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_async_anthropic_client):
        """Test handling multiple concurrent requests."""
        async_client = toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=False)
        
        # Mock responses for concurrent requests
        responses = [
            create_mock_anthropic_response(content=f"Response {i}")
            for i in range(3)
        ]
        
        mock_async_anthropic_client.messages.create.side_effect = responses
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(3):
            task = async_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"Request {i}"}]
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)
        
        # All requests should complete successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == f"Response {i}"


class TestAnthropicAsyncFullResponseParameter:
    """Test full_response parameter with async client."""
    
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_full_response_true(self, mock_async_anthropic_client):
        """Test async client with full_response=True."""
        mock_response = create_mock_anthropic_response(content="Full response test")
        mock_async_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=True)
        
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test full response"}]
        )
        
        # Should return the full mock response object
        assert response == mock_response
        assert hasattr(response, 'content')

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_full_response_false(self, mock_async_anthropic_client):
        """Test async client with full_response=False."""
        mock_response = create_mock_anthropic_response(content="Simplified async response")
        mock_async_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=False)
        
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test simplified response"}]
        )
        
        # Should return just the text content
        assert response == "Simplified async response"

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_method_level_override(self, mock_async_anthropic_client):
        """Test method-level full_response override with async client."""
        mock_response = create_mock_anthropic_response(content="Override test async")
        mock_async_anthropic_client.messages.create.return_value = mock_response
        
        # Client set to full_response=False
        client = toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=False)
        
        # Override at method level to full_response=True
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test override"}],
            full_response=True
        )
        
        # Should return full response due to method override
        assert response == mock_response
        assert hasattr(response, 'content')


class TestAnthropicAsyncSystemMessages:
    """Test system message handling with async client."""
    
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_system_message_handling(self, mock_async_anthropic_client):
        """Test that system messages work correctly with async client."""
        mock_response = create_mock_anthropic_response(content="I'm an async math assistant!")
        mock_async_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=False)
        
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are a helpful math assistant that works asynchronously.",
            messages=[{"role": "user", "content": "Hello, are you ready to help?"}],
            tools=[calculator_tool]
        )
        
        # Verify system message was passed correctly
        call_args = mock_async_anthropic_client.messages.create.call_args
        assert call_args[1]["system"] == "You are a helpful math assistant that works asynchronously."
        assert response == "I'm an async math assistant!"


class TestAnthropicAsyncBasicFunctionality:
    """Test basic async Anthropic functionality."""
    
    @pytest.mark.asyncio
    async def test_simple_async_message_without_tools(self):
        """Test simple async message without tools."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        # Mock response
        text_block = create_mock_anthropic_text_block("Hello from async!")
        mock_response = create_mock_anthropic_response([text_block])
        mock_messages.create = AsyncMock(return_value=mock_response)
        
        # Create async wrapper
        client = toolflow.from_anthropic_async(mock_client)
        
        # Test async call
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result == "Hello from async!"
        mock_messages.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_single_tool_call(self):
        """Test async single tool call execution."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        # Mock first response with tool call
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", "async_get_weather", {"city": "Seattle"}
        )
        first_response = create_mock_anthropic_response([tool_block])
        
        # Mock second response with final answer
        text_block = create_mock_anthropic_text_block("The weather in Seattle is 75°F, partly cloudy.")
        second_response = create_mock_anthropic_response([text_block])
        
        mock_messages.create = AsyncMock(side_effect=[first_response, second_response])
        
        # Create async wrapper
        client = toolflow.from_anthropic_async(mock_client)
        
        # Test async call
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
            tools=[async_get_weather]
        )
        
        assert result == "The weather in Seattle is 75°F, partly cloudy."
        assert mock_messages.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_tools(self):
        """Test mixing sync and async tools."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        # Mock responses for both tool calls
        sync_tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", "sync_add", {"a": 10, "b": 5}
        )
        first_response = create_mock_anthropic_response([sync_tool_block])
        
        async_tool_block = create_mock_anthropic_tool_use_block(
            "tool_2", "async_multiply", {"a": 15, "b": 3}
        )
        second_response = create_mock_anthropic_response([async_tool_block])
        
        text_block = create_mock_anthropic_text_block("10+5=15 and 15*3=45")
        final_response = create_mock_anthropic_response([text_block])
        
        mock_messages.create = AsyncMock(side_effect=[first_response, second_response, final_response])
        
        # Create async wrapper
        client = toolflow.from_anthropic_async(mock_client)
        
        # Test mixed tools
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Add 10+5 and multiply 15*3"}],
            tools=[sync_add, async_multiply]
        )
        
        assert result == "10+5=15 and 15*3=45"
        assert mock_messages.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_parallel_tool_execution(self):
        """Test parallel execution of async tools."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        # Mock response with multiple tool calls
        tool_block1 = create_mock_anthropic_tool_use_block(
            "tool_1", "async_multiply", {"a": 6, "b": 7}
        )
        tool_block2 = create_mock_anthropic_tool_use_block(
            "tool_2", "async_multiply", {"a": 8, "b": 9}
        )
        first_response = create_mock_anthropic_response([tool_block1, tool_block2])
        
        text_block = create_mock_anthropic_text_block("Results: 42 and 72")
        second_response = create_mock_anthropic_response([text_block])
        
        mock_messages.create = AsyncMock(side_effect=[first_response, second_response])
        
        # Create async wrapper
        client = toolflow.from_anthropic_async(mock_client)
        
        # Test parallel execution
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Calculate 6*7 and 8*9"}],
            tools=[async_multiply],
            parallel_tool_execution=True
        )
        
        assert result == "Results: 42 and 72"
        assert mock_messages.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_full_response_mode(self):
        """Test async full response mode."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        text_block = create_mock_anthropic_text_block("Async response!")
        mock_response = create_mock_anthropic_response([text_block])
        mock_messages.create = AsyncMock(return_value=mock_response)
        
        # Create async wrapper with full_response=True
        client = toolflow.from_anthropic_async(mock_client, full_response=True)
        
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result == mock_response


class TestAnthropicAsyncStructuredOutput:
    """Test async structured output functionality."""
    
    @pytest.mark.asyncio
    async def test_async_structured_output_integration(self):
        """Test end-to-end async structured output integration."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        # Mock response with structured tool call
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1",
            "final_response_tool_internal",
            {
                "response": {
                    "location": "Portland",
                    "temperature": 68,
                    "description": "Rainy with light showers"
                }
            }
        )
        mock_response = create_mock_anthropic_response([tool_block])
        mock_messages.create = AsyncMock(return_value=mock_response)
        
        # Create async wrapper
        client = toolflow.from_anthropic_async(mock_client)
        
        # Test async structured output call
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "What's the weather in Portland?"}],
            tools=[async_get_weather],
            response_format=WeatherReport
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, WeatherReport)
        assert result.location == "Portland"
        assert result.temperature == 68
        assert result.description == "Rainy with light showers"
    
    @pytest.mark.asyncio
    async def test_async_structured_output_with_tools(self):
        """Test async structured output with tool execution."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        # Mock first response with regular tool call
        calc_tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", "sync_add", {"a": 25, "b": 17}
        )
        first_response = create_mock_anthropic_response([calc_tool_block])
        
        # Mock second response with structured output tool call
        structured_tool_block = create_mock_anthropic_tool_use_block(
            "tool_2",
            "final_response_tool_internal",
            {
                "response": {
                    "operation": "addition",
                    "operands": [25, 17],
                    "result": 42
                }
            }
        )
        second_response = create_mock_anthropic_response([structured_tool_block])
        
        mock_messages.create = AsyncMock(side_effect=[first_response, second_response])
        
        # Create async wrapper
        client = toolflow.from_anthropic_async(mock_client)
        
        # Test structured output with tool execution
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Calculate 25 + 17"}],
            tools=[sync_add],
            response_format=CalculationResult
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, CalculationResult)
        assert result.operation == "addition"
        assert result.operands == [25, 17]
        assert result.result == 42
    
    @pytest.mark.asyncio
    async def test_async_structured_output_full_response_mode(self):
        """Test async structured output in full response mode."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        # Mock response with structured tool call
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1",
            "final_response_tool_internal",
            {
                "response": {
                    "location": "Denver",
                    "temperature": 55,
                    "description": "Sunny and clear"
                }
            }
        )
        mock_response = create_mock_anthropic_response([tool_block])
        mock_messages.create = AsyncMock(return_value=mock_response)
        
        # Create async wrapper with full_response=True
        client = toolflow.from_anthropic_async(mock_client, full_response=True)
        
        # Test async structured output call
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Weather in Denver?"}],
            response_format=WeatherReport
        )
        
        # Should return full response object with parsed attribute
        assert hasattr(result, 'parsed')
        
        parsed = result.parsed
        assert isinstance(parsed, WeatherReport)
        assert parsed.location == "Denver"
        assert parsed.temperature == 55
        assert parsed.description == "Sunny and clear"
    
    @pytest.mark.asyncio
    async def test_async_structured_output_with_streaming_raises_error(self):
        """Test that async structured output with streaming raises an error."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        client = toolflow.from_anthropic_async(mock_client)
        
        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            await client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=WeatherReport,
                stream=True
            )


class TestAnthropicAsyncErrorHandling:
    """Test async error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_async_graceful_tool_error_handling(self):
        """Test graceful handling of async tool execution errors."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        @toolflow.tool
        async def async_failing_tool(x: int) -> int:
            """An async tool that always fails."""
            await asyncio.sleep(0.01)
            raise ValueError("Async tool failed!")
        
        # Mock response with tool call
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", "async_failing_tool", {"x": 10}
        )
        first_response = create_mock_anthropic_response([tool_block])
        
        # Mock final response
        text_block = create_mock_anthropic_text_block("I encountered an async error.")
        second_response = create_mock_anthropic_response([text_block])
        
        mock_messages.create = AsyncMock(side_effect=[first_response, second_response])
        
        client = toolflow.from_anthropic_async(mock_client)
        
        # Should not raise exception due to graceful error handling
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Test async"}],
            tools=[async_failing_tool],
            graceful_error_handling=True
        )
        
        assert result == "I encountered an async error."
    
    @pytest.mark.asyncio
    async def test_async_non_graceful_tool_error_handling(self):
        """Test non-graceful handling of async tool execution errors."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        @toolflow.tool
        async def async_failing_tool(x: int) -> int:
            """An async tool that always fails."""
            await asyncio.sleep(0.01)
            raise ValueError("Async tool failed!")
        
        # Mock response with tool call
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", "async_failing_tool", {"x": 10}
        )
        mock_response = create_mock_anthropic_response([tool_block])
        mock_messages.create = AsyncMock(return_value=mock_response)
        
        client = toolflow.from_anthropic_async(mock_client)
        
        # Should raise exception when graceful_error_handling=False
        with pytest.raises(ValueError, match="Async tool failed!"):
            await client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Test async"}],
                tools=[async_failing_tool],
                graceful_error_handling=False
            )
    
    @pytest.mark.asyncio
    async def test_async_max_tool_calls_limit(self):
        """Test async max tool calls limit is enforced."""
        mock_client, mock_messages = create_mock_anthropic_async_client()
        
        # Always return a tool call to trigger infinite loop
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", "async_get_weather", {"city": "NYC"}
        )
        mock_response = create_mock_anthropic_response([tool_block])
        mock_messages.create = AsyncMock(return_value=mock_response)
        
        client = toolflow.from_anthropic_async(mock_client)
        
        with pytest.raises(Exception, match="Max tool calls reached"):
            await client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Weather?"}],
                tools=[async_get_weather],
                max_tool_calls=2
            ) 