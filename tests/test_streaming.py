"""
Tests for streaming functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock
import toolflow

def test_sync_streaming_without_tools():
    """Test sync streaming without tools."""
    # Mock OpenAI client
    mock_client = Mock()
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="Hello", tool_calls=None))]),
        Mock(choices=[Mock(delta=Mock(content=" world", tool_calls=None))]),
        Mock(choices=[Mock(delta=Mock(content="!", tool_calls=None))])
    ]
    mock_client.chat.completions.create.return_value = iter(mock_stream)
    
    client = toolflow.from_openai(mock_client)
    
    # Test streaming without tools
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    
    # Collect all chunks
    chunks = list(stream)
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].delta.content == " world"
    assert chunks[2].choices[0].delta.content == "!"
    
    # Verify the correct parameters were passed
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["stream"] is True
    assert call_args[1]["model"] == "gpt-4o-mini"


def test_sync_streaming_with_tools():
    """Test sync streaming with tool calls."""
    @toolflow.tool
    def test_tool(message: str) -> str:
        return f"Tool response: {message}"
    
    # Mock OpenAI client
    mock_client = Mock()
    
    # First stream with tool call
    # Create proper mock for function that returns actual string values
    mock_function = Mock()
    mock_function.name = "test_tool"
    mock_function.arguments = '{"message": "test"}'
    
    first_stream = [
        Mock(choices=[Mock(delta=Mock(content="I'll call", tool_calls=None))]),
        Mock(choices=[Mock(delta=Mock(
            content=None,
            tool_calls=[Mock(
                index=0,
                id="call_123",
                function=mock_function
            )]
        ))])
    ]
    
    # Second stream after tool execution
    second_stream = [
        Mock(choices=[Mock(delta=Mock(content="The tool", tool_calls=None))]),
        Mock(choices=[Mock(delta=Mock(content=" returned: Tool response: test", tool_calls=None))])
    ]
    
    mock_client.chat.completions.create.side_effect = [
        iter(first_stream),
        iter(second_stream)
    ]
    
    client = toolflow.from_openai(mock_client)
    
    # Test streaming with tools
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Call the test tool"}],
        tools=[test_tool],
        stream=True
    )
    
    # Collect all chunks
    chunks = list(stream)
    assert len(chunks) == 4  # 2 from first stream + 2 from second stream
    
    # Verify multiple calls were made (one for tool call, one after tool execution)
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_async_streaming_without_tools():
    """Test async streaming without tools."""
    # Mock AsyncOpenAI client with proper AsyncMock
    mock_client = Mock()
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="Hello", tool_calls=None))]),
        Mock(choices=[Mock(delta=Mock(content=" async", tool_calls=None))]),
        Mock(choices=[Mock(delta=Mock(content=" world!", tool_calls=None))])
    ]
    
    async def async_iter():
        for chunk in mock_stream:
            yield chunk
    
    # Use AsyncMock for the create method to properly handle async calls
    mock_client.chat.completions.create = AsyncMock(return_value=async_iter())
    
    client = toolflow.from_openai_async(mock_client)
    
    # Test async streaming without tools
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    
    # Collect all chunks
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].delta.content == " async"
    assert chunks[2].choices[0].delta.content == " world!"
    
    # Verify the correct parameters were passed
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["stream"] is True
    assert call_args[1]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_async_streaming_with_tools():
    """Test async streaming with tool calls."""
    @toolflow.tool
    async def async_test_tool(message: str) -> str:
        return f"Async tool response: {message}"
    
    # Mock AsyncOpenAI client with proper AsyncMock
    mock_client = Mock()
    
    # First stream with tool call
    # Create proper mock for function that returns actual string values
    mock_function = Mock()
    mock_function.name = "async_test_tool"
    mock_function.arguments = '{"message": "async test"}'
    
    first_stream = [
        Mock(choices=[Mock(delta=Mock(content="I'll call", tool_calls=None))]),
        Mock(choices=[Mock(delta=Mock(
            content=None,
            tool_calls=[Mock(
                index=0,
                id="call_456",
                function=mock_function
            )]
        ))])
    ]
    
    # Second stream after tool execution
    second_stream = [
        Mock(choices=[Mock(delta=Mock(content="The async tool", tool_calls=None))]),
        Mock(choices=[Mock(delta=Mock(content=" returned successfully", tool_calls=None))])
    ]
    
    async def first_async_iter():
        for chunk in first_stream:
            yield chunk
            
    async def second_async_iter():
        for chunk in second_stream:
            yield chunk
    
    # Use AsyncMock with side_effect for multiple calls
    mock_client.chat.completions.create = AsyncMock(side_effect=[
        first_async_iter(),
        second_async_iter()
    ])
    
    client = toolflow.from_openai_async(mock_client)
    
    # Test async streaming with tools
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Call the async test tool"}],
        tools=[async_test_tool],
        stream=True
    )
    
    # Collect all chunks
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    assert len(chunks) == 4  # 2 from first stream + 2 from second stream
    
    # Verify multiple calls were made (one for tool call, one after tool execution)
    assert mock_client.chat.completions.create.call_count == 2


def test_streaming_returns_iterator():
    """Test that streaming returns an iterator."""
    mock_client = Mock()
    mock_stream = [Mock(choices=[Mock(delta=Mock(content="test", tool_calls=None))])]
    mock_client.chat.completions.create.return_value = iter(mock_stream)
    
    client = toolflow.from_openai(mock_client)
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    
    # Should be an iterator/generator
    assert hasattr(stream, '__iter__')
    assert hasattr(stream, '__next__')


@pytest.mark.asyncio 
async def test_async_streaming_returns_async_iterator():
    """Test that async streaming returns an async iterator."""
    mock_client = Mock()
    
    async def empty_async_iter():
        return
        yield  # Unreachable, but makes this an async generator
    
    # Use AsyncMock for proper async behavior
    mock_client.chat.completions.create = AsyncMock(return_value=empty_async_iter())
    
    client = toolflow.from_openai_async(mock_client)
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    
    # Should be an async iterator/generator
    assert hasattr(stream, '__aiter__')
    assert hasattr(stream, '__anext__') 