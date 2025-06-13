"""
Test streaming functionality of the toolflow library with Anthropic.

This module tests:
- Streaming responses without tools
- Streaming responses with tool calls
- Async streaming functionality
- Error handling in streaming contexts
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Iterator, AsyncIterator

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
    calculator_tool,
    weather_tool,
    create_mock_anthropic_tool_call,
    create_mock_anthropic_response,
    create_mock_anthropic_streaming_chunk
)


class TestAnthropicStreamingBasics:
    """Test basic streaming functionality with Anthropic."""
    
    def test_streaming_text_only(self, sync_anthropic_client, mock_anthropic_client):
        """Test streaming text response without tools."""
        # Create mock streaming chunks
        chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Hello")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text=" world")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="!")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        mock_anthropic_client.messages.create.return_value = iter(chunks)
        
        # Create client with full_response=False for text streaming
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True
        )
        
        content_parts = []
        for chunk in stream:
            if chunk:
                content_parts.append(chunk)
        
        full_content = "".join(content_parts)
        assert full_content == "Hello world!"

    def test_streaming_with_full_response_true(self, mock_anthropic_client):
        """Test streaming with full_response=True returns chunks as-is."""
        chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Chunk 1")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text=" Chunk 2"))
        ]
        
        mock_anthropic_client.messages.create.return_value = iter(chunks)
        
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=True)
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test"}],
            stream=True
        )
        
        # Should return the actual chunk objects
        chunk_list = list(stream)
        assert len(chunk_list) == 3  # Original chunks
        assert chunk_list[0].type == "message_start"
        assert chunk_list[1].type == "content_block_delta"

    def test_streaming_with_tools_basic(self, sync_anthropic_client, mock_anthropic_client):
        """Test streaming with simple tool usage."""
        # Mock streaming response with tool call - properly structured for streaming
        tool_call_start = Mock(type="tool_use", id="call_123", name="simple_math_tool")
        
        chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="I'll calculate that for you.")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("content_block_start", index=1, content_block=tool_call_start),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=1, delta=Mock(type="input_json_delta", partial_json='{"a": 5.0, "b": 3.0}')),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=1),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        # Follow-up streaming chunks after tool execution
        follow_up_chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="The result is 8.0")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        # First call: streaming with tool call, Second call: streaming follow-up
        mock_anthropic_client.messages.create.side_effect = [
            iter(chunks),
            iter(follow_up_chunks)  # Follow-up response should also be streaming
        ]
        
        # Create client with full_response=False for text streaming
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 5 + 3?"}],
            tools=[simple_math_tool],
            stream=True
        )
        
        content_parts = []
        for chunk in stream:
            if chunk:
                content_parts.append(chunk)
        
        full_content = "".join(content_parts)
        assert "I'll calculate that for you." in full_content
        assert "The result is 8.0" in full_content
        
        # Should have made two calls: streaming + follow-up
        assert mock_anthropic_client.messages.create.call_count == 2


class TestAnthropicStreamingWithTools:
    """Test streaming functionality with tool execution."""
    
    def test_streaming_multiple_tool_calls(self, sync_anthropic_client, mock_anthropic_client):
        """Test streaming with multiple tool calls."""
        tool_call_1 = Mock(type="tool_use", id="call_1", name="simple_math_tool")
        tool_call_2 = Mock(type="tool_use", id="call_2", name="calculator_tool")
        
        chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Let me calculate both operations.")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("content_block_start", index=1, content_block=tool_call_1),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=1, delta=Mock(type="input_json_delta", partial_json='{"a": 10.0, "b": 5.0}')),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=1),
            create_mock_anthropic_streaming_chunk("content_block_start", index=2, content_block=tool_call_2),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=2, delta=Mock(type="input_json_delta", partial_json='{"operation": "multiply", "a": 3.0, "b": 4.0}')),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=2),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        # Follow-up streaming chunks after tool execution
        follow_up_chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Addition result: 15.0, Multiplication result: 12.0")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        mock_anthropic_client.messages.create.side_effect = [
            iter(chunks),
            iter(follow_up_chunks)
        ]
        
        # Create client with full_response=False for text streaming
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Add 10+5 and multiply 3*4"}],
            tools=[simple_math_tool, calculator_tool],
            stream=True,
            parallel_tool_execution=True
        )
        
        content_parts = []
        for chunk in stream:
            if chunk:
                content_parts.append(chunk)
        
        full_content = "".join(content_parts)
        assert "Let me calculate both operations." in full_content
        assert "15.0" in full_content  # Addition result
        assert "12.0" in full_content  # Multiplication result

    def test_streaming_tool_error_handling(self, sync_anthropic_client, mock_anthropic_client):
        """Test error handling during streaming with tools."""
        # Create a tool call that will cause an error
        tool_call_error = Mock(type="tool_use", id="call_error", name="simple_math_tool")
        
        chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Processing your request.")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("content_block_start", index=1, content_block=tool_call_error),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=1, delta=Mock(type="input_json_delta", partial_json='{"a": 10.0, "b": 0.0}')),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=1),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        # Follow-up streaming chunks after tool execution
        follow_up_chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="The calculation completed successfully: 10.0")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        # Mock the streaming and follow-up response
        mock_anthropic_client.messages.create.side_effect = [
            iter(chunks),
            iter(follow_up_chunks)
        ]
        
        # Create client with full_response=False for text streaming
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Add 10 and 0"}],
            tools=[simple_math_tool],
            stream=True,
            graceful_error_handling=True
        )
        
        content_parts = []
        for chunk in stream:
            if chunk:
                content_parts.append(chunk)
        
        full_content = "".join(content_parts)
        assert "Processing your request." in full_content
        assert "10.0" in full_content


class TestAnthropicAsyncStreaming:
    """Test async streaming functionality with Anthropic."""
    
    @pytest.fixture
    def async_anthropic_client(self, mock_async_anthropic_client):
        """Create async toolflow Anthropic client."""
        return toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=True)
    
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_streaming_text_only(self, async_anthropic_client, mock_async_anthropic_client):
        """Test async streaming without tools."""
        chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Async")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text=" streaming")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text=" works!")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        async def async_iter_chunks():
            for chunk in chunks:
                yield chunk
        
        mock_async_anthropic_client.messages.create.return_value = async_iter_chunks()
        
        stream = await async_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test async streaming"}],
            stream=True
        )
        
        content_parts = []
        async for chunk in stream:
            if hasattr(chunk, 'type') and chunk.type == "content_block_delta":
                if hasattr(chunk.delta, 'text'):
                    content_parts.append(chunk.delta.text)
        
        full_content = "".join(content_parts)
        assert full_content == "Async streaming works!"

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio 
    async def test_async_streaming_with_tools(self, mock_async_anthropic_client):
        """Test async streaming with tool execution."""
        # Create async toolflow client
        async_client = toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=False)
        
        tool_call_mock = Mock(type="tool_use", id="call_async", name="weather_tool")
        
        chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Getting weather info.")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("content_block_start", index=1, content_block=tool_call_mock),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=1, delta=Mock(type="input_json_delta", partial_json='{"city": "Tokyo"}')),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=1),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        # Follow-up streaming chunks after tool execution
        follow_up_chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Weather in Tokyo: Sunny, 72°F with light winds")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        async def async_iter_chunks():
            for chunk in chunks:
                yield chunk
        
        async def async_iter_follow_up():
            for chunk in follow_up_chunks:
                yield chunk
        
        # Mock streaming response and follow-up
        mock_async_anthropic_client.messages.create.side_effect = [
            async_iter_chunks(),
            async_iter_follow_up()
        ]
        
        stream = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[weather_tool],
            stream=True
        )
        
        content_parts = []
        async for chunk in stream:
            if chunk:
                content_parts.append(chunk)
        
        full_content = "".join(content_parts)
        assert "Getting weather info." in full_content
        assert "Weather in Tokyo" in full_content


class TestAnthropicStreamingErrorHandling:
    """Test error handling in streaming contexts."""
    
    def test_streaming_network_interruption(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of network interruption during streaming."""
        def failing_stream():
            yield create_mock_anthropic_streaming_chunk("message_start", message=Mock())
            yield create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text"))
            yield create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Starting"))
            raise ConnectionError("Network interrupted")
        
        mock_anthropic_client.messages.create.return_value = failing_stream()
        
        # Create client with full_response=False for text streaming
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test interruption"}],
            stream=True
        )
        
        content_parts = []
        with pytest.raises(ConnectionError):
            for chunk in stream:
                if chunk:
                    content_parts.append(chunk)
        
        # Should have processed at least the first chunk
        assert len(content_parts) >= 1
        assert "Starting" in "".join(content_parts)

    def test_streaming_malformed_chunks(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of malformed streaming chunks."""
        def malformed_stream():
            yield create_mock_anthropic_streaming_chunk("message_start", message=Mock())
            
            # Malformed chunk without required attributes
            malformed_chunk = Mock()
            malformed_chunk.type = "content_block_delta"
            # Missing index and delta attributes
            yield malformed_chunk
            
            yield create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Recovery"))
            yield create_mock_anthropic_streaming_chunk("message_stop")
        
        mock_anthropic_client.messages.create.return_value = malformed_stream()
        
        # Create client with full_response=False for text streaming
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test malformed chunks"}],
            stream=True
        )
        
        content_parts = []
        # Should handle malformed chunks gracefully
        for chunk in stream:
            if chunk:
                content_parts.append(chunk)
        
        full_content = "".join(content_parts)
        assert "Recovery" in full_content


class TestAnthropicStreamingPerformance:
    """Test performance characteristics of streaming."""
    
    @pytest.mark.skipif(True, reason="Performance test - run manually if needed")
    def test_streaming_vs_non_streaming_comparison(self, sync_anthropic_client, mock_anthropic_client):
        """Compare streaming vs non-streaming response times."""
        import time
        
        # Mock responses
        mock_response = create_mock_anthropic_response(content="Non-streaming response")
        
        chunks = [
            create_mock_anthropic_streaming_chunk("message_start", message=Mock()),
            create_mock_anthropic_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text="Streaming")),
            create_mock_anthropic_streaming_chunk("content_block_delta", index=0, delta=Mock(type="text_delta", text=" response")),
            create_mock_anthropic_streaming_chunk("content_block_stop", index=0),
            create_mock_anthropic_streaming_chunk("message_stop")
        ]
        
        # Test non-streaming
        mock_anthropic_client.messages.create.return_value = mock_response
        start_time = time.time()
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test performance"}],
            stream=False
        )
        
        non_streaming_time = time.time() - start_time
        
        # Test streaming
        mock_anthropic_client.messages.create.return_value = iter(chunks)
        start_time = time.time()
        
        stream = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022", 
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test performance"}],
            stream=True
        )
        
        content_parts = []
        for chunk in stream:
            if chunk:
                content_parts.append(chunk)
        
        streaming_time = time.time() - start_time
        
        print(f"Non-streaming time: {non_streaming_time:.4f}s")
        print(f"Streaming time: {streaming_time:.4f}s")
        
        # Both should produce valid responses
        assert response.content[0].text == "Non-streaming response"
        assert "".join(content_parts) == "Streaming response" 