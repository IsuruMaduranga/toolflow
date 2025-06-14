"""
Test async functionality of the toolflow library.

This module tests:
- Async tool execution
- Mixed sync/async tool handling
- Async client behavior
- Error handling in async context
"""
import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock
from toolflow import tool, from_openai_async
from ..conftest import (
    async_math_tool,
    simple_math_tool,
    slow_async_tool,
    failing_tool,
    create_mock_openai_tool_call as create_mock_tool_call,
    create_mock_openai_response as create_mock_response,
    create_mock_openai_streaming_chunk as create_mock_streaming_chunk
)


class TestAsyncToolDecorator:
    """Test @tool decorator with async functions."""
    
    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test that async tools execute correctly."""
        result = await async_math_tool(10.0, 2.0)
        assert result == 12.0
    
    @pytest.mark.asyncio
    async def test_async_tool_with_delay(self):
        """Test async tool that includes delay."""
        result = await slow_async_tool("test", 0.001)
        assert result == "Async result from test after 0.001s"
        assert isinstance(result, str)


class TestAsyncClientBasic:
    """Test basic async client functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_async_call_no_tools(self, async_toolflow_client, mock_async_openai_client):
        """Test basic async call without tools."""
        mock_response = create_mock_response(content="Hello, async world!")
        mock_async_openai_client.chat.completions.create.return_value = mock_response
        
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.choices[0].message.content == "Hello, async world!"
        mock_async_openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_client_delegation(self, mock_async_openai_client):
        """Test that async client properly delegates to OpenAI client."""
        client = from_openai_async(mock_async_openai_client)
        
        # Test that the wrapped client maintains reference to original
        assert client._client is mock_async_openai_client
        assert hasattr(client.chat, 'completions')


class TestAsyncToolExecution:
    """Test async tool execution with mocked responses."""
    
    @pytest.mark.asyncio
    async def test_async_call_with_sync_tool(self, async_toolflow_client, mock_async_openai_client):
        """Test async client executing sync tools."""
        # First call - model wants to use tool
        tool_call = create_mock_tool_call("call_123", "simple_math_tool", {"a": 10, "b": 2})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        # Second call - model responds with result
        mock_response_2 = create_mock_response(content="The result is 12.0")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 10 + 2?"}],
            tools=[simple_math_tool]
        )
        
        assert response.choices[0].message.content == "The result is 12.0"
        assert mock_async_openai_client.chat.completions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_call_with_async_tool(self, async_toolflow_client, mock_async_openai_client):
        """Test async client executing async tools."""
        # First call - model wants to use tool
        tool_call = create_mock_tool_call("call_456", "async_math_tool", {"a": 15, "b": 3})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        # Second call - model responds with result
        mock_response_2 = create_mock_response(content="The result is 18.0")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 15 + 3?"}],
            tools=[async_math_tool]
        )
        
        assert response.choices[0].message.content == "The result is 18.0"
        assert mock_async_openai_client.chat.completions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_call_with_mixed_tools(self, async_toolflow_client, mock_async_openai_client):
        """Test async client with both sync and async tools."""
        # First call - model wants to use sync tool
        tool_call_1 = create_mock_tool_call("call_sync", "simple_math_tool", {"a": 4, "b": 5})
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1])
        
        # Second call - model wants to use async tool
        tool_call_2 = create_mock_tool_call("call_async", "async_math_tool", {"a": 20, "b": 4})
        mock_response_2 = create_mock_response(tool_calls=[tool_call_2])
        
        # Final response
        mock_response_3 = create_mock_response(content="The calculations are complete")
        
        mock_async_openai_client.chat.completions.create.side_effect = [
            mock_response_1, mock_response_2, mock_response_3
        ]
        
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 4+5 and 20+4"}],
            tools=[simple_math_tool, async_math_tool]
        )
        
        assert response.choices[0].message.content == "The calculations are complete"
        assert mock_async_openai_client.chat.completions.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_response(self, async_toolflow_client, mock_async_openai_client):
        """Test handling multiple tool calls in a single response."""
        # Create multiple tool calls
        tool_call_1 = create_mock_tool_call("call_1", "simple_math_tool", {"a": 1, "b": 2})
        tool_call_2 = create_mock_tool_call("call_2", "async_math_tool", {"a": 3, "b": 4})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Both calculations complete")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Do multiple calculations"}],
            tools=[simple_math_tool, async_math_tool]
        )
        
        assert response.choices[0].message.content == "Both calculations complete"
        assert mock_async_openai_client.chat.completions.create.call_count == 2


class TestAsyncErrorHandling:
    """Test error handling in async context."""
    
    @pytest.mark.asyncio
    async def test_async_tool_execution_error(self, async_toolflow_client, mock_async_openai_client):
        """Test handling of tool execution errors in async context with graceful error handling."""
        # Tool call that will fail
        tool_call = create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Async tool error was handled gracefully")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test error handling"}],
            tools=[failing_tool]
        )
        
        # Should get response after error is handled gracefully
        assert response.choices[0].message.content == "Async tool error was handled gracefully"
        
        # Check that error was passed to the model in second call
        second_call_args = mock_async_openai_client.chat.completions.create.call_args_list[1]
        tool_messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in tool_messages if msg.get('role') == 'tool']
        
        assert len(tool_result_messages) == 1
        assert "Error executing tool failing_tool" in tool_result_messages[0]['content']
        assert "This tool failed intentionally" in tool_result_messages[0]['content']
    
    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, async_toolflow_client, mock_async_openai_client):
        """Test handling of unknown tool calls."""
        # Tool call for non-existent tool
        tool_call = create_mock_tool_call("call_unknown", "nonexistent_tool", {})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when tool is not found
        with pytest.raises(ValueError) as exc_info:
            await async_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test unknown tool"}],
                tools=[simple_math_tool]  # Available tool, but call is for different one
            )
        
        # Verify the exception contains tool not found error
        assert "Tool nonexistent_tool not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_invalid_tool_arguments(self, async_toolflow_client, mock_async_openai_client):
        """Test handling of invalid tool arguments with graceful error handling."""
        # Tool call with invalid JSON arguments
        tool_call = Mock()
        tool_call.id = "call_invalid"
        tool_call.function.name = "simple_math_tool"
        tool_call.function.arguments = '{"a": "not_a_number", "b": 5}'  # Invalid argument type
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Async type error was handled gracefully")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test invalid args"}],
            tools=[simple_math_tool]
        )
        
        # Should get response after error is handled gracefully
        assert response.choices[0].message.content == "Async type error was handled gracefully"
        
        # Check that error was passed to the model in second call
        second_call_args = mock_async_openai_client.chat.completions.create.call_args_list[1]
        tool_messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in tool_messages if msg.get('role') == 'tool']
        
        assert len(tool_result_messages) == 1
        assert "Error executing tool simple_math_tool" in tool_result_messages[0]['content']
        assert "can only concatenate str" in tool_result_messages[0]['content']


class TestAsyncLimits:
    """Test async functionality with limits and constraints."""
    
    @pytest.mark.asyncio
    async def test_max_tool_calls_limit(self, async_toolflow_client, mock_async_openai_client):
        """Test max_tool_calls limit in async context."""
        # Create a scenario that would exceed max_tool_calls if not limited
        tool_call = create_mock_tool_call("call_1", "simple_math_tool", {"a": 1, "b": 1})
        
        # Mock response that keeps requesting tools
        mock_response_with_tools = create_mock_response(tool_calls=[tool_call])
        
        # Set up to return tool calls that will exceed the limit
        mock_async_openai_client.chat.completions.create.side_effect = [
            mock_response_with_tools,  # First tool call executes
            mock_response_with_tools   # Second attempt should hit limit
        ]
        
        # Should raise an exception when max tool calls is exceeded
        with pytest.raises(Exception) as exc_info:
            await async_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test limits"}],
                tools=[simple_math_tool],
                max_tool_calls=1  # Low limit to test
            )
        
        # Verify the exception indicates max tool calls reached
        assert "Max tool calls reached without finding a solution" in str(exc_info.value)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestAsyncRealIntegration:
    """Test async functionality with real OpenAI API."""
    
    @pytest.mark.asyncio
    async def test_real_openai_async_integration(self):
        """Test real async OpenAI integration (requires API key)."""
        import openai
        
        client = from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 10 + 5?"}],
            tools=[simple_math_tool],
            max_tool_calls=5
        )
        
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_real_openai_async_with_async_tool(self):
        """Test real async OpenAI integration with async tools."""
        import openai
        
        client = from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 7 + 8?"}],
            tools=[async_math_tool],
            max_tool_calls=5
        )
        
        assert response is not None


class TestAsyncFullResponseParameter:
    """Test the full_response parameter functionality in async context."""
    
    @pytest.mark.asyncio
    async def test_async_full_response_true(self, mock_async_openai_client):
        """Test that full_response=True returns the complete response object in async context."""
        from toolflow import from_openai_async
        client = from_openai_async(mock_async_openai_client, full_response=True)
        
        mock_response = create_mock_response(content="Async test content")
        mock_async_openai_client.chat.completions.create.return_value = mock_response
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return the full response object
        assert hasattr(response, 'choices')
        assert response.choices[0].message.content == "Async test content"
    
    @pytest.mark.asyncio
    async def test_async_full_response_false(self, mock_async_openai_client):
        """Test that full_response=False returns only the content in async context."""
        from toolflow import from_openai_async
        client = from_openai_async(mock_async_openai_client, full_response=False)
        
        mock_response = create_mock_response(content="Async test content")
        mock_async_openai_client.chat.completions.create.return_value = mock_response
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return only the content string
        assert response == "Async test content"
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_async_method_level_override(self, mock_async_openai_client):
        """Test that full_response can be overridden at the method level in async context."""
        from toolflow import from_openai_async
        client = from_openai_async(mock_async_openai_client, full_response=False)
        
        mock_response = create_mock_response(content="Async test content")
        mock_async_openai_client.chat.completions.create.return_value = mock_response
        
        # Override client-level setting
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            full_response=True  # Override the client setting
        )
        
        # Should return the full response object due to override
        assert hasattr(response, 'choices')
        assert response.choices[0].message.content == "Async test content"
    
    @pytest.mark.asyncio
    async def test_async_beta_full_response_parameter(self, mock_async_openai_client):
        """Test that full_response parameter works with beta API in async context."""
        from toolflow import from_openai_async
        
        # Set up beta mock properly for async
        mock_async_openai_client.beta = Mock()
        mock_async_openai_client.beta.chat = Mock()
        mock_async_openai_client.beta.chat.completions = AsyncMock()
        
        client = from_openai_async(mock_async_openai_client, full_response=False)
        
        mock_response = create_mock_response(content="Async beta test content")
        mock_async_openai_client.beta.chat.completions.parse.return_value = mock_response
        
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return only the content string
        assert response == "Async beta test content"
        assert isinstance(response, str)


class TestAsyncStreamingFullResponse:
    """Test async streaming with full_response parameter."""
    
    @pytest.mark.asyncio
    async def test_async_streaming_full_response_false(self, mock_async_openai_client):
        """Test that async streaming with full_response=False yields content only."""
        from toolflow import from_openai_async
        client = from_openai_async(mock_async_openai_client, full_response=False)
        
        # Mock streaming chunks
        chunk1 = create_mock_streaming_chunk(content="Hello")
        chunk2 = create_mock_streaming_chunk(content=" async")
        chunk3 = create_mock_streaming_chunk(content=" world")
        
        async def mock_async_stream():
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk
        
        mock_async_openai_client.chat.completions.create.return_value = mock_async_stream()
        
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        content_pieces = []
        async for content in stream:
            content_pieces.append(content)
        
        # Should yield content strings directly
        assert content_pieces == ["Hello", " async", " world"]
        for piece in content_pieces:
            assert isinstance(piece, str)
    
    @pytest.mark.asyncio
    async def test_async_streaming_full_response_true(self, mock_async_openai_client):
        """Test that async streaming with full_response=True yields full chunks."""
        from toolflow import from_openai_async
        client = from_openai_async(mock_async_openai_client, full_response=False)
        
        # Mock streaming chunks
        chunk1 = create_mock_streaming_chunk(content="Hello")
        chunk2 = create_mock_streaming_chunk(content=" async")
        chunk3 = create_mock_streaming_chunk(content=" chunks")
        
        async def mock_async_stream():
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk
        
        mock_async_openai_client.chat.completions.create.return_value = mock_async_stream()
        
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            full_response=True  # Override client setting
        )
        
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Should yield full chunk objects
        assert len(chunks) == 3
        for chunk in chunks:
            assert hasattr(chunk, 'choices')
            assert hasattr(chunk.choices[0], 'delta')
    
    @pytest.mark.asyncio
    async def test_async_streaming_client_level_full_response_true(self, mock_async_openai_client):
        """Test async streaming with client-level full_response=True."""
        from toolflow import from_openai_async
        client = from_openai_async(mock_async_openai_client, full_response=True)
        
        # Mock streaming chunks
        chunk1 = create_mock_streaming_chunk(content="Hello")
        chunk2 = create_mock_streaming_chunk(content=" client")
        
        async def mock_async_stream():
            for chunk in [chunk1, chunk2]:
                yield chunk
        
        mock_async_openai_client.chat.completions.create.return_value = mock_async_stream()
        
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Should yield full chunk objects
        assert len(chunks) == 2
        for chunk in chunks:
            assert hasattr(chunk, 'choices')
            assert hasattr(chunk.choices[0], 'delta')
    
    @pytest.mark.asyncio
    async def test_async_streaming_method_level_override(self, mock_async_openai_client):
        """Test that method-level full_response parameter overrides client setting in async."""
        from toolflow import from_openai_async
        client = from_openai_async(mock_async_openai_client, full_response=True)
        
        # Mock streaming chunks
        chunk1 = create_mock_streaming_chunk(content="Test")
        chunk2 = create_mock_streaming_chunk(content=" override")
        
        async def mock_async_stream():
            for chunk in [chunk1, chunk2]:
                yield chunk
        
        mock_async_openai_client.chat.completions.create.return_value = mock_async_stream()
        
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            full_response=False  # Override client setting
        )
        
        content_pieces = []
        async for content in stream:
            content_pieces.append(content)
        
        # Should yield content strings due to method override
        assert content_pieces == ["Test", " override"]
        for piece in content_pieces:
            assert isinstance(piece, str)
