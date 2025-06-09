import pytest
import asyncio
import json
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from toolflow import tool, from_openai_async
import datetime


@tool
def sync_get_current_time():
    """Get the current time (sync version)."""
    return str(datetime.datetime.now())


@tool
async def async_get_current_time():
    """Get the current time (async version)."""
    await asyncio.sleep(0.01)  # Simulate async work
    return str(datetime.datetime.now())


@tool
def sync_divide(a: float, b: float) -> float:
    """Divide two numbers (sync version)."""
    return a / b


@tool
async def async_divide(a: float, b: float) -> float:
    """Divide two numbers (async version)."""
    await asyncio.sleep(0.01)  # Simulate async work
    return a / b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
async def async_multiply(a: float, b: float) -> float:
    """Multiply two numbers (async version)."""
    await asyncio.sleep(0.01)  # Simulate async work
    return a * b


class TestAsyncToolDecorator:
    """Test the @tool decorator with async functions."""
    
    def test_sync_tool_decorator(self):
        """Test that the @tool decorator works properly with sync functions."""
        result = sync_divide(10.0, 2.0)
        assert result == 5.0
        
        time_result = sync_get_current_time()
        assert isinstance(time_result, str)
    
    @pytest.mark.asyncio
    async def test_async_tool_decorator(self):
        """Test that the @tool decorator works properly with async functions."""
        result = await async_divide(10.0, 2.0)
        assert result == 5.0
        
        time_result = await async_get_current_time()
        assert isinstance(time_result, str)


class TestAsyncClientMocked:
    """Test async client with mocked OpenAI responses."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock OpenAI async client
        self.mock_openai_client = Mock()
        self.mock_completions = AsyncMock()
        self.mock_openai_client.chat.completions = self.mock_completions
        
        # Create wrapped client
        self.client = from_openai_async(self.mock_openai_client)
    
    @pytest.mark.asyncio
    async def test_basic_async_call_no_tools(self):
        """Test basic async call without tools."""
        # Mock response without tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Hello, world!"
        
        self.mock_completions.create.return_value = mock_response
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.choices[0].message.content == "Hello, world!"
        self.mock_completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_call_with_sync_tool(self):
        """Test async call with sync tool execution."""
        # First call - model wants to use tool
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "sync_divide"
        mock_tool_call.function.arguments = '{"a": 10, "b": 2}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call]
        mock_response_1.choices[0].message.content = None
        
        # Second call - model responds with result
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "The result is 5.0"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 10 divided by 2?"}],
            tools=[sync_divide]
        )
        
        assert response.choices[0].message.content == "The result is 5.0"
        assert self.mock_completions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_call_with_async_tool(self):
        """Test async call with async tool execution."""
        # First call - model wants to use tool
        mock_tool_call = Mock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = "async_divide"
        mock_tool_call.function.arguments = '{"a": 15, "b": 3}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call]
        mock_response_1.choices[0].message.content = None
        
        # Second call - model responds with result
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "The result is 5.0"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 15 divided by 3?"}],
            tools=[async_divide]
        )
        
        assert response.choices[0].message.content == "The result is 5.0"
        assert self.mock_completions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_call_with_mixed_tools(self):
        """Test async call with both sync and async tools."""
        # First call - model wants to use sync tool
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_sync"
        mock_tool_call_1.function.name = "multiply"
        mock_tool_call_1.function.arguments = '{"a": 4, "b": 5}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1]
        mock_response_1.choices[0].message.content = None
        
        # Second call - model wants to use async tool
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_async"
        mock_tool_call_2.function.name = "async_divide"
        mock_tool_call_2.function.arguments = '{"a": 20, "b": 4}'
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = [mock_tool_call_2]
        mock_response_2.choices[0].message.content = None
        
        # Final response
        mock_response_3 = Mock()
        mock_response_3.choices = [Mock()]
        mock_response_3.choices[0].message.tool_calls = None
        mock_response_3.choices[0].message.content = "The calculations are complete"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2, mock_response_3]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 4*5 and 20/4"}],
            tools=[multiply, async_divide]
        )
        
        assert response.choices[0].message.content == "The calculations are complete"
        assert self.mock_completions.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_response(self):
        """Test handling multiple tool calls in a single response."""
        # Mock multiple tool calls
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "multiply"
        mock_tool_call_1.function.arguments = '{"a": 2, "b": 3}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function.name = "async_multiply"
        mock_tool_call_2.function.arguments = '{"a": 4, "b": 5}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response_1.choices[0].message.content = None
        
        # Final response
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Both calculations complete"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 2*3 and 4*5"}],
            tools=[multiply, async_multiply]
        )
        
        assert response.choices[0].message.content == "Both calculations complete"
        assert self.mock_completions.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_tool_calls_limit(self):
        """Test that max_tool_calls limit is respected."""
        # Mock tool call that keeps triggering
        mock_tool_call = Mock()
        mock_tool_call.id = "call_loop"
        mock_tool_call.function.name = "multiply"
        mock_tool_call.function.arguments = '{"a": 1, "b": 1}'
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].message.content = None
        
        # Always return the same response to create a loop
        self.mock_completions.create.return_value = mock_response
        
        with pytest.raises(Exception, match="Max tool calls reached"):
            await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test"}],
                tools=[multiply],
                max_tool_calls=2
            )
    
    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test error handling during tool execution."""
        # Mock tool call with invalid arguments
        mock_tool_call = Mock()
        mock_tool_call.id = "call_error"
        mock_tool_call.function.name = "sync_divide"
        mock_tool_call.function.arguments = '{"a": 10, "b": 0}'  # Division by zero
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].message.content = None
        
        self.mock_completions.create.return_value = mock_response
        
        with pytest.raises(Exception, match="Error executing tool sync_divide"):
            await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Divide by zero"}],
                tools=[sync_divide]
            )
    
    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test error handling for unknown tool calls."""
        # Mock tool call for non-existent tool
        mock_tool_call = Mock()
        mock_tool_call.id = "call_unknown"
        mock_tool_call.function.name = "non_existent_tool"
        mock_tool_call.function.arguments = '{}'
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].message.content = None
        
        self.mock_completions.create.return_value = mock_response
        
        with pytest.raises(ValueError, match="Tool non_existent_tool not found"):
            await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Use unknown tool"}],
                tools=[multiply]
            )
    
    @pytest.mark.asyncio
    async def test_invalid_tool_arguments(self):
        """Test error handling for invalid tool arguments."""
        # Mock tool call with malformed JSON
        mock_tool_call = Mock()
        mock_tool_call.id = "call_invalid"
        mock_tool_call.function.name = "multiply"
        mock_tool_call.function.arguments = '{"a": 10, "b":}'  # Invalid JSON
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].message.content = None
        
        self.mock_completions.create.return_value = mock_response
        
        with pytest.raises(Exception, match="Error executing tool multiply"):
            await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Multiply with bad args"}],
                tools=[multiply]
            )


class TestAsyncClientIntegration:
    """Integration tests with real OpenAI API (requires API key)."""
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_real_openai_async_integration(self):
        """Test integration with real OpenAI async API."""
        import openai
        
        client = from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 3.145 divided by 2? Use the tool."}],
            tools=[sync_divide],
            max_tool_calls=5,
        )
        
        assert response.choices[0].message.content is not None
        # Should contain the result of 3.145 / 2 = 1.5725
        assert "1.5725" in response.choices[0].message.content
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_real_openai_async_with_async_tool(self):
        """Test integration with real OpenAI async API using async tools."""
        import openai
        
        client = from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 8 multiplied by 7? Use the tool."}],
            tools=[async_multiply],
            max_tool_calls=5,
        )
        
        assert response.choices[0].message.content is not None
        # Should contain the result of 8 * 7 = 56
        assert "56" in response.choices[0].message.content


def test_import_async_functionality():
    """Test that the async toolflow imports work correctly."""
    from toolflow import from_openai_async
    
    # Basic smoke test - if we get here, imports worked
    assert from_openai_async is not None


class TestAsyncClientProperties:
    """Test that async client properly delegates properties."""
    
    def test_async_client_delegation(self):
        """Test that the async wrapper properly delegates attributes."""
        mock_openai_client = Mock()
        mock_openai_client.some_property = "test_value"
        mock_openai_client.some_method.return_value = "method_result"
        
        client = from_openai_async(mock_openai_client)
        
        # Test property delegation
        assert client.some_property == "test_value"
        
        # Test method delegation
        result = client.some_method()
        assert result == "method_result"
        mock_openai_client.some_method.assert_called_once() 