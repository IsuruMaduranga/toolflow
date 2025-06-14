"""
Test core functionality of the toolflow library.

This module tests:
- @tool decorator functionality
- Basic tool execution
- Import functionality  
- Core integration with OpenAI
"""
import pytest
import os
import openai
from unittest.mock import Mock
from toolflow import tool, from_openai
from ..conftest import (
    simple_math_tool, 
    divide_tool, 
    get_current_time_tool,
    create_mock_openai_tool_call as create_mock_tool_call,
    create_mock_openai_response as create_mock_response,
    create_mock_openai_streaming_chunk as create_mock_streaming_chunk
)


class TestToolDecorator:
    """Test the @tool decorator functionality."""
    
    def test_decorator_preserves_function_behavior(self):
        """Test that the @tool decorator doesn't break basic function behavior."""
        result = simple_math_tool(10.0, 2.0)
        assert result == 12.0
        
        result = divide_tool(10.0, 2.0)
        assert result == 5.0
    
    def test_decorator_without_parentheses(self):
        """Test @tool decorator used without parentheses."""
        @tool
        def simple_tool(x: int) -> int:
            """A simple tool."""
            return x * 2
        
        assert simple_tool(5) == 10
        assert hasattr(simple_tool, '_tool_metadata')
    
    def test_decorator_with_parentheses(self):
        """Test @tool decorator used with parentheses."""
        @tool()
        def simple_tool_with_parens(x: int) -> int:
            """A simple tool with parentheses."""
            return x * 3
        
        assert simple_tool_with_parens(5) == 15
        assert hasattr(simple_tool_with_parens, '_tool_metadata')
    
    def test_decorator_with_custom_params(self):
        """Test @tool decorator with custom name and description."""
        @tool(name="custom_name", description="Custom description")
        def tool_with_custom_params(x: int) -> int:
            """Original description."""
            return x
        
        assert tool_with_custom_params(42) == 42
        metadata = tool_with_custom_params._tool_metadata
        assert metadata['function']['name'] == 'custom_name'
        assert metadata['function']['description'] == 'Custom description'
    
    def test_decorator_preserves_function_attributes(self):
        """Test that decorator preserves original function attributes."""
        def original_func(x: int) -> int:
            """Original docstring."""
            return x
        
        # Add custom attribute
        original_func.custom_attr = "test"
        
        # Apply decorator
        decorated_func = tool(original_func)
        
        # Check preserved attributes
        assert decorated_func.__name__ == 'original_func'
        assert decorated_func.__doc__ == 'Original docstring.'
        assert hasattr(decorated_func, '_tool_metadata')
    
    def test_async_tool_decorator(self):
        """Test @tool decorator with async functions."""
        @tool
        async def async_tool_func(x: int) -> int:
            """An async tool."""
            return x * 2
        
        # Check metadata is attached
        assert hasattr(async_tool_func, '_tool_metadata')
        
        # Test execution
        import asyncio
        async def test_async():
            result = await async_tool_func(5)
            assert result == 10
        
        asyncio.run(test_async())


class TestBasicToolExecution:
    """Test basic tool execution functionality."""
    
    def test_simple_math_operations(self):
        """Test basic math tools work correctly."""
        assert simple_math_tool(5, 3) == 8
        assert simple_math_tool(10.5, 2.5) == 13.0
        assert divide_tool(15, 3) == 5.0
        assert divide_tool(7, 2) == 3.5
    
    def test_division_by_zero_error(self):
        """Test that division by zero raises appropriate error."""
        with pytest.raises(ZeroDivisionError):
            divide_tool(10, 0)
    
    def test_time_tool_returns_string(self):
        """Test that time tool returns string representation."""
        result = get_current_time_tool()
        assert isinstance(result, str)
        assert len(result) > 10  # Basic sanity check


class TestImportFunctionality:
    """Test that imports work correctly."""
    
    def test_basic_imports(self):
        """Test that core imports work."""
        from toolflow import tool, from_openai
        
        # Basic smoke test - if we get here, imports worked
        assert tool is not None
        assert from_openai is not None
    
    def test_async_imports(self):
        """Test that async imports work."""
        from toolflow import from_openai_async
        
        assert from_openai_async is not None
    
    def test_utils_imports(self):
        """Test that utility imports work."""
        from toolflow.utils import get_tool_schema
        
        assert get_tool_schema is not None


class TestOpenAIIntegration:
    """Test integration with OpenAI API."""
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_real_openai_integration(self):
        """Test integration with real OpenAI API (requires API key)."""
        client = from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 10 divided by 2?"}],
            tools=[divide_tool],
            max_tool_calls=5,
        )
        
        assert response is not None
    
    def test_mock_openai_integration(self, sync_toolflow_client, mock_openai_client):
        """Test basic integration with mocked OpenAI client."""
        # Mock a simple interaction without tool calls
        mock_response = create_mock_response(content="Hello, world!")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.choices[0].message.content == "Hello, world!"
        mock_openai_client.chat.completions.create.assert_called_once()
    
    def test_tool_execution_flow(self, sync_toolflow_client, mock_openai_client):
        """Test the complete tool execution flow with mocked client."""
        # First response - model wants to use tool
        tool_call = create_mock_tool_call("call_123", "divide_tool", {"a": 10, "b": 2})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        # Second response - model responds with result
        mock_response_2 = create_mock_response(content="The result is 5.0")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 10 divided by 2?"}],
            tools=[divide_tool]
        )
        
        assert response.choices[0].message.content == "The result is 5.0"
        assert mock_openai_client.chat.completions.create.call_count == 2


class TestFullResponseParameter:
    """Test the full_response parameter functionality."""
    
    def test_full_response_true(self, mock_openai_client):
        """Test that full_response=True returns the complete response object."""
        client = from_openai(mock_openai_client, full_response=True)
        
        mock_response = create_mock_response(content="Test content")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return the full response object
        assert hasattr(response, 'choices')
        assert response.choices[0].message.content == "Test content"
    
    def test_full_response_false(self, mock_openai_client):
        """Test that full_response=False returns only the content."""
        client = from_openai(mock_openai_client, full_response=False)
        
        mock_response = create_mock_response(content="Test content")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return only the content string
        assert response == "Test content"
        assert isinstance(response, str)
    
    def test_method_level_override(self, mock_openai_client):
        """Test that full_response can be overridden at the method level."""
        client = from_openai(mock_openai_client, full_response=False)
        
        mock_response = create_mock_response(content="Test content")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Override client-level setting
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            full_response=True  # Override the client setting
        )
        
        # Should return the full response object due to override
        assert hasattr(response, 'choices')
        assert response.choices[0].message.content == "Test content"
    
    def test_create_method_with_response_format_full_response_false(self, mock_openai_client):
        """Test that create() method with response_format respects full_response=False for simple responses."""
        client = from_openai(mock_openai_client, full_response=False)
        
        # Simple case without tools - should just return content
        mock_response = create_mock_response(content="Simple response")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return only the content string (no tools involved)
        assert response == "Simple response"
        assert isinstance(response, str)
    
    def test_beta_full_response_parameter(self, mock_openai_client):
        """Test that full_response parameter works with beta API."""
        client = from_openai(mock_openai_client, full_response=False)
        
        mock_response = create_mock_response(content="Beta test content")
        mock_openai_client.beta.chat.completions.parse.return_value = mock_response
        
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return only the content string
        assert response == "Beta test content"
        assert isinstance(response, str)


class TestStreamingFullResponse:
    """Test streaming with full_response parameter."""
    
    def test_streaming_full_response_false(self, mock_openai_client):
        """Test that streaming with full_response=False yields content only."""
        client = from_openai(mock_openai_client, full_response=False)
        
        # Mock streaming chunks
        chunk1 = create_mock_streaming_chunk(content="Hello")
        chunk2 = create_mock_streaming_chunk(content=" world")
        chunk3 = create_mock_streaming_chunk(content="!")
        
        mock_openai_client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        content_pieces = list(stream)
        
        # Should yield content strings directly
        assert content_pieces == ["Hello", " world", "!"]
        for piece in content_pieces:
            assert isinstance(piece, str)
    
    def test_streaming_full_response_true(self, mock_openai_client):
        """Test that streaming with full_response=True yields full chunks."""
        client = from_openai(mock_openai_client, full_response=False)
        
        # Mock streaming chunks
        chunk1 = create_mock_streaming_chunk(content="Hello")
        chunk2 = create_mock_streaming_chunk(content=" world")
        chunk3 = create_mock_streaming_chunk(content="!")
        
        mock_openai_client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            full_response=True  # Override client setting
        )
        
        chunks = list(stream)
        
        # Should yield full chunk objects
        assert len(chunks) == 3
        for chunk in chunks:
            assert hasattr(chunk, 'choices')
            assert hasattr(chunk.choices[0], 'delta')
    
    def test_streaming_client_level_full_response_true(self, mock_openai_client):
        """Test streaming with client-level full_response=True."""
        client = from_openai(mock_openai_client, full_response=True)
        
        # Mock streaming chunks
        chunk1 = create_mock_streaming_chunk(content="Hello")
        chunk2 = create_mock_streaming_chunk(content=" world")
        
        mock_openai_client.chat.completions.create.return_value = [chunk1, chunk2]
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        chunks = list(stream)
        
        # Should yield full chunk objects
        assert len(chunks) == 2
        for chunk in chunks:
            assert hasattr(chunk, 'choices')
            assert hasattr(chunk.choices[0], 'delta')
    
    def test_streaming_method_level_override(self, mock_openai_client):
        """Test that method-level full_response parameter overrides client setting."""
        client = from_openai(mock_openai_client, full_response=True)
        
        # Mock streaming chunks
        chunk1 = create_mock_streaming_chunk(content="Test")
        chunk2 = create_mock_streaming_chunk(content=" content")
        
        mock_openai_client.chat.completions.create.return_value = [chunk1, chunk2]
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            full_response=False  # Override client setting
        )
        
        content_pieces = list(stream)
        
        # Should yield content strings due to method override
        assert content_pieces == ["Test", " content"]
        for piece in content_pieces:
            assert isinstance(piece, str)
 