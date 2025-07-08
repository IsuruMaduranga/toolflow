"""
Unit tests for Gemini wrapper functionality.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import asyncio
from typing import Generator

from toolflow.providers.gemini.wrappers import GeminiWrapper
from toolflow.providers.gemini import from_gemini
from toolflow import tool

# Test fixtures
def create_gemini_response(text: str = None, function_calls: list = None):
    """Create a mock Gemini response."""
    mock_response = Mock()
    mock_response.text = text or ""
    mock_response.candidates = []
    
    if text or function_calls:
        candidate = Mock()
        candidate.content = Mock()
        candidate.content.parts = []
        
        if text:
            text_part = Mock()
            text_part.text = text
            if hasattr(text_part, 'function_call'):
                delattr(text_part, 'function_call')
            candidate.content.parts.append(text_part)
        
        if function_calls:
            for func_call in function_calls:
                func_part = Mock()
                func_part.function_call = Mock()
                func_part.function_call.name = func_call["name"]
                func_part.function_call.args = func_call["args"]
                if hasattr(func_part, 'text'):
                    delattr(func_part, 'text')
                candidate.content.parts.append(func_part)
        
        candidate.finish_reason = "STOP"
        mock_response.candidates.append(candidate)
    
    return mock_response

@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client."""
    mock = Mock()
    mock.generate_content = Mock()
    return mock

@pytest.fixture
def wrapper(mock_gemini_client):
    """Create GeminiWrapper instance."""
    return GeminiWrapper(mock_gemini_client, full_response=False)

@pytest.fixture
def wrapper_full_response(mock_gemini_client):
    """Create GeminiWrapper instance with full_response=True."""
    return GeminiWrapper(mock_gemini_client, full_response=True)

# Test tools
@tool
def add_tool(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def greet_tool(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

@tool
def error_tool() -> str:
    """Tool that raises an error."""
    raise ValueError("Tool error")

class TestGeminiWrapperInitialization:
    """Test GeminiWrapper initialization and configuration."""
    
    def test_wrapper_creation(self, mock_gemini_client):
        """Test wrapper is created correctly."""
        wrapper = GeminiWrapper(mock_gemini_client, full_response=False)
        
        assert wrapper._client == mock_gemini_client
        assert wrapper.full_response == False
        assert wrapper.original_generate_content == mock_gemini_client.generate_content
        assert hasattr(wrapper, 'handler')
    
    def test_wrapper_full_response_mode(self, mock_gemini_client):
        """Test wrapper with full_response=True."""
        wrapper = GeminiWrapper(mock_gemini_client, full_response=True)
        
        assert wrapper.full_response == True
    
    def test_from_gemini_function(self, mock_gemini_client):
        """Test from_gemini function creates wrapper correctly."""
        wrapper = from_gemini(mock_gemini_client)
        
        assert isinstance(wrapper, GeminiWrapper)
        assert wrapper._client == mock_gemini_client
    
    def test_from_gemini_with_options(self, mock_gemini_client):
        """Test from_gemini with options."""
        wrapper = from_gemini(mock_gemini_client, full_response=True)
        
        assert wrapper.full_response == True
    
    def test_wrapper_attribute_delegation(self, mock_gemini_client, wrapper):
        """Test that wrapper delegates attributes to client."""
        mock_gemini_client.some_attribute = "test_value"
        
        assert wrapper.some_attribute == "test_value"

class TestBasicContentGeneration:
    """Test basic content generation without tools."""
    
    def test_simple_text_generation(self, wrapper, mock_gemini_client):
        """Test simple text generation."""
        mock_response = create_gemini_response(text="Hello world")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content("Say hello")
        
        assert response == "Hello world"
        mock_gemini_client.generate_content.assert_called_once()
    
    def test_text_generation_full_response(self, wrapper_full_response, mock_gemini_client):
        """Test text generation with full_response=True."""
        mock_response = create_gemini_response(text="Hello world")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper_full_response.generate_content("Say hello")
        
        assert response == mock_response  # Should return full response object
    
    def test_text_generation_with_params(self, wrapper, mock_gemini_client):
        """Test text generation with additional parameters."""
        mock_response = create_gemini_response(text="Response")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content(
            "Test prompt",
            generation_config={"temperature": 0.7},
            safety_settings=[]
        )
        
        assert response == "Response"
        
        # Check that parameters were passed through
        call_args = mock_gemini_client.generate_content.call_args[1]
        assert "generation_config" in call_args
        assert call_args["generation_config"]["temperature"] == 0.7
    
    def test_empty_response(self, wrapper, mock_gemini_client):
        """Test handling of empty response."""
        mock_response = create_gemini_response(text="")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content("Say something")
        
        assert response == ""

class TestToolCalling:
    """Test tool calling functionality."""
    
    def test_single_tool_call(self, wrapper, mock_gemini_client):
        """Test calling a single tool."""
        # First response: model calls tool
        function_calls = [{"name": "add_tool", "args": {"a": 5, "b": 3}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        
        # Second response: model responds with result
        mock_response_2 = create_gemini_response(text="The sum is 8")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = wrapper.generate_content(
            "What is 5 + 3?",
            tools=[add_tool]
        )
        
        assert response == "The sum is 8"
        assert mock_gemini_client.generate_content.call_count == 2
    
    def test_multiple_tool_calls(self, wrapper, mock_gemini_client):
        """Test multiple tool calls in sequence."""
        # First response: model calls multiple tools
        function_calls = [
            {"name": "add_tool", "args": {"a": 5, "b": 3}},
            {"name": "greet_tool", "args": {"name": "Alice"}}
        ]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        
        # Second response: model responds with results
        mock_response_2 = create_gemini_response(text="Results: 8 and Hello, Alice!")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = wrapper.generate_content(
            "Add 5+3 and greet Alice",
            tools=[add_tool, greet_tool]
        )
        
        assert response == "Results: 8 and Hello, Alice!"
        assert mock_gemini_client.generate_content.call_count == 2
    
    def test_tool_call_with_text(self, wrapper, mock_gemini_client):
        """Test tool call with accompanying text."""
        # Response with both text and tool call
        function_calls = [{"name": "add_tool", "args": {"a": 2, "b": 2}}]
        mock_response_1 = create_gemini_response(
            text="Let me calculate that for you",
            function_calls=function_calls
        )
        
        # Final response
        mock_response_2 = create_gemini_response(text="The result is 4")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = wrapper.generate_content(
            "What is 2 + 2?",
            tools=[add_tool]
        )
        
        assert response == "The result is 4"
    
    def test_no_tool_calls_needed(self, wrapper, mock_gemini_client):
        """Test when model doesn't need to call tools."""
        mock_response = create_gemini_response(text="I can answer without tools")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content(
            "What's your name?",
            tools=[add_tool]
        )
        
        assert response == "I can answer without tools"
        assert mock_gemini_client.generate_content.call_count == 1
    
    def test_max_tool_call_rounds(self, wrapper, mock_gemini_client):
        """Test max tool call rounds limit."""
        from toolflow.core.exceptions import MaxToolCallsError
        
        # Always return tool calls to exceed limit
        function_calls = [{"name": "add_tool", "args": {"a": 1, "b": 1}}]
        mock_response = create_gemini_response(function_calls=function_calls)
        mock_gemini_client.generate_content.return_value = mock_response
        
        with pytest.raises(MaxToolCallsError):
            wrapper.generate_content(
                "Keep adding",
                tools=[add_tool],
                max_tool_call_rounds=2
            )
    
    def test_tool_error_handling(self, wrapper, mock_gemini_client):
        """Test tool error handling."""
        # First response: model calls error tool
        function_calls = [{"name": "error_tool", "args": {}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        
        # Second response: model handles error
        mock_response_2 = create_gemini_response(text="Tool failed but I handled it")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = wrapper.generate_content(
            "Try the error tool",
            tools=[error_tool],
            graceful_error_handling=True
        )
        
        assert response == "Tool failed but I handled it"
    
    def test_parallel_tool_execution(self, wrapper, mock_gemini_client):
        """Test parallel tool execution."""
        function_calls = [
            {"name": "add_tool", "args": {"a": 1, "b": 1}},
            {"name": "add_tool", "args": {"a": 2, "b": 2}}
        ]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(text="Both calculations done")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = wrapper.generate_content(
            "Calculate 1+1 and 2+2",
            tools=[add_tool],
            parallel_tool_execution=True
        )
        
        assert response == "Both calculations done"

class TestStreamingGeneration:
    """Test streaming content generation."""
    
    def test_basic_streaming(self, wrapper, mock_gemini_client):
        """Test basic streaming without tools."""
        def mock_stream():
            yield create_gemini_response(text="Hello ")
            yield create_gemini_response(text="streaming ")
            yield create_gemini_response(text="world")
        
        mock_gemini_client.generate_content.return_value = mock_stream()
        
        response = wrapper.generate_content("Stream hello", stream=True)
        
        # Response should be a generator
        assert hasattr(response, '__iter__')
        
        # Collect streamed content
        content_parts = list(response)
        assert "Hello " in content_parts
        assert "streaming " in content_parts
        assert "world" in content_parts
    
    def test_streaming_full_response(self, wrapper_full_response, mock_gemini_client):
        """Test streaming with full_response=True."""
        def mock_stream():
            yield create_gemini_response(text="Hello")
            yield create_gemini_response(text=" world")
        
        mock_gemini_client.generate_content.return_value = mock_stream()
        
        response = wrapper_full_response.generate_content("Stream", stream=True)
        
        # Should return generator of full response objects
        response_objects = list(response)
        assert len(response_objects) == 2
        assert hasattr(response_objects[0], 'text')
    
    def test_streaming_with_tools(self, wrapper, mock_gemini_client):
        """Test streaming with tool calls."""
        def mock_stream_1():
            # First call: model wants to use tool
            function_calls = [{"name": "add_tool", "args": {"a": 5, "b": 3}}]
            yield create_gemini_response(function_calls=function_calls)
        
        def mock_stream_2():
            # Second call: model streams final response
            yield create_gemini_response(text="The result ")
            yield create_gemini_response(text="is 8")
        
        mock_gemini_client.generate_content.side_effect = [mock_stream_1(), mock_stream_2()]
        
        response = wrapper.generate_content(
            "What is 5 + 3?",
            tools=[add_tool],
            stream=True
        )
        
        # Should get final streamed response
        content_parts = list(response)
        assert "The result " in content_parts
        assert "is 8" in content_parts

class TestContentFormatting:
    """Test content formatting and message handling."""
    
    def test_string_content(self, wrapper, mock_gemini_client):
        """Test passing string content directly."""
        mock_response = create_gemini_response(text="Response")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content("Hello")
        
        assert response == "Response"
        
        # Check that content was converted to message format
        call_args = mock_gemini_client.generate_content.call_args[1]
        assert call_args["contents"] == [{'parts': [{'text': 'Hello'}], 'role': 'user'}]
    
    def test_contents_parameter(self, wrapper, mock_gemini_client):
        """Test passing contents parameter."""
        mock_response = create_gemini_response(text="Response")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content(contents="Test content")
        
        assert response == "Response"
    
    def test_positional_and_keyword_content(self, wrapper, mock_gemini_client):
        """Test positional content with keyword args."""
        mock_response = create_gemini_response(text="Response")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content(
            "Positional content",
            generation_config={"temperature": 0.5}
        )
        
        assert response == "Response"
        
        call_args = mock_gemini_client.generate_content.call_args[1]
        assert call_args["contents"] == [{'parts': [{'text': 'Positional content'}], 'role': 'user'}]
        assert call_args["generation_config"]["temperature"] == 0.5

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_api_error_propagation(self, wrapper, mock_gemini_client):
        """Test that API errors are properly propagated."""
        mock_gemini_client.generate_content.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            wrapper.generate_content("Test")
    
    def test_invalid_tool_error(self, wrapper, mock_gemini_client):
        """Test error when model calls non-existent tool."""
        function_calls = [{"name": "nonexistent_tool", "args": {}}]
        mock_response = create_gemini_response(function_calls=function_calls)
        mock_gemini_client.generate_content.return_value = mock_response
        
        # Should handle gracefully with error in tool result
        with pytest.raises(Exception):  # Should raise when tool doesn't exist
            wrapper.generate_content(
                "Use fake tool",
                tools=[add_tool],
                graceful_error_handling=False
            )
    
    def test_empty_tools_list(self, wrapper, mock_gemini_client):
        """Test with empty tools list."""
        mock_response = create_gemini_response(text="No tools needed")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content("Test", tools=[])
        
        assert response == "No tools needed"
    
    def test_none_tools(self, wrapper, mock_gemini_client):
        """Test with None tools."""
        mock_response = create_gemini_response(text="No tools")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content("Test", tools=None)
        
        assert response == "No tools"

class TestResponseFormatting:
    """Test response formatting options."""
    
    def test_structured_output(self, wrapper, mock_gemini_client):
        """Test structured output with response_format."""
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            name: str
            value: int
        
        # Mock response that calls the structured output tool
        from toolflow.core.constants import RESPONSE_FORMAT_TOOL_NAME
        function_calls = [{
            "name": RESPONSE_FORMAT_TOOL_NAME,
            "args": {"response": {"name": "test", "value": 42}}
        }]
        mock_response = create_gemini_response(function_calls=function_calls)
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = wrapper.generate_content(
            "Generate structured data",
            response_format=TestModel
        )
        
        assert isinstance(response, TestModel)
        assert response.name == "test"
        assert response.value == 42
    
    def test_full_response_mode_with_tools(self, wrapper_full_response, mock_gemini_client):
        """Test full_response mode with tool calling."""
        function_calls = [{"name": "add_tool", "args": {"a": 1, "b": 1}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(text="Result: 2")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = wrapper_full_response.generate_content(
            "Calculate 1+1",
            tools=[add_tool],
            full_response=True
        )
        
        assert response == mock_response_2  # Should return full response object

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
