"""
Comprehensive unit tests for Gemini handler functionality.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any
import asyncio

from toolflow.providers.gemini.handler import GeminiHandler
from toolflow.core.exceptions import MaxToolCallsError, MaxTokensError
from toolflow import tool

# Test fixtures and helpers
def create_gemini_response(text: str = None, function_calls: list = None, finish_reason: str = "STOP"):
    """Create a mock Gemini response."""
    mock_response = Mock()
    mock_response.text = text or ""
    mock_response.candidates = []
    
    if text or function_calls:
        candidate = Mock()
        candidate.content = Mock()
        candidate.content.parts = []
        candidate.finish_reason = finish_reason
        
        if text:
            text_part = Mock()
            text_part.text = text
            # Ensure function_call attribute doesn't exist for text parts
            if hasattr(text_part, 'function_call'):
                delattr(text_part, 'function_call')
            candidate.content.parts.append(text_part)
        
        if function_calls:
            for func_call in function_calls:
                func_part = Mock()
                func_part.function_call = Mock()
                func_part.function_call.name = func_call["name"]
                func_part.function_call.args = func_call["args"]
                # Ensure text attribute doesn't exist for function call parts
                if hasattr(func_part, 'text'):
                    delattr(func_part, 'text')
                candidate.content.parts.append(func_part)
        
        mock_response.candidates.append(candidate)
    
    return mock_response

def create_gemini_stream_chunk(text: str = None, function_calls: list = None):
    """Create a mock Gemini streaming chunk."""
    return create_gemini_response(text=text, function_calls=function_calls)

@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client."""
    mock = Mock()
    mock.generate_content = Mock()
    return mock

@pytest.fixture
def handler(mock_gemini_client):
    """Create GeminiHandler instance."""
    return GeminiHandler(mock_gemini_client, mock_gemini_client.generate_content)

# Test tools
@tool
def math_tool(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def weather_tool(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 75°F"

@tool
def error_tool(should_fail: bool = True) -> str:
    """Tool that can fail."""
    if should_fail:
        raise ValueError("Tool error")
    return "Success"

class TestGeminiHandlerInitialization:
    """Test GeminiHandler initialization."""
    
    def test_handler_creation(self, mock_gemini_client):
        """Test handler is created correctly."""
        handler = GeminiHandler(mock_gemini_client, mock_gemini_client.generate_content)
        
        assert handler.client == mock_gemini_client
        assert handler.original_generate_content == mock_gemini_client.generate_content
        assert handler._tool_call_name_map == {}

class TestResponseParsing:
    """Test response parsing functionality."""
    
    def test_parse_text_only_response(self, handler):
        """Test parsing response with only text."""
        response = create_gemini_response(text="Hello world")
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == "Hello world"
        assert tool_calls == []
        assert raw == response
    
    def test_parse_empty_response(self, handler):
        """Test parsing empty response."""
        response = create_gemini_response()
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == ""
        assert tool_calls == []
        assert raw == response
    
    def test_parse_function_call_response(self, handler):
        """Test parsing response with function calls."""
        function_calls = [
            {"name": "math_tool", "args": {"a": 5, "b": 3}}
        ]
        response = create_gemini_response(function_calls=function_calls)
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "math_tool"
        assert tool_calls[0]["function"]["arguments"] == {"a": 5, "b": 3}
        assert tool_calls[0]["type"] == "function"
        assert "id" in tool_calls[0]
    
    def test_parse_mixed_response(self, handler):
        """Test parsing response with both text and function calls."""
        function_calls = [
            {"name": "weather_tool", "args": {"city": "NYC"}}
        ]
        response = create_gemini_response(text="Let me check that", function_calls=function_calls)
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == "Let me check that"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "weather_tool"
    
    def test_parse_multiple_function_calls(self, handler):
        """Test parsing response with multiple function calls."""
        function_calls = [
            {"name": "math_tool", "args": {"a": 5, "b": 3}},
            {"name": "weather_tool", "args": {"city": "Paris"}}
        ]
        response = create_gemini_response(function_calls=function_calls)
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "math_tool"
        assert tool_calls[1]["function"]["name"] == "weather_tool"
    
    def test_parse_response_no_text_attribute(self, handler):
        """Test parsing response when .text raises ValueError."""
        response = Mock()
        # Simulate the case where response.text raises ValueError
        def text_property():
            raise ValueError("No text available")
        
        type(response).text = property(lambda self: text_property())
        response.candidates = []
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == ""
        assert tool_calls == []

class TestMessageBuilding:
    """Test message building functionality."""
    
    def test_build_assistant_message_text_only(self, handler):
        """Test building assistant message with text only."""
        message = handler.build_assistant_message("Hello", [])
        
        assert message["role"] == "model"
        assert len(message["parts"]) == 1
        assert message["parts"][0]["text"] == "Hello"
    
    def test_build_assistant_message_tool_calls_only(self, handler):
        """Test building assistant message with tool calls only."""
        tool_calls = [{
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": {"param": "value"}}
        }]
        
        message = handler.build_assistant_message(None, tool_calls)
        
        assert message["role"] == "model"
        assert len(message["parts"]) == 1
        assert message["parts"][0]["function_call"]["name"] == "test_tool"
        assert message["parts"][0]["function_call"]["args"] == {"param": "value"}
    
    def test_build_assistant_message_empty_text(self, handler):
        """Test building assistant message with empty text."""
        tool_calls = [{
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": {}}
        }]
        
        message = handler.build_assistant_message("", tool_calls)
        
        assert message["role"] == "model"
        assert len(message["parts"]) == 1  # Only function call, no empty text
        assert message["parts"][0]["function_call"]["name"] == "test_tool"
    
    def test_build_assistant_message_mixed(self, handler):
        """Test building assistant message with text and tool calls."""
        tool_calls = [{
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": {"param": "value"}}
        }]
        
        message = handler.build_assistant_message("Let me help", tool_calls)
        
        assert message["role"] == "model"
        assert len(message["parts"]) == 2
        assert message["parts"][0]["text"] == "Let me help"
        assert message["parts"][1]["function_call"]["name"] == "test_tool"
    
    def test_build_assistant_message_no_content(self, handler):
        """Test building assistant message with no content."""
        message = handler.build_assistant_message(None, [])
        
        assert message["role"] == "model"
        assert len(message["parts"]) == 1
        assert message["parts"][0]["text"] == ""  # Empty text added to avoid validation error

class TestToolResultMessages:
    """Test tool result message building."""
    
    def test_build_single_tool_result(self, handler):
        """Test building single tool result message."""
        # First register a tool call to create the mapping manually
        # In real usage, this mapping is created during response parsing
        handler._tool_call_name_map["call_123"] = "math_tool"
        
        tool_results = [{
            "tool_call_id": "call_123",
            "output": "8"
        }]
        
        messages = handler.build_tool_result_messages(tool_results)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "function"
        assert len(messages[0]["parts"]) == 1
        assert messages[0]["parts"][0]["function_response"]["name"] == "math_tool"
        assert messages[0]["parts"][0]["function_response"]["response"]["result"] == "8"
    
    def test_build_multiple_tool_results(self, handler):
        """Test building multiple tool result messages in single message."""
        # Register tool calls first
        handler._tool_call_name_map["call_123"] = "math_tool"
        handler._tool_call_name_map["call_456"] = "weather_tool"
        
        tool_results = [
            {"tool_call_id": "call_123", "output": "8"},
            {"tool_call_id": "call_456", "output": "Sunny"}
        ]
        
        messages = handler.build_tool_result_messages(tool_results)
        
        assert len(messages) == 1  # All results in single message
        assert len(messages[0]["parts"]) == 2  # Two function response parts
        assert messages[0]["parts"][0]["function_response"]["name"] == "math_tool"
        assert messages[0]["parts"][1]["function_response"]["name"] == "weather_tool"
    
    def test_build_tool_result_unknown_id(self, handler):
        """Test building tool result with unknown tool call ID."""
        tool_results = [{
            "tool_call_id": "unknown_id",
            "output": "result"
        }]
        
        messages = handler.build_tool_result_messages(tool_results)
        
        assert len(messages) == 1
        assert messages[0]["parts"][0]["function_response"]["name"] == "unknown_id"
    
    def test_build_tool_result_non_string_output(self, handler):
        """Test building tool result with non-string output."""
        handler._tool_call_name_map["call_123"] = "test_tool"
        
        tool_results = [{
            "tool_call_id": "call_123",
            "output": 42  # Non-string output
        }]
        
        messages = handler.build_tool_result_messages(tool_results)
        
        assert messages[0]["parts"][0]["function_response"]["response"]["result"] == "42"

class TestMessageConversion:
    """Test message format conversion."""
    
    def test_convert_string_content(self, handler):
        """Test converting string content."""
        result = handler._convert_messages_to_gemini_format("Hello world")
        assert result == "Hello world"
    
    def test_convert_user_message(self, handler):
        """Test converting user message format."""
        messages = [{"role": "user", "content": "Hello"}]
        
        result = handler._convert_messages_to_gemini_format(messages)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "Hello"
    
    def test_convert_assistant_message(self, handler):
        """Test converting assistant message format."""
        messages = [{"role": "assistant", "content": "Hi there"}]
        
        result = handler._convert_messages_to_gemini_format(messages)
        
        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert result[0]["parts"][0]["text"] == "Hi there"
    
    def test_convert_gemini_format_preserved(self, handler):
        """Test that existing Gemini format is preserved."""
        messages = [{
            "role": "model",
            "parts": [{"function_call": {"name": "test", "args": {}}}]
        }]
        
        result = handler._convert_messages_to_gemini_format(messages)
        
        assert result == messages  # Should be unchanged
    
    def test_convert_function_message(self, handler):
        """Test converting function message format."""
        messages = [{
            "role": "function",
            "parts": [{"function_response": {"name": "test", "response": "result"}}]
        }]
        
        result = handler._convert_messages_to_gemini_format(messages)
        
        assert result == messages  # Should be unchanged
    
    def test_convert_system_message(self, handler):
        """Test converting system message format."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        result = handler._convert_messages_to_gemini_format(messages)
        
        assert len(result) == 2  # System creates separate message, then user message
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "System: You are helpful"
        assert result[1]["role"] == "user"
        assert result[1]["parts"][0]["text"] == "Hello"

class TestToolSchemaPreparation:
    """Test tool schema preparation."""
    
    def test_prepare_function_tools(self, handler):
        """Test preparing schemas from function tools."""
        tools = [math_tool, weather_tool]
        
        schemas, tool_map = handler.prepare_tool_schemas(tools)
        
        assert len(schemas) == 2
        assert "math_tool" in tool_map
        assert "weather_tool" in tool_map
        
        # Check schema structure
        math_schema = next(s for s in schemas if s["function"]["name"] == "math_tool")
        assert math_schema["function"]["description"] == "Add two numbers."
        assert "parameters" in math_schema["function"]
    
    def test_prepare_dict_tools(self, handler):
        """Test preparing schemas from dictionary tools."""
        tool_schema = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test tool",
                "parameters": {"type": "object"}
            }
        }
        tools = [tool_schema]
        
        schemas, tool_map = handler.prepare_tool_schemas(tools)
        
        assert len(schemas) == 1
        assert schemas[0] == tool_schema
        assert tool_map["test_tool"] == tool_schema

class TestTokenLimits:
    """Test token limit checking."""
    
    def test_check_max_tokens_reached_true(self, handler):
        """Test detecting when max tokens is reached."""
        response = create_gemini_response(text="Hello", finish_reason="MAX_TOKENS")
        
        result = handler.check_max_tokens_reached(response)
        
        assert result is True
    
    def test_check_max_tokens_reached_false(self, handler):
        """Test detecting when max tokens is not reached."""
        response = create_gemini_response(text="Hello", finish_reason="STOP")
        
        result = handler.check_max_tokens_reached(response)
        
        assert result is False
    
    def test_check_max_tokens_no_candidates(self, handler):
        """Test checking max tokens with no candidates."""
        response = Mock()
        response.candidates = []
        
        result = handler.check_max_tokens_reached(response)
        
        assert result is False

class TestStreaming:
    """Test streaming functionality."""
    
    def test_parse_stream_chunk_text(self, handler):
        """Test parsing streaming chunk with text."""
        chunk = create_gemini_stream_chunk(text="Hello")
        
        result = handler.parse_stream_chunk(chunk)
        
        assert result["text"] == "Hello"
        assert result["tool_calls"] == []
        assert result["raw"] == chunk
    
    def test_parse_stream_chunk_function_call(self, handler):
        """Test parsing streaming chunk with function call."""
        function_calls = [{"name": "test_tool", "args": {"param": "value"}}]
        chunk = create_gemini_stream_chunk(function_calls=function_calls)
        
        result = handler.parse_stream_chunk(chunk)
        
        assert result["text"] == ""
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "test_tool"
    
    def test_accumulate_streaming_response(self, handler):
        """Test accumulating streaming responses."""
        chunks = [
            create_gemini_stream_chunk(text="Hello "),
            create_gemini_stream_chunk(text="world")
        ]
        
        # Convert to generator
        def chunk_generator():
            for chunk in chunks:
                yield chunk
        
        results = list(handler.accumulate_streaming_response(chunk_generator()))
        
        assert len(results) == 2
        assert results[0][0] == "Hello "  # First text
        assert results[1][0] == "world"   # Second text

class TestAPIFormatConversion:
    """Test API format conversion."""
    
    def test_convert_to_gemini_format_basic(self, handler):
        """Test basic conversion to Gemini format."""
        kwargs = {
            "contents": "Hello world",
            "tools": [],
            "generation_config": {"temperature": 0.7}
        }
        
        result = handler._convert_to_gemini_format(**kwargs)
        
        assert result["contents"] == "Hello world"
        assert result["generation_config"]["temperature"] == 0.7
        assert "tools" not in result  # Empty tools should not be included
    
    def test_convert_to_gemini_format_with_tools(self, handler):
        """Test conversion with tools."""
        kwargs = {
            "contents": "Hello",
            "tools": [math_tool]
        }
        
        result = handler._convert_to_gemini_format(**kwargs)
        
        assert "tools" in result
        assert len(result["tools"]) == 1
        # Tools are now wrapped in function_declarations
        assert "function_declarations" in result["tools"][0]
        assert len(result["tools"][0]["function_declarations"]) == 1
        assert result["tools"][0]["function_declarations"][0]["name"] == "math_tool"
    
    def test_convert_schema_to_gemini_format(self, handler):
        """Test schema conversion to Gemini format."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "A name"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        
        result = handler._convert_schema_to_gemini_format(schema)
        
        assert result["type"] == "OBJECT"  # Uppercase
        assert "properties" in result
        assert result["properties"]["name"]["type"] == "STRING"
        assert result["required"] == ["name"]

class TestAsyncOperations:
    """Test async operations."""
    
    @pytest.mark.asyncio
    async def test_call_api_async(self, handler, mock_gemini_client):
        """Test async API calling."""
        # Mock sync generate_content since Gemini doesn't have native async
        def mock_generate(**kwargs):
            return create_gemini_response(text="Async response")
        
        handler.original_generate_content = mock_generate
        
        result = await handler.call_api_async(contents="Hello")
        
        assert result.text == "Async response"
    
    @pytest.mark.asyncio
    async def test_accumulate_streaming_response_async(self, handler):
        """Test async streaming accumulation."""
        chunks = [
            create_gemini_stream_chunk(text="Hello "),
            create_gemini_stream_chunk(text="async world")
        ]
        
        # Convert to async generator
        async def async_chunk_generator():
            for chunk in chunks:
                yield chunk
        
        results = []
        async for result in handler.accumulate_streaming_response_async(async_chunk_generator()):
            results.append(result)
        
        assert len(results) == 2
        assert results[0][0] == "Hello "
        assert results[1][0] == "async world"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
