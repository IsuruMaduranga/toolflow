"""
Basic functionality tests for Gemini provider.
"""
import pytest
from unittest.mock import Mock, MagicMock
from toolflow.providers.gemini import from_gemini
from toolflow.providers.gemini.handler import GeminiHandler
from toolflow.providers.gemini.wrappers import GeminiWrapper
from toolflow import tool

# Test fixtures
@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client."""
    mock = Mock()
    mock.generate_content = Mock()
    return mock

@pytest.fixture
def toolflow_gemini_client(mock_gemini_client):
    """Create toolflow wrapped Gemini client."""
    return from_gemini(mock_gemini_client)

# Test tools
@tool
def simple_math_tool(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def weather_tool(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny."

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
            # Important: Remove function_call attribute for text parts
            if hasattr(text_part, 'function_call'):
                delattr(text_part, 'function_call')
            candidate.content.parts.append(text_part)
        
        if function_calls:
            for func_call in function_calls:
                func_part = Mock()
                func_part.function_call = Mock()
                func_part.function_call.name = func_call["name"]
                func_part.function_call.args = func_call["args"]
                # Important: Remove text attribute for function call parts
                if hasattr(func_part, 'text'):
                    delattr(func_part, 'text')
                candidate.content.parts.append(func_part)
        
        candidate.finish_reason = "STOP"
        mock_response.candidates.append(candidate)
    
    return mock_response

class TestGeminiProvider:
    """Test Gemini provider initialization."""
    
    def test_from_gemini_creates_wrapper(self, mock_gemini_client):
        """Test that from_gemini creates a GeminiWrapper."""
        wrapper = from_gemini(mock_gemini_client)
        assert isinstance(wrapper, GeminiWrapper)
        assert wrapper._client == mock_gemini_client
        assert wrapper.full_response == False
    
    def test_from_gemini_full_response_mode(self, mock_gemini_client):
        """Test from_gemini with full_response=True."""
        wrapper = from_gemini(mock_gemini_client, full_response=True)
        assert wrapper.full_response == True
    
    def test_from_gemini_invalid_client(self):
        """Test from_gemini with invalid client type."""
        invalid_client = "not a client"
        with pytest.raises(TypeError):
            from_gemini(invalid_client)

class TestGeminiHandler:
    """Test Gemini handler functionality."""
    
    def test_handler_initialization(self, mock_gemini_client):
        """Test GeminiHandler initialization."""
        handler = GeminiHandler(mock_gemini_client, mock_gemini_client.generate_content)
        assert handler.client == mock_gemini_client
        assert handler.original_generate_content == mock_gemini_client.generate_content
        assert handler._tool_call_name_map == {}
    
    def test_parse_response_text_only(self):
        """Test parsing response with text only."""
        handler = GeminiHandler(Mock(), Mock())
        
        response = create_gemini_response(text="Hello world")
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == "Hello world"
        assert tool_calls == []
        assert raw == response
    
    def test_parse_response_with_function_calls(self):
        """Test parsing response with function calls."""
        handler = GeminiHandler(Mock(), Mock())
        
        function_calls = [
            {"name": "simple_math_tool", "args": {"a": 5, "b": 3}}
        ]
        response = create_gemini_response(function_calls=function_calls)
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "simple_math_tool"
        assert tool_calls[0]["function"]["arguments"] == {"a": 5, "b": 3}
        assert tool_calls[0]["type"] == "function"
        assert "id" in tool_calls[0]
    
    def test_parse_response_mixed_content(self):
        """Test parsing response with both text and function calls."""
        handler = GeminiHandler(Mock(), Mock())
        
        function_calls = [{"name": "weather_tool", "args": {"city": "NYC"}}]
        response = create_gemini_response(text="Let me check", function_calls=function_calls)
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == "Let me check"
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "weather_tool"
    
    def test_parse_response_no_text_error(self):
        """Test parsing response when .text raises ValueError."""
        handler = GeminiHandler(Mock(), Mock())
        
        response = Mock()
        # Simulate the case where response.text raises ValueError
        def text_property():
            raise ValueError("No text")
        
        type(response).text = property(lambda self: text_property())
        response.candidates = []
        
        text, tool_calls, raw = handler.parse_response(response)
        
        assert text == ""
        assert tool_calls == []
    
    def test_build_assistant_message(self):
        """Test building assistant message."""
        handler = GeminiHandler(Mock(), Mock())
        
        tool_calls = [{
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": {"param": "value"}}
        }]
        
        message = handler.build_assistant_message("Hello", tool_calls)
        
        assert message["role"] == "model"
        assert len(message["parts"]) == 2  # text + function call
        assert message["parts"][0]["text"] == "Hello"
        assert message["parts"][1]["function_call"]["name"] == "test_tool"
    
    def test_build_assistant_message_no_text(self):
        """Test building assistant message with no text."""
        handler = GeminiHandler(Mock(), Mock())
        
        tool_calls = [{
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": {}}
        }]
        
        message = handler.build_assistant_message(None, tool_calls)
        
        assert message["role"] == "model"
        assert len(message["parts"]) == 1  # Only function call
        assert message["parts"][0]["function_call"]["name"] == "test_tool"
    
    def test_build_assistant_message_empty(self):
        """Test building assistant message with no content."""
        handler = GeminiHandler(Mock(), Mock())
        
        message = handler.build_assistant_message(None, [])
        
        assert message["role"] == "model"
        assert len(message["parts"]) == 1
        assert message["parts"][0]["text"] == ""  # Empty text to avoid validation error
    
    def test_build_tool_result_messages_single(self):
        """Test building tool result messages."""
        handler = GeminiHandler(Mock(), Mock())
        
        # Register tool call mapping first
        handler._tool_call_name_map["call_123"] = "test_tool"
        
        tool_results = [{
            "tool_call_id": "call_123",
            "output": "test result"
        }]
        
        messages = handler.build_tool_result_messages(tool_results)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "function"
        assert len(messages[0]["parts"]) == 1
        assert messages[0]["parts"][0]["function_response"]["name"] == "test_tool"
        assert messages[0]["parts"][0]["function_response"]["response"]["result"] == "test result"
    
    def test_build_tool_result_messages_multiple(self):
        """Test building multiple tool result messages."""
        handler = GeminiHandler(Mock(), Mock())
        
        # Register tool call mappings
        handler._tool_call_name_map["call_123"] = "tool_1"
        handler._tool_call_name_map["call_456"] = "tool_2"
        
        tool_results = [
            {"tool_call_id": "call_123", "output": "result 1"},
            {"tool_call_id": "call_456", "output": "result 2"}
        ]
        
        messages = handler.build_tool_result_messages(tool_results)
        
        assert len(messages) == 1  # All results in single message
        assert len(messages[0]["parts"]) == 2  # Two function response parts
        assert messages[0]["parts"][0]["function_response"]["name"] == "tool_1"
        assert messages[0]["parts"][1]["function_response"]["name"] == "tool_2"
    
    def test_check_max_tokens_reached(self):
        """Test max tokens checking."""
        handler = GeminiHandler(Mock(), Mock())
        
        # Response with MAX_TOKENS finish reason
        response = create_gemini_response(text="Incomplete")
        response.candidates[0].finish_reason = "MAX_TOKENS"
        
        assert handler.check_max_tokens_reached(response) == True
        
        # Response with STOP finish reason
        response.candidates[0].finish_reason = "STOP"
        assert handler.check_max_tokens_reached(response) == False
    
    def test_extract_text_from_response(self):
        """Test text extraction from response."""
        handler = GeminiHandler(Mock(), Mock())
        
        response = create_gemini_response(text="Test text")
        text = handler.extract_text_from_response(response)
        
        assert text == "Test text"
        
        # Test with ValueError on .text
        response = Mock()
        def text_property():
            raise ValueError("No text")
        
        type(response).text = property(lambda self: text_property())
        text = handler.extract_text_from_response(response)
        
        assert text == ""

class TestGeminiIntegration:
    """Test Gemini integration with toolflow."""
    
    def test_simple_text_generation(self, toolflow_gemini_client, mock_gemini_client):
        """Test simple text generation."""
        mock_response = create_gemini_response(text="Hello world")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = toolflow_gemini_client.generate_content("Say hello")
        
        assert response == "Hello world"
        mock_gemini_client.generate_content.assert_called_once()
    
    def test_full_response_mode(self, mock_gemini_client):
        """Test full response mode."""
        client = from_gemini(mock_gemini_client, full_response=True)
        mock_response = create_gemini_response(text="Hello")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = client.generate_content("Say hello")
        
        assert response == mock_response  # Should return full response object
    
    def test_tool_calling_basic(self, toolflow_gemini_client, mock_gemini_client):
        """Test basic tool calling."""
        # First response: model wants to call a tool
        function_calls = [{"name": "simple_math_tool", "args": {"a": 5, "b": 3}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        
        # Second response: model responds with result
        mock_response_2 = create_gemini_response(text="The sum is 8")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_gemini_client.generate_content(
            "What is 5 + 3?",
            tools=[simple_math_tool]
        )
        
        assert response == "The sum is 8"
        assert mock_gemini_client.generate_content.call_count == 2
    
    def test_multiple_tool_calls(self, toolflow_gemini_client, mock_gemini_client):
        """Test multiple tool calls in one turn."""
        function_calls = [
            {"name": "simple_math_tool", "args": {"a": 5, "b": 3}},
            {"name": "weather_tool", "args": {"city": "NYC"}}
        ]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(text="Math result: 8, Weather: sunny")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_gemini_client.generate_content(
            "Calculate 5+3 and get NYC weather",
            tools=[simple_math_tool, weather_tool]
        )
        
        assert "8" in response
        assert "sunny" in response
    
    def test_no_tools_needed(self, toolflow_gemini_client, mock_gemini_client):
        """Test when no tools are needed."""
        mock_response = create_gemini_response(text="I can answer directly")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = toolflow_gemini_client.generate_content(
            "What's your name?",
            tools=[simple_math_tool]
        )
        
        assert response == "I can answer directly"
        assert mock_gemini_client.generate_content.call_count == 1
    
    def test_streaming_generation(self, toolflow_gemini_client, mock_gemini_client):
        """Test streaming generation."""
        def mock_stream():
            yield create_gemini_response(text="Hello ")
            yield create_gemini_response(text="streaming ")
            yield create_gemini_response(text="world")
        
        mock_gemini_client.generate_content.return_value = mock_stream()
        
        response = toolflow_gemini_client.generate_content("Stream hello", stream=True)
        
        # Should return a generator
        assert hasattr(response, '__iter__')
        
        # Collect content
        content = list(response)
        assert "Hello " in content
        assert "streaming " in content
        assert "world" in content
    
    def test_parameter_passing(self, toolflow_gemini_client, mock_gemini_client):
        """Test that parameters are passed through correctly."""
        mock_response = create_gemini_response(text="Response")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = toolflow_gemini_client.generate_content(
            "Test",
            generation_config={"temperature": 0.7, "max_output_tokens": 100},
            safety_settings=[],
            max_tool_call_rounds=5,
            parallel_tool_execution=False
        )
        
        assert response == "Response"
        
        # Check parameters were passed
        call_args = mock_gemini_client.generate_content.call_args[1]
        assert call_args["generation_config"]["temperature"] == 0.7
        assert call_args["generation_config"]["max_output_tokens"] == 100
        assert call_args["safety_settings"] == []

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_response(self, toolflow_gemini_client, mock_gemini_client):
        """Test empty response handling."""
        mock_response = create_gemini_response()
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = toolflow_gemini_client.generate_content("Test")
        
        assert response == ""
    
    def test_api_error_handling(self, toolflow_gemini_client, mock_gemini_client):
        """Test API error propagation."""
        mock_gemini_client.generate_content.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            toolflow_gemini_client.generate_content("Test")
    
    def test_invalid_function_call(self, toolflow_gemini_client, mock_gemini_client):
        """Test handling of invalid function calls."""
        # Model tries to call non-existent tool
        function_calls = [{"name": "nonexistent_tool", "args": {}}]
        mock_response = create_gemini_response(function_calls=function_calls)
        mock_gemini_client.generate_content.return_value = mock_response
        
        with pytest.raises(Exception):  # Should raise when tool doesn't exist
            toolflow_gemini_client.generate_content(
                "Use fake tool",
                tools=[simple_math_tool],
                graceful_error_handling=False
            )
    
    def test_malformed_function_args(self, toolflow_gemini_client, mock_gemini_client):
        """Test handling of malformed function arguments."""
        # Function call with missing required arguments
        function_calls = [{"name": "simple_math_tool", "args": {"a": 5}}]  # Missing 'b'
        mock_response = create_gemini_response(function_calls=function_calls)
        mock_gemini_client.generate_content.return_value = mock_response
        
        with pytest.raises(Exception):  # Should raise due to missing argument
            toolflow_gemini_client.generate_content(
                "Add numbers",
                tools=[simple_math_tool],
                graceful_error_handling=False
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
