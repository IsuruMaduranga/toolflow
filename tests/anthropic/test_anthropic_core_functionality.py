"""
Test core functionality of the toolflow library with Anthropic.

This module tests:
- @tool decorator functionality (shared)
- Basic tool execution with Anthropic
- Anthropic-specific integration features
- Mock-based testing of Anthropic client
"""
import pytest
import os
from unittest.mock import Mock
from typing import List
from pydantic import BaseModel

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

import toolflow
from ..conftest import (
    simple_math_tool, 
    divide_tool, 
    get_current_time_tool,
    calculator_tool,
    weather_tool,
    create_mock_anthropic_tool_call,
    create_mock_anthropic_response as conftest_create_mock_anthropic_response,
    create_mock_anthropic_streaming_chunk
)
from toolflow.providers.anthropic.structured_output import (
    create_anthropic_response_tool,
    handle_anthropic_structured_response,
    validate_response_format
)


# Test Pydantic models for structured output
class WeatherInfo(BaseModel):
    city: str
    temperature: float
    condition: str


class TaskResult(BaseModel):
    task: str
    completed: bool
    result: str


@toolflow.tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F, sunny"


@toolflow.tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def create_mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock_client = Mock()
    mock_messages = Mock()
    mock_client.messages = mock_messages
    return mock_client, mock_messages


def create_mock_anthropic_response(content_blocks):
    """Create a mock Anthropic response with given content blocks."""
    mock_response = Mock()
    mock_response.content = content_blocks
    return mock_response


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


class TestAnthropicToolExecution:
    """Test tool execution functionality with mocked Anthropic client."""
    
    def test_simple_math_operations(self, sync_anthropic_client, mock_anthropic_client):
        """Test basic math tools work correctly with Anthropic client."""
        # Mock a tool call response
        tool_call = create_mock_anthropic_tool_call("call_123", "simple_math_tool", {"a": 10.0, "b": 2.0})
        mock_response_1 = conftest_create_mock_anthropic_response(tool_calls=[tool_call])
        
        # Mock the final response
        mock_response_2 = conftest_create_mock_anthropic_response(content="The result is 12.0")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 10 + 2?"}],
            tools=[simple_math_tool]
        )
        
        assert response.content[0].text == "The result is 12.0"
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_tool_execution_flow(self, sync_anthropic_client, mock_anthropic_client):
        """Test the complete tool execution flow with mocked Anthropic client."""
        # First response - model wants to use tool
        tool_call = create_mock_anthropic_tool_call("call_456", "divide_tool", {"a": 20.0, "b": 4.0})
        mock_response_1 = conftest_create_mock_anthropic_response(tool_calls=[tool_call])
        
        # Second response - model responds with result  
        mock_response_2 = conftest_create_mock_anthropic_response(content="The division result is 5.0")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 20 divided by 4?"}],
            tools=[divide_tool]
        )
        
        assert response.content[0].text == "The division result is 5.0"
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_multiple_tool_calls(self, sync_anthropic_client, mock_anthropic_client):
        """Test multiple tool calls in sequence."""
        # First response with multiple tool calls
        tool_call_1 = create_mock_anthropic_tool_call("call_1", "simple_math_tool", {"a": 5.0, "b": 3.0})
        tool_call_2 = create_mock_anthropic_tool_call("call_2", "calculator_tool", {"operation": "multiply", "a": 8.0, "b": 2.0})
        
        # Create mock content blocks
        text_block = Mock(type="text", text="Let me calculate both:")
        
        mock_response_1 = Mock()
        mock_response_1.content = [text_block, tool_call_1, tool_call_2]
        
        # Second response
        mock_response_2 = conftest_create_mock_anthropic_response(content="First result is 8.0 and second is 16.0")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022", 
            max_tokens=1024,
            messages=[{"role": "user", "content": "Add 5+3 and multiply 8*2"}],
            tools=[simple_math_tool, calculator_tool]
        )
        
        assert response.content[0].text == "First result is 8.0 and second is 16.0"
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_system_message_handling(self, sync_anthropic_client, mock_anthropic_client):
        """Test that system messages are properly passed to Anthropic."""
        mock_response = conftest_create_mock_anthropic_response(content="Hello! I'm ready to help with math.")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are a helpful math assistant.",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[calculator_tool]
        )
        
        # Verify system message was passed
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["system"] == "You are a helpful math assistant."
        assert response.content[0].text == "Hello! I'm ready to help with math."

    def test_no_tool_calls_response(self, sync_anthropic_client, mock_anthropic_client):
        """Test response when model doesn't use any tools."""
        mock_response = conftest_create_mock_anthropic_response(content="I don't need any tools for this simple question.")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is your name?"}],
            tools=[calculator_tool]
        )
        
        assert response.content[0].text == "I don't need any tools for this simple question."
        assert mock_anthropic_client.messages.create.call_count == 1


class TestAnthropicFullResponseParameter:
    """Test the full_response parameter with Anthropic."""
    
    def test_full_response_true(self, mock_anthropic_client):
        """Test full_response=True returns complete Anthropic response."""
        mock_response = conftest_create_mock_anthropic_response(content="Test response")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=True)
        
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return the full mock response object
        assert response == mock_response
        assert hasattr(response, 'content')

    def test_full_response_false(self, mock_anthropic_client):
        """Test full_response=False returns simplified response."""
        mock_response = conftest_create_mock_anthropic_response(content="Simplified response")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return just the text content
        assert response == "Simplified response"

    def test_method_level_override(self, mock_anthropic_client):
        """Test that method-level full_response parameter overrides client-level."""
        mock_response = conftest_create_mock_anthropic_response(content="Override test")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Client set to full_response=False
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        
        # Override at method level to full_response=True
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            full_response=True
        )
        
        # Should return full response due to method override
        assert response == mock_response
        assert hasattr(response, 'content')


class TestAnthropicErrorHandling:
    """Test error handling with Anthropic integration."""
    
    def test_graceful_tool_error_handling(self, sync_anthropic_client, mock_anthropic_client):
        """Test graceful handling of tool execution errors."""
        # Mock a tool call that will cause an error
        tool_call = create_mock_anthropic_tool_call("call_error", "divide_tool", {"a": 10.0, "b": 0.0})
        mock_response_1 = conftest_create_mock_anthropic_response(tool_calls=[tool_call])
        
        # Second response after error handling
        mock_response_2 = conftest_create_mock_anthropic_response(content="I encountered an error with division by zero.")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Divide 10 by 0"}],
            tools=[divide_tool],
            graceful_error_handling=True
        )
        
        assert response.content[0].text == "I encountered an error with division by zero."
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_max_tool_calls_limit(self, sync_anthropic_client, mock_anthropic_client):
        """Test that max_tool_calls limit is enforced."""
        # Always return a tool call to trigger infinite loop
        tool_call = create_mock_anthropic_tool_call("call_loop", "simple_math_tool", {"a": 1.0, "b": 1.0})
        mock_response = conftest_create_mock_anthropic_response(tool_calls=[tool_call])
        mock_anthropic_client.messages.create.return_value = mock_response
        
        with pytest.raises(Exception, match="Max tool calls reached"):
            sync_anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Keep calculating"}],
                tools=[simple_math_tool],
                max_tool_calls=2  # Low limit to trigger error
            )

    def test_invalid_tool_name_error(self, sync_anthropic_client, mock_anthropic_client):
        """Test error when model tries to call non-existent tool."""
        # Mock a tool call with invalid tool name
        tool_call = create_mock_anthropic_tool_call("call_invalid", "non_existent_tool", {"param": "value"})
        mock_response = conftest_create_mock_anthropic_response(tool_calls=[tool_call])
        mock_anthropic_client.messages.create.return_value = mock_response
        
        with pytest.raises(ValueError, match="Tool non_existent_tool not found"):
            sync_anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Use invalid tool"}],
                tools=[simple_math_tool],
                graceful_error_handling=False  # Disable graceful handling to get exception
            )


class TestAnthropicParallelExecution:
    """Test parallel tool execution with Anthropic."""
    
    def test_parallel_execution_enabled(self, sync_anthropic_client, mock_anthropic_client):
        """Test that parallel_tool_execution parameter is passed correctly."""
        tool_call_1 = create_mock_anthropic_tool_call("call_1", "simple_math_tool", {"a": 5.0, "b": 3.0})
        tool_call_2 = create_mock_anthropic_tool_call("call_2", "simple_math_tool", {"a": 7.0, "b": 2.0})
        
        text_block = Mock(type="text", text="Calculating both:")
        mock_response_1 = Mock()
        mock_response_1.content = [text_block, tool_call_1, tool_call_2]
        
        mock_response_2 = conftest_create_mock_anthropic_response(content="Results: 8.0 and 9.0")
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate 5+3 and 7+2"}],
            tools=[simple_math_tool],
            parallel_tool_execution=True,
            max_workers=4
        )
        
        assert response.content[0].text == "Results: 8.0 and 9.0"
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_sequential_execution_default(self, sync_anthropic_client, mock_anthropic_client):
        """Test sequential execution as default behavior."""
        tool_call = create_mock_anthropic_tool_call("call_seq", "simple_math_tool", {"a": 10.0, "b": 5.0})
        mock_response_1 = conftest_create_mock_anthropic_response(tool_calls=[tool_call])
        mock_response_2 = conftest_create_mock_anthropic_response(content="Sequential result: 15.0")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Add 10 and 5"}],
            tools=[simple_math_tool]
            # parallel_tool_execution defaults to False
        )
        
        assert response.content[0].text == "Sequential result: 15.0"


class TestAnthropicMessageFormatting:
    """Test Anthropic-specific message formatting."""
    
    def test_tool_result_formatting(self, sync_anthropic_client, mock_anthropic_client):
        """Test that tool results are formatted correctly for Anthropic."""
        tool_call = create_mock_anthropic_tool_call("call_format", "weather_tool", {"city": "Tokyo"})
        mock_response_1 = conftest_create_mock_anthropic_response(tool_calls=[tool_call])
        mock_response_2 = conftest_create_mock_anthropic_response(content="Weather info received")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[weather_tool]
        )
        
        # Check that the second call includes properly formatted tool results
        call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = call_args[1]["messages"]
        
        # Should have original user message, assistant response with tool call, and tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"  
        assert messages[2]["role"] == "user"  # Tool results come as user messages in Anthropic
        
        # Tool result should be properly formatted
        tool_result_content = messages[2]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "tool_result"

    def test_empty_messages_list(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of edge cases with message formatting."""
        mock_response = conftest_create_mock_anthropic_response(content="Empty messages handled")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Simple question"}]
        )
        
        assert response.content[0].text == "Empty messages handled"
        call_args = mock_anthropic_client.messages.create.call_args
        assert len(call_args[1]["messages"]) == 1


class TestAnthropicStructuredOutput:
    """Test structured output functionality."""
    
    def test_validate_response_format_valid_pydantic(self):
        """Test validation passes for valid Pydantic model."""
        # Should not raise
        validate_response_format(WeatherInfo)
    
    def test_validate_response_format_invalid_type(self):
        """Test validation fails for non-Pydantic type."""
        with pytest.raises(ValueError, match="response_format must be a Pydantic model"):
            validate_response_format(dict)
    
    def test_create_anthropic_response_tool(self):
        """Test creation of response tool."""
        response_tool = create_anthropic_response_tool(WeatherInfo)
        
        # Check tool properties
        assert hasattr(response_tool, '_tool_metadata')
        assert response_tool._tool_metadata['function']['name'] == 'final_response_tool_internal'
        # Note: The internal flag would be in the description or function logic, not in metadata
    
    def test_handle_anthropic_structured_response_success(self):
        """Test successful structured response handling."""
        # Create mock response with structured tool call
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", 
            "final_response_tool_internal", 
            {
                "response": {
                    "city": "San Francisco",
                    "temperature": 72.0,
                    "condition": "sunny"
                }
            }
        )
        mock_response = create_mock_anthropic_response([tool_block])
        
        # Handle structured response
        result = handle_anthropic_structured_response(mock_response, WeatherInfo)
        
        assert result is not None
        assert hasattr(result, 'parsed')
        
        parsed = result.parsed
        assert isinstance(parsed, WeatherInfo)
        assert parsed.city == "San Francisco"
        assert parsed.temperature == 72.0
        assert parsed.condition == "sunny"
    
    def test_handle_anthropic_structured_response_no_tool_call(self):
        """Test structured response handling when no tool call present."""
        text_block = create_mock_anthropic_text_block("Just regular text")
        mock_response = create_mock_anthropic_response([text_block])
        
        result = handle_anthropic_structured_response(mock_response, WeatherInfo)
        assert result is None
    
    def test_structured_output_integration(self):
        """Test end-to-end structured output integration."""
        mock_client, mock_messages = create_mock_anthropic_client()
        
        # Mock response with structured tool call only (no other tools)
        structured_tool_block = create_mock_anthropic_tool_use_block(
            "tool_1",
            "final_response_tool_internal",
            {
                "response": {
                    "task": "Calculate sum",
                    "completed": True,
                    "result": "The sum is 42"
                }
            }
        )
        mock_response = create_mock_anthropic_response([structured_tool_block])
        mock_messages.create.return_value = mock_response
        
        # Create wrapper
        client = toolflow.from_anthropic(mock_client)
        
        # Test structured output call (no other tools, just response_format)
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Provide a task result"}],
            response_format=TaskResult
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, TaskResult)
        assert result.task == "Calculate sum"
        assert result.completed == True
        assert result.result == "The sum is 42"
    
    def test_structured_output_with_streaming_raises_error(self):
        """Test that structured output with streaming raises an error."""
        mock_client, mock_messages = create_mock_anthropic_client()
        client = toolflow.from_anthropic(mock_client)
        
        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=WeatherInfo,
                stream=True
            )
    
    def test_structured_output_full_response_mode(self):
        """Test structured output in full response mode."""
        mock_client, mock_messages = create_mock_anthropic_client()
        
        # Mock response with structured tool call (no other tools, so this should be the only call)
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1",
            "final_response_tool_internal",
            {
                "response": {
                    "city": "Boston",
                    "temperature": 65.0,
                    "condition": "cloudy"
                }
            }
        )
        mock_response = create_mock_anthropic_response([tool_block])
        mock_messages.create.return_value = mock_response
        
        # Create wrapper with full_response=True
        client = toolflow.from_anthropic(mock_client, full_response=True)
        
        # Test structured output call (no other tools, just response_format)
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Weather in Boston?"}],
            response_format=WeatherInfo
        )
        
        # Should return full response object with parsed attribute
        assert hasattr(result, 'parsed')
        
        parsed = result.parsed
        assert isinstance(parsed, WeatherInfo)
        assert parsed.city == "Boston"
        assert parsed.temperature == 65.0
        assert parsed.condition == "cloudy"


class TestAnthropicErrorHandling:
    """Test error handling scenarios."""
    
    def test_graceful_tool_error_handling(self):
        """Test graceful handling of tool execution errors."""
        mock_client, mock_messages = create_mock_anthropic_client()
        
        @toolflow.tool
        def failing_tool(x: int) -> int:
            """A tool that always fails."""
            raise ValueError("Tool failed!")
        
        # Mock response with tool call
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", "failing_tool", {"x": 5}
        )
        first_response = create_mock_anthropic_response([tool_block])
        
        # Mock final response
        text_block = create_mock_anthropic_text_block("I encountered an error.")
        second_response = create_mock_anthropic_response([text_block])
        
        mock_messages.create.side_effect = [first_response, second_response]
        
        client = toolflow.from_anthropic(mock_client)
        
        # Should not raise exception due to graceful error handling
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Test"}],
            tools=[failing_tool],
            graceful_error_handling=True
        )
        
        assert result == "I encountered an error."
    
    def test_non_graceful_tool_error_handling(self):
        """Test non-graceful handling of tool execution errors."""
        mock_client, mock_messages = create_mock_anthropic_client()
        
        @toolflow.tool
        def failing_tool(x: int) -> int:
            """A tool that always fails."""
            raise ValueError("Tool failed!")
        
        # Mock response with tool call
        tool_block = create_mock_anthropic_tool_use_block(
            "tool_1", "failing_tool", {"x": 5}
        )
        mock_response = create_mock_anthropic_response([tool_block])
        mock_messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic(mock_client)
        
        # Should raise exception when graceful_error_handling=False
        with pytest.raises(ValueError, match="Tool failed!"):
            client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Test"}],
                tools=[failing_tool],
                graceful_error_handling=False
            ) 