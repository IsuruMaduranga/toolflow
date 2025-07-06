"""
Tests for basic Anthropic provider functionality.
"""
import pytest
from unittest.mock import Mock, patch
import json

from toolflow import from_anthropic, tool
from tests.conftest import (
    create_anthropic_response,
    create_anthropic_tool_call,
    simple_math_tool,
    weather_tool
)


class TestAnthropicWrapper:
    """Test the Anthropic client wrapper."""
    
    def test_from_anthropic_creates_wrapper(self, mock_anthropic_client):
        """Test that from_anthropic creates a proper wrapper."""
        client = from_anthropic(mock_anthropic_client)
        
        # Should have messages attribute
        assert hasattr(client, 'messages')
        
        # Should preserve original client
        assert hasattr(client, '_original_client')
    
    def test_wrapper_preserves_client_methods(self, mock_anthropic_client):
        """Test that wrapper preserves original client methods."""
        # Add a custom method to mock client
        mock_anthropic_client.custom_method = Mock(return_value="custom_result")
        
        client = from_anthropic(mock_anthropic_client)
        
        # Should still be accessible
        assert hasattr(client, 'custom_method')
        assert client.custom_method() == "custom_result"
    
    def test_full_response_parameter(self, mock_anthropic_client):
        """Test full_response parameter in from_anthropic."""
        # Default should be simplified response
        client_simple = from_anthropic(mock_anthropic_client)
        assert client_simple.full_response is False
        
        # Explicit full response
        client_full = from_anthropic(mock_anthropic_client, full_response=True)
        assert client_full.full_response is True


class TestBasicMessageCreation:
    """Test basic message creation without tools."""
    
    def test_simple_message_creation(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test simple message creation without tools."""
        # Mock response
        mock_response = create_anthropic_response(content="Hello, world!")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return simplified response (just content)
        assert response == "Hello, world!"
        
        # Verify original client was called
        mock_anthropic_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
        )
    
    def test_message_creation_full_response(self, mock_anthropic_client):
        """Test message creation with full_response=True."""
        client = from_anthropic(mock_anthropic_client, full_response=True)
        
        mock_response = create_anthropic_response(content="Hello, world!")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return full response object
        assert response == mock_response
        assert hasattr(response, 'content')
        assert response.content[0].text == "Hello, world!"
    
    def test_message_creation_method_level_full_response(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test method-level full_response override."""
        mock_response = create_anthropic_response(content="Hello, world!")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Override at method level
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            full_response=True
        )
        
        # Should return full response despite client default
        assert response == mock_response
    
    def test_standard_parameters_passthrough(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test that standard Anthropic parameters pass through correctly."""
        mock_response = create_anthropic_response(content="Response")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            top_p=0.9,
            system="You are a helpful assistant"
        )
        
        # Verify all parameters were passed through
        mock_anthropic_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            top_p=0.9,
            system="You are a helpful assistant"
        )


class TestToolExecution:
    """Test tool execution with Anthropic."""
    
    def test_single_tool_execution(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test execution of a single tool."""
        # First response: model wants to call tool
        tool_call = create_anthropic_tool_call("toolu_123", "simple_math_tool", {"a": 5, "b": 3})
        mock_response_1 = create_anthropic_response(tool_calls=[tool_call])
        
        # Second response: model responds with result
        mock_response_2 = create_anthropic_response(content="The answer is 8")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "What is 5 + 3?"}],
            tools=[simple_math_tool]
        )
        
        assert response == "The answer is 8"
        
        # Should have made two calls
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Check second call includes tool result
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        
        # Should have original message, assistant message, and tool result
        assert len(messages) == 3
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == "What is 5 + 3?"
        assert messages[1]['role'] == 'assistant'
        assert messages[2]['role'] == 'user'
        assert messages[2]['content'][0]['type'] == 'tool_result'
        assert messages[2]['content'][0]['tool_use_id'] == 'toolu_123'
        assert messages[2]['content'][0]['content'] == '8.0'  # Result of 5 + 3 (float precision preserved)
    
    def test_multiple_tool_execution(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test execution of multiple tools in sequence."""
        # First response: model wants to call multiple tools
        tool_call_1 = create_anthropic_tool_call("toolu_123", "simple_math_tool", {"a": 5, "b": 3})
        tool_call_2 = create_anthropic_tool_call("toolu_456", "weather_tool", {"city": "NYC"})
        mock_response_1 = create_anthropic_response(tool_calls=[tool_call_1, tool_call_2])
        
        # Second response: model responds with results
        mock_response_2 = create_anthropic_response(content="Math result is 8, NYC weather is sunny")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "What is 5 + 3 and what's the weather in NYC?"}],
            tools=[simple_math_tool, weather_tool]
        )
        
        assert response == "Math result is 8, NYC weather is sunny"
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Check that both tool results are included
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        
        # Should have user message, assistant message, and tool result message with multiple results
        assert len(messages) == 3
        tool_message = messages[2]
        assert tool_message['role'] == 'user'
        assert len(tool_message['content']) == 2  # Two tool results
    
    def test_parallel_tool_execution(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test parallel tool execution enabled."""
        tool_call_1 = create_anthropic_tool_call("toolu_123", "simple_math_tool", {"a": 10, "b": 5})
        tool_call_2 = create_anthropic_tool_call("toolu_456", "weather_tool", {"city": "Boston"})
        mock_response_1 = create_anthropic_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_anthropic_response(content="Results computed")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # Enable parallel execution
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Calculate 10+5 and get Boston weather"}],
            tools=[simple_math_tool, weather_tool],
            parallel_tool_execution=True
        )
        
        assert response == "Results computed"
        assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_tool_execution_without_tools_parameter(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test that tool execution is skipped when no tools provided."""
        mock_response = create_anthropic_response(content="No tools available")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
            # No tools parameter
        )
        
        assert response == "No tools available"
        assert mock_anthropic_client.messages.create.call_count == 1
    
    def test_no_tool_calls_in_response(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test when model doesn't call any tools."""
        mock_response = create_anthropic_response(content="I don't need to use any tools")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Just say hello"}],
            tools=[simple_math_tool]
        )
        
        assert response == "I don't need to use any tools"
        assert mock_anthropic_client.messages.create.call_count == 1


class TestErrorHandling:
    """Test error handling in Anthropic provider."""
    
    def test_tool_execution_error(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test handling of tool execution errors."""
        @tool
        def failing_tool(should_fail: bool = True) -> str:
            """A tool that fails."""
            if should_fail:
                raise ValueError("Tool failed intentionally")
            return "Success"
        
        # Model wants to call failing tool
        tool_call = create_anthropic_tool_call("toolu_123", "failing_tool", {"should_fail": True})
        mock_response_1 = create_anthropic_response(tool_calls=[tool_call])
        
        # Model responds after receiving error
        mock_response_2 = create_anthropic_response(content="I encountered an error")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Use the failing tool"}],
            tools=[failing_tool]
        )
        
        assert response == "I encountered an error"
        
        # Check that error was passed to model
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        tool_message = messages[2]  # Tool result is the 3rd message
        assert "Tool failed intentionally" in tool_message['content'][0]['content']
    
    def test_invalid_tool_arguments(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test handling of invalid tool arguments."""
        # Model tries to call tool with wrong arguments
        tool_call = create_anthropic_tool_call("toolu_123", "simple_math_tool", {"x": 5})  # Wrong param name
        mock_response_1 = create_anthropic_response(tool_calls=[tool_call])
        mock_response_2 = create_anthropic_response(content="There was an argument error")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Add numbers"}],
            tools=[simple_math_tool]
        )
        
        assert response == "There was an argument error"
        
        # Should still make second call with error message
        assert mock_anthropic_client.messages.create.call_count == 2
    
    def test_unknown_tool_call(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test handling of calls to unknown tools."""
        # Model tries to call non-existent tool
        tool_call = create_anthropic_tool_call("toolu_123", "unknown_tool", {"param": "value"})
        mock_response_1 = create_anthropic_response(tool_calls=[tool_call])
        mock_response_2 = create_anthropic_response(content="Unknown tool error handled")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Use unknown tool"}],
            tools=[simple_math_tool]
        )
        
        assert response == "Unknown tool error handled"
        assert mock_anthropic_client.messages.create.call_count == 2


class TestAnthropicSpecificFeatures:
    """Test Anthropic-specific features."""
    
    def test_system_parameter(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test Anthropic system parameter handling."""
        mock_response = create_anthropic_response(content="Response with system message")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant."
        )
        
        # Verify system parameter was passed through
        mock_anthropic_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant."
        )
    
    def test_mixed_content_response(self, toolflow_anthropic_client, mock_anthropic_client):
        """Test handling of mixed content responses."""
        # Create response with both text and tool calls
        text_content = Mock()
        text_content.type = "text"
        text_content.text = "I'll help you with that."
        
        tool_call = create_anthropic_tool_call("toolu_123", "simple_math_tool", {"a": 2, "b": 3})
        
        mock_response = Mock()
        mock_response.content = [text_content, tool_call]
        mock_response.stop_reason = "tool_use"
        
        # Second response after tool execution
        mock_response_2 = create_anthropic_response(content="The result is 5")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response, mock_response_2]
        
        response = toolflow_anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Calculate 2+3"}],
            tools=[simple_math_tool]
        )
        
        assert response == "The result is 5"
        assert mock_anthropic_client.messages.create.call_count == 2 