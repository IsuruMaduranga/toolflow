"""
Tests for basic OpenAI provider functionality.
"""
import pytest
from unittest.mock import Mock, patch
import json

from toolflow import from_openai, tool
from tests.conftest import (
    create_openai_response,
    create_openai_tool_call,
    simple_math_tool,
    weather_tool,
    BASIC_TOOLS
)


class TestOpenAIWrapper:
    """Test the OpenAI client wrapper."""
    
    def test_from_openai_creates_wrapper(self, mock_openai_client):
        """Test that from_openai creates a proper wrapper."""
        client = from_openai(mock_openai_client)
        
        # Should have chat attribute
        assert hasattr(client, 'chat')
        assert hasattr(client.chat, 'completions')
        
        # Should preserve original client
        assert hasattr(client, '_original_client')
    
    def test_wrapper_preserves_client_methods(self, mock_openai_client):
        """Test that wrapper preserves original client methods."""
        # Add a custom method to mock client
        mock_openai_client.custom_method = Mock(return_value="custom_result")
        
        client = from_openai(mock_openai_client)
        
        # Should still be accessible
        assert hasattr(client, 'custom_method')
        assert client.custom_method() == "custom_result"
    
    def test_full_response_parameter(self, mock_openai_client):
        """Test full_response parameter in from_openai."""
        # Default should be simplified response
        client_simple = from_openai(mock_openai_client)
        assert client_simple.full_response is False
        
        # Explicit full response
        client_full = from_openai(mock_openai_client, full_response=True)
        assert client_full.full_response is True


class TestBasicChatCompletion:
    """Test basic chat completion without tools."""
    
    def test_simple_chat_completion(self, toolflow_openai_client, mock_openai_client):
        """Test simple chat completion without tools."""
        # Mock response
        mock_response = create_openai_response(content="Hello, world!")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return simplified response (just content)
        assert response == "Hello, world!"
        
        # Verify original client was called
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
    
    def test_chat_completion_full_response(self, mock_openai_client):
        """Test chat completion with full_response=True."""
        client = from_openai(mock_openai_client, full_response=True)
        
        mock_response = create_openai_response(content="Hello, world!")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should return full response object
        assert response == mock_response
        assert hasattr(response, 'choices')
        assert response.choices[0].message.content == "Hello, world!"
    
    def test_chat_completion_method_level_full_response(self, toolflow_openai_client, mock_openai_client):
        """Test method-level full_response override."""
        mock_response = create_openai_response(content="Hello, world!")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Override at method level
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            full_response=True
        )
        
        # Should return full response despite client default
        assert response == mock_response
    
    def test_standard_parameters_passthrough(self, toolflow_openai_client, mock_openai_client):
        """Test that standard OpenAI parameters pass through correctly."""
        mock_response = create_openai_response(content="Response")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )
        
        # Verify all parameters were passed through
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )


class TestToolExecution:
    """Test tool execution with OpenAI."""
    
    def test_single_tool_execution(self, toolflow_openai_client, mock_openai_client):
        """Test execution of a single tool."""
        # First response: model wants to call tool
        tool_call = create_openai_tool_call("call_123", "simple_math_tool", {"a": 5, "b": 3})
        mock_response_1 = create_openai_response(tool_calls=[tool_call])
        
        # Second response: model responds with result
        mock_response_2 = create_openai_response(content="The answer is 8")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 5 + 3?"}],
            tools=[simple_math_tool]
        )
        
        assert response == "The answer is 8"
        
        # Should have made two calls
        assert mock_openai_client.chat.completions.create.call_count == 2
        
        # Check second call includes tool result
        second_call_args = mock_openai_client.chat.completions.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        
        # Should have original message, assistant message, and tool result
        assert len(messages) == 3
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'
        assert messages[2]['role'] == 'tool'
        assert messages[2]['tool_call_id'] == 'call_123'
        assert messages[2]['content'] == '8.0'  # Result of 5 + 3 (float precision preserved)
    
    def test_multiple_tool_execution(self, toolflow_openai_client, mock_openai_client):
        """Test execution of multiple tools in sequence."""
        # First response: model wants to call multiple tools
        tool_call_1 = create_openai_tool_call("call_123", "simple_math_tool", {"a": 5, "b": 3})
        tool_call_2 = create_openai_tool_call("call_456", "weather_tool", {"city": "NYC"})
        mock_response_1 = create_openai_response(tool_calls=[tool_call_1, tool_call_2])
        
        # Second response: model responds with results
        mock_response_2 = create_openai_response(content="Math result is 8, NYC weather is sunny")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 5 + 3 and what's the weather in NYC?"}],
            tools=[simple_math_tool, weather_tool]
        )
        
        assert response == "Math result is 8, NYC weather is sunny"
        assert mock_openai_client.chat.completions.create.call_count == 2
        
        # Check that both tool results are included
        second_call_args = mock_openai_client.chat.completions.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        
        # Should have user message, assistant message, and two tool results
        assert len(messages) == 4
        tool_messages = [msg for msg in messages if msg['role'] == 'tool']
        assert len(tool_messages) == 2
    
    def test_parallel_tool_execution(self, toolflow_openai_client, mock_openai_client):
        """Test parallel tool execution enabled."""
        tool_call_1 = create_openai_tool_call("call_123", "simple_math_tool", {"a": 10, "b": 5})
        tool_call_2 = create_openai_tool_call("call_456", "weather_tool", {"city": "Boston"})
        mock_response_1 = create_openai_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_openai_response(content="Results computed")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # Enable parallel execution
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 10+5 and get Boston weather"}],
            tools=[simple_math_tool, weather_tool],
            parallel_tool_execution=True
        )
        
        assert response == "Results computed"
        assert mock_openai_client.chat.completions.create.call_count == 2
    
    def test_tool_execution_without_tools_parameter(self, toolflow_openai_client, mock_openai_client):
        """Test that tool execution is skipped when no tools provided."""
        mock_response = create_openai_response(content="No tools available")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
            # No tools parameter
        )
        
        assert response == "No tools available"
        assert mock_openai_client.chat.completions.create.call_count == 1
    
    def test_no_tool_calls_in_response(self, toolflow_openai_client, mock_openai_client):
        """Test when model doesn't call any tools."""
        mock_response = create_openai_response(content="I don't need to use any tools")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Just say hello"}],
            tools=[simple_math_tool]
        )
        
        assert response == "I don't need to use any tools"
        assert mock_openai_client.chat.completions.create.call_count == 1


class TestErrorHandling:
    """Test error handling in OpenAI provider."""
    
    def test_tool_execution_error(self, toolflow_openai_client, mock_openai_client):
        """Test handling of tool execution errors."""
        @tool
        def failing_tool(should_fail: bool = True) -> str:
            """A tool that fails."""
            if should_fail:
                raise ValueError("Tool failed intentionally")
            return "Success"
        
        # Model wants to call failing tool
        tool_call = create_openai_tool_call("call_123", "failing_tool", {"should_fail": True})
        mock_response_1 = create_openai_response(tool_calls=[tool_call])
        
        # Model responds after receiving error
        mock_response_2 = create_openai_response(content="I encountered an error")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Use the failing tool"}],
            tools=[failing_tool]
        )
        
        assert response == "I encountered an error"
        
        # Check that error was passed to model
        second_call_args = mock_openai_client.chat.completions.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        tool_message = [msg for msg in messages if msg['role'] == 'tool'][0]
        assert "Tool failed intentionally" in tool_message['content']
    
    def test_invalid_tool_arguments(self, toolflow_openai_client, mock_openai_client):
        """Test handling of invalid tool arguments."""
        # Model tries to call tool with wrong arguments
        tool_call = create_openai_tool_call("call_123", "simple_math_tool", {"x": 5})  # Wrong param name
        mock_response_1 = create_openai_response(tool_calls=[tool_call])
        mock_response_2 = create_openai_response(content="There was an argument error")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Add numbers"}],
            tools=[simple_math_tool]
        )
        
        assert response == "There was an argument error"
        
        # Should still make second call with error message
        assert mock_openai_client.chat.completions.create.call_count == 2
    
    def test_unknown_tool_call(self, toolflow_openai_client, mock_openai_client):
        """Test handling of calls to unknown tools."""
        # Model tries to call non-existent tool
        tool_call = create_openai_tool_call("call_123", "unknown_tool", {"param": "value"})
        mock_response_1 = create_openai_response(tool_calls=[tool_call])
        mock_response_2 = create_openai_response(content="Unknown tool error handled")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Use unknown tool"}],
            tools=[simple_math_tool]
        )
        
        assert response == "Unknown tool error handled"
        assert mock_openai_client.chat.completions.create.call_count == 2 