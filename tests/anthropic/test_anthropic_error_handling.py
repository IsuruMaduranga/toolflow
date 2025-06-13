"""
Test error handling functionality of the toolflow library with Anthropic.

This module tests:
- Tool execution errors
- Invalid tool arguments
- Unknown tool calls
- JSON parsing errors
- Library import errors
- Client delegation errors
"""
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from toolflow import tool, from_anthropic, from_anthropic_async
from ..conftest import (
    failing_tool,
    simple_math_tool,
    async_math_tool,
    divide_tool,
    create_mock_anthropic_tool_call as create_mock_tool_call,
    create_mock_anthropic_response as create_mock_response
)


class TestAnthropicToolExecutionErrors:
    """Test errors that occur during tool execution."""
    
    def test_tool_function_raises_exception(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling when tool function raises an exception with graceful error handling."""
        tool_call = create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Tool error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test tool error"}],
            tools=[failing_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Tool error was handled gracefully" for block in response.content if hasattr(block, 'text'))
        
        # Check that error was passed to the model in second call
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in messages if msg.get('role') == 'user' and 'tool_result' in str(msg.get('content', []))]
        
        assert len(tool_result_messages) == 1
        tool_results = tool_result_messages[0]['content']
        error_found = any('Error executing tool failing_tool' in str(result) for result in tool_results if isinstance(result, dict))
        assert error_found
    
    def test_tool_function_raises_exception_without_graceful_handling(self, sync_anthropic_client, mock_anthropic_client):
        """Test that exceptions are raised when graceful error handling is disabled."""
        tool_call = create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1]
        
        # Should raise an exception when graceful error handling is disabled
        with pytest.raises(Exception) as exc_info:
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "Test tool error"}],
                tools=[failing_tool],
                graceful_error_handling=False
            )
        
        # Verify the exception contains tool error information
        assert "Error executing tool failing_tool" in str(exc_info.value)
        assert "This tool failed intentionally" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_async_tool_execution_error(self, async_anthropic_client, mock_async_anthropic_client):
        """Test async tool execution error handling with graceful error handling."""
        tool_call = create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Async tool error was handled gracefully")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test async error"}],
            tools=[failing_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Async tool error was handled gracefully" for block in response.content if hasattr(block, 'text'))
        
        # Check that error was passed to the model in second call
        second_call_args = mock_async_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in messages if msg.get('role') == 'user' and 'tool_result' in str(msg.get('content', []))]
        
        assert len(tool_result_messages) == 1
    
    def test_division_by_zero_error_handling(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of division by zero errors with graceful error handling."""
        tool_call = create_mock_tool_call("call_div", "divide_tool", {"a": 10, "b": 0})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Division error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Divide 10 by 0"}],
            tools=[divide_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Division error was handled gracefully" for block in response.content if hasattr(block, 'text'))
        
        # Check that error was passed to the model in second call
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in messages if msg.get('role') == 'user' and 'tool_result' in str(msg.get('content', []))]
        
        assert len(tool_result_messages) == 1


class TestAnthropicInvalidArguments:
    """Test handling of invalid tool arguments."""
    
    def test_malformed_json_arguments(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of malformed JSON in tool arguments with graceful error handling."""
        # Create tool call with invalid JSON input
        tool_call = Mock()
        tool_call.id = "call_bad_json"
        tool_call.name = "simple_math_tool"
        tool_call.input = {"a": 10, "b": None}  # Invalid input that will cause issues
        tool_call.type = "tool_use"
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="JSON error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test bad JSON"}],
            tools=[simple_math_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "JSON error was handled gracefully" for block in response.content if hasattr(block, 'text'))
    
    def test_wrong_argument_types(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of wrong argument types with graceful error handling."""
        # Valid input but wrong types
        tool_call = create_mock_tool_call("call_wrong_type", "simple_math_tool", {"a": "not_a_number", "b": 5})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Type error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test wrong types"}],
            tools=[simple_math_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Type error was handled gracefully" for block in response.content if hasattr(block, 'text'))
    
    def test_missing_required_arguments(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of missing required arguments with graceful error handling."""
        tool_call = create_mock_tool_call("call_missing", "simple_math_tool", {"a": 5})  # Missing 'b'
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Missing argument error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test missing args"}],
            tools=[simple_math_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Missing argument error was handled gracefully" for block in response.content if hasattr(block, 'text'))
    
    def test_empty_tool_arguments(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of empty tool arguments with graceful error handling."""
        tool_call = create_mock_tool_call("call_empty", "simple_math_tool", {})  # Empty arguments
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Empty argument error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test empty args"}],
            tools=[simple_math_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Empty argument error was handled gracefully" for block in response.content if hasattr(block, 'text'))


class TestAnthropicUnknownTools:
    """Test handling of unknown tools."""
    
    def test_non_decorated_function_error(self, sync_anthropic_client, mock_anthropic_client):
        """Test that non-decorated functions can't be used as tools."""
        def regular_function(x: int) -> int:
            return x * 2
        
        # Should raise an error when trying to use non-decorated function
        with pytest.raises(ValueError, match="All tools must be decorated with @tool"):
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "Test"}],
                tools=[regular_function]
            )
    
    def test_non_callable_tool_error(self):
        """Test that non-callable objects can't be used as tools."""
        with pytest.raises(ValueError, match="All tools must be decorated with @tool"):
            # Should raise an error during tool validation
            from toolflow.providers.anthropic.tool_execution import validate_and_prepare_anthropic_tools
            validate_and_prepare_anthropic_tools(["not_a_function"])
    
    def test_unknown_tool_call(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of unknown tool calls with graceful error handling."""
        # Tool call for non-existent tool
        tool_call = create_mock_tool_call("call_unknown", "nonexistent_tool", {})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Unknown tool error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test unknown tool"}],
            tools=[simple_math_tool]  # Only provide simple_math_tool, not nonexistent_tool
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Unknown tool error was handled gracefully" for block in response.content if hasattr(block, 'text'))


class TestAnthropicLibraryImportErrors:
    """Test handling of library import errors."""
    
    def test_anthropic_import_error_simulation(self):
        """Test handling when Anthropic library is not available."""
        with patch('toolflow.providers.anthropic.ANTHROPIC_AVAILABLE', False):
            with pytest.raises(ImportError, match="Anthropic library not installed"):
                from toolflow.providers.anthropic import from_anthropic
                from unittest.mock import Mock
                mock_client = Mock()
                from_anthropic(mock_client)
    
    def test_async_anthropic_import_handling(self):
        """Test handling when Anthropic library is not available for async client."""
        with patch('toolflow.providers.anthropic.ANTHROPIC_AVAILABLE', False):
            with pytest.raises(ImportError, match="Anthropic library not installed"):
                from toolflow.providers.anthropic import from_anthropic_async
                from unittest.mock import Mock
                mock_client = Mock()
                from_anthropic_async(mock_client)


class TestAnthropicClientDelegation:
    """Test client delegation and validation errors."""
    
    def test_sync_client_delegation_error(self):
        """Test error when wrong client type is passed to sync wrapper."""
        try:
            import anthropic
            with pytest.raises(TypeError, match="Expected synchronous Anthropic client, got AsyncAnthropic"):
                from_anthropic(anthropic.AsyncAnthropic(api_key="test"))
        except ImportError:
            pytest.skip("Anthropic library not available")
    
    def test_async_client_delegation_error(self):
        """Test error when wrong client type is passed to async wrapper."""
        try:
            import anthropic
            with pytest.raises(TypeError, match="Expected asynchronous AsyncAnthropic client, got Anthropic"):
                from_anthropic_async(anthropic.Anthropic(api_key="test"))
        except ImportError:
            pytest.skip("Anthropic library not available")


class TestAnthropicEdgeCaseErrors:
    """Test edge case error scenarios."""
    
    def test_circular_tool_dependencies(self):
        """Test that circular tool dependencies don't cause infinite loops."""
        # This is more of a theoretical test since tools don't call each other directly
        # But we can test that the system handles complex tool scenarios
        
        @tool
        def tool_a(x: int) -> int:
            return x + 1
        
        @tool  
        def tool_b(x: int) -> int:
            return x - 1
        
        # Should not raise any errors during tool preparation
        from toolflow.providers.anthropic.tool_execution import validate_and_prepare_anthropic_tools
        tool_functions, tool_schemas = validate_and_prepare_anthropic_tools([tool_a, tool_b])
        assert len(tool_functions) == 2
        assert len(tool_schemas) == 2
    
    def test_very_large_arguments(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of very large arguments with graceful error handling."""
        # Create a very large string
        large_string = "x" * 100000  # 100k characters
        
        tool_call = create_mock_tool_call("call_large", "simple_math_tool", {"a": large_string, "b": 5})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Large argument error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test large args"}],
            tools=[simple_math_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Large argument error was handled gracefully" for block in response.content if hasattr(block, 'text'))
    
    def test_null_and_undefined_values(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of null and undefined values with graceful error handling."""
        tool_call = create_mock_tool_call("call_null", "simple_math_tool", {"a": None, "b": 5})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Null value error was handled gracefully")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Test null values"}],
            tools=[simple_math_tool]
        )
        
        # Should get response after error is handled gracefully
        assert any(block.text == "Null value error was handled gracefully" for block in response.content if hasattr(block, 'text'))


class TestAnthropicMaxTokensHandling:
    """Test handling of max tokens limit scenarios."""
    
    def test_max_tokens_reached_error(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling when max tokens limit is reached."""
        # Create a response that indicates max_tokens was reached
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Partial response")]
        mock_response.stop_reason = "max_tokens"
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Should raise an exception when max tokens is reached
        with pytest.raises(Exception, match="Max tokens reached without finding a solution"):
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,  # Very low limit
                messages=[{"role": "user", "content": "Long response needed"}],
                tools=[simple_math_tool]
            )
    
    def test_max_tool_calls_limit(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling when max tool calls limit is reached."""
        # Create responses that always include tool calls to trigger the limit
        tool_call = create_mock_tool_call("call_loop", "simple_math_tool", {"a": 1, "b": 2})
        mock_response = create_mock_response(tool_calls=[tool_call])
        
        # Make the client always return the same response with tool calls
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Should raise an exception when max tool calls is reached
        with pytest.raises(Exception, match="Max tool calls reached"):
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "Keep calling tools"}],
                tools=[simple_math_tool],
                max_tool_calls=2  # Low limit to trigger quickly
            ) 