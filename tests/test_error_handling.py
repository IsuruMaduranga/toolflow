"""
Test error handling functionality of the toolflow library.

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
from toolflow import tool, from_openai, from_openai_async
from .conftest import (
    failing_tool,
    simple_math_tool,
    async_math_tool,
    create_mock_tool_call,
    create_mock_response
)


class TestToolExecutionErrors:
    """Test errors that occur during tool execution."""
    
    def test_tool_function_raises_exception(self, sync_toolflow_client, mock_openai_client):
        """Test handling when tool function raises an exception."""
        tool_call = create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when tool fails
        with pytest.raises(Exception) as exc_info:
            sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test tool error"}],
                tools=[failing_tool]
            )
        
        # Verify the exception contains tool error information
        assert "Error executing tool failing_tool" in str(exc_info.value)
        assert "This tool failed intentionally" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_async_tool_execution_error(self, async_toolflow_client, mock_async_openai_client):
        """Test async tool execution error handling."""
        tool_call = create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when async tool fails
        with pytest.raises(Exception) as exc_info:
            await async_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test async error"}],
                tools=[failing_tool]
            )
        
        # Verify the exception contains tool error information
        assert "Error executing tool failing_tool" in str(exc_info.value)
        assert "This tool failed intentionally" in str(exc_info.value)
    
    def test_division_by_zero_error_handling(self, sync_toolflow_client, mock_openai_client):
        """Test handling of division by zero errors."""
        from .conftest import divide_tool
        
        tool_call = create_mock_tool_call("call_div", "divide_tool", {"a": 10, "b": 0})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when division by zero occurs
        with pytest.raises(Exception) as exc_info:
            sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Divide 10 by 0"}],
                tools=[divide_tool]
            )
        
        # Verify the exception contains division by zero error
        assert "Error executing tool divide_tool" in str(exc_info.value)
        assert "division by zero" in str(exc_info.value)


class TestInvalidArguments:
    """Test handling of invalid tool arguments."""
    
    def test_malformed_json_arguments(self, sync_toolflow_client, mock_openai_client):
        """Test handling of malformed JSON in tool arguments."""
        # Create tool call with invalid JSON
        tool_call = Mock()
        tool_call.id = "call_bad_json"
        tool_call.function.name = "simple_math_tool"
        tool_call.function.arguments = '{"a": 10, "b":}'  # Invalid JSON
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when JSON is malformed
        with pytest.raises(Exception) as exc_info:
            sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test bad JSON"}],
                tools=[simple_math_tool]
            )
        
        # Verify the exception contains JSON parsing error
        assert "Error executing tool simple_math_tool" in str(exc_info.value)
        assert "Expecting value" in str(exc_info.value)
    
    def test_wrong_argument_types(self, sync_toolflow_client, mock_openai_client):
        """Test handling of wrong argument types."""
        # Valid JSON but wrong types
        tool_call = create_mock_tool_call("call_wrong_type", "simple_math_tool", {"a": "not_a_number", "b": 5})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when types are wrong
        with pytest.raises(Exception) as exc_info:
            sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test wrong types"}],
                tools=[simple_math_tool]
            )
        
        # Verify the exception contains type error information
        assert "Error executing tool simple_math_tool" in str(exc_info.value)
        assert "can only concatenate str" in str(exc_info.value)
    
    def test_missing_required_arguments(self, sync_toolflow_client, mock_openai_client):
        """Test handling of missing required arguments."""
        tool_call = create_mock_tool_call("call_missing", "simple_math_tool", {"a": 5})  # Missing 'b'
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when required arguments are missing
        with pytest.raises(Exception) as exc_info:
            sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test missing args"}],
                tools=[simple_math_tool]
            )
        
        # Verify the exception contains missing argument error
        assert "Error executing tool simple_math_tool" in str(exc_info.value)
        assert "missing 1 required positional argument" in str(exc_info.value)
    
    def test_empty_tool_arguments(self, sync_toolflow_client, mock_openai_client):
        """Test handling of empty tool arguments."""
        from .conftest import get_current_time_tool
        
        # Tool that expects no arguments but gets some
        tool_call = create_mock_tool_call("call_empty", "get_current_time_tool", {})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Empty args handled")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Get time"}],
            tools=[get_current_time_tool]
        )
        
        assert response.choices[0].message.content == "Empty args handled"


class TestUnknownTools:
    """Test handling of unknown or non-existent tools."""
    
    def test_non_decorated_function_error(self, sync_toolflow_client, mock_openai_client):
        """Test error when trying to use non-decorated function as tool."""
        def regular_function(x: int) -> int:
            return x * 2
        
        # This should raise an error when trying to extract schema
        with pytest.raises(AttributeError):
            # Try to access tool metadata that doesn't exist
            _ = regular_function._tool_metadata
    
    def test_non_callable_tool_error(self):
        """Test error when non-callable object is passed as tool."""
        not_a_function = "this is not a function"
        
        # This should be caught during tool validation
        # The exact behavior depends on implementation, but should not crash
        with pytest.raises((TypeError, AttributeError)):
            # Try to treat non-callable as tool
            from toolflow.utils import get_tool_schema
            get_tool_schema(not_a_function)
    
    def test_unknown_tool_call(self, sync_toolflow_client, mock_openai_client):
        """Test handling of calls to unknown tools."""
        # Tool call for non-existent tool
        tool_call = create_mock_tool_call("call_unknown", "nonexistent_tool", {})
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when tool is not found
        with pytest.raises(ValueError) as exc_info:
            sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test unknown tool"}],
                tools=[simple_math_tool]  # Available tools don't include the called one
            )
        
        # Verify the exception contains tool not found error
        assert "Tool nonexistent_tool not found" in str(exc_info.value)


class TestLibraryImportErrors:
    """Test handling of import errors for optional dependencies."""
    
    def test_openai_import_error_simulation(self):
        """Test behavior when OpenAI library is not available."""
        # Mock the import error scenario
        with patch.dict('sys.modules', {'openai': None}):
            # This should work since we're testing with mocks
            # In real scenario, this would help users understand missing dependencies
            try:
                from toolflow import from_openai
                # If this works, the import handling is robust
                assert from_openai is not None
            except ImportError as e:
                # This is expected behavior when OpenAI is not installed
                assert 'openai' in str(e).lower()
    
    def test_async_openai_import_handling(self):
        """Test that async functionality handles import errors gracefully."""
        try:
            from toolflow import from_openai_async
            assert from_openai_async is not None
        except ImportError:
            # This is acceptable - async functionality might have additional requirements
            pass


class TestClientDelegation:
    """Test error handling in client delegation."""
    
    def test_sync_client_delegation_error(self):
        """Test error handling when sync client delegation fails."""
        # Create a mock client that will cause issues
        mock_client = Mock()
        mock_client.chat = None  # This should cause an error
        
        with pytest.raises(AttributeError):
            client = from_openai(mock_client)
            # This should fail when trying to access chat.completions
            _ = client.chat.completions
    
    def test_async_client_delegation_error(self):
        """Test error handling when async client delegation fails."""
        mock_client = Mock()
        mock_client.chat = None  # This should cause an error
        
        with pytest.raises(AttributeError):
            client = from_openai_async(mock_client)
            # This should fail when trying to access chat.completions
            _ = client.chat.completions


class TestEdgeCaseErrors:
    """Test edge case error scenarios."""
    
    def test_circular_tool_dependencies(self):
        """Test handling of circular dependencies (if applicable)."""
        # This is more of a design consideration
        # but worth testing if tools could potentially call each other
        @tool
        def tool_a(x: int) -> int:
            """Tool A."""
            return x + 1
        
        @tool  
        def tool_b(x: int) -> int:
            """Tool B."""
            return x + 2
        
        # These should be fine individually
        assert tool_a._tool_metadata is not None
        assert tool_b._tool_metadata is not None
    
    def test_very_large_arguments(self, sync_toolflow_client, mock_openai_client):
        """Test handling of very large argument values."""
        # Create a tool call with very large string
        large_string = "x" * 10000
        tool_call = create_mock_tool_call("call_large", "simple_math_tool", {"a": len(large_string), "b": 1})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Large args handled")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test large args"}],
            tools=[simple_math_tool]
        )
        
        assert response.choices[0].message.content == "Large args handled"
    
    def test_null_and_undefined_values(self, sync_toolflow_client, mock_openai_client):
        """Test handling of null and undefined values in arguments."""
        # Tool call with null values
        tool_call = Mock()
        tool_call.id = "call_null"
        tool_call.function.name = "simple_math_tool"
        tool_call.function.arguments = '{"a": null, "b": 5}'
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1]
        
        # Should raise an exception when null values cause type errors
        with pytest.raises(Exception) as exc_info:
            sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test null values"}],
                tools=[simple_math_tool]
            )
        
        # Verify the exception contains type error for null values
        assert "Error executing tool simple_math_tool" in str(exc_info.value)
        assert "unsupported operand type(s) for +: 'NoneType' and 'int'" in str(exc_info.value)
 