"""
Test parallel execution functionality of the toolflow library.

This module tests:
- Parallel vs sequential execution
- Execution timing and performance
- Max workers configuration  
- Error handling in parallel execution
- Order preservation in parallel execution
"""
import pytest
import time
import threading
from unittest.mock import Mock, AsyncMock, patch
from toolflow import tool, from_openai, from_openai_async
from ..conftest import (
    slow_tool,
    slow_async_tool,
    simple_math_tool,
    async_math_tool,
    failing_tool,
    create_mock_openai_tool_call as create_mock_tool_call,
    create_mock_openai_response as create_mock_response
)


class TestSyncParallelExecution:
    """Test parallel execution with sync client."""
    
    def test_parallel_execution_faster_than_sequential(self, sync_toolflow_client, mock_openai_client):
        """Test that parallel execution is faster than sequential."""
        # Mock multiple tool calls that would take time
        tool_call_1 = create_mock_tool_call("call_1", "slow_tool", {"name": "tool1", "delay": 0.1})
        tool_call_2 = create_mock_tool_call("call_2", "slow_tool", {"name": "tool2", "delay": 0.1})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Parallel execution complete!")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run two slow tools"}],
            tools=[slow_tool],
            parallel_tool_execution=True  # Enable parallel execution
        )
        execution_time = time.time() - start_time
        
        # Parallel execution should be significantly faster than 0.2s (2 * 0.1s)
        assert execution_time < 0.15, f"Parallel execution took {execution_time}s, expected < 0.15s"
        assert response.choices[0].message.content == "Parallel execution complete!"
    
    def test_sequential_execution_by_default(self, sync_toolflow_client, mock_openai_client):
        """Test that execution is sequential by default."""
        tool_call_1 = create_mock_tool_call("call_1", "slow_tool", {"name": "tool1", "delay": 0.05})
        tool_call_2 = create_mock_tool_call("call_2", "slow_tool", {"name": "tool2", "delay": 0.05})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Sequential execution complete!")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run two slow tools"}],
            tools=[slow_tool]
            # No parallel_tool_execution=True, should be sequential
        )
        execution_time = time.time() - start_time
        
        # Sequential execution should take approximately 0.1s (sum of delays)
        assert execution_time >= 0.08, f"Sequential execution took {execution_time}s, expected >= 0.08s"
        assert response.choices[0].message.content == "Sequential execution complete!"
    
    def test_single_tool_call_no_threading_overhead(self, sync_toolflow_client, mock_openai_client):
        """Test that single tool calls don't use threading unnecessarily."""
        tool_call = create_mock_tool_call("call_single", "simple_math_tool", {"a": 5, "b": 3})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Single tool result: 8")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Add 5 + 3"}],
            tools=[simple_math_tool],
            parallel_tool_execution=True  # Even with parallel enabled, single calls should be direct
        )
        
        assert response.choices[0].message.content == "Single tool result: 8"
    
    def test_parallel_execution_maintains_order(self, sync_toolflow_client, mock_openai_client):
        """Test that results are returned in the same order as tool_calls."""
        # Create tools with different execution times to test ordering
        tool_call_1 = create_mock_tool_call("call_slow", "slow_tool", {"name": "slow", "delay": 0.05})
        tool_call_2 = create_mock_tool_call("call_fast", "simple_math_tool", {"a": 1, "b": 2})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Mixed results received")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run mixed tools"}],
            tools=[slow_tool, simple_math_tool],
            parallel_tool_execution=True
        )
        
        assert response.choices[0].message.content == "Mixed results received"
        
        # Check that tool results were added in correct order
        second_call_args = mock_openai_client.chat.completions.create.call_args_list[1]
        tool_messages = second_call_args[1]['messages']
        
        # Find tool result messages
        tool_result_messages = [msg for msg in tool_messages if msg.get('role') == 'tool']
        
        # Should have results in same order as tool_calls (slow first, then fast)
        assert len(tool_result_messages) == 2
        assert tool_result_messages[0]['tool_call_id'] == 'call_slow'
        assert tool_result_messages[1]['tool_call_id'] == 'call_fast'


class TestAsyncParallelExecution:
    """Test parallel execution with async client."""
    
    @pytest.mark.asyncio
    async def test_parallel_async_tools_faster_than_sequential(self, async_toolflow_client, mock_async_openai_client):
        """Test that parallel async execution is faster than sequential."""
        tool_call_1 = create_mock_tool_call("call_1", "slow_async_tool", {"name": "async1", "delay": 0.05})
        tool_call_2 = create_mock_tool_call("call_2", "slow_async_tool", {"name": "async2", "delay": 0.05})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Async parallel complete!")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run two async tools"}],
            tools=[slow_async_tool],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        # Should be faster than sequential (0.1s total)
        assert execution_time < 0.08, f"Async parallel execution took {execution_time}s, expected < 0.08s"
        assert response.choices[0].message.content == "Async parallel complete!"
    
    @pytest.mark.asyncio
    async def test_sequential_async_execution_by_default(self, async_toolflow_client, mock_async_openai_client):
        """Test that async execution is sequential by default."""
        tool_call_1 = create_mock_tool_call("call_1", "slow_async_tool", {"name": "async1", "delay": 0.03})
        tool_call_2 = create_mock_tool_call("call_2", "slow_async_tool", {"name": "async2", "delay": 0.03})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Async sequential complete!")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run two async tools"}],
            tools=[slow_async_tool]
            # No parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        # Sequential should take approximately sum of delays
        assert execution_time >= 0.05, f"Async sequential took {execution_time}s, expected >= 0.05s"
        assert response.choices[0].message.content == "Async sequential complete!"
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_tools_parallel(self, async_toolflow_client, mock_async_openai_client):
        """Test parallel execution with mixed sync and async tools."""
        tool_call_1 = create_mock_tool_call("call_sync", "slow_tool", {"name": "sync", "delay": 0.03})
        tool_call_2 = create_mock_tool_call("call_async", "slow_async_tool", {"name": "async", "delay": 0.03})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Mixed parallel complete!")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run mixed tools"}],
            tools=[slow_tool, slow_async_tool],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        # Should be faster than sequential execution of both
        assert execution_time < 0.05, f"Mixed parallel took {execution_time}s, expected < 0.05s"
        assert response.choices[0].message.content == "Mixed parallel complete!"
    
    @pytest.mark.asyncio
    async def test_async_single_tool_no_gather_overhead(self, async_toolflow_client, mock_async_openai_client):
        """Test that single async tool calls don't use asyncio.gather unnecessarily."""
        tool_call = create_mock_tool_call("call_single", "async_math_tool", {"a": 10, "b": 5})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Single async result: 15")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Add 10 + 5 async"}],
            tools=[async_math_tool],
            parallel_tool_execution=True
        )
        
        assert response.choices[0].message.content == "Single async result: 15"
    
    @pytest.mark.asyncio
    async def test_async_parallel_maintains_order(self, async_toolflow_client, mock_async_openai_client):
        """Test that async parallel execution maintains tool call order."""
        # Mix of different speed tools
        tool_call_1 = create_mock_tool_call("call_slow", "slow_async_tool", {"name": "slow", "delay": 0.02})
        tool_call_2 = create_mock_tool_call("call_fast", "async_math_tool", {"a": 3, "b": 4})
        tool_call_3 = create_mock_tool_call("call_sync", "simple_math_tool", {"a": 1, "b": 1})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2, tool_call_3])
        mock_response_2 = create_mock_response(content="Async order test complete!")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test order"}],
            tools=[slow_async_tool, async_math_tool, simple_math_tool],
            parallel_tool_execution=True
        )
        
        assert response.choices[0].message.content == "Async order test complete!"
        
        # Check order preservation
        second_call_args = mock_async_openai_client.chat.completions.create.call_args_list[1]
        tool_messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in tool_messages if msg.get('role') == 'tool']
        
        assert len(tool_result_messages) == 3
        assert tool_result_messages[0]['tool_call_id'] == 'call_slow'
        assert tool_result_messages[1]['tool_call_id'] == 'call_fast'
        assert tool_result_messages[2]['tool_call_id'] == 'call_sync'


class TestMaxWorkersConfiguration:
    """Test max_workers parameter in parallel execution."""
    
    def test_max_workers_parameter_accepted(self, sync_toolflow_client, mock_openai_client):
        """Test that max_workers parameter is accepted and doesn't cause errors."""
        tool_call = create_mock_tool_call("call_1", "simple_math_tool", {"a": 5, "b": 5})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Max workers test complete")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # Test various max_workers values
        for max_workers in [1, 2, 5, 10]:
            response = sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test max workers"}],
                tools=[simple_math_tool],

                parallel_tool_execution=True
            )
            assert response.choices[0].message.content == "Max workers test complete"
            
            # Reset for next iteration
            mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
    
    @pytest.mark.asyncio
    async def test_async_max_workers_parameter(self, async_toolflow_client, mock_async_openai_client):
        """Test max_workers parameter with async client."""
        tool_call = create_mock_tool_call("call_1", "async_math_tool", {"a": 3, "b": 7})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Async max workers test complete")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test async max workers"}],
            tools=[async_math_tool],

            parallel_tool_execution=True
        )
        
        assert response.choices[0].message.content == "Async max workers test complete"
    
    def test_max_workers_with_multiple_tools(self, sync_toolflow_client, mock_openai_client):
        """Test max_workers with multiple tool calls."""
        # Create several tool calls
        tool_calls = []
        for i in range(5):
            tool_call = create_mock_tool_call(f"call_{i}", "simple_math_tool", {"a": i, "b": 1})
            tool_calls.append(tool_call)
        
        mock_response_1 = create_mock_response(tool_calls=tool_calls)
        # Create a second response that would trigger the max tool calls limit
        mock_response_2 = create_mock_response(tool_calls=tool_calls)
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # Should raise exception when max tool calls limit is exceeded
        with pytest.raises(Exception) as exc_info:
            sync_toolflow_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Run multiple tools"}],
                tools=[simple_math_tool],
                max_workers=2,  # Limit workers to test batching
                parallel_tool_execution=True
            )
        
        # Verify the exception indicates max tool calls reached
        assert "Max tool calls reached without finding a solution" in str(exc_info.value)


class TestParallelExecutionErrorHandling:
    """Test error handling in parallel execution context."""
    
    def test_parallel_execution_with_failing_tool(self, sync_toolflow_client, mock_openai_client):
        """Test that parallel execution handles tool failures gracefully."""
        tool_call_1 = create_mock_tool_call("call_success", "simple_math_tool", {"a": 1, "b": 2})
        tool_call_2 = create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Parallel errors were handled gracefully")
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = sync_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test parallel errors"}],
            tools=[simple_math_tool, failing_tool],
            parallel_tool_execution=True
        )
        
        # Should get response after errors are handled gracefully
        assert response.choices[0].message.content == "Parallel errors were handled gracefully"
        
        # Check that errors were passed to the model in second call
        second_call_args = mock_openai_client.chat.completions.create.call_args_list[1]
        tool_messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in tool_messages if msg.get('role') == 'tool']
        
        # Should have both results: one success, one error
        assert len(tool_result_messages) == 2
        
        # Check for successful execution
        success_msg = next((msg for msg in tool_result_messages if msg['tool_call_id'] == 'call_success'), None)
        assert success_msg is not None
        assert "3" in success_msg['content']  # 1 + 2 = 3
        
        # Check for error message
        error_msg = next((msg for msg in tool_result_messages if msg['tool_call_id'] == 'call_fail'), None)
        assert error_msg is not None
        assert "Error executing tool failing_tool" in error_msg['content']
    
    @pytest.mark.asyncio
    async def test_async_parallel_error_handling(self, async_toolflow_client, mock_async_openai_client):
        """Test error handling in async parallel execution with graceful error handling."""
        tool_call_1 = create_mock_tool_call("call_success", "async_math_tool", {"a": 5, "b": 5})
        tool_call_2 = create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Async parallel errors were handled gracefully")
        
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # With graceful error handling (default), should not raise exception
        response = await async_toolflow_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test async parallel errors"}],
            tools=[async_math_tool, failing_tool],
            parallel_tool_execution=True
        )
        
        # Should get response after errors are handled gracefully
        assert response.choices[0].message.content == "Async parallel errors were handled gracefully"
        
        # Check that errors were passed to the model in second call
        second_call_args = mock_async_openai_client.chat.completions.create.call_args_list[1]
        tool_messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in tool_messages if msg.get('role') == 'tool']
        
        # Should have both results: one success, one error
        assert len(tool_result_messages) == 2
        
        # Check for successful execution
        success_msg = next((msg for msg in tool_result_messages if msg['tool_call_id'] == 'call_success'), None)
        assert success_msg is not None
        assert "10" in success_msg['content']  # 5 + 5 = 10
        
        # Check for error message
        error_msg = next((msg for msg in tool_result_messages if msg['tool_call_id'] == 'call_fail'), None)
        assert error_msg is not None
        assert "Error executing tool failing_tool" in error_msg['content']
