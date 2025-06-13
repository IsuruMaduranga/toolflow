"""
Test parallel execution functionality of the toolflow library with Anthropic.

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
from toolflow import tool, from_anthropic, from_anthropic_async
from ..conftest import (
    slow_tool,
    slow_async_tool,
    simple_math_tool,
    async_math_tool,
    failing_tool,
    create_mock_anthropic_tool_call as create_mock_tool_call,
    create_mock_anthropic_response as create_mock_response
)


class TestAnthropicSyncParallelExecution:
    """Test parallel execution with sync Anthropic client."""
    
    def test_parallel_execution_faster_than_sequential(self, sync_anthropic_client, mock_anthropic_client):
        """Test that parallel execution is faster than sequential."""
        # Mock multiple tool calls that would take time
        tool_call_1 = create_mock_tool_call("call_1", "slow_tool", {"name": "tool1", "delay": 0.1})
        tool_call_2 = create_mock_tool_call("call_2", "slow_tool", {"name": "tool2", "delay": 0.1})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Parallel execution complete!")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run two slow tools"}],
            tools=[slow_tool],
            parallel_tool_execution=True  # Enable parallel execution
        )
        execution_time = time.time() - start_time
        
        # Parallel execution should be significantly faster than 0.2s (2 * 0.1s)
        assert execution_time < 0.15, f"Parallel execution took {execution_time}s, expected < 0.15s"
        assert any(block.text == "Parallel execution complete!" for block in response.content if hasattr(block, 'text'))
    
    def test_sequential_execution_by_default(self, sync_anthropic_client, mock_anthropic_client):
        """Test that execution is sequential by default."""
        tool_call_1 = create_mock_tool_call("call_1", "slow_tool", {"name": "tool1", "delay": 0.05})
        tool_call_2 = create_mock_tool_call("call_2", "slow_tool", {"name": "tool2", "delay": 0.05})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Sequential execution complete!")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run two slow tools"}],
            tools=[slow_tool]
            # No parallel_tool_execution=True, should be sequential
        )
        execution_time = time.time() - start_time
        
        # Sequential execution should take approximately 0.1s (sum of delays)
        assert execution_time >= 0.08, f"Sequential execution took {execution_time}s, expected >= 0.08s"
        assert any(block.text == "Sequential execution complete!" for block in response.content if hasattr(block, 'text'))
    
    def test_single_tool_call_no_threading_overhead(self, sync_anthropic_client, mock_anthropic_client):
        """Test that single tool calls don't use threading unnecessarily."""
        tool_call = create_mock_tool_call("call_single", "simple_math_tool", {"a": 5, "b": 3})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Single tool result: 8")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Add 5 + 3"}],
            tools=[simple_math_tool],
            parallel_tool_execution=True  # Even with parallel enabled, single calls should be direct
        )
        
        assert any(block.text == "Single tool result: 8" for block in response.content if hasattr(block, 'text'))
    
    def test_parallel_execution_maintains_order(self, sync_anthropic_client, mock_anthropic_client):
        """Test that results are returned in the same order as tool_calls."""
        # Create tools with different execution times to test ordering
        tool_call_1 = create_mock_tool_call("call_slow", "slow_tool", {"name": "slow", "delay": 0.05})
        tool_call_2 = create_mock_tool_call("call_fast", "simple_math_tool", {"a": 1, "b": 2})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Mixed results received")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run mixed tools"}],
            tools=[slow_tool, simple_math_tool],
            parallel_tool_execution=True
        )
        
        assert any(block.text == "Mixed results received" for block in response.content if hasattr(block, 'text'))
        
        # Check that tool results were added in correct order
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        
        # Find tool result messages (Anthropic format)
        tool_result_messages = [msg for msg in messages if msg.get('role') == 'user' and 'tool_result' in str(msg.get('content', []))]
        
        # Should have results in same order as tool_calls (slow first, then fast)
        assert len(tool_result_messages) == 1  # Anthropic groups tool results into single message
        
        # Check the content structure for ordered results
        tool_results = tool_result_messages[0]['content']
        tool_result_ids = [result.get('tool_use_id') for result in tool_results if isinstance(result, dict) and 'tool_use_id' in result]
        
        # Results should be in the same order as the tool calls
        assert tool_result_ids == ['call_slow', 'call_fast']


class TestAnthropicAsyncParallelExecution:
    """Test parallel execution with async Anthropic client."""
    
    @pytest.mark.asyncio
    async def test_parallel_async_tools_faster_than_sequential(self, async_anthropic_client, mock_async_anthropic_client):
        """Test that parallel async execution is faster than sequential."""
        tool_call_1 = create_mock_tool_call("call_1", "slow_async_tool", {"name": "async1", "delay": 0.05})
        tool_call_2 = create_mock_tool_call("call_2", "slow_async_tool", {"name": "async2", "delay": 0.05})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Async parallel complete!")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run two async tools"}],
            tools=[slow_async_tool],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        # Should be faster than sequential (0.1s total)
        assert execution_time < 0.08, f"Async parallel execution took {execution_time}s, expected < 0.08s"
        assert any(block.text == "Async parallel complete!" for block in response.content if hasattr(block, 'text'))
    
    @pytest.mark.asyncio
    async def test_sequential_async_execution_by_default(self, async_anthropic_client, mock_async_anthropic_client):
        """Test that async execution is sequential by default."""
        tool_call_1 = create_mock_tool_call("call_1", "slow_async_tool", {"name": "async1", "delay": 0.03})
        tool_call_2 = create_mock_tool_call("call_2", "slow_async_tool", {"name": "async2", "delay": 0.03})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Async sequential complete!")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run two async tools"}],
            tools=[slow_async_tool]
            # No parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        # Sequential should take approximately sum of delays
        assert execution_time >= 0.05, f"Async sequential took {execution_time}s, expected >= 0.05s"
        assert any(block.text == "Async sequential complete!" for block in response.content if hasattr(block, 'text'))
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_tools_parallel(self, async_anthropic_client, mock_async_anthropic_client):
        """Test parallel execution with mixed sync and async tools."""
        tool_call_1 = create_mock_tool_call("call_sync", "slow_tool", {"name": "sync", "delay": 0.03})
        tool_call_2 = create_mock_tool_call("call_async", "slow_async_tool", {"name": "async", "delay": 0.03})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Mixed parallel complete!")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run mixed tools"}],
            tools=[slow_tool, slow_async_tool],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        # Should be faster than sequential execution of both
        assert execution_time < 0.05, f"Mixed parallel took {execution_time}s, expected < 0.05s"
        assert any(block.text == "Mixed parallel complete!" for block in response.content if hasattr(block, 'text'))
    
    @pytest.mark.asyncio
    async def test_async_single_tool_no_gather_overhead(self, async_anthropic_client, mock_async_anthropic_client):
        """Test that single async tool calls don't use asyncio.gather unnecessarily."""
        tool_call = create_mock_tool_call("call_single", "async_math_tool", {"a": 10, "b": 5})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call])
        mock_response_2 = create_mock_response(content="Single async result: 15")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Add 10 + 5"}],
            tools=[async_math_tool],
            parallel_tool_execution=True
        )
        
        assert any(block.text == "Single async result: 15" for block in response.content if hasattr(block, 'text'))
    
    @pytest.mark.asyncio
    async def test_async_parallel_maintains_order(self, async_anthropic_client, mock_async_anthropic_client):
        """Test that async parallel execution maintains tool call order."""
        tool_call_1 = create_mock_tool_call("call_slow", "slow_async_tool", {"name": "slow", "delay": 0.03})
        tool_call_2 = create_mock_tool_call("call_fast", "async_math_tool", {"a": 2, "b": 3})
        
        mock_response_1 = create_mock_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_mock_response(content="Async mixed results")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run mixed async tools"}],
            tools=[slow_async_tool, async_math_tool],
            parallel_tool_execution=True
        )
        
        assert any(block.text == "Async mixed results" for block in response.content if hasattr(block, 'text'))
        
        # Check that tool results maintain order
        second_call_args = mock_async_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        
        tool_result_messages = [msg for msg in messages if msg.get('role') == 'user' and 'tool_result' in str(msg.get('content', []))]
        assert len(tool_result_messages) == 1
        
        tool_results = tool_result_messages[0]['content']
        tool_result_ids = [result.get('tool_use_id') for result in tool_results if isinstance(result, dict) and 'tool_use_id' in result]
        
        # Order should be preserved
        assert tool_result_ids == ['call_slow', 'call_fast']


class TestAnthropicMaxWorkersConfiguration:
    """Test max workers configuration for parallel execution."""
    
    def test_max_workers_parameter_accepted(self, sync_anthropic_client, mock_anthropic_client):
        """Test that max_workers parameter is accepted and used."""
        # Create many tool calls to test worker limiting
        tool_calls = [
            create_mock_tool_call(f"call_{i}", "simple_math_tool", {"a": i, "b": 1})
            for i in range(5)
        ]
        
        mock_response_1 = create_mock_response(tool_calls=tool_calls)
        mock_response_2 = create_mock_response(content="Max workers test complete")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run many tools"}],
            tools=[simple_math_tool],
            parallel_tool_execution=True,
            max_workers=2  # Limit to 2 workers
        )
        
        assert any(block.text == "Max workers test complete" for block in response.content if hasattr(block, 'text'))
    
    @pytest.mark.asyncio
    async def test_async_max_workers_parameter(self, async_anthropic_client, mock_async_anthropic_client):
        """Test max_workers parameter with async execution."""
        tool_calls = [
            create_mock_tool_call(f"call_{i}", "async_math_tool", {"a": i, "b": 2})
            for i in range(4)
        ]
        
        mock_response_1 = create_mock_response(tool_calls=tool_calls)
        mock_response_2 = create_mock_response(content="Async max workers complete")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run many async tools"}],
            tools=[async_math_tool],
            parallel_tool_execution=True,
            max_workers=3  # Limit to 3 workers
        )
        
        assert any(block.text == "Async max workers complete" for block in response.content if hasattr(block, 'text'))
    
    def test_max_workers_with_multiple_tools(self, sync_anthropic_client, mock_anthropic_client):
        """Test max_workers with different tool types."""
        tool_calls = [
            create_mock_tool_call("call_math1", "simple_math_tool", {"a": 1, "b": 2}),
            create_mock_tool_call("call_slow1", "slow_tool", {"name": "test1", "delay": 0.01}),
            create_mock_tool_call("call_math2", "simple_math_tool", {"a": 3, "b": 4}),
            create_mock_tool_call("call_slow2", "slow_tool", {"name": "test2", "delay": 0.01}),
        ]
        
        mock_response_1 = create_mock_response(tool_calls=tool_calls)
        mock_response_2 = create_mock_response(content="Multiple tools with max workers")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run multiple tool types"}],
            tools=[simple_math_tool, slow_tool],
            parallel_tool_execution=True,
            max_workers=2
        )
        
        assert any(block.text == "Multiple tools with max workers" for block in response.content if hasattr(block, 'text'))


class TestAnthropicParallelExecutionErrorHandling:
    """Test error handling in parallel execution scenarios."""
    
    def test_parallel_execution_with_failing_tool(self, sync_anthropic_client, mock_anthropic_client):
        """Test that parallel execution handles tool failures gracefully."""
        tool_calls = [
            create_mock_tool_call("call_good", "simple_math_tool", {"a": 5, "b": 3}),
            create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True}),
            create_mock_tool_call("call_good2", "simple_math_tool", {"a": 10, "b": 2})
        ]
        
        mock_response_1 = create_mock_response(tool_calls=tool_calls)
        mock_response_2 = create_mock_response(content="Parallel with errors handled")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        # Should handle errors gracefully in parallel execution
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run tools with one failure"}],
            tools=[simple_math_tool, failing_tool],
            parallel_tool_execution=True,
            graceful_error_handling=True
        )
        
        assert any(block.text == "Parallel with errors handled" for block in response.content if hasattr(block, 'text'))
        
        # Check that error was included in tool results
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in messages if msg.get('role') == 'user' and 'tool_result' in str(msg.get('content', []))]
        
        assert len(tool_result_messages) == 1
        tool_results = tool_result_messages[0]['content']
        
        # Should have results for all three tools (including error)
        tool_result_count = sum(1 for result in tool_results if isinstance(result, dict) and 'tool_use_id' in result)
        assert tool_result_count == 3
        
        # Check that error result is present
        error_results = [result for result in tool_results if isinstance(result, dict) and 'Error executing tool' in str(result.get('content', ''))]
        assert len(error_results) == 1
    
    @pytest.mark.asyncio
    async def test_async_parallel_error_handling(self, async_anthropic_client, mock_async_anthropic_client):
        """Test error handling in async parallel execution."""
        tool_calls = [
            create_mock_tool_call("call_good", "async_math_tool", {"a": 7, "b": 3}),
            create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True}),
        ]
        
        mock_response_1 = create_mock_response(tool_calls=tool_calls)
        mock_response_2 = create_mock_response(content="Async parallel errors handled")
        
        mock_async_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Async tools with failure"}],
            tools=[async_math_tool, failing_tool],
            parallel_tool_execution=True,
            graceful_error_handling=True
        )
        
        assert any(block.text == "Async parallel errors handled" for block in response.content if hasattr(block, 'text'))
        
        # Verify error handling
        second_call_args = mock_async_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        tool_result_messages = [msg for msg in messages if msg.get('role') == 'user' and 'tool_result' in str(msg.get('content', []))]
        
        assert len(tool_result_messages) == 1
        tool_results = tool_result_messages[0]['content']
        tool_result_count = sum(1 for result in tool_results if isinstance(result, dict) and 'tool_use_id' in result)
        assert tool_result_count == 2  # Both tools should have results (one success, one error)
    
    def test_parallel_execution_no_graceful_error_handling(self, sync_anthropic_client, mock_anthropic_client):
        """Test that parallel execution raises errors when graceful handling is disabled."""
        tool_calls = [
            create_mock_tool_call("call_good", "simple_math_tool", {"a": 5, "b": 3}),
            create_mock_tool_call("call_fail", "failing_tool", {"should_fail": True})
        ]
        
        mock_response_1 = create_mock_response(tool_calls=tool_calls)
        mock_anthropic_client.messages.create.side_effect = [mock_response_1]
        
        # Should raise exception when graceful error handling is disabled
        with pytest.raises(Exception) as exc_info:
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "Run tools with failure"}],
                tools=[simple_math_tool, failing_tool],
                parallel_tool_execution=True,
                graceful_error_handling=False
            )
        
        assert "Error executing tool failing_tool" in str(exc_info.value)
        assert "This tool failed intentionally" in str(exc_info.value)


class TestAnthropicParallelExecutionPerformance:
    """Test performance characteristics of parallel execution."""
    
    def test_parallel_shows_clear_performance_benefit(self, sync_anthropic_client, mock_anthropic_client):
        """Test that parallel execution shows clear performance benefits with multiple slow tools."""
        # Create several slow tool calls
        num_tools = 4
        delay_per_tool = 0.02
        tool_calls = [
            create_mock_tool_call(f"call_{i}", "slow_tool", {"name": f"tool{i}", "delay": delay_per_tool})
            for i in range(num_tools)
        ]
        
        mock_response_1 = create_mock_response(tool_calls=tool_calls)
        mock_response_2 = create_mock_response(content="Performance test complete")
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run performance test"}],
            tools=[slow_tool],
            parallel_tool_execution=True
        )
        parallel_time = time.time() - start_time
        
        # Reset mocks for sequential test
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run performance test"}],
            tools=[slow_tool],
            parallel_tool_execution=False
        )
        sequential_time = time.time() - start_time
        
        # Parallel should be significantly faster
        speedup_ratio = sequential_time / parallel_time
        assert speedup_ratio > 1.5, f"Parallel execution speedup ratio {speedup_ratio} should be > 1.5"
        
        # Parallel should be close to single tool time, not sum of all tools
        expected_parallel_time = delay_per_tool + 0.01  # Allow some overhead
        assert parallel_time < expected_parallel_time * 2, f"Parallel time {parallel_time} should be close to single tool time {expected_parallel_time}" 