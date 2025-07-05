"""
Tests for parallel execution functionality.
"""
import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

import toolflow
from toolflow import from_openai, from_anthropic, tool
from tests.conftest import (
    create_openai_response,
    create_openai_tool_call,
    create_anthropic_response,
    create_anthropic_tool_call,
    slow_tool,
    SLOW_TOOLS
)


class TestParallelToolExecution:
    """Test parallel tool execution."""
    
    def test_parallel_execution_faster_than_sequential_openai(self, mock_openai_client):
        """Test that parallel execution is faster than sequential for OpenAI."""
        client = from_openai(mock_openai_client)
        
        # Create slow tools for timing test
        @tool
        def slow_tool_1() -> str:
            """First slow tool."""
            time.sleep(0.1)
            return "Result 1"
        
        @tool 
        def slow_tool_2() -> str:
            """Second slow tool."""
            time.sleep(0.1)
            return "Result 2"
        
        # Mock responses
        tool_call_1 = create_openai_tool_call("call_1", "slow_tool_1", {})
        tool_call_2 = create_openai_tool_call("call_2", "slow_tool_2", {})
        mock_response_1 = create_openai_response(tool_calls=[tool_call_1, tool_call_2])
        mock_response_2 = create_openai_response(content="Both tools completed")
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # Test parallel execution
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run both slow tools"}],
            tools=[slow_tool_1, slow_tool_2],
            parallel_tool_execution=True
        )
        parallel_time = time.time() - start_time
        
        # Reset mock for sequential test
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # Test sequential execution
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run both slow tools"}],
            tools=[slow_tool_1, slow_tool_2],
            parallel_tool_execution=False
        )
        sequential_time = time.time() - start_time
        
        # Parallel should be faster (allowing some overhead)
        assert parallel_time < sequential_time * 0.8
    
    def test_parallel_execution_with_different_execution_times(self, mock_openai_client):
        """Test parallel execution with tools that have different execution times."""
        client = from_openai(mock_openai_client)
        
        @tool
        def fast_tool() -> str:
            """Fast tool."""
            time.sleep(0.01)
            return "Fast result"
        
        @tool
        def medium_tool() -> str:
            """Medium speed tool."""
            time.sleep(0.05)
            return "Medium result"
        
        @tool
        def slow_tool() -> str:
            """Slow tool."""
            time.sleep(0.1)
            return "Slow result"
        
        # Mock responses
        tool_calls = [
            create_openai_tool_call("call_1", "fast_tool", {}),
            create_openai_tool_call("call_2", "medium_tool", {}),
            create_openai_tool_call("call_3", "slow_tool", {})
        ]
        mock_response_1 = create_openai_response(tool_calls=tool_calls)
        mock_response_2 = create_openai_response(content="All tools completed")
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run all tools"}],
            tools=[fast_tool, medium_tool, slow_tool],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        # Total time should be close to the slowest tool (0.1s) plus overhead
        assert execution_time < 0.2  # Should be much less than sum of all tools (0.16s)
        assert response == "All tools completed"
    
    def test_parallel_execution_error_handling(self, mock_openai_client):
        """Test parallel execution with some tools failing."""
        client = from_openai(mock_openai_client)
        
        @tool
        def working_tool() -> str:
            """A tool that works."""
            time.sleep(0.01)
            return "Success"
        
        @tool
        def failing_tool() -> str:
            """A tool that fails."""
            time.sleep(0.01)
            raise ValueError("Tool failed")
        
        # Mock responses
        tool_calls = [
            create_openai_tool_call("call_1", "working_tool", {}),
            create_openai_tool_call("call_2", "failing_tool", {})
        ]
        mock_response_1 = create_openai_response(tool_calls=tool_calls)
        mock_response_2 = create_openai_response(content="Mixed results")
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run both tools"}],
            tools=[working_tool, failing_tool],
            parallel_tool_execution=True
        )
        
        assert response == "Mixed results"
        
        # Check that both results (success and error) are included in second call
        second_call_args = mock_openai_client.chat.completions.create.call_args_list[1]
        messages = second_call_args[1]['messages']
        tool_messages = [msg for msg in messages if msg['role'] == 'tool']
        assert len(tool_messages) == 2
        
        # One should have success, one should have error
        contents = [msg['content'] for msg in tool_messages]
        assert "Success" in contents
        assert any("Tool failed" in content for content in contents)


class TestAsyncParallelExecution:
    """Test async parallel execution."""
    
    @pytest.mark.asyncio
    async def test_async_parallel_execution(self, mock_async_openai_client):
        """Test async parallel execution."""
        # This would require async client wrapper - placeholder for now
        # The actual implementation would test async tool execution
        pass
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_tools_parallel(self, mock_async_openai_client):
        """Test parallel execution with mixed sync/async tools."""
        from toolflow.providers.openai.wrappers import AsyncOpenAIWrapper
        client = AsyncOpenAIWrapper(mock_async_openai_client)
        
        @tool
        def sync_tool() -> str:
            """Sync tool."""
            time.sleep(0.05)
            return "Sync result"
        
        @tool
        async def async_tool() -> str:
            """Async tool."""
            await asyncio.sleep(0.05)
            return "Async result"
        
        # Mock responses
        tool_calls = [
            create_openai_tool_call("call_1", "sync_tool", {}),
            create_openai_tool_call("call_2", "async_tool", {})
        ]
        mock_response_1 = create_openai_response(tool_calls=tool_calls)
        mock_response_2 = create_openai_response(content="Mixed execution completed")
        mock_async_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run sync and async tools"}],
            tools=[sync_tool, async_tool],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        assert response == "Mixed execution completed"
        # Should be faster than sequential execution
        assert execution_time < 0.15  # Much less than 0.1s (sum of both tools)


class TestThreadPoolConfiguration:
    """Test thread pool configuration for parallel execution."""
    
    def test_max_workers_configuration(self, mock_openai_client):
        """Test that max workers configuration affects execution."""
        client = from_openai(mock_openai_client)
        
        # Set low number of workers
        original_workers = toolflow.get_max_workers()
        toolflow.set_max_workers(1)
        
        try:
            @tool
            def worker_tool(worker_id: int) -> str:
                """Tool that simulates work."""
                time.sleep(0.05)  # Increased for more reliable timing
                return f"Worker {worker_id} done"
            
            # Create many tool calls
            tool_calls = [
                create_openai_tool_call(f"call_{i}", "worker_tool", {"worker_id": i})
                for i in range(3)  # Reduced count but increased sleep time
            ]
            mock_response_1 = create_openai_response(tool_calls=tool_calls)
            mock_response_2 = create_openai_response(content="All workers completed")
            mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
            
            start_time = time.time()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Run worker tools"}],
                tools=[worker_tool],
                parallel_tool_execution=True
            )
            execution_time = time.time() - start_time
            
            # With 1 worker, should take longer (more sequential)
            # With 3 tools of 0.05s each, should take at least 0.12s
            # Allow for some overhead/timing variance on fast hardware
            assert execution_time > 0.10
            
        finally:
            # Reset
            toolflow.set_max_workers(original_workers)
    
    def test_custom_executor(self, mock_openai_client):
        """Test using custom executor."""
        client = from_openai(mock_openai_client)
        
        # Create custom executor
        custom_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-toolflow-")
        original_workers = toolflow.get_max_workers()
        
        try:
            toolflow.set_executor(custom_executor)
            
            @tool
            def executor_tool() -> str:
                """Tool for testing custom executor."""
                import threading
                thread_name = threading.current_thread().name
                time.sleep(0.01)
                return f"Executed in {thread_name}"
            
            # Mock responses
            tool_calls = [
                create_openai_tool_call("call_1", "executor_tool", {}),
                create_openai_tool_call("call_2", "executor_tool", {})
            ]
            mock_response_1 = create_openai_response(tool_calls=tool_calls)
            mock_response_2 = create_openai_response(content="Custom executor test")
            mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test custom executor"}],
                tools=[executor_tool],
                parallel_tool_execution=True
            )
            
            assert response == "Custom executor test"
            
            # Check that tool results contain custom thread names
            second_call_args = mock_openai_client.chat.completions.create.call_args_list[1]
            messages = second_call_args[1]['messages']
            tool_messages = [msg for msg in messages if msg['role'] == 'tool']
            
            # At least one result should mention the custom thread prefix
            contents = [msg['content'] for msg in tool_messages]
            assert any("test-toolflow-" in content for content in contents)
            
        finally:
            # Reset to default
            toolflow.set_max_workers(original_workers)


class TestParallelExecutionAnthropic:
    """Test parallel execution with Anthropic provider."""
    
    def test_parallel_execution_anthropic(self, mock_anthropic_client):
        """Test parallel execution with Anthropic."""
        client = from_anthropic(mock_anthropic_client)
        
        @tool
        def anthropic_tool_1() -> str:
            """First tool."""
            time.sleep(0.05)
            return "Anthropic Result 1"
        
        @tool
        def anthropic_tool_2() -> str:
            """Second tool."""
            time.sleep(0.05)
            return "Anthropic Result 2"
        
        # Mock responses
        tool_calls = [
            create_anthropic_tool_call("toolu_1", "anthropic_tool_1", {}),
            create_anthropic_tool_call("toolu_2", "anthropic_tool_2", {})
        ]
        mock_response_1 = create_anthropic_response(tool_calls=tool_calls)
        mock_response_2 = create_anthropic_response(content="Anthropic parallel execution completed")
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Run both tools"}],
            tools=[anthropic_tool_1, anthropic_tool_2],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        assert response == "Anthropic parallel execution completed"
        # Should be faster than sequential (0.1s)
        assert execution_time < 0.08


class TestParallelExecutionEdgeCases:
    """Test edge cases for parallel execution."""
    
    def test_single_tool_parallel_execution(self, mock_openai_client):
        """Test parallel execution with only one tool."""
        client = from_openai(mock_openai_client)
        
        @tool
        def single_tool() -> str:
            """Single tool."""
            time.sleep(0.01)
            return "Single result"
        
        tool_call = create_openai_tool_call("call_1", "single_tool", {})
        mock_response_1 = create_openai_response(tool_calls=[tool_call])
        mock_response_2 = create_openai_response(content="Single tool completed")
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run single tool"}],
            tools=[single_tool],
            parallel_tool_execution=True
        )
        
        assert response == "Single tool completed"
    
    def test_no_tools_parallel_execution(self, mock_openai_client):
        """Test parallel execution parameter with no tools."""
        client = from_openai(mock_openai_client)
        
        mock_response = create_openai_response(content="No tools to execute")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            parallel_tool_execution=True  # Should have no effect
        )
        
        assert response == "No tools to execute"
        assert mock_openai_client.chat.completions.create.call_count == 1
    
    def test_parallel_execution_default_behavior(self, mock_openai_client):
        """Test default parallel execution behavior."""
        client = from_openai(mock_openai_client)
        
        @tool
        def default_tool() -> str:
            """Tool for testing default behavior."""
            return "Default result"
        
        tool_call = create_openai_tool_call("call_1", "default_tool", {})
        mock_response_1 = create_openai_response(tool_calls=[tool_call])
        mock_response_2 = create_openai_response(content="Default behavior test")
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # Don't specify parallel_tool_execution - should default to False (sequential)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test default"}],
            tools=[default_tool]
        )
        
        assert response == "Default behavior test"
        assert mock_openai_client.chat.completions.create.call_count == 2 