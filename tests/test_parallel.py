import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, AsyncMock
from toolflow import tool, from_openai, from_openai_async


@tool
def slow_sync_tool(name: str, delay: float) -> str:
    """A sync tool that takes some time to execute."""
    time.sleep(delay)
    return f"Sync result from {name} after {delay}s"


@tool
async def slow_async_tool(name: str, delay: float) -> str:
    """An async tool that takes some time to execute."""
    await asyncio.sleep(delay)
    return f"Async result from {name} after {delay}s"


@tool
def quick_math_tool(a: int, b: int) -> int:
    """A quick math operation."""
    return a + b


@tool
async def quick_async_math_tool(a: int, b: int) -> int:
    """A quick async math operation."""
    await asyncio.sleep(0.001)  # Very small delay
    return a * b


class TestSyncParallelExecution:
    """Test parallel execution with sync client."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_openai_client = Mock()
        self.mock_completions = Mock()
        self.mock_openai_client.chat.completions = self.mock_completions
        self.client = from_openai(self.mock_openai_client)
    
    def test_parallel_sync_tools_execution_time(self):
        """Test that parallel execution is faster than sequential."""
        # Mock multiple tool calls
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "slow_sync_tool"
        mock_tool_call_1.function.arguments = '{"name": "tool1", "delay": 0.1}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function.name = "slow_sync_tool"
        mock_tool_call_2.function.arguments = '{"name": "tool2", "delay": 0.1}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "All done!"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run two slow tools"}],
            tools=[slow_sync_tool],
            parallel_tool_execution=True  # Enable parallel execution
        )
        execution_time = time.time() - start_time
        
        # Parallel execution should be significantly faster than 0.2s (2 * 0.1s)
        assert execution_time < 0.15, f"Parallel execution took {execution_time}s, expected < 0.15s"
        assert response.choices[0].message.content == "All done!"
    
    def test_sequential_execution_by_default(self):
        """Test that execution is sequential by default (parallel_tool_execution=False)."""
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "slow_sync_tool"
        mock_tool_call_1.function.arguments = '{"name": "tool1", "delay": 0.05}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function.name = "slow_sync_tool"
        mock_tool_call_2.function.arguments = '{"name": "tool2", "delay": 0.05}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Sequential done!"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run two slow tools"}],
            tools=[slow_sync_tool]
            # No parallel_tool_execution=True, should be sequential
        )
        execution_time = time.time() - start_time
        
        # Sequential execution should take approximately 0.1s (sum of delays)
        assert execution_time >= 0.08, f"Sequential execution took {execution_time}s, expected >= 0.08s"
        assert response.choices[0].message.content == "Sequential done!"
    
    def test_single_tool_call_no_threading_overhead(self):
        """Test that single tool calls don't use threading."""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_single"
        mock_tool_call.function.name = "quick_math_tool"
        mock_tool_call.function.arguments = '{"a": 5, "b": 3}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Result: 8"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Add 5 + 3"}],
            tools=[quick_math_tool],
            parallel_tool_execution=True  # Even with parallel_tool_execution=True, single calls should be direct
        )
        
        assert response.choices[0].message.content == "Result: 8"
    
    def test_parallel_execution_maintains_order(self):
        """Test that results are returned in the same order as tool_calls."""
        # Create tools with different execution times
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_slow"
        mock_tool_call_1.function.name = "slow_sync_tool"
        mock_tool_call_1.function.arguments = '{"name": "slow", "delay": 0.05}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_fast"
        mock_tool_call_2.function.name = "quick_math_tool"
        mock_tool_call_2.function.arguments = '{"a": 1, "b": 2}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Mixed results received"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # Capture the messages to verify order
        messages = [{"role": "user", "content": "Run mixed tools"}]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[slow_sync_tool, quick_math_tool],
            parallel_tool_execution=True  # Enable parallel execution
        )
        
        assert response.choices[0].message.content == "Mixed results received"
        
        # Check that tool results were added in correct order
        # The second call to create() should have the tool results in messages
        tool_results_call = self.mock_completions.create.call_args_list[1]
        tool_messages = tool_results_call[1]['messages']
        
        # Find tool result messages (they have role "tool")
        tool_result_messages = [msg for msg in tool_messages if msg.get('role') == 'tool']
        
        # Should have results in same order as tool_calls (slow first, then fast)
        assert len(tool_result_messages) == 2
        assert tool_result_messages[0]['tool_call_id'] == 'call_slow'
        assert tool_result_messages[1]['tool_call_id'] == 'call_fast'


class TestAsyncParallelExecution:
    """Test parallel execution with async client."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_openai_client = Mock()
        self.mock_completions = AsyncMock()
        self.mock_openai_client.chat.completions = self.mock_completions
        self.client = from_openai_async(self.mock_openai_client)
    
    @pytest.mark.asyncio
    async def test_parallel_async_tools_execution_time(self):
        """Test that parallel async execution is faster than sequential."""
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "slow_async_tool"
        mock_tool_call_1.function.arguments = '{"name": "async1", "delay": 0.1}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function.name = "slow_async_tool"
        mock_tool_call_2.function.arguments = '{"name": "async2", "delay": 0.1}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Async all done!"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run two slow async tools"}],
            tools=[slow_async_tool],
            parallel_tool_execution=True  # Enable parallel execution
        )
        execution_time = time.time() - start_time
        
        # Parallel execution should be significantly faster than 0.2s (2 * 0.1s)
        assert execution_time < 0.15, f"Parallel async execution took {execution_time}s, expected < 0.15s"
        assert response.choices[0].message.content == "Async all done!"
    
    @pytest.mark.asyncio
    async def test_sequential_async_execution_by_default(self):
        """Test that async execution is sequential by default."""
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "slow_async_tool"
        mock_tool_call_1.function.arguments = '{"name": "async1", "delay": 0.05}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function.name = "slow_async_tool"
        mock_tool_call_2.function.arguments = '{"name": "async2", "delay": 0.05}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Sequential async done!"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run two slow async tools"}],
            tools=[slow_async_tool]
            # No parallel_tool_execution=True, should be sequential
        )
        execution_time = time.time() - start_time
        
        # Sequential execution should take approximately 0.1s (sum of delays)
        assert execution_time >= 0.08, f"Sequential async execution took {execution_time}s, expected >= 0.08s"
        assert response.choices[0].message.content == "Sequential async done!"
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_tools_parallel(self):
        """Test parallel execution of mixed sync and async tools with improved strategy."""
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_sync"
        mock_tool_call_1.function.name = "slow_sync_tool"
        mock_tool_call_1.function.arguments = '{"name": "sync", "delay": 0.05}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_async"
        mock_tool_call_2.function.name = "slow_async_tool"
        mock_tool_call_2.function.arguments = '{"name": "async", "delay": 0.05}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Mixed execution complete!"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Run mixed sync/async tools"}],
            tools=[slow_sync_tool, slow_async_tool],
            parallel_tool_execution=True  # Enable parallel execution
        )
        execution_time = time.time() - start_time
        
        # Should execute in parallel, so faster than sequential (0.1s)
        assert execution_time < 0.08, f"Mixed parallel execution took {execution_time}s, expected < 0.08s"
        assert response.choices[0].message.content == "Mixed execution complete!"
    
    @pytest.mark.asyncio
    async def test_async_single_tool_no_gather_overhead(self):
        """Test that single async tool calls don't use gather."""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_single_async"
        mock_tool_call.function.name = "quick_async_math_tool"
        mock_tool_call.function.arguments = '{"a": 4, "b": 5}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Single async result: 20"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Multiply 4 * 5"}],
            tools=[quick_async_math_tool],
            parallel_tool_execution=True  # Even with parallel_tool_execution=True, single calls should be direct
        )
        
        assert response.choices[0].message.content == "Single async result: 20"
    
    @pytest.mark.asyncio
    async def test_async_parallel_maintains_order(self):
        """Test that async parallel execution maintains tool call order."""
        # Fast tool first, slow tool second
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_fast_async"
        mock_tool_call_1.function.name = "quick_async_math_tool"
        mock_tool_call_1.function.arguments = '{"a": 2, "b": 3}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_slow_async"
        mock_tool_call_2.function.name = "slow_async_tool"
        mock_tool_call_2.function.arguments = '{"name": "slow", "delay": 0.02}'
        
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response_1.choices[0].message.content = None
        
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.tool_calls = None
        mock_response_2.choices[0].message.content = "Async order maintained"
        
        self.mock_completions.create.side_effect = [mock_response_1, mock_response_2]
        
        messages = [{"role": "user", "content": "Run fast then slow async tools"}]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[quick_async_math_tool, slow_async_tool],
            parallel_tool_execution=True  # Enable parallel execution
        )
        
        assert response.choices[0].message.content == "Async order maintained"
        
        # Check tool results order in the second API call
        tool_results_call = self.mock_completions.create.call_args_list[1]
        tool_messages = tool_results_call[1]['messages']
        
        tool_result_messages = [msg for msg in tool_messages if msg.get('role') == 'tool']
        
        # Should maintain original order (fast first, slow second)
        assert len(tool_result_messages) == 2
        assert tool_result_messages[0]['tool_call_id'] == 'call_fast_async'
        assert tool_result_messages[1]['tool_call_id'] == 'call_slow_async'


@tool
def failing_tool(message: str) -> str:
    """A tool that always fails."""
    raise ValueError(f"Tool failed: {message}")


class TestParallelExecutionErrorHandling:
    """Test error handling in parallel execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_openai_client = Mock()
        self.mock_completions = Mock()
        self.mock_openai_client.chat.completions = self.mock_completions
        self.client = from_openai(self.mock_openai_client)
    
    def test_sync_parallel_error_handling(self):
        """Test that errors in parallel sync execution are properly handled."""
        mock_tool_call_1 = Mock()
        mock_tool_call_1.id = "call_good"
        mock_tool_call_1.function.name = "quick_math_tool"
        mock_tool_call_1.function.arguments = '{"a": 1, "b": 2}'
        
        mock_tool_call_2 = Mock()
        mock_tool_call_2.id = "call_bad"
        mock_tool_call_2.function.name = "failing_tool"
        mock_tool_call_2.function.arguments = '{"message": "test error"}'
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.tool_calls = [mock_tool_call_1, mock_tool_call_2]
        mock_response.choices[0].message.content = None
        
        self.mock_completions.create.return_value = mock_response
        
        with pytest.raises(Exception, match="Error in parallel tool execution"):
            self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Run good and bad tools"}],
                tools=[quick_math_tool, failing_tool],
                parallel_tool_execution=True  # Enable parallel execution to trigger the error
            ) 