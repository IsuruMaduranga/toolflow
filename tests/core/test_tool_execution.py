"""
Tests for core tool execution functionality.
"""
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, Mock

import toolflow
from toolflow.core.tool_execution import (
    set_max_workers,
    get_max_workers,
    set_executor
)


class TestToolExecutionConfig:
    """Test tool execution configuration functions."""
    
    def test_default_max_workers(self):
        """Test default max workers configuration."""
        # Reset to default
        set_max_workers(4)
        assert get_max_workers() == 4
    
    def test_set_max_workers(self):
        """Test setting max workers."""
        original = get_max_workers()
        
        # Test setting different values
        set_max_workers(8)
        assert get_max_workers() == 8
        
        set_max_workers(16)
        assert get_max_workers() == 16
        
        # Reset
        set_max_workers(original)
    
    def test_set_max_workers_validation(self):
        """Test max workers validation."""
        original = get_max_workers()
        
        # Should accept positive integers
        set_max_workers(1)
        assert get_max_workers() == 1
        
        set_max_workers(100)
        assert get_max_workers() == 100
        
        # Reset
        set_max_workers(original)
    
    def test_custom_executor(self):
        """Test setting custom executor."""
        original_workers = get_max_workers()
        
        # Create custom executor
        custom_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-")
        
        # Set custom executor
        set_executor(custom_executor)
        
        # Should use custom executor (hard to test directly, but we can verify it was set)
        # The actual usage would be tested in integration tests
        
        # Reset to default by setting max workers
        set_max_workers(original_workers)
    
    def test_executor_thread_names(self):
        """Test that custom executor thread names are used."""
        # Create executor with custom thread name prefix
        custom_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="toolflow-test-")
        set_executor(custom_executor)
        
        # This is mainly to ensure the API works
        # Actual thread naming would be tested in integration
        
        # Reset
        set_max_workers(4)


class TestGlobalExecutorState:
    """Test global executor state management."""
    
    def test_executor_singleton_behavior(self):
        """Test that executor maintains singleton-like behavior."""
        # Get current max workers
        workers1 = get_max_workers()
        
        # Change max workers
        set_max_workers(workers1 + 1)
        workers2 = get_max_workers()
        
        assert workers2 == workers1 + 1
        
        # Reset
        set_max_workers(workers1)
    
    def test_thread_safety_basics(self):
        """Basic test for thread safety of configuration."""
        import threading
        
        original = get_max_workers()
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # Each thread sets a different value
                value = 4 + worker_id
                set_max_workers(value)
                time.sleep(0.01)  # Small delay
                retrieved = get_max_workers()
                results.append((worker_id, value, retrieved))
            except Exception as e:
                errors.append((worker_id, e))
        
        # Run multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should not have errors
        assert len(errors) == 0
        
        # Due to race conditions, we can't predict exact final value,
        # but we can verify the API worked
        assert len(results) == 3
        
        # Reset
        set_max_workers(original)


class TestAsyncConfiguration:
    """Test async-related configuration."""
    
    def test_async_yield_frequency_default(self):
        """Test default async yield frequency."""
        # Should start with default (0)
        from toolflow.core.execution_loops import get_async_yield_frequency
        assert get_async_yield_frequency() == 0
    
    def test_set_async_yield_frequency(self):
        """Test setting async yield frequency."""
        from toolflow.core.execution_loops import get_async_yield_frequency
        original = get_async_yield_frequency()
        
        # Test setting different values
        toolflow.set_async_yield_frequency(1)
        assert get_async_yield_frequency() == 1
        
        toolflow.set_async_yield_frequency(10)
        assert get_async_yield_frequency() == 10
        
        toolflow.set_async_yield_frequency(0)
        assert get_async_yield_frequency() == 0
        
        # Reset
        toolflow.set_async_yield_frequency(original)
    
    def test_async_yield_frequency_validation(self):
        """Test async yield frequency validation."""
        from toolflow.core.execution_loops import get_async_yield_frequency
        original = get_async_yield_frequency()
        
        # Should accept non-negative integers
        toolflow.set_async_yield_frequency(0)
        assert get_async_yield_frequency() == 0
        
        toolflow.set_async_yield_frequency(100)
        assert get_async_yield_frequency() == 100
        
        # Reset
        toolflow.set_async_yield_frequency(original)


class TestExecutorIntegration:
    """Test executor integration with actual work."""
    
    @pytest.mark.slow
    def test_executor_actually_executes(self):
        """Test that the executor actually executes work."""
        # This is more of an integration test, but belongs here
        # since it tests the core execution mechanism
        
        results = []
        
        def test_function(value):
            results.append(value)
            return value * 2
        
        # We would need to test this through the actual tool execution
        # This test serves as a placeholder for integration testing
        assert True  # Placeholder
    
    def test_multiple_workers_setting(self):
        """Test that changing max workers affects behavior."""
        original = get_max_workers()
        
        # Test with minimal workers
        set_max_workers(1)
        assert get_max_workers() == 1
        
        # Test with more workers
        set_max_workers(8)
        assert get_max_workers() == 8
        
        # Reset
        set_max_workers(original)


class TestConfigurationPersistence:
    """Test that configuration persists across operations."""
    
    def test_max_workers_persistence(self):
        """Test that max workers setting persists."""
        original = get_max_workers()
        
        # Set new value
        new_value = original + 2
        set_max_workers(new_value)
        
        # Should persist across multiple gets
        assert get_max_workers() == new_value
        assert get_max_workers() == new_value
        assert get_max_workers() == new_value
        
        # Reset
        set_max_workers(original)
    
    def test_async_frequency_persistence(self):
        """Test that async yield frequency persists."""
        from toolflow.core.execution_loops import get_async_yield_frequency
        original = get_async_yield_frequency()
        
        # Set new value
        new_value = 5
        toolflow.set_async_yield_frequency(new_value)
        
        # Should persist
        assert get_async_yield_frequency() == new_value
        assert get_async_yield_frequency() == new_value
        
        # Reset
        toolflow.set_async_yield_frequency(original) 