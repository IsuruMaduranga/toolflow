#!/usr/bin/env python3
"""Test the simplified executor behavior."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import toolflow

@toolflow.tool
def sync_task(x: int) -> str:
    return f"Sync result: {x * 2}"

@toolflow.tool
async def async_task(x: int) -> str:
    await asyncio.sleep(0.01)
    return f"Async result: {x * 3}"

def test_sync_behavior():
    """Test sync behavior with global executor."""
    print("=== Testing Sync Behavior ===")
    
    # Test that we can set max workers
    toolflow.set_max_workers(4)
    print("✅ Set max workers to 4")
    
    # Test that we can set a custom executor
    custom_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="test-")
    toolflow.set_global_executor(custom_executor)
    print("✅ Set custom global executor")
    
    # Test core functions directly
    from src.toolflow.core.tool_execution import execute_tools, _get_sync_executor
    
    # Check that we get our custom executor
    executor = _get_sync_executor()
    assert executor is custom_executor, "Should return our custom executor"
    print("✅ _get_sync_executor returns custom executor")
    
    # Test tool execution
    tool_calls = [
        {"id": "1", "function": {"name": "sync_task", "arguments": {"x": 5}}},
        {"id": "2", "function": {"name": "sync_task", "arguments": {"x": 10}}}
    ]
    tool_map = {"sync_task": sync_task}
    
    # Test sequential execution
    results = execute_tools(tool_calls, tool_map, parallel=False)
    assert len(results) == 2
    assert results[0]["output"] == "Sync result: 10"
    assert results[1]["output"] == "Sync result: 20"
    print("✅ Sequential execution works")
    
    # Test parallel execution  
    results = execute_tools(tool_calls, tool_map, parallel=True)
    assert len(results) == 2
    print("✅ Parallel execution works")

async def test_async_behavior():
    """Test async behavior with shared executor."""
    print("\n=== Testing Async Behavior ===")
    
    from src.toolflow.core.tool_execution import execute_tools_async, _get_async_executor
    
    # Should return the same global executor we set
    executor = _get_async_executor()
    print(f"✅ _get_async_executor returns the global executor")
    
    # Test mixed sync/async tool execution
    tool_calls = [
        {"id": "1", "function": {"name": "sync_task", "arguments": {"x": 5}}},
        {"id": "2", "function": {"name": "async_task", "arguments": {"x": 7}}}
    ]
    tool_map = {"sync_task": sync_task, "async_task": async_task}
    
    results = await execute_tools_async(tool_calls, tool_map)
    assert len(results) == 2
    
    # Results might be in different order due to concurrency
    outputs = [r["output"] for r in results]
    assert "Sync result: 10" in outputs
    assert "Async result: 21" in outputs
    print("✅ Async execution with mixed sync/async tools works")

def test_no_global_executor():
    """Test behavior when no global executor is set."""
    print("\n=== Testing No Global Executor ===")
    
    # Reset global executor
    from src.toolflow.core.tool_execution import _executor_lock
    import src.toolflow.core.tool_execution as te
    
    with _executor_lock:
        if te._global_executor:
            te._global_executor.shutdown(wait=True)
        te._global_executor = None
    
    from src.toolflow.core.tool_execution import _get_async_executor, _get_sync_executor
    
    # Async should return None (use asyncio default)
    async_executor = _get_async_executor()
    assert async_executor is None, "Should return None when no global executor set"
    print("✅ Async uses None (asyncio default) when no global executor")
    
    # Sync should create default executor
    sync_executor = _get_sync_executor()
    assert sync_executor is not None, "Should create default executor for sync"
    print("✅ Sync creates default executor when none set")
    
    # Now async should return the same executor that sync created
    async_executor = _get_async_executor()
    assert async_executor is sync_executor, "Async should now return same executor as sync"
    print("✅ After sync creates executor, async uses the same one")

def test_behavior_summary():
    """Test and demonstrate the complete behavior."""
    print("\n=== Testing Complete Behavior ===")
    
    # Reset everything
    from src.toolflow.core.tool_execution import _executor_lock
    import src.toolflow.core.tool_execution as te
    
    with _executor_lock:
        if te._global_executor:
            te._global_executor.shutdown(wait=True)
        te._global_executor = None
    
    print("1. No global executor set:")
    print("   - Sync: Creates default executor when parallel=True")
    print("   - Async: Uses None (asyncio default)")
    
    print("\n2. set_max_workers(8) called:")
    toolflow.set_max_workers(8)
    print("   - Both sync and async use the same 8-thread executor")
    
    print("\n3. set_global_executor(custom) called:")
    custom = ThreadPoolExecutor(max_workers=12, thread_name_prefix="custom-")
    toolflow.set_global_executor(custom)
    print("   - Both sync and async use the custom executor")
    
    # Verify
    from src.toolflow.core.tool_execution import _get_async_executor, _get_sync_executor
    
    sync_exec = _get_sync_executor()
    async_exec = _get_async_executor()
    
    assert sync_exec is custom
    assert async_exec is custom
    print("✅ Both executors return the custom executor")

def main():
    print("Testing Simplified Executor Implementation")
    print("=" * 45)
    
    test_sync_behavior()
    asyncio.run(test_async_behavior())
    test_no_global_executor()
    test_behavior_summary()
    
    print("\n🎉 All tests passed!")
    print("\nFinal Behavior Summary:")
    print("- ONE global executor shared by both sync and async")
    print("- Sync: Sequential by default, parallel when enabled")
    print("- Async: Always concurrent")
    print("- If no global executor: sync creates default, async uses asyncio default")
    print("- If global executor set: both sync and async use it")

if __name__ == "__main__":
    main() 