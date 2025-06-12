"""
Basic smoke tests for toolflow library.

This module contains simple smoke tests to verify basic functionality.
More comprehensive tests are in dedicated test modules:
- test_core_functionality.py: Core decorator and integration tests
- test_async_functionality.py: Async-specific tests
- test_parallel_execution.py: Parallel execution tests  
- test_schema_generation.py: Schema generation and validation
- test_error_handling.py: Error handling and edge cases
"""
import pytest
from toolflow import tool, from_openai


def test_smoke_basic_imports():
    """Smoke test: verify basic imports work."""
    assert tool is not None
    assert from_openai is not None


def test_smoke_decorator_works():
    """Smoke test: verify @tool decorator works."""
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    # Basic functionality
    assert add(2, 3) == 5
    assert hasattr(add, '_tool_metadata')
    assert add._tool_metadata['function']['name'] == 'add'


def test_smoke_async_import():
    """Smoke test: verify async imports work."""
    try:
        from toolflow import from_openai_async
        assert from_openai_async is not None
    except ImportError:
        pytest.skip("Async functionality not available")


if __name__ == "__main__":
    # Quick manual smoke test
    print("Running smoke tests...")
    test_smoke_basic_imports()
    test_smoke_decorator_works()
    test_smoke_async_import()
    print("✓ All smoke tests passed!")
