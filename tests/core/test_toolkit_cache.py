"""
Tests for ToolKit schema caching functionality.
"""
import pytest
from toolflow.core.utils import (
    extract_toolkit_methods, 
    clear_toolkit_schema_cache, 
    get_toolkit_schema_cache_size,
    _get_toolkit_cache_key,
    _get_cached_toolkit_schema
)
from toolflow.core.adapters import MessageAdapter


class MockAdapter(MessageAdapter):
    """Mock adapter for testing."""
    
    def parse_response(self, response):
        return None, [], response
    
    def parse_stream_chunk(self, chunk):
        return None, None, chunk
    
    def build_assistant_message(self, text, tool_calls, original_response=None):
        return {"role": "assistant", "content": text}
    
    def build_tool_result_messages(self, tool_results):
        return [{"role": "tool", "content": str(result["output"])} for result in tool_results]


class TestToolKit:
    """A test ToolKit for caching tests."""
    
    def __init__(self, precision: int = 2):
        self.precision = precision
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return round(a + b, self.precision)
    
    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers."""
        return round(x * y, self.precision)


class TestToolKitSchemaCaching:
    """Test ToolKit schema caching functionality."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_toolkit_schema_cache()
    
    def test_cache_key_generation(self):
        """Test cache key generation for ToolKit methods."""
        toolkit = TestToolKit()
        
        # Test cache key format
        cache_key = _get_toolkit_cache_key(toolkit, "add")
        expected_key = f"{TestToolKit.__module__}.{TestToolKit.__name__}:add"
        assert cache_key == expected_key
        
        cache_key = _get_toolkit_cache_key(toolkit, "multiply")
        expected_key = f"{TestToolKit.__module__}.{TestToolKit.__name__}:multiply"
        assert cache_key == expected_key
    
    def test_cache_initial_state(self):
        """Test initial cache state."""
        assert get_toolkit_schema_cache_size() == 0
        
        # Test getting non-existent cached schema
        toolkit = TestToolKit()
        cached_schema = _get_cached_toolkit_schema(toolkit, "add")
        assert cached_schema is None
    
    def test_schema_caching_on_first_use(self):
        """Test that schemas are cached on first use."""
        adapter = MockAdapter()
        toolkit = TestToolKit()
        
        # First use - should generate and cache schemas
        tool_schemas, tool_map = adapter.prepare_tool_schemas([toolkit])
        
        # Should have 2 schemas (add and multiply)
        assert len(tool_schemas) == 2
        assert get_toolkit_schema_cache_size() == 2
        
        # Verify both methods are cached
        add_schema = _get_cached_toolkit_schema(toolkit, "add")
        multiply_schema = _get_cached_toolkit_schema(toolkit, "multiply")
        
        assert add_schema is not None
        assert multiply_schema is not None
        assert add_schema["function"]["name"] == "add"
        assert multiply_schema["function"]["name"] == "multiply"
    
    def test_schema_reuse_on_second_use(self):
        """Test that cached schemas are reused on subsequent calls."""
        adapter = MockAdapter()
        toolkit = TestToolKit()
        
        # First use - populate cache
        tool_schemas1, _ = adapter.prepare_tool_schemas([toolkit])
        cache_size_after_first = get_toolkit_schema_cache_size()
        
        # Second use - should reuse cached schemas
        tool_schemas2, _ = adapter.prepare_tool_schemas([toolkit])
        cache_size_after_second = get_toolkit_schema_cache_size()
        
        # Cache size should remain the same (no new entries)
        assert cache_size_after_second == cache_size_after_first
        
        # Schemas should be identical
        assert tool_schemas1 == tool_schemas2
    
    def test_cache_with_multiple_toolkit_instances(self):
        """Test caching with multiple instances of the same ToolKit class."""
        adapter = MockAdapter()
        toolkit1 = TestToolKit(precision=2)
        toolkit2 = TestToolKit(precision=3)
        
        # Use first instance
        tool_schemas1, _ = adapter.prepare_tool_schemas([toolkit1])
        cache_size_after_first = get_toolkit_schema_cache_size()
        
        # Use second instance - should reuse cache since it's the same class
        tool_schemas2, _ = adapter.prepare_tool_schemas([toolkit2])
        cache_size_after_second = get_toolkit_schema_cache_size()
        
        # Cache size should remain the same (same class, different instances)
        assert cache_size_after_second == cache_size_after_first
        
        # Schemas should be identical (same method signatures)
        assert tool_schemas1 == tool_schemas2
    
    def test_cache_with_different_toolkit_classes(self):
        """Test caching with different ToolKit classes."""
        adapter = MockAdapter()
        
        class DifferentToolKit:
            def different_method(self, x: int) -> int:
                """A different method."""
                return x * 2
        
        toolkit1 = TestToolKit()
        toolkit2 = DifferentToolKit()
        
        # Use first ToolKit
        tool_schemas1, _ = adapter.prepare_tool_schemas([toolkit1])
        cache_size_after_first = get_toolkit_schema_cache_size()
        
        # Use second ToolKit - should add new cache entries
        tool_schemas2, _ = adapter.prepare_tool_schemas([toolkit2])
        cache_size_after_second = get_toolkit_schema_cache_size()
        
        # Cache size should increase (different classes)
        assert cache_size_after_second > cache_size_after_first
        
        # Schemas should be different
        assert tool_schemas1 != tool_schemas2
    
    def test_cache_clear_functionality(self):
        """Test cache clearing functionality."""
        adapter = MockAdapter()
        toolkit = TestToolKit()
        
        # Populate cache
        adapter.prepare_tool_schemas([toolkit])
        assert get_toolkit_schema_cache_size() > 0
        
        # Clear cache
        clear_toolkit_schema_cache()
        assert get_toolkit_schema_cache_size() == 0
        
        # Verify cached schemas are gone
        cached_schema = _get_cached_toolkit_schema(toolkit, "add")
        assert cached_schema is None
    
    def test_mixed_toolkit_and_regular_functions(self):
        """Test caching works correctly with mixed ToolKits and regular functions."""
        adapter = MockAdapter()
        toolkit = TestToolKit()
        
        def regular_function(x: int) -> int:
            """A regular function."""
            return x + 1
        
        # Use mixed tools
        tool_schemas, tool_map = adapter.prepare_tool_schemas([toolkit, regular_function])
        
        # Should have 3 schemas total (2 from toolkit + 1 from function)
        assert len(tool_schemas) == 3
        
        # Only ToolKit methods should be cached
        assert get_toolkit_schema_cache_size() == 2
        
        # Verify ToolKit methods are cached
        add_schema = _get_cached_toolkit_schema(toolkit, "add")
        multiply_schema = _get_cached_toolkit_schema(toolkit, "multiply")
        assert add_schema is not None
        assert multiply_schema is not None
    
    def test_cache_performance_benefit(self):
        """Test that caching provides performance benefit."""
        adapter = MockAdapter()
        toolkit = TestToolKit()
        
        import time
        
        # First call - should be slower (schema generation)
        start_time = time.time()
        adapter.prepare_tool_schemas([toolkit])
        first_call_time = time.time() - start_time
        
        # Second call - should be faster (cached)
        start_time = time.time()
        adapter.prepare_tool_schemas([toolkit])
        second_call_time = time.time() - start_time
        
        # Second call should be significantly faster
        # (Note: This is a basic test, actual performance may vary)
        assert second_call_time < first_call_time * 0.5  # At least 50% faster 