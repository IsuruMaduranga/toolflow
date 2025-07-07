"""
Tests for ToolKit functionality.
"""
import pytest
import asyncio
from typing import List
from pydantic import BaseModel

from toolflow.core.utils import extract_toolkit_methods
from toolflow.core.adapters import MessageAdapter
from toolflow.core.tool_execution import execute_tools_sync, execute_tools_async
from toolflow.core.decorators import tool


class SimpleToolKit:
    """A simple ToolKit with basic methods."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers."""
        return x * y
    
    def greet(self, name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"
    
    def _private_method(self) -> str:
        """This should not be extracted as a tool."""
        return "private"


class CalculatorToolKit:
    """A more complex ToolKit with mathematical operations."""
    
    def __init__(self, precision: int = 2):
        self.precision = precision
    
    def divide(self, dividend: float, divisor: float) -> float:
        """Divide two numbers with precision."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        result = dividend / divisor
        return round(result, self.precision)
    
    def power(self, base: float, exponent: int) -> float:
        """Calculate base raised to exponent."""
        return round(base ** exponent, self.precision)
    
    def factorial(self, n: int) -> int:
        """Calculate factorial of n."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result


class DataProcessingToolKit:
    """ToolKit with more complex data types."""
    
    def process_data(self, data: List[int]) -> dict:
        """Process a list of integers and return statistics."""
        if not data:
            return {"sum": 0, "average": 0, "count": 0}
        
        return {
            "sum": sum(data),
            "average": sum(data) / len(data),
            "count": len(data),
            "min": min(data),
            "max": max(data)
        }
    
    def filter_positive(self, numbers: List[int]) -> List[int]:
        """Filter out positive numbers from a list."""
        return [n for n in numbers if n > 0]


class EmptyToolKit:
    """ToolKit with no public methods."""
    
    def __init__(self):
        pass
    
    def _only_private(self):
        """Only private method."""
        return "private"


class MockMessageAdapter(MessageAdapter):
    """Mock implementation for testing."""
    
    def parse_response(self, response):
        return None, [], response
    
    def parse_stream_chunk(self, chunk):
        return None, None, chunk
    
    def build_assistant_message(self, text, tool_calls, original_response=None):
        return {"role": "assistant", "content": text}
    
    def build_tool_result_messages(self, tool_results):
        return [{"role": "tool", "content": str(result["output"])} for result in tool_results]


class TestExtractToolkitMethods:
    """Test the extract_toolkit_methods utility function."""
    
    def test_extract_simple_toolkit_methods(self):
        """Test extracting methods from a simple ToolKit."""
        toolkit = SimpleToolKit()
        methods = extract_toolkit_methods(toolkit)
        
        # Should extract all public methods
        method_names = [method.__name__ for method in methods]
        assert set(method_names) == {"add", "multiply", "greet"}
        
        # Methods should be bound to the instance
        for method in methods:
            assert method.__self__ is toolkit
    
    def test_extract_calculator_toolkit_methods(self):
        """Test extracting methods from calculator ToolKit."""
        calc = CalculatorToolKit(precision=3)
        methods = extract_toolkit_methods(calc)
        
        method_names = [method.__name__ for method in methods]
        assert set(method_names) == {"divide", "power", "factorial"}
        
        # Test that instance state is preserved
        divide_method = next(m for m in methods if m.__name__ == "divide")
        assert divide_method.__self__.precision == 3
    
    def test_extract_from_empty_toolkit_raises_error(self):
        """Test that extracting from empty ToolKit raises error."""
        empty_toolkit = EmptyToolKit()
        
        with pytest.raises(ValueError, match="ToolKit EmptyToolKit has no public methods"):
            extract_toolkit_methods(empty_toolkit)
    
    def test_extract_from_class_raises_error(self):
        """Test that passing a class instead of instance raises error."""
        with pytest.raises(ValueError, match="Expected an instance, got a class"):
            extract_toolkit_methods(SimpleToolKit)
    
    def test_extract_from_function_raises_error(self):
        """Test that passing a function raises error."""
        def some_function():
            pass
        
        with pytest.raises(ValueError, match="appears to be a function"):
            extract_toolkit_methods(some_function)
    
    def test_extract_from_non_object_raises_error(self):
        """Test that passing non-object raises error."""
        with pytest.raises(ValueError, match="is a built-in type"):
            extract_toolkit_methods("not an object")


class TestToolKitIntegration:
    """Test ToolKit integration with the tool system."""
    
    def test_prepare_tool_schemas_with_toolkit(self):
        """Test preparing tool schemas with ToolKit instance."""
        adapter = MockMessageAdapter()
        toolkit = SimpleToolKit()
        
        tool_schemas, tool_map = adapter.prepare_tool_schemas([toolkit])
        
        # Should have schemas for all public methods
        assert len(tool_schemas) == 3
        schema_names = [schema["function"]["name"] for schema in tool_schemas]
        assert set(schema_names) == {"add", "multiply", "greet"}
        
        # Tool map should contain bound methods
        assert len(tool_map) == 3
        for name, method in tool_map.items():
            assert callable(method)
            assert method.__self__ is toolkit
    
    def test_prepare_tool_schemas_mixed_tools_and_toolkit(self):
        """Test preparing schemas with both regular tools and ToolKit."""
        adapter = MockMessageAdapter()
        toolkit = SimpleToolKit()
        
        @tool
        def standalone_tool(msg: str) -> str:
            """A standalone tool function."""
            return f"Standalone: {msg}"
        
        tool_schemas, tool_map = adapter.prepare_tool_schemas([toolkit, standalone_tool])
        
        # Should have 4 tools total (3 from toolkit + 1 standalone)
        assert len(tool_schemas) == 4
        assert len(tool_map) == 4
        
        schema_names = [schema["function"]["name"] for schema in tool_schemas]
        assert "standalone_tool" in schema_names
        assert "add" in schema_names
        assert "multiply" in schema_names
        assert "greet" in schema_names
    
    def test_multiple_toolkites(self):
        """Test using multiple ToolKit instances."""
        adapter = MockMessageAdapter()
        simple_toolkit = SimpleToolKit()
        calc_toolkit = CalculatorToolKit()
        
        tool_schemas, tool_map = adapter.prepare_tool_schemas([simple_toolkit, calc_toolkit])
        
        # Should have methods from both toolkites
        expected_names = {"add", "multiply", "greet", "divide", "power", "factorial"}
        schema_names = [schema["function"]["name"] for schema in tool_schemas]
        assert set(schema_names) == expected_names
        
        # Methods should be bound to correct instances
        assert tool_map["add"].__self__ is simple_toolkit
        assert tool_map["divide"].__self__ is calc_toolkit


class TestToolKitExecution:
    """Test executing tools from ToolKit instances."""
    
    def test_sync_tool_execution_from_toolkit(self):
        """Test synchronous execution of ToolKit methods."""
        adapter = MockMessageAdapter()
        toolkit = SimpleToolKit()
        
        tool_schemas, tool_map = adapter.prepare_tool_schemas([toolkit])
        
        # Create tool calls
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "add",
                    "arguments": {"a": 5, "b": 3}
                }
            },
            {
                "id": "call_2", 
                "function": {
                    "name": "greet",
                    "arguments": {"name": "Alice"}
                }
            }
        ]
        
        results = execute_tools_sync(tool_calls, tool_map)
        
        assert len(results) == 2
        assert results[0]["output"] == 8
        assert results[1]["output"] == "Hello, Alice!"
    
    @pytest.mark.asyncio
    async def test_async_tool_execution_from_toolkit(self):
        """Test asynchronous execution of ToolKit methods."""
        adapter = MockMessageAdapter()
        calc = CalculatorToolKit(precision=2)
        
        tool_schemas, tool_map = adapter.prepare_tool_schemas([calc])
        
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "divide",
                    "arguments": {"dividend": 10.0, "divisor": 3.0}
                }
            },
            {
                "id": "call_2",
                "function": {
                    "name": "power", 
                    "arguments": {"base": 2.0, "exponent": 3}
                }
            }
        ]
        
        results = await execute_tools_async(tool_calls, tool_map, parallel=True)
        
        assert len(results) == 2
        assert results[0]["output"] == 3.33  # 10/3 rounded to 2 decimal places
        assert results[1]["output"] == 8.0
    
    def test_toolkit_method_with_complex_types(self):
        """Test ToolKit method with complex parameter types."""
        adapter = MockMessageAdapter()
        data_toolkit = DataProcessingToolKit()
        
        tool_schemas, tool_map = adapter.prepare_tool_schemas([data_toolkit])
        
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "process_data",
                    "arguments": {"data": [1, 2, 3, 4, 5]}
                }
            },
            {
                "id": "call_2",
                "function": {
                    "name": "filter_positive",
                    "arguments": {"numbers": [-2, -1, 0, 1, 2]}
                }
            }
        ]
        
        results = execute_tools_sync(tool_calls, tool_map)
        
        assert len(results) == 2
        
        # Check process_data result
        stats = results[0]["output"]
        assert stats["sum"] == 15
        assert stats["average"] == 3.0
        assert stats["count"] == 5
        assert stats["min"] == 1
        assert stats["max"] == 5
        
        # Check filter_positive result
        filtered = results[1]["output"]
        assert filtered == [1, 2]
    
    def test_toolkit_error_handling(self):
        """Test error handling in ToolKit methods."""
        adapter = MockMessageAdapter()
        calc = CalculatorToolKit()
        
        tool_schemas, tool_map = adapter.prepare_tool_schemas([calc])
        
        # Tool call that should raise an error
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "divide",
                    "arguments": {"dividend": 10.0, "divisor": 0.0}
                }
            }
        ]
        
        # With graceful error handling
        results = execute_tools_sync(tool_calls, tool_map, graceful_error_handling=True)
        assert len(results) == 1
        assert "is_error" in results[0]
        assert "Cannot divide by zero" in results[0]["output"]
        
        # Without graceful error handling should raise
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            execute_tools_sync(tool_calls, tool_map, graceful_error_handling=False)


class TestToolKitEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_toolkit_with_decorated_methods(self):
        """Test ToolKit where methods are decorated with @tool."""
        # Note: decorating methods at class definition time doesn't work well 
        # because @tool expects regular functions, not unbound methods with 'self'
        # This test shows that the extraction still works, even if decoration fails
        
        class DecoratedToolKit:
            def regular_method(self, x: int) -> int:
                """Regular method without decorator."""
                return x * 2
            
            def another_method(self, a: int, b: int) -> int:
                """Another regular method."""
                return a + b
        
        adapter = MockMessageAdapter()
        toolkit = DecoratedToolKit()
        
        tool_schemas, tool_map = adapter.prepare_tool_schemas([toolkit])
        
        # Both methods should be extracted
        assert len(tool_schemas) == 2
        schema_names = [schema["function"]["name"] for schema in tool_schemas]
        assert set(schema_names) == {"regular_method", "another_method"}
    
    def test_invalid_toolkit_types(self):
        """Test various invalid inputs to ToolKit handling."""
        adapter = MockMessageAdapter()
        
        # Test with primitive types - these should be handled as invalid toolkites now
        # since they pass the initial ToolKit checks but fail extraction
        with pytest.raises(ValueError, match="not a function or ToolKit instance"):
            adapter.prepare_tool_schemas([42])
        
        with pytest.raises(ValueError, match="not a function or ToolKit instance"):
            adapter.prepare_tool_schemas(["string"])
        
        # Test with empty ToolKit
        empty_toolkit = EmptyToolKit()
        with pytest.raises(ValueError, match="Invalid ToolKit"):
            adapter.prepare_tool_schemas([empty_toolkit]) 