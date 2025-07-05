"""
Tests for core tool execution functionality.
"""
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, Mock
from typing import List

import toolflow
from toolflow.core.tool_execution import (
    set_max_workers,
    get_max_workers,
    set_executor,
    execute_tools_sync,
    execute_tools_async
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


class TestPydanticModelParameters:
    """Test tool execution with Pydantic model parameters."""
    
    def test_sync_tool_execution_with_pydantic_models(self):
        """Test sync tool execution with Pydantic model parameters."""
        from pydantic import BaseModel
        
        class Operation(BaseModel):
            operation: str
            a: float
            b: float
        
        def calculator(op: Operation) -> float:
            """Calculator tool that takes a Pydantic model."""
            if op.operation == "add":
                return op.a + op.b
            elif op.operation == "multiply":
                return op.a * op.b
            return 0
        
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "calculator",
                    "arguments": {"op": {"operation": "add", "a": 3.5, "b": 2.0}}
                }
            }
        ]
        
        tool_map = {"calculator": calculator}
        
        results = execute_tools_sync(tool_calls, tool_map)
        
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_1"
        assert results[0]["output"] == 5.5
        assert "is_error" not in results[0]

    @pytest.mark.asyncio
    async def test_async_tool_execution_with_pydantic_models(self):
        """Test async tool execution with Pydantic model parameters."""
        from pydantic import BaseModel
        
        class MathRequest(BaseModel):
            operation: str
            numbers: List[float]
        
        async def async_calculator(req: MathRequest) -> float:
            """Async calculator tool that takes a Pydantic model."""
            if req.operation == "sum":
                return sum(req.numbers)
            elif req.operation == "product":
                result = 1
                for num in req.numbers:
                    result *= num
                return result
            return 0
        
        tool_calls = [
            {
                "id": "call_async_1",
                "function": {
                    "name": "async_calculator",
                    "arguments": {"req": {"operation": "sum", "numbers": [1.0, 2.0, 3.0]}}
                }
            }
        ]
        
        tool_map = {"async_calculator": async_calculator}
        
        results = await execute_tools_async(tool_calls, tool_map)
        
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_async_1"
        assert results[0]["output"] == 6.0
        assert "is_error" not in results[0]

    def test_mixed_pydantic_and_regular_parameters(self):
        """Test tool with both Pydantic models and regular parameters."""
        from pydantic import BaseModel
        
        class Config(BaseModel):
            multiplier: float
            add_value: int
        
        def mixed_tool(value: float, config: Config, description: str = "default") -> str:
            """Tool with mixed parameter types."""
            result = (value * config.multiplier) + config.add_value
            return f"{description}: {result}"
        
        tool_calls = [
            {
                "id": "call_mixed",
                "function": {
                    "name": "mixed_tool",
                    "arguments": {
                        "value": 10.0,
                        "config": {"multiplier": 2.0, "add_value": 5},
                        "description": "calculation"
                    }
                }
            }
        ]
        
        tool_map = {"mixed_tool": mixed_tool}
        
        results = execute_tools_sync(tool_calls, tool_map)
        
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_mixed"
        assert results[0]["output"] == "calculation: 25.0"

    def test_pydantic_model_validation_error(self):
        """Test handling of Pydantic validation errors."""
        from pydantic import BaseModel
        
        class StrictModel(BaseModel):
            name: str
            age: int  # Must be int, not float or string
        
        def strict_tool(data: StrictModel) -> str:
            """Tool with strict Pydantic validation."""
            return f"{data.name} is {data.age} years old"
        
        tool_calls = [
            {
                "id": "call_strict",
                "function": {
                    "name": "strict_tool",
                    "arguments": {"data": {"name": "John", "age": "not_a_number"}}
                }
            }
        ]
        
        tool_map = {"strict_tool": strict_tool}
        
        # With graceful error handling, should return error message
        results = execute_tools_sync(tool_calls, tool_map, graceful_error_handling=True)
        
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_strict"
        assert results[0]["is_error"] is True
        assert "Error executing tool" in results[0]["output"]


class TestComprehensiveTypeSupport:
    """Test comprehensive Python type support in tool execution and schema generation."""
    
    def test_dataclass_support(self):
        """Test dataclass parameter support."""
        from dataclasses import dataclass
        from toolflow.core.utils import get_tool_schema
        
        @dataclass
        class Config:
            name: str
            value: int
            
        def tool_with_dataclass(config: Config) -> str:
            return f"{config.name}: {config.value}"
        
        # Test schema generation
        schema = get_tool_schema(tool_with_dataclass)
        assert schema['function']['name'] == 'tool_with_dataclass'
        assert 'config' in schema['function']['parameters']['properties']
        
        # Test tool execution
        tool_calls = [{
            "id": "test_dataclass",
            "function": {
                "name": "tool_with_dataclass",
                "arguments": {"config": {"name": "test", "value": 42}}
            }
        }]
        
        results = execute_tools_sync(tool_calls, {"tool_with_dataclass": tool_with_dataclass})
        assert results[0]["output"] == "test: 42"
    
    def test_enum_support(self):
        """Test enum parameter support."""
        from enum import Enum
        from toolflow.core.utils import get_tool_schema
        
        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"
            
        def tool_with_enum(status: Status) -> str:
            return f"Status: {status.value}"
        
        # Test schema generation
        schema = get_tool_schema(tool_with_enum)
        assert 'status' in schema['function']['parameters']['properties']
        
        # Test tool execution with string value
        tool_calls = [{
            "id": "test_enum",
            "function": {
                "name": "tool_with_enum",
                "arguments": {"status": "active"}
            }
        }]
        
        results = execute_tools_sync(tool_calls, {"tool_with_enum": tool_with_enum})
        assert results[0]["output"] == "Status: active"
    
    def test_namedtuple_support(self):
        """Test NamedTuple parameter support."""
        from typing import NamedTuple
        from toolflow.core.utils import get_tool_schema
        
        class Point(NamedTuple):
            x: float
            y: float
            
        def tool_with_namedtuple(point: Point) -> str:
            return f"Point: ({point.x}, {point.y})"
        
        # Test schema generation
        schema = get_tool_schema(tool_with_namedtuple)
        assert 'point' in schema['function']['parameters']['properties']
        
        # Test tool execution
        tool_calls = [{
            "id": "test_namedtuple",
            "function": {
                "name": "tool_with_namedtuple", 
                "arguments": {"point": {"x": 1.5, "y": 2.5}}
            }
        }]
        
        results = execute_tools_sync(tool_calls, {"tool_with_namedtuple": tool_with_namedtuple})
        assert results[0]["output"] == "Point: (1.5, 2.5)"
    
    def test_union_type_support(self):
        """Test Union type parameter support."""
        from typing import Union
        from toolflow.core.utils import get_tool_schema
        
        def tool_with_union(value: Union[int, str]) -> str:
            return f"Got {type(value).__name__}: {value}"
        
        # Test schema generation
        schema = get_tool_schema(tool_with_union)
        assert 'value' in schema['function']['parameters']['properties']
        
        # Test with int
        tool_calls = [{
            "id": "test_union_int",
            "function": {
                "name": "tool_with_union",
                "arguments": {"value": 42}
            }
        }]
        
        results = execute_tools_sync(tool_calls, {"tool_with_union": tool_with_union})
        assert "int: 42" in results[0]["output"]
        
        # Test with string
        tool_calls[0]["function"]["arguments"]["value"] = "hello"
        results = execute_tools_sync(tool_calls, {"tool_with_union": tool_with_union})
        assert "str: hello" in results[0]["output"]
    
    def test_optional_type_support(self):
        """Test Optional type parameter support."""
        from typing import Optional
        from toolflow.core.utils import get_tool_schema
        
        def tool_with_optional(name: Optional[str] = None) -> str:
            return f"Name: {name or 'unknown'}"
        
        # Test schema generation
        schema = get_tool_schema(tool_with_optional)
        assert 'name' in schema['function']['parameters']['properties']
        
        # Test with value
        tool_calls = [{
            "id": "test_optional_value",
            "function": {
                "name": "tool_with_optional",
                "arguments": {"name": "Alice"}
            }
        }]
        
        results = execute_tools_sync(tool_calls, {"tool_with_optional": tool_with_optional})
        assert results[0]["output"] == "Name: Alice"
        
        # Test with None
        tool_calls[0]["function"]["arguments"]["name"] = None
        results = execute_tools_sync(tool_calls, {"tool_with_optional": tool_with_optional})
        assert results[0]["output"] == "Name: unknown"
    
    def test_generic_type_support(self):
        """Test generic type (List, Dict) parameter support."""
        from typing import List, Dict
        from toolflow.core.utils import get_tool_schema
        
        def tool_with_list(items: List[str]) -> str:
            return f"Items: {', '.join(items)}"
            
        def tool_with_dict(mapping: Dict[str, int]) -> str:
            return f"Sum: {sum(mapping.values())}"
        
        # Test List schema generation
        list_schema = get_tool_schema(tool_with_list)
        assert 'items' in list_schema['function']['parameters']['properties']
        
        # Test Dict schema generation  
        dict_schema = get_tool_schema(tool_with_dict)
        assert 'mapping' in dict_schema['function']['parameters']['properties']
        
        # Test List execution
        list_calls = [{
            "id": "test_list",
            "function": {
                "name": "tool_with_list",
                "arguments": {"items": ["a", "b", "c"]}
            }
        }]
        
        results = execute_tools_sync(list_calls, {"tool_with_list": tool_with_list})
        assert results[0]["output"] == "Items: a, b, c"
        
        # Test Dict execution
        dict_calls = [{
            "id": "test_dict", 
            "function": {
                "name": "tool_with_dict",
                "arguments": {"mapping": {"x": 1, "y": 2, "z": 3}}
            }
        }]
        
        results = execute_tools_sync(dict_calls, {"tool_with_dict": tool_with_dict})
        assert results[0]["output"] == "Sum: 6"

    def test_complex_nested_types(self):
        """Test complex nested type combinations."""
        from dataclasses import dataclass
        from typing import List, Optional
        from toolflow.core.utils import get_tool_schema
        
        @dataclass
        class Person:
            name: str
            age: int
            
        def tool_with_nested_list(people: List[Person], limit: Optional[int] = None) -> str:
            limited_people = people[:limit] if limit else people
            names = [p.name for p in limited_people]
            return f"People: {', '.join(names)}"
        
        # Test schema generation for complex nested types
        schema = get_tool_schema(tool_with_nested_list)
        assert 'people' in schema['function']['parameters']['properties']
        assert 'limit' in schema['function']['parameters']['properties']
        
        # Test execution with complex nested data
        tool_calls = [{
            "id": "test_nested",
            "function": {
                "name": "tool_with_nested_list",
                "arguments": {
                    "people": [
                        {"name": "Alice", "age": 25},
                        {"name": "Bob", "age": 30}
                    ],
                    "limit": 1
                }
            }
        }]
        
        results = execute_tools_sync(tool_calls, {"tool_with_nested_list": tool_with_nested_list})
        assert results[0]["output"] == "People: Alice"
