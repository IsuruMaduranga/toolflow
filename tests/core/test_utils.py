"""
Tests for core utility functions.
"""
import pytest
from typing import List, Optional, Dict, Any
from unittest.mock import Mock
from pydantic import BaseModel

from toolflow.core.utils import (
    get_tool_schema,
    filter_toolflow_params,
    get_structured_output_tool
)
from toolflow import tool


class TestGetToolSchema:
    """Test tool schema generation."""
    
    def test_simple_function_schema(self):
        """Test schema generation for simple function."""
        def simple_func(x: int, y: str) -> bool:
            """A simple function."""
            return True
        
        schema = get_tool_schema(simple_func)
        
        assert schema['type'] == 'function'
        assert schema['function']['name'] == 'simple_func'
        assert 'A simple function.' in schema['function']['description']
        
        params = schema['function']['parameters']
        assert params['type'] == 'object'
        assert 'x' in params['properties']
        assert 'y' in params['properties']
    
    def test_function_with_optional_params(self):
        """Test schema generation with optional parameters."""
        def func_with_defaults(required: str, optional: int = 42) -> str:
            """Function with optional parameter."""
            return f"{required}-{optional}"
        
        schema = get_tool_schema(func_with_defaults)
        
        params = schema['function']['parameters']
        assert 'required' in params['properties']
        assert 'optional' in params['properties']
    
    def test_function_with_complex_types(self):
        """Test schema generation with complex types."""
        def complex_func(items: List[str], config: Dict[str, Any], count: Optional[int] = None) -> dict:
            """Function with complex types."""
            return {}
        
        schema = get_tool_schema(complex_func)
        
        params = schema['function']['parameters']
        assert 'items' in params['properties']
        assert 'config' in params['properties']
        assert 'count' in params['properties']
    
    def test_function_with_no_docstring(self):
        """Test schema generation for function without docstring."""
        def no_docs(x: int) -> int:
            return x * 2
        
        schema = get_tool_schema(no_docs)
        
        assert schema['function']['name'] == 'no_docs'
        # Should have some description
        assert isinstance(schema['function']['description'], str)
        assert len(schema['function']['description']) > 0
    
    def test_function_with_no_annotations(self):
        """Test schema generation for function without type annotations."""
        def no_annotations(x, y=None):
            """Function without annotations."""
            return x
        
        schema = get_tool_schema(no_annotations)
        
        assert schema['function']['name'] == 'no_annotations'
        params = schema['function']['parameters']
        
        # Should still include parameters
        assert 'x' in params['properties']
        assert 'y' in params['properties']
    
    def test_decorated_function_schema(self):
        """Test schema generation for @tool decorated function."""
        @tool
        def decorated_func(value: float) -> str:
            """A decorated function."""
            return str(value)
        
        schema = get_tool_schema(decorated_func)
        
        assert schema['function']['name'] == 'decorated_func'
        assert 'A decorated function.' in schema['function']['description']
        assert 'value' in schema['function']['parameters']['properties']
    
    def test_custom_name_and_description(self):
        """Test schema generation with custom name and description."""
        def test_func(x: int) -> str:
            """Original description."""
            return str(x)
        
        schema = get_tool_schema(test_func, name="custom_name", description="Custom description")
        
        assert schema['function']['name'] == 'custom_name'
        assert schema['function']['description'] == 'Custom description'


class TestFilterToolflowParams:
    """Test toolflow parameter filtering."""
    
    def test_filter_basic_params(self):
        """Test filtering basic toolflow parameters."""
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tool_calls": 5,
            "parallel_tool_execution": True,
            "temperature": 0.7
        }
        
        filtered_kwargs, max_tool_calls, parallel_tool_execution, response_format, full_response, graceful_error_handling = filter_toolflow_params(**kwargs)
        
        # Check that toolflow params were extracted
        assert max_tool_calls == 5
        assert parallel_tool_execution is True
        assert response_format is None  # Default
        assert full_response is False  # Default
        assert graceful_error_handling is True  # Default
        
        # Check that other params remain
        assert filtered_kwargs["model"] == "gpt-4o-mini"
        assert filtered_kwargs["temperature"] == 0.7
        assert "max_tool_calls" not in filtered_kwargs
        assert "parallel_tool_execution" not in filtered_kwargs
    
    def test_filter_all_params(self):
        """Test filtering all toolflow parameters."""
        class MockModel(BaseModel):
            name: str
        
        kwargs = {
            "model": "gpt-4o-mini",
            "max_tool_calls": 10,
            "parallel_tool_execution": False,
            "response_format": MockModel,
            "full_response": True,
            "graceful_error_handling": False,
            "temperature": 0.8
        }
        
        filtered_kwargs, max_tool_calls, parallel_tool_execution, response_format, full_response, graceful_error_handling = filter_toolflow_params(**kwargs)
        
        assert max_tool_calls == 10
        assert parallel_tool_execution is False
        assert response_format == MockModel
        assert full_response is True
        assert graceful_error_handling is False
        
        # Only non-toolflow params should remain
        assert len(filtered_kwargs) == 2
        assert filtered_kwargs["model"] == "gpt-4o-mini"
        assert filtered_kwargs["temperature"] == 0.8
    
    def test_filter_default_values(self):
        """Test that default values are used when params not provided."""
        kwargs = {"model": "gpt-4o-mini"}
        
        filtered_kwargs, max_tool_calls, parallel_tool_execution, response_format, full_response, graceful_error_handling = filter_toolflow_params(**kwargs)
        
        # Should use defaults from DEFAULT_PARAMS
        assert max_tool_calls == 5  # From constants
        assert parallel_tool_execution is False  # From constants
        assert response_format is None
        assert full_response is False
        assert graceful_error_handling is True


class TestGetStructuredOutputTool:
    """Test structured output tool generation."""
    
    def test_create_structured_output_tool(self):
        """Test creating a structured output tool."""
        class TestModel(BaseModel):
            name: str
            value: int
        
        tool_func = get_structured_output_tool(TestModel)
        
        # Check that it's a callable function
        assert callable(tool_func)
        
        # Check function properties
        assert tool_func.__name__ == "final_response_tool_internal"  # From constants
        assert hasattr(tool_func, "__internal_tool__")
        assert tool_func.__internal_tool__ is True
        
        # Check docstring includes model name
        assert "TestModel" in tool_func.__doc__
        assert "final response" in tool_func.__doc__.lower()
    
    def test_structured_output_tool_with_different_models(self):
        """Test creating tools for different Pydantic models."""
        class ModelA(BaseModel):
            field_a: str
        
        class ModelB(BaseModel):
            field_b: int
        
        tool_a = get_structured_output_tool(ModelA)
        tool_b = get_structured_output_tool(ModelB)
        
        # Both should be callable but different instances
        assert callable(tool_a)
        assert callable(tool_b)
        assert tool_a != tool_b
        
        # Docstrings should reference correct model names
        assert "ModelA" in tool_a.__doc__
        assert "ModelB" in tool_b.__doc__


class TestUtilityHelpers:
    """Test utility helper functions."""
    
    def test_error_handling_in_get_tool_schema(self):
        """Test error handling in get_tool_schema."""
        # Test with non-function input
        with pytest.raises((ValueError, TypeError, AttributeError)):
            get_tool_schema("not_a_function")
    
    def test_schema_consistency(self):
        """Test that schema generation is consistent."""
        def test_func(x: int) -> str:
            """Test function."""
            return str(x)
        
        # Generate schema multiple times
        schema1 = get_tool_schema(test_func)
        schema2 = get_tool_schema(test_func)
        
        # Should be identical
        assert schema1 == schema2
        assert schema1['type'] == 'function'
        assert schema1['function']['name'] == 'test_func'
    
    def test_schema_strict_parameter(self):
        """Test schema generation with strict parameter."""
        def test_func(x: int) -> str:
            """Test function."""
            return str(x)
        
        # Test with strict=True
        schema_strict = get_tool_schema(test_func, strict=True)
        assert schema_strict['function']['strict'] is True
        
        # Test with strict=False (default)
        schema_normal = get_tool_schema(test_func, strict=False)
        assert schema_normal['function']['strict'] is False 