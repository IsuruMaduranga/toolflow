"""
Test schema generation and validation functionality.

This module tests:
- Tool schema generation
- Type annotation handling
- Parameter validation
- Metadata attachment
- Pydantic integration
"""
import pytest
from typing import List, Dict, Optional, Union, Any
from toolflow import tool
from toolflow.utils import get_tool_schema
import inspect


class TestBasicSchemaGeneration:
    """Test basic schema generation functionality."""
    
    def test_simple_function_schema(self):
        """Test schema generation for simple function."""
        @tool
        def simple_tool(a: int, b: str) -> str:
            """A simple tool."""
            return f"{a}: {b}"
        
        schema = simple_tool._tool_metadata
        assert schema['type'] == 'function'
        assert schema['function']['name'] == 'simple_tool'
        assert schema['function']['description'] == 'A simple tool.'
        
        properties = schema['function']['parameters']['properties']
        assert 'a' in properties
        assert 'b' in properties
    
    def test_function_with_no_parameters(self):
        """Test schema generation for function without parameters."""
        @tool
        def no_params_tool():
            """Tool with no parameters."""
            return "no params"
        
        schema = no_params_tool._tool_metadata
        parameters = schema['function']['parameters']
        assert parameters['properties'] == {}
        assert parameters['required'] == []
    
    def test_function_without_docstring(self):
        """Test schema generation for function without docstring."""
        @tool
        def no_docstring_tool(x: int):
            return x * 2
        
        schema = no_docstring_tool._tool_metadata
        # Should have default description or function name
        assert 'description' in schema['function']
        assert len(schema['function']['description']) > 0
    
    def test_custom_name_and_description(self):
        """Test schema generation with custom name and description."""
        @tool(name="custom_name", description="Custom description")
        def original_name():
            """Original description."""
            return "test"
        
        schema = original_name._tool_metadata
        assert schema['function']['name'] == 'custom_name'
        assert schema['function']['description'] == 'Custom description'


class TestComplexTypeHandling:
    """Test schema generation with complex types."""
    
    def test_function_with_union_types(self):
        """Test schema generation with Union types."""
        @tool
        def union_tool(value: Union[str, int]) -> str:
            """Tool accepting string or int."""
            return str(value)
        
        schema = union_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        assert 'value' in properties
        # Should handle Union type appropriately
    
    def test_function_with_any_type(self):
        """Test schema generation with Any type."""
        @tool
        def any_tool(data: Any) -> str:
            """Tool accepting any type."""
            return str(data)
        
        schema = any_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        assert 'data' in properties
    
    def test_function_with_nested_types(self):
        """Test schema generation with nested complex types."""
        @tool
        def nested_tool(items: List[Dict[str, Optional[int]]]) -> int:
            """Tool with nested type annotations."""
            return len(items)
        
        schema = nested_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        assert 'items' in properties
    
    def test_function_with_complex_defaults(self):
        """Test function with multiple default parameters."""
        @tool
        def multiple_defaults_tool(
            a: str,
            b: int = 10,
            c: float = 3.14,
            d: bool = True,
            e: List[str] = None
        ) -> str:
            """Tool with multiple default parameters."""
            return f"{a}-{b}-{c}-{d}-{len(e or [])}"
        
        schema = multiple_defaults_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        
        # All parameters should be in schema
        expected_params = ['a', 'b', 'c', 'd', 'e']
        for param in expected_params:
            assert param in properties


class TestParameterHandling:
    """Test parameter handling in schema generation."""
    
    def test_function_with_default_none(self):
        """Test function parameter with default None value."""
        @tool
        def default_none_tool(required: str, optional: str = None) -> str:
            """Tool with optional parameter defaulting to None."""
            return required + (optional or "")
        
        schema = default_none_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        required = schema['function']['parameters']['required']
        
        assert 'required' in properties
        assert 'optional' in properties
        assert 'required' in required
        assert 'optional' not in required
    
    def test_function_with_keyword_only_params(self):
        """Test function with keyword-only parameters."""
        @tool
        def keyword_only_tool(a: int, *, b: int, c: int = 20) -> int:
            """Tool with keyword-only params."""
            return a + b + c
        
        schema = keyword_only_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        required = schema['function']['parameters']['required']
        
        # Should include all parameters
        assert 'a' in properties
        assert 'b' in properties
        assert 'c' in properties
        assert 'a' in required
        assert 'b' in required
        assert 'c' not in required  # Has default value
    
    def test_function_without_type_annotations(self):
        """Test function without type annotations."""
        @tool
        def no_annotations_tool(x, y=5):
            """Tool without type annotations."""
            return x + y
        
        schema = no_annotations_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        
        # Should still include parameters
        assert 'x' in properties
        assert 'y' in properties
    
    def test_function_with_mixed_annotations(self):
        """Test function with some type annotations and some without."""
        @tool
        def mixed_annotations_tool(typed: str, untyped, default_typed: int = 10, default_untyped=20):
            """Tool with mixed type annotations."""
            return f"{typed}-{untyped}-{default_typed}-{default_untyped}"
        
        schema = mixed_annotations_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        
        # Should include all parameters
        expected_params = ['typed', 'untyped', 'default_typed', 'default_untyped']
        for param in expected_params:
            assert param in properties
    
    def test_complex_function_signature(self):
        """Test complex function signature with all parameter types."""
        @tool
        def complex_tool(
            pos_required: str,
            pos_default: int = 10,
            *args,
            kw_required: float,
            kw_default: bool = True,
            **kwargs
        ) -> str:
            """Complex tool with all parameter types."""
            return f"complex-{pos_required}-{pos_default}-{kw_required}-{kw_default}"
        
        schema = complex_tool._tool_metadata
        properties = schema['function']['parameters']['properties']
        
        # Should only include regular parameters, not *args/**kwargs
        assert 'pos_required' in properties
        assert 'pos_default' in properties
        assert 'kw_required' in properties
        assert 'kw_default' in properties
        assert len(properties) == 4  # No *args or **kwargs


class TestSchemaProperties:
    """Test schema properties and structure."""
    
    def test_schema_additional_properties_false(self):
        """Test that schema has additionalProperties set to False."""
        @tool
        def test_tool(param: str) -> str:
            return param
        
        schema = test_tool._tool_metadata
        parameters = schema['function']['parameters']
        
        # Should prevent additional properties
        assert parameters.get('additionalProperties') is False
    
    def test_schema_title_removed(self):
        """Test that 'title' is removed from schema."""
        @tool
        def test_tool(param: str) -> str:
            return param
        
        schema = test_tool._tool_metadata
        parameters = schema['function']['parameters']
        
        # Title should be removed for cleaner schema
        assert 'title' not in parameters
    
    def test_metadata_structure(self):
        """Test the overall structure of tool metadata."""
        @tool
        def test_tool(param: str) -> str:
            """Test tool."""
            return param
        
        schema = test_tool._tool_metadata
        
        # Check required top-level fields
        assert 'type' in schema
        assert 'function' in schema
        assert schema['type'] == 'function'
        
        # Check function structure
        function = schema['function']
        assert 'name' in function
        assert 'description' in function
        assert 'parameters' in function
        
        # Check parameters structure
        parameters = function['parameters']
        assert 'type' in parameters
        assert 'properties' in parameters
        assert 'required' in parameters
        assert parameters['type'] == 'object'


class TestSchemaGenerationUtils:
    """Test the utility functions for schema generation."""
    
    def test_get_tool_schema_function(self):
        """Test get_tool_schema utility function."""
        def test_func(a: int, b: str = "default") -> str:
            """Test function."""
            return f"{a}: {b}"

        schema = get_tool_schema(test_func, "test_func", "Test function.")
        
        assert schema['function']['name'] == 'test_func'
        assert schema['function']['description'] == 'Test function.'
        
        properties = schema['function']['parameters']['properties']
        assert 'a' in properties
        assert 'b' in properties
    
    def test_get_tool_schema_custom_name_description(self):
        """Test get_tool_schema with custom name and description."""
        def test_func(param: int) -> int:
            return param
        
        schema = get_tool_schema(test_func, name="custom", description="Custom desc")
        assert schema['function']['name'] == 'custom'
        assert schema['function']['description'] == 'Custom desc'
    
    def test_get_tool_schema_no_docstring(self):
        """Test get_tool_schema with function that has no docstring."""
        def no_doc_func(param: int) -> int:
            return param
        
        schema = get_tool_schema(no_doc_func, "no_doc_func", "No docstring function.")
        assert 'description' in schema['function']
        assert len(schema['function']['description']) > 0


class TestToolMetadataAttachment:
    """Test that metadata is properly attached to decorated functions."""
    
    def test_metadata_persistence_after_decoration(self):
        """Test that metadata persists after decoration."""
        @tool(name="custom", description="Custom desc")
        def test_tool(x: int) -> int:
            """Original desc."""
            return x
        
        # Metadata should be attached
        assert hasattr(test_tool, '_tool_metadata')
        
        # Should use custom values
        metadata = test_tool._tool_metadata
        assert metadata['function']['name'] == 'custom'
        assert metadata['function']['description'] == 'Custom desc'
    
    def test_wrapper_preserves_metadata(self):
        """Test that function wrapper preserves metadata."""
        @tool
        async def async_tool(x: int) -> int:
            """Async tool."""
            return x
        
        # Should have metadata
        assert hasattr(async_tool, '_tool_metadata')
        
        # Should preserve function properties
        assert async_tool.__name__ == 'async_tool'
        
        # Metadata should be correct
        metadata = async_tool._tool_metadata
        assert metadata['function']['name'] == 'async_tool'
        assert metadata['function']['description'] == 'Async tool.'


class TestPositionalOnlyParameters:
    """Test handling of positional-only parameters (Python 3.8+)."""
    
    def test_function_with_positional_only_params(self):
        """Test function with positional-only parameters (Python 3.8+)."""
        # Create function with positional-only parameters using exec
        # since syntax might not be supported in all environments
        func_code = '''
@tool
def pos_only_tool(a: int, b: str, /, c: float, d: int = 10):
    """Tool with positional-only params."""
    return a + len(b) + c + d
'''
        try:
            exec(func_code, globals())
            schema = pos_only_tool._tool_metadata  # noqa: F821
            properties = schema['function']['parameters']['properties']
            
            # Should include all parameters
            assert 'a' in properties
            assert 'b' in properties
            assert 'c' in properties
            assert 'd' in properties
        except SyntaxError:
            # Skip test if positional-only syntax not supported
            pytest.skip("Positional-only parameters not supported in this Python version")
