"""
Enhanced test suite for advanced schema generation functionality.

This module tests new schema generation features including:
- Strict vs non-strict schema generation
- Internal tool functionality
- Enhanced schema properties
- Tool metadata variations
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import List, Optional, Union

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

import toolflow
from toolflow.utils import get_tool_schema
from toolflow.decorators import tool


class TestStrictSchemaGeneration:
    """Test strict vs non-strict schema generation differences."""

    def test_strict_vs_non_strict_additional_properties(self):
        """Test that both strict and non-strict modes set additionalProperties to False."""
        def test_func(param: str) -> str:
            """Test function."""
            return param

        # Non-strict schema
        schema = get_tool_schema(test_func, strict=False)
        params = schema['function']['parameters']
        assert params.get('additionalProperties') is False

        # Strict schema
        strict_schema = get_tool_schema(test_func, strict=True)
        strict_params = strict_schema['function']['parameters']
        assert strict_params.get('additionalProperties') is False

    def test_strict_flag_in_schema(self):
        """Test that strict flag is included in the schema when strict=True."""
        def test_func(param: str) -> str:
            """Test function."""
            return param

        # Non-strict schema
        schema = get_tool_schema(test_func, strict=False)
        assert schema['function']['strict'] is False

        # Strict schema
        strict_schema = get_tool_schema(test_func, strict=True)
        assert strict_schema['function']['strict'] is True

    def test_decorator_creates_both_metadata_versions(self):
        """Test that @tool decorator creates both strict and non-strict metadata."""
        @tool
        def test_func(param: str) -> str:
            """Test function."""
            return param

        # Check both metadata versions exist
        assert hasattr(test_func, '_tool_metadata')
        assert hasattr(test_func, '_tool_metadata_strict')

        # Check strict flag differences
        assert test_func._tool_metadata['function']['strict'] is False
        assert test_func._tool_metadata_strict['function']['strict'] is True

    def test_strict_schema_consistency(self):
        """Test that strict schemas are consistent across different parameter types."""
        @tool
        def complex_func(
            required_str: str,
            optional_int: int = 10,
            optional_list: List[str] = None
        ) -> str:
            """Complex function with various parameter types."""
            return f"{required_str}-{optional_int}-{len(optional_list or [])}"

        strict_schema = complex_func._tool_metadata_strict
        params = strict_schema['function']['parameters']
        
        # All strict schemas should have additionalProperties False
        assert params.get('additionalProperties') is False
        assert strict_schema['function']['strict'] is True
        
        # Required parameters should be properly identified
        required = params.get('required', [])
        assert 'required_str' in required
        assert 'optional_int' not in required
        assert 'optional_list' not in required

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_strict_schema_with_pydantic_models(self):
        """Test strict schema generation with Pydantic model parameters."""
        class UserInfo(BaseModel):
            name: str
            age: int
            email: Optional[str] = None

        @tool
        def process_user(user: UserInfo, validate: bool = True) -> str:
            """Process user information."""
            return f"Processing {user.name}"

        strict_schema = process_user._tool_metadata_strict
        params = strict_schema['function']['parameters']
        
        assert params.get('additionalProperties') is False
        assert strict_schema['function']['strict'] is True

        # Check that Pydantic model is properly included
        properties = params.get('properties', {})
        assert 'user' in properties
        assert 'validate' in properties

    def test_empty_function_strict_schema(self):
        """Test strict schema for function with no parameters."""
        @tool
        def no_params() -> str:
            """Function with no parameters."""
            return "test"

        strict_schema = no_params._tool_metadata_strict
        params = strict_schema['function']['parameters']

        assert params == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
        assert strict_schema['function']['strict'] is True


class TestInternalToolFunctionality:
    """Test internal tool functionality and restrictions."""

    def test_internal_tool_creation(self):
        """Test that internal tools can be created with internal=True."""
        @tool(internal=True)
        def internal_func() -> str:
            """Internal function."""
            return "internal"

        # Internal tools should still have metadata
        assert hasattr(internal_func, '_tool_metadata')
        assert internal_func._tool_metadata['function']['name'] == 'internal_func'

    def test_internal_tool_name_restriction(self):
        """Test that final_response_tool_internal name is restricted for non-internal tools."""
        with pytest.raises(ValueError, match="final_response_tool_internal is an internally used tool"):
            @tool(name="final_response_tool_internal")
            def restricted_func() -> str:
                """This should fail."""
                return "fail"

    def test_internal_tool_allows_restricted_name(self):
        """Test that internal tools can use the restricted name."""
        @tool(name="final_response_tool_internal", internal=True)
        def allowed_internal_func() -> str:
            """This should work."""
            return "allowed"

        assert allowed_internal_func._tool_metadata['function']['name'] == 'final_response_tool_internal'

    def test_structured_output_internal_tool_creation(self):
        """Test that structured output creates internal response tools correctly."""
        from toolflow.providers.openai.structured_output import create_openai_response_tool

        if not PYDANTIC_AVAILABLE:
            pytest.skip("Pydantic not available")

        class TestModel(BaseModel):
            value: str

        response_tool = create_openai_response_tool(TestModel)
        
        # Should be an internal tool with the specific name
        assert response_tool._tool_metadata['function']['name'] == 'final_response_tool_internal'
        assert hasattr(response_tool, '_tool_metadata')
        assert hasattr(response_tool, '_tool_metadata_strict')


class TestEnhancedSchemaProperties:
    """Test enhanced schema properties and validation."""

    def test_schema_required_field_always_present(self):
        """Test that required field is always present in schema."""
        # Function with no parameters
        @tool
        def no_params() -> str:
            return "test"

        schema = no_params._tool_metadata
        assert 'required' in schema['function']['parameters']
        assert schema['function']['parameters']['required'] == []

        # Function with parameters
        @tool
        def with_params(a: str, b: int = 10) -> str:
            return f"{a}-{b}"

        schema_with_params = with_params._tool_metadata
        assert 'required' in schema_with_params['function']['parameters']
        assert 'a' in schema_with_params['function']['parameters']['required']
        assert 'b' not in schema_with_params['function']['parameters']['required']

    def test_schema_title_removal(self):
        """Test that title is properly removed from generated schemas."""
        @tool
        def test_func(param: str) -> str:
            """Test function."""
            return param

        schema = test_func._tool_metadata
        params = schema['function']['parameters']
        
        # Title should be removed from the main parameters object
        assert 'title' not in params

    def test_function_description_fallback(self):
        """Test function description fallback mechanism."""
        # Function without docstring
        @tool
        def no_docstring(param: str) -> str:
            return param

        schema = no_docstring._tool_metadata
        # Should use function name as fallback
        assert schema['function']['description'] == 'no_docstring'

        # Function with custom description
        @tool(description="Custom description")
        def custom_desc(param: str) -> str:
            """Original docstring."""
            return param

        custom_schema = custom_desc._tool_metadata
        assert custom_schema['function']['description'] == 'Custom description'

        # Function with docstring
        @tool
        def with_docstring(param: str) -> str:
            """Function with docstring."""
            return param

        docstring_schema = with_docstring._tool_metadata
        assert docstring_schema['function']['description'] == 'Function with docstring.'

    def test_complex_type_handling_in_strict_mode(self):
        """Test that complex types are handled correctly in strict mode."""
        @tool
        def complex_types(
            union_param: Union[str, int],
            list_param: List[str],
            optional_param: Optional[int] = None
        ) -> str:
            """Function with complex type annotations."""
            return "test"

        strict_schema = complex_types._tool_metadata_strict
        params = strict_schema['function']['parameters']
        
        assert params.get('additionalProperties') is False
        assert strict_schema['function']['strict'] is True
        
        properties = params.get('properties', {})
        assert 'union_param' in properties
        assert 'list_param' in properties
        assert 'optional_param' in properties

    def test_schema_structure_validation(self):
        """Test that generated schemas have correct OpenAI structure."""
        @tool
        def test_func(param: str) -> str:
            """Test function."""
            return param

        schema = test_func._tool_metadata
        
        # Check top-level structure
        assert schema['type'] == 'function'
        assert 'function' in schema
        
        # Check function structure
        function = schema['function']
        assert 'name' in function
        assert 'description' in function
        assert 'parameters' in function
        assert 'strict' in function
        
        # Check parameters structure
        parameters = function['parameters']
        assert parameters['type'] == 'object'
        assert 'properties' in parameters
        assert 'required' in parameters
        assert 'additionalProperties' in parameters


class TestSchemaGenerationEdgeCases:
    """Test edge cases in schema generation."""

    def test_async_function_schema_generation(self):
        """Test schema generation for async functions."""
        @tool
        async def async_func(param: str) -> str:
            """Async function."""
            return param

        # Async functions should generate schemas normally
        assert hasattr(async_func, '_tool_metadata')
        assert hasattr(async_func, '_tool_metadata_strict')
        
        schema = async_func._tool_metadata
        assert schema['function']['name'] == 'async_func'
        assert schema['function']['description'] == 'Async function.'

    def test_function_with_decorators_preservation(self):
        """Test that function properties are preserved after decoration."""
        @tool
        def test_func(param: str) -> str:
            """Test function."""
            return param

        # Function name should be preserved
        assert test_func.__name__ == 'test_func'
        
        # Metadata should be properly attached
        assert hasattr(test_func, '_tool_metadata')
        assert test_func._tool_metadata['function']['name'] == 'test_func'

    def test_custom_name_and_description_override(self):
        """Test that custom names and descriptions properly override defaults."""
        @tool(name="custom_name", description="Custom description")
        def original_func() -> str:
            """Original description."""
            return "test"

        schema = original_func._tool_metadata
        assert schema['function']['name'] == 'custom_name'
        assert schema['function']['description'] == 'Custom description'

        # Both strict and non-strict should use the custom values
        strict_schema = original_func._tool_metadata_strict
        assert strict_schema['function']['name'] == 'custom_name'
        assert strict_schema['function']['description'] == 'Custom description'

    def test_parameter_kind_filtering(self):
        """Test that VAR_POSITIONAL and VAR_KEYWORD parameters are filtered out."""
        @tool
        def var_params_func(normal: str, *args, **kwargs) -> str:
            """Function with *args and **kwargs."""
            return normal

        schema = var_params_func._tool_metadata
        properties = schema['function']['parameters']['properties']
        
        # Only normal parameter should be included
        assert 'normal' in properties
        assert len(properties) == 1  # args and kwargs should be filtered out

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_field_metadata_integration(self):
        """Test integration with Pydantic Field metadata using Annotated."""
        from pydantic import Field
        from typing import Annotated

        @tool
        def field_func(
            param1: Annotated[str, Field(description="Parameter with Field description")],
            param2: Annotated[int, Field(default=10, description="Parameter with default and description")]
        ) -> str:
            """Function with Annotated Pydantic Field parameters."""
            return f"{param1}-{param2}"

        schema = field_func._tool_metadata
        properties = schema['function']['parameters']['properties']
        
        # Field descriptions should be included when using Annotated
        assert properties['param1'].get('description') == "Parameter with Field description"
        assert properties['param2'].get('description') == "Parameter with default and description"
        
        # Defaults should be handled correctly
        required = schema['function']['parameters']['required']
        assert 'param1' in required  # No default value
        assert 'param2' not in required  # Has default value

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_field_as_default_limitation(self):
        """Test current limitation with Field() as default values."""
        from pydantic import Field

        @tool
        def field_func(
            param1: str = Field(description="Parameter with Field description"),
            param2: int = Field(default=10, description="Parameter with default and description")
        ) -> str:
            """Function with Field as default values (current limitation)."""
            return f"{param1}-{param2}"

        schema = field_func._tool_metadata
        properties = schema['function']['parameters']['properties']
        
        # Currently, Field() as default values are not properly handled for descriptions
        # This is a known limitation that could be improved in the future
        assert 'param1' in properties
        assert 'param2' in properties
        
        # The function should still work even if Field metadata isn't extracted
        assert callable(field_func) 