"""
Tests for the @tool decorator functionality.
"""
import pytest
import asyncio
from typing import List
from toolflow import tool


class TestToolDecorator:
    """Test the @tool decorator functionality."""
    
    def test_decorator_without_parentheses(self):
        """Test @tool decorator used without parentheses."""
        @tool
        def simple_tool(x: int) -> int:
            """A simple tool."""
            return x * 2
        
        # Test function still works
        assert simple_tool(5) == 10
        
        # Test metadata is attached
        assert hasattr(simple_tool, '_tool_metadata')
        metadata = simple_tool._tool_metadata
        assert metadata['function']['name'] == 'simple_tool'
        assert 'A simple tool.' in metadata['function']['description']
    
    def test_decorator_with_parentheses(self):
        """Test @tool decorator used with parentheses."""
        @tool()
        def simple_tool_with_parens(x: int) -> int:
            """A simple tool with parentheses."""
            return x * 3
        
        assert simple_tool_with_parens(5) == 15
        assert hasattr(simple_tool_with_parens, '_tool_metadata')
    
    def test_decorator_with_custom_name(self):
        """Test @tool decorator with custom name."""
        @tool(name="custom_name")
        def tool_with_custom_name(x: int) -> int:
            """Tool with custom name."""
            return x
        
        assert tool_with_custom_name(42) == 42
        metadata = tool_with_custom_name._tool_metadata
        assert metadata['function']['name'] == 'custom_name'
    
    def test_decorator_with_custom_description(self):
        """Test @tool decorator with custom description."""
        @tool(description="Custom description")
        def tool_with_custom_desc(x: int) -> int:
            """Original description."""
            return x
        
        assert tool_with_custom_desc(42) == 42
        metadata = tool_with_custom_desc._tool_metadata
        assert metadata['function']['description'] == 'Custom description'
    
    def test_decorator_with_custom_name_and_description(self):
        """Test @tool decorator with both custom name and description."""
        @tool(name="custom_name", description="Custom description")
        def tool_with_custom_params(x: int) -> int:
            """Original description."""
            return x
        
        assert tool_with_custom_params(42) == 42
        metadata = tool_with_custom_params._tool_metadata
        assert metadata['function']['name'] == 'custom_name'
        assert metadata['function']['description'] == 'Custom description'
    
    def test_decorator_preserves_function_attributes(self):
        """Test that decorator preserves original function attributes."""
        def original_func(x: int) -> int:
            """Original docstring."""
            return x
        
        # Add custom attribute
        original_func.custom_attr = "test"
        
        # Apply decorator
        decorated_func = tool(original_func)
        
        # Check preserved attributes
        assert decorated_func.__name__ == 'original_func'
        assert decorated_func.__doc__ == 'Original docstring.'
        assert hasattr(decorated_func, '_tool_metadata')
    
    def test_decorator_with_complex_types(self):
        """Test @tool decorator with complex type annotations."""
        @tool
        def complex_tool(items: List[str], count: int = 5) -> dict:
            """A tool with complex types."""
            return {"items": items[:count], "total": len(items)}
        
        result = complex_tool(["a", "b", "c"], 2)
        assert result == {"items": ["a", "b"], "total": 3}
        
        # Check that metadata includes proper schema
        assert hasattr(complex_tool, '_tool_metadata')
        metadata = complex_tool._tool_metadata
        assert 'parameters' in metadata['function']
    
    def test_async_tool_decorator(self):
        """Test @tool decorator with async functions."""
        @tool
        async def async_tool_func(x: int) -> int:
            """An async tool."""
            await asyncio.sleep(0.001)  # Small delay
            return x * 2
        
        # Check metadata is attached
        assert hasattr(async_tool_func, '_tool_metadata')
        
        # Test execution
        async def test_async():
            result = await async_tool_func(5)
            assert result == 10
        
        asyncio.run(test_async())
    
    def test_decorator_with_no_docstring(self):
        """Test @tool decorator with function that has no docstring."""
        @tool
        def no_docstring_tool(x: int) -> int:
            return x * 2
        
        assert no_docstring_tool(5) == 10
        assert hasattr(no_docstring_tool, '_tool_metadata')
        # Should have some default description
        metadata = no_docstring_tool._tool_metadata
        assert metadata['function']['description'] is not None
    
    def test_decorator_with_multiple_parameters(self):
        """Test @tool decorator with multiple parameter types."""
        @tool
        def multi_param_tool(
            text: str, 
            number: float, 
            flag: bool = True,
            items: List[int] = None
        ) -> str:
            """Tool with multiple parameter types."""
            items = items or []
            return f"{text}: {number}, flag={flag}, items={len(items)}"
        
        result = multi_param_tool("test", 3.14, False, [1, 2, 3])
        assert result == "test: 3.14, flag=False, items=3"
        
        assert hasattr(multi_param_tool, '_tool_metadata')
    
    def test_decorator_error_handling(self):
        """Test that decorator doesn't interfere with error handling."""
        @tool
        def error_tool(should_fail: bool) -> str:
            """Tool that can fail."""
            if should_fail:
                raise ValueError("Tool failed intentionally")
            return "Success"
        
        # Should work normally
        assert error_tool(False) == "Success"
        
        # Should still raise errors
        with pytest.raises(ValueError, match="Tool failed intentionally"):
            error_tool(True)
    
    @pytest.mark.asyncio
    async def test_async_decorator_error_handling(self):
        """Test that async decorator doesn't interfere with error handling."""
        @tool
        async def async_error_tool(should_fail: bool) -> str:
            """Async tool that can fail."""
            await asyncio.sleep(0.001)
            if should_fail:
                raise ValueError("Async tool failed")
            return "Async success"
        
        # Should work normally
        result = await async_error_tool(False)
        assert result == "Async success"
        
        # Should still raise errors
        with pytest.raises(ValueError, match="Async tool failed"):
            await async_error_tool(True) 