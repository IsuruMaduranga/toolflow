"""
Comprehensive integration tests for toolflow functionality.

This module tests how different new features work together:
- Structured output with tools
- Beta API with structured output
- Strict schema validation in real scenarios
- Enhanced parameter handling
- Error handling integration
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

import toolflow


# Test models for integration testing
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class WeatherData(BaseModel):
    city: str
    temperature: float
    humidity: int
    conditions: str
    forecast: Optional[List[str]] = None


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class AnalysisResult(BaseModel):
    summary: str
    data_points: List[Dict[str, Any]]
    confidence: float
    recommendations: List[str]


class TestStructuredOutputIntegration:
    """Test structured output working with various tool combinations."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        client.chat = Mock()
        client.chat.completions = Mock()
        client.beta = Mock()
        client.beta.chat = Mock()
        client.beta.chat.completions = Mock()
        return client

    @pytest.fixture
    def toolflow_client(self, mock_openai_client):
        """Toolflow wrapped client for testing."""
        return toolflow.from_openai(mock_openai_client)

    @pytest.fixture
    def weather_tools(self):
        """Collection of weather-related tools."""
        @toolflow.tool
        def get_weather(city: str) -> str:
            """Get current weather for a city."""
            return f"Weather in {city}: 72°F, 65% humidity, sunny"

        @toolflow.tool
        def get_forecast(city: str, days: int = 3) -> str:
            """Get weather forecast for a city."""
            return f"3-day forecast for {city}: Sunny, Cloudy, Rainy"

        return [get_weather, get_forecast]

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_functionality_exists(self, toolflow_client):
        """Test that structured output functionality exists via beta API."""
        # Test that beta parse method exists
        assert hasattr(toolflow_client.beta.chat.completions, 'parse')

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_create_method_adds_response_tool(self, toolflow_client, mock_openai_client, weather_tools):
        """Test that create method adds response format tool when response_format is provided."""
        # Mock response to prevent actual tool execution
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        toolflow_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            tools=weather_tools,
            response_format=WeatherData
        )

        # Verify API was called
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        
        # Should have original tools + response format tool
        tools = call_args[1]['tools']
        assert len(tools) == len(weather_tools) + 1  # Original tools + response format tool
        
        # Check that final response tool was added
        tool_names = [tool['function']['name'] for tool in tools]
        assert 'final_response_tool_internal' in tool_names

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_beta_api_uses_strict_schemas(self, toolflow_client, mock_openai_client, weather_tools):
        """Test that beta API uses strict schema validation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.beta.chat.completions.parse.return_value = mock_response

        toolflow_client.beta.chat.completions.parse(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            tools=weather_tools,
            response_format=AnalysisResult
        )

        # Verify beta API was called
        mock_openai_client.beta.chat.completions.parse.assert_called_once()
        call_args = mock_openai_client.beta.chat.completions.parse.call_args
        
        # Check that tools have strict validation
        tools = call_args[1]['tools']
        for tool in tools:
            params = tool['function']['parameters']
            assert params.get('additionalProperties') is False


class TestEnhancedParameterHandling:
    """Test enhanced parameter handling in various scenarios."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        client.chat = Mock()
        client.chat.completions = Mock()
        return client

    @pytest.fixture
    def toolflow_client(self, mock_openai_client):
        """Toolflow wrapped client for testing."""
        return toolflow.from_openai(mock_openai_client)

    def test_max_workers_parameter_acceptance(self, toolflow_client, mock_openai_client):
        """Test that max_workers parameter is accepted without errors."""
        @toolflow.tool
        def test_tool(param: str) -> str:
            return param

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Test different max_workers values
        for max_workers in [1, 5, 20]:
            toolflow_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                tools=[test_tool],
                max_workers=max_workers,
                parallel_tool_execution=True
            )

        # Should not raise any errors
        assert mock_openai_client.chat.completions.create.call_count == 3

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_parameter_filtering_in_api_calls(self, toolflow_client, mock_openai_client):
        """Test that toolflow parameters are properly filtered from OpenAI calls."""
        @toolflow.tool
        def test_tool(param: str) -> str:
            return param

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        toolflow_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            tools=[test_tool],
            parallel_tool_execution=True,
            max_tool_calls=15
        )

        # Verify toolflow-specific parameters were filtered out
        call_args = mock_openai_client.chat.completions.create.call_args
        call_kwargs = call_args[1]
        
        # These should not be in the call to OpenAI
        assert 'parallel_tool_execution' not in call_kwargs
        assert 'max_workers' not in call_kwargs
        assert 'max_tool_calls' not in call_kwargs


class TestErrorHandlingIntegration:
    """Test error handling across different feature combinations."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        client.chat = Mock()
        client.chat.completions = Mock()
        return client

    @pytest.fixture
    def toolflow_client(self, mock_openai_client):
        """Toolflow wrapped client for testing."""
        return toolflow.from_openai(mock_openai_client)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_streaming_not_supported_with_response_format(self, toolflow_client):
        """Test that streaming is not supported with response_format."""
        @toolflow.tool
        def test_tool(param: str) -> str:
            return param

        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            toolflow_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                tools=[test_tool],
                response_format=WeatherData,
                stream=True
            )

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_beta_api_error_handling(self, toolflow_client, mock_openai_client):
        """Test error handling in beta API with structured output."""
        @toolflow.tool
        def test_tool(param: str) -> str:
            return param

        # Mock beta API throwing an error
        mock_openai_client.beta.chat.completions.parse.side_effect = Exception("Beta API error")

        with pytest.raises(Exception, match="Beta API error"):
            toolflow_client.beta.chat.completions.parse(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                tools=[test_tool],
                response_format=WeatherData
            )


class TestSchemaGenerationIntegration:
    """Test schema generation integration with real usage patterns."""

    def test_strict_vs_non_strict_metadata_generation(self):
        """Test that both strict and non-strict metadata are generated."""
        @toolflow.tool
        def test_tool(param: str, optional: int = 10) -> str:
            return f"{param}-{optional}"

        # Check that both metadata versions exist
        assert hasattr(test_tool, '_tool_metadata')
        assert hasattr(test_tool, '_tool_metadata_strict')

        # Check strict property differences
        main_schema = test_tool._tool_metadata
        strict_schema = test_tool._tool_metadata_strict

        assert main_schema['function']['strict'] is False
        assert strict_schema['function']['strict'] is True

        # Both should have additionalProperties False
        assert main_schema['function']['parameters']['additionalProperties'] is False
        assert strict_schema['function']['parameters']['additionalProperties'] is False

    def test_internal_tool_name_protection(self):
        """Test that internal tool name is protected from regular usage."""
        # This should work (internal=True)
        @toolflow.tool(name="final_response_tool_internal", internal=True)
        def internal_ok() -> str:
            return "ok"

        assert internal_ok._tool_metadata['function']['name'] == 'final_response_tool_internal'

        # This should fail (internal=False or not specified)
        with pytest.raises(ValueError, match="final_response_tool_internal is an internally used tool"):
            @toolflow.tool(name="final_response_tool_internal")
            def internal_fail() -> str:
                return "fail"

    def test_required_field_always_present(self):
        """Test that required field is always present in schemas."""
        @toolflow.tool
        def no_params() -> str:
            return "test"

        @toolflow.tool
        def with_params(required: str, optional: int = 10) -> str:
            return f"{required}-{optional}"

        # Both should have required field
        no_params_schema = no_params._tool_metadata
        with_params_schema = with_params._tool_metadata

        assert 'required' in no_params_schema['function']['parameters']
        assert no_params_schema['function']['parameters']['required'] == []

        assert 'required' in with_params_schema['function']['parameters']
        assert 'required' in with_params_schema['function']['parameters']['required']
        assert 'optional' not in with_params_schema['function']['parameters']['required']

    def test_description_fallback_mechanism(self):
        """Test function description fallback to function name."""
        @toolflow.tool
        def no_docstring_func(param: str) -> str:
            return param

        @toolflow.tool
        def with_docstring_func(param: str) -> str:
            """Function with docstring."""
            return param

        # No docstring should fallback to function name
        no_doc_schema = no_docstring_func._tool_metadata
        assert no_doc_schema['function']['description'] == 'no_docstring_func'

        # With docstring should use docstring
        with_doc_schema = with_docstring_func._tool_metadata
        assert with_doc_schema['function']['description'] == 'Function with docstring.'


class TestFeatureInteroperability:
    """Test how different features work together."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        client.chat = Mock()
        client.chat.completions = Mock()
        client.beta = Mock()
        client.beta.chat = Mock()
        client.beta.chat.completions = Mock()
        return client

    @pytest.fixture
    def toolflow_client(self, mock_openai_client):
        """Toolflow wrapped client for testing."""
        return toolflow.from_openai(mock_openai_client)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_multiple_features_together(self, toolflow_client, mock_openai_client):
        """Test multiple new features working together."""
        @toolflow.tool
        def data_tool(query: str, format: str = "json") -> str:
            """Fetch data in specified format."""
            return f"Data for {query} in {format} format"

        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.beta.chat.completions.parse.return_value = mock_response

        # Test with multiple enhanced features
        result = toolflow_client.beta.chat.completions.parse(
            model="gpt-4",
            messages=[{"role": "user", "content": "Get analysis"}],
            tools=[data_tool],
            response_format=AnalysisResult,
            parallel_tool_execution=True,
            max_workers=3,
            max_tool_calls=5
        )

        # Verify all parameters were handled correctly
        mock_openai_client.beta.chat.completions.parse.assert_called_once()
        call_args = mock_openai_client.beta.chat.completions.parse.call_args

        # Check tools were processed correctly (beta API doesn't auto-add response format tool)
        tools = call_args[1]['tools']
        assert len(tools) == 1  # Only original tool (beta API handles response_format natively)

        # Check toolflow parameters were filtered out
        call_kwargs = call_args[1]
        assert 'parallel_tool_execution' not in call_kwargs
        assert 'max_workers' not in call_kwargs
        assert 'max_tool_calls' not in call_kwargs
        # response_format should be passed through to beta API
        assert 'response_format' in call_kwargs

    def test_schema_metadata_consistency(self):
        """Test that schema metadata is consistent across features."""
        @toolflow.tool
        def consistent_tool(param1: str, param2: int = 10) -> str:
            """Tool for testing consistency."""
            return f"{param1}-{param2}"

        # Both metadata versions should exist
        assert hasattr(consistent_tool, '_tool_metadata')
        assert hasattr(consistent_tool, '_tool_metadata_strict')

        main_meta = consistent_tool._tool_metadata
        strict_meta = consistent_tool._tool_metadata_strict

        # Should have same basic structure
        assert main_meta['type'] == 'function'
        assert strict_meta['type'] == 'function'
        assert main_meta['function']['name'] == strict_meta['function']['name']
        assert main_meta['function']['description'] == strict_meta['function']['description']

        # Should have different strict flags
        assert main_meta['function']['strict'] is False
        assert strict_meta['function']['strict'] is True

        # Both should have proper schema structure
        for meta in [main_meta, strict_meta]:
            params = meta['function']['parameters']
            assert params['type'] == 'object'
            assert 'properties' in params
            assert 'required' in params
            assert 'additionalProperties' in params
            assert params['additionalProperties'] is False 