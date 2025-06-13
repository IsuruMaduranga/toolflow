"""
Test suite for structured output functionality.

This module tests the parse method and structured output capabilities
for both sync and async OpenAI wrappers.
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import List, Optional

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

import toolflow
from toolflow.providers.openai.structured_output import (
    create_openai_response_tool,
    handle_openai_structured_response,
    validate_response_format
)


# Test Pydantic models for structured outputs
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str
    humidity: Optional[int] = None


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class CalculationResult(BaseModel):
    operation: str
    operands: List[float]
    result: float
    explanation: str


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class ComplexResponse(BaseModel):
    summary: str
    details: List[WeatherReport]
    count: int


class TestStructuredOutputUtils:
    """Test structured output utility functions."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_validate_response_format_valid(self):
        """Test response format validation with valid Pydantic model."""
        validate_response_format(WeatherReport)  # Should not raise

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_validate_response_format_invalid(self):
        """Test response format validation with invalid format."""
        with pytest.raises(ValueError, match="response_format must be a Pydantic model"):
            validate_response_format(dict)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available") 
    def test_create_openai_response_tool(self):
        """Test creation of dynamic response tool."""
        response_tool = create_openai_response_tool(WeatherReport)
        
        # Check that it's a decorated function
        assert hasattr(response_tool, '_tool_metadata')
        assert response_tool._tool_metadata['function']['name'] == 'final_response_tool_internal'
        
        # Check that the tool accepts the response format
        properties = response_tool._tool_metadata['function']['parameters']['properties']
        assert 'response' in properties

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_handle_openai_structured_response(self):
        """Test handling of structured response from OpenAI."""
        # Create mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = [Mock()]
        mock_response.choices[0].message.tool_calls[0].function.name = 'final_response_tool_internal'
        mock_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            'response': {
                'city': 'New York',
                'temperature': 72.5,
                'conditions': 'Sunny',
                'humidity': 65
            }
        })

        result = handle_openai_structured_response(mock_response, WeatherReport)
        
        assert result is not None
        assert result.choices[0].message.parsed.city == 'New York'
        assert result.choices[0].message.parsed.temperature == 72.5
        assert result.choices[0].message.parsed.conditions == 'Sunny'
        assert result.choices[0].message.parsed.humidity == 65

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_handle_openai_structured_response_no_tool_calls(self):
        """Test handling when response has no tool calls."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None
    
        result = handle_openai_structured_response(mock_response, WeatherReport)
        assert result is None


class TestStructuredOutputSyncWrapper:
    """Test structured output functionality with sync OpenAI wrapper."""

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

    @pytest.fixture
    def weather_tool(self):
        """Sample weather tool for testing."""
        @toolflow.tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"
        return get_weather



    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_with_create_method(self, toolflow_client, mock_openai_client):
        """Test structured output via create method with response_format."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = [Mock()]
        mock_response.choices[0].message.tool_calls[0].function.name = 'final_response_tool_internal'
        mock_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            'response': {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear'}
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = toolflow_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What's the weather?"}],
            response_format=WeatherReport
        )

        # Verify the response format tool was added and API was called
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        assert len(call_args[1]['tools']) == 1  # Response format tool was added

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_create_with_tools(self, toolflow_client, mock_openai_client, weather_tool):
        """Test create method with both tools and structured output."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = [Mock()]
        mock_response.choices[0].message.tool_calls[0].function.name = 'final_response_tool_internal'
        mock_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            'response': {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear'}
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = toolflow_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[weather_tool],
            response_format=WeatherReport
        )

        # Verify both tools and response format tool were included
        call_args = mock_openai_client.chat.completions.create.call_args
        assert len(call_args[1]['tools']) == 2  # Original tool + response format tool

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_create_streaming_not_supported_with_response_format(self, toolflow_client):
        """Test that create method doesn't support streaming with response_format."""
        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            toolflow_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                response_format=WeatherReport,
                stream=True
            )


class TestStructuredOutputBetaWrapper:
    """Test structured output functionality with beta OpenAI wrapper."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client with beta support."""
        client = Mock()
        client.beta = Mock()
        client.beta.chat = Mock()
        client.beta.chat.completions = Mock()
        return client

    @pytest.fixture
    def toolflow_client(self, mock_openai_client):
        """Toolflow wrapped client for testing."""
        return toolflow.from_openai(mock_openai_client)

    @pytest.fixture
    def math_tool(self):
        """Sample math tool for testing."""
        @toolflow.tool
        def calculate(operation: str, a: float, b: float) -> float:
            """Perform mathematical operations."""
            if operation == "add":
                return a + b
            elif operation == "multiply": 
                return a * b
            else:
                raise ValueError(f"Unknown operation: {operation}")
        return calculate

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_beta_parse_method_exists(self, toolflow_client):
        """Test that parse method is available on beta completions."""
        assert hasattr(toolflow_client.beta.chat.completions, 'parse')

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_beta_parse_with_strict_schema(self, toolflow_client, mock_openai_client, math_tool):
        """Test beta parse method uses strict schema validation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None  # No tool calls
        mock_openai_client.beta.chat.completions.parse.return_value = mock_response

        result = toolflow_client.beta.chat.completions.parse(
            model="gpt-4",
            messages=[{"role": "user", "content": "Calculate 2 + 3"}],
            tools=[math_tool],
            response_format=CalculationResult
        )

        # Verify strict validation was used for tools
        mock_openai_client.beta.chat.completions.parse.assert_called_once()
        call_args = mock_openai_client.beta.chat.completions.parse.call_args
        
        # Check that tools have strict schema properties
        tools = call_args[1]['tools']
        for tool in tools:
            if tool['function']['name'] != 'final_response_tool_internal':
                params = tool['function']['parameters']
                assert params.get('additionalProperties') is False

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_beta_create_with_tools(self, toolflow_client, mock_openai_client, math_tool):
        """Test beta create method with tools."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None
        mock_openai_client.beta.chat.completions.create.return_value = mock_response

        result = toolflow_client.beta.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Calculate something"}],
            tools=[math_tool]
        )

        # Verify strict validation was used
        mock_openai_client.beta.chat.completions.create.assert_called_once()


class TestStructuredOutputErrorHandling:
    """Test error handling in structured output functionality."""

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

    def test_create_without_pydantic(self, toolflow_client):
        """Test create method with response_format when Pydantic is not available."""
        with patch('toolflow.providers.openai.structured_output.PYDANTIC_AVAILABLE', False):
            with pytest.raises(ImportError, match="Pydantic is required for structured output"):
                toolflow_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Test"}],
                    response_format=dict  # Not a Pydantic model
                )

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_create_invalid_response_format(self, toolflow_client):
        """Test create method with invalid response format."""
        with pytest.raises(ValueError, match="response_format must be a Pydantic model"):
            toolflow_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                response_format=str  # Not a Pydantic model
            )

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_create_malformed_tool_response(self, toolflow_client, mock_openai_client):
        """Test handling of malformed tool response in create method."""
        # Setup mock response with malformed JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = [Mock()]
        mock_response.choices[0].message.tool_calls[0].function.name = 'final_response_tool_internal'
        mock_response.choices[0].message.tool_calls[0].function.arguments = "invalid json"
        
        mock_openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            toolflow_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                response_format=WeatherReport
            )


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestStructuredOutputAsyncWrapper:
    """Test structured output functionality with async OpenAI wrapper."""

    @pytest.fixture
    def mock_async_openai_client(self):
        """Mock async OpenAI client for testing."""
        client = Mock()
        client.chat = Mock()
        client.chat.completions = Mock()
        client.beta = Mock()
        client.beta.chat = Mock()
        client.beta.chat.completions = Mock()
        return client

    @pytest.fixture
    def async_toolflow_client(self, mock_async_openai_client):
        """Async toolflow wrapped client for testing."""
        return toolflow.from_openai_async(mock_async_openai_client)

    @pytest.fixture
    def async_weather_tool(self):
        """Sample async weather tool for testing."""
        @toolflow.tool
        async def get_weather_async(city: str) -> str:
            """Get weather for a city asynchronously."""
            return f"Weather in {city}: Sunny, 72°F"
        return get_weather_async



    def test_async_beta_parse_method_exists(self, async_toolflow_client):
        """Test that parse method is available on async beta completions."""
        assert hasattr(async_toolflow_client.beta.chat.completions, 'parse')

    # Note: More comprehensive async tests would require proper async test setup
    # which would need additional fixtures and async test methods 