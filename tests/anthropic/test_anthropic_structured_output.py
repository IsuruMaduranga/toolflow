"""
Test suite for structured output functionality with Anthropic.

This module tests structured output capabilities for both sync and async Anthropic wrappers.
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
from toolflow.providers.anthropic.structured_output import (
    create_anthropic_response_tool,
    handle_anthropic_structured_response,
    validate_response_format
)
from ..conftest import (
    create_mock_anthropic_tool_call as create_mock_tool_call,
    create_mock_anthropic_response as create_mock_response
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


class TestAnthropicStructuredOutputUtils:
    """Test structured output utility functions for Anthropic."""

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
    def test_create_anthropic_response_tool(self):
        """Test creation of dynamic response tool for Anthropic."""
        response_tool = create_anthropic_response_tool(WeatherReport)
        
        # Check that it's a decorated function
        assert hasattr(response_tool, '_tool_metadata')
        assert response_tool._tool_metadata['function']['name'] == 'final_response_tool_internal'
        
        # Check that the tool accepts the response format
        properties = response_tool._tool_metadata['function']['parameters']['properties']
        assert 'response' in properties

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_handle_anthropic_structured_response(self):
        """Test handling of structured response from Anthropic."""
        # Create mock tool call with structured response
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {
                'city': 'New York',
                'temperature': 72.5,
                'conditions': 'Sunny',
                'humidity': 65
            }
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])

        result = handle_anthropic_structured_response(mock_response, WeatherReport)
        
        assert result is not None
        assert result.parsed.city == 'New York'
        assert result.parsed.temperature == 72.5
        assert result.parsed.conditions == 'Sunny'
        assert result.parsed.humidity == 65

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_handle_anthropic_structured_response_no_tool_calls(self):
        """Test handling when response has no tool calls."""
        mock_response = create_mock_response(content="Regular response")
    
        result = handle_anthropic_structured_response(mock_response, WeatherReport)
        assert result is None


class TestAnthropicStructuredOutputSyncWrapper:
    """Test structured output functionality with sync Anthropic wrapper."""

    @pytest.fixture
    def weather_tool(self):
        """Sample weather tool for testing."""
        @toolflow.tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"
        return get_weather

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_with_create_method(self, sync_anthropic_client, mock_anthropic_client):
        """Test structured output via create method with response_format."""
        # Setup mock response with structured tool call
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear'}
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])
        mock_anthropic_client.messages.create.return_value = mock_response

        result = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "What's the weather?"}],
            response_format=WeatherReport
        )

        # Verify the response format tool was added and API was called
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args
        assert len(call_args[1]['tools']) == 1  # Response format tool was added
        
        # Check that the result is parsed correctly
        assert result.city == 'NYC'
        assert result.temperature == 75.0
        assert result.conditions == 'Clear'

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_create_with_tools(self, sync_anthropic_client, mock_anthropic_client, weather_tool):
        """Test create method with both tools and structured output."""
        # Setup mock response with structured tool call
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear'}
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])
        mock_anthropic_client.messages.create.return_value = mock_response

        result = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[weather_tool],
            response_format=WeatherReport
        )

        # Verify both tools and response format tool were included
        call_args = mock_anthropic_client.messages.create.call_args
        assert len(call_args[1]['tools']) == 2  # Original tool + response format tool
        
        # Check that the result is parsed correctly
        assert result.city == 'NYC'
        assert result.temperature == 75.0
        assert result.conditions == 'Clear'

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_create_streaming_not_supported_with_response_format(self, sync_anthropic_client):
        """Test that create method doesn't support streaming with response_format."""
        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "What's the weather?"}],
                response_format=WeatherReport,
                stream=True
            )

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_with_complex_model(self, sync_anthropic_client, mock_anthropic_client):
        """Test structured output with complex nested Pydantic model."""
        # Setup mock response with complex structured data
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {
                'summary': 'Weather analysis for multiple cities',
                'details': [
                    {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear', 'humidity': 60},
                    {'city': 'LA', 'temperature': 80.0, 'conditions': 'Sunny', 'humidity': 45}
                ],
                'count': 2
            }
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])
        mock_anthropic_client.messages.create.return_value = mock_response

        result = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": "Analyze weather for multiple cities"}],
            response_format=ComplexResponse
        )

        # Check that the complex result is parsed correctly
        assert result.summary == 'Weather analysis for multiple cities'
        assert len(result.details) == 2
        assert result.details[0].city == 'NYC'
        assert result.details[1].city == 'LA'
        assert result.count == 2

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_full_response_mode(self, mock_anthropic_client):
        """Test structured output with full response mode."""
        # Create client in full response mode
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=True)
        
        # Setup mock response
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear'}
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])
        mock_anthropic_client.messages.create.return_value = mock_response

        result = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "What's the weather?"}],
            response_format=WeatherReport
        )

        # In full response mode, should return the full mock response with parsed data
        assert hasattr(result, 'parsed')
        assert result.parsed.city == 'NYC'
        assert result.parsed.temperature == 75.0


class TestAnthropicStructuredOutputErrorHandling:
    """Test error handling for structured output functionality."""

    def test_create_without_pydantic(self, sync_anthropic_client):
        """Test that structured output requires Pydantic."""
        if PYDANTIC_AVAILABLE:
            pytest.skip("Pydantic is available, can't test missing Pydantic scenario")
        
        with pytest.raises(ValueError, match="response_format requires Pydantic"):
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "Test"}],
                response_format=dict  # Invalid format when Pydantic not available
            )

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_create_invalid_response_format(self, sync_anthropic_client):
        """Test create method with invalid response_format."""
        with pytest.raises(ValueError, match="response_format must be a Pydantic model"):
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "Test"}],
                response_format=str  # Invalid format
            )

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_create_malformed_tool_response(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of malformed structured response from model."""
        # Setup mock response with malformed tool call
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {'invalid': 'data'}  # Missing required fields for WeatherReport
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])
        mock_anthropic_client.messages.create.return_value = mock_response

        # Should handle the parsing error gracefully
        with pytest.raises(Exception):  # Pydantic validation error
            sync_anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": "What's the weather?"}],
                response_format=WeatherReport
            )


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestAnthropicStructuredOutputAsyncWrapper:
    """Test structured output functionality with async Anthropic wrapper."""

    @pytest.fixture
    def async_weather_tool(self):
        """Sample async weather tool for testing."""
        @toolflow.tool
        async def get_weather_async(city: str) -> str:
            """Get weather for a city asynchronously."""
            return f"Weather in {city}: Sunny, 72°F"
        return get_weather_async

    @pytest.mark.asyncio
    async def test_async_structured_output_with_create_method(self, async_anthropic_client, mock_async_anthropic_client):
        """Test async structured output via create method with response_format."""
        # Setup mock response with structured tool call
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear'}
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])
        mock_async_anthropic_client.messages.create.return_value = mock_response

        result = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "What's the weather?"}],
            response_format=WeatherReport
        )

        # Verify the response format tool was added and API was called
        mock_async_anthropic_client.messages.create.assert_called_once()
        call_args = mock_async_anthropic_client.messages.create.call_args
        assert len(call_args[1]['tools']) == 1  # Response format tool was added
        
        # Check that the result is parsed correctly
        assert result.city == 'NYC'
        assert result.temperature == 75.0
        assert result.conditions == 'Clear'

    @pytest.mark.asyncio
    async def test_async_structured_output_with_tools(self, async_anthropic_client, mock_async_anthropic_client, async_weather_tool):
        """Test async create method with both tools and structured output."""
        # Setup mock response with structured tool call
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear'}
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])
        mock_async_anthropic_client.messages.create.return_value = mock_response

        result = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[async_weather_tool],
            response_format=WeatherReport
        )

        # Verify both tools and response format tool were included
        call_args = mock_async_anthropic_client.messages.create.call_args
        assert len(call_args[1]['tools']) == 2  # Original tool + response format tool
        
        # Check that the result is parsed correctly
        assert result.city == 'NYC'
        assert result.temperature == 75.0
        assert result.conditions == 'Clear'

    @pytest.mark.asyncio
    async def test_async_structured_output_full_response_mode(self, mock_async_anthropic_client):
        """Test async structured output with full response mode."""
        # Create async client in full response mode
        client = toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=True)
        
        # Setup mock response
        tool_call = Mock()
        tool_call.id = "call_response"
        tool_call.name = "final_response_tool_internal"
        tool_call.input = {
            'response': {'city': 'NYC', 'temperature': 75.0, 'conditions': 'Clear'}
        }
        tool_call.type = "tool_use"
        
        mock_response = create_mock_response(tool_calls=[tool_call])
        mock_async_anthropic_client.messages.create.return_value = mock_response

        result = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "What's the weather?"}],
            response_format=WeatherReport
        )

        # In full response mode, should return the full mock response with parsed data
        assert hasattr(result, 'parsed')
        assert result.parsed.city == 'NYC'
        assert result.parsed.temperature == 75.0


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestAnthropicStructuredOutputIntegration:
    """Integration tests for structured output functionality."""

    def test_structured_output_with_calculation_tool(self, sync_anthropic_client, mock_anthropic_client):
        """Test structured output working with calculation tools."""
        @toolflow.tool
        def calculate(operation: str, a: float, b: float) -> float:
            """Perform mathematical calculations."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            else:
                raise ValueError(f"Unknown operation: {operation}")

        # Setup tool call response first, then structured response
        calc_tool_call = create_mock_tool_call("call_calc", "calculate", {"operation": "add", "a": 10, "b": 5})
        calc_response = create_mock_response(tool_calls=[calc_tool_call])
        
        final_tool_call = Mock()
        final_tool_call.id = "call_response"
        final_tool_call.name = "final_response_tool_internal"
        final_tool_call.input = {
            'response': {
                'operation': 'addition',
                'operands': [10.0, 5.0],
                'result': 15.0,
                'explanation': '10 + 5 = 15'
            }
        }
        final_tool_call.type = "tool_use"
        
        final_response = create_mock_response(tool_calls=[final_tool_call])
        
        mock_anthropic_client.messages.create.side_effect = [calc_response, final_response]

        result = sync_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": "Calculate 10 + 5 and explain"}],
            tools=[calculate],
            response_format=CalculationResult
        )

        # Check that the structured result includes the calculation
        assert result.operation == 'addition'
        assert result.operands == [10.0, 5.0]
        assert result.result == 15.0
        assert '10 + 5 = 15' in result.explanation 