"""
Tests for structured outputs functionality across providers.
"""
import pytest
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from unittest.mock import Mock

from toolflow import from_openai, from_anthropic
from tests.conftest import (
    create_openai_response,
    create_anthropic_response,
    Person,
    TeamAnalysis,
    WeatherInfo
)


class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    age: int

class NestedModel(BaseModel):
    """Model with nested structures."""
    users: List[Person]
    total_count: int
    metadata: Optional[dict] = None


class TestStructuredOutputsOpenAI:
    """Test structured outputs with OpenAI provider."""
    
    def test_simple_structured_output(self, mock_openai_client):
        """Test simple structured output with OpenAI."""
        from tests.conftest import create_openai_structured_response
        client = from_openai(mock_openai_client)
        
        # Mock response with structured output tool call
        json_data = {"name": "John", "age": 30}
        mock_response = create_openai_structured_response(json_data)
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Create a person"}],
            response_format=SimpleModel
        )
        
        # Should return parsed Pydantic model
        assert isinstance(response, SimpleModel)
        assert response.name == "John"
        assert response.age == 30
    
    def test_complex_structured_output(self, mock_openai_client):
        """Test complex structured output with nested data."""
        from tests.conftest import create_openai_structured_response
        client = from_openai(mock_openai_client)
        
        json_data = {
            "people": [
                {"name": "Alice", "age": 25, "skills": ["Python", "React"]},
                {"name": "Bob", "age": 30, "skills": ["Go", "Docker"]}
            ],
            "average_age": 27.5,
            "top_skills": ["Python", "React", "Go", "Docker"]
        }
        mock_response = create_openai_structured_response(json_data)
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Analyze this team"}],
            response_format=TeamAnalysis
        )
        
        assert isinstance(response, TeamAnalysis)
        assert len(response.people) == 2
        assert response.people[0].name == "Alice"
        assert response.average_age == 27.5
        assert "Python" in response.top_skills
    
    def test_structured_output_with_full_response(self, mock_openai_client):
        """Test structured output with full_response=True."""
        from tests.conftest import create_openai_structured_response
        client = from_openai(mock_openai_client)
        
        json_data = {"name": "Jane", "age": 25}
        mock_response = create_openai_structured_response(json_data)
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Create a person"}],
            response_format=SimpleModel,
            full_response=True
        )
        
        # Should return full response object when full_response=True
        assert hasattr(response, 'choices')
        # With structured output, the parsed result should be available through the response
        # The exact structure may depend on how full_response handles structured output
    
    def test_invalid_json_response(self, mock_openai_client):
        """Test handling of invalid JSON in response."""
        from tests.conftest import create_openai_tool_call, create_openai_response
        client = from_openai(mock_openai_client)
        
        # Create a tool call with invalid JSON arguments
        tool_call = create_openai_tool_call(
            "call_invalid", 
            "final_response_tool_internal", 
            {"response": {"name": "John", "age": None}}  # Invalid age type
        )
        mock_response = create_openai_response(content=None, tool_calls=[tool_call])
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises((ValidationError, ValueError, TypeError)):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Create a person"}],
                response_format=SimpleModel
            )
    
    def test_schema_validation_error(self, mock_openai_client):
        """Test handling of schema validation errors."""
        from tests.conftest import create_openai_structured_response
        client = from_openai(mock_openai_client)
        
        # JSON that doesn't match schema (missing required field)
        invalid_data = {"name": "John"}  # Missing age
        mock_response = create_openai_structured_response(invalid_data)
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises(ValidationError):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Create a person"}],
                response_format=SimpleModel
            )


class TestStructuredOutputsAnthropic:
    """Test structured outputs with Anthropic provider."""
    
    def test_simple_structured_output(self, mock_anthropic_client):
        """Test simple structured output with Anthropic."""
        from tests.conftest import create_anthropic_structured_response
        client = from_anthropic(mock_anthropic_client)
        
        # Mock response with structured output tool call
        json_data = {"name": "John", "age": 30}
        mock_response = create_anthropic_structured_response(json_data)
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Create a person"}],
            response_format=SimpleModel
        )
        
        # Should return parsed Pydantic model
        assert isinstance(response, SimpleModel)
        assert response.name == "John"
        assert response.age == 30
    
    def test_complex_structured_output(self, mock_anthropic_client):
        """Test complex structured output with Anthropic."""
        from tests.conftest import create_anthropic_structured_response
        client = from_anthropic(mock_anthropic_client)
        
        json_data = {
            "city": "New York",
            "temperature": 72.5,
            "condition": "Sunny",
            "humidity": 45
        }
        mock_response = create_anthropic_structured_response(json_data)
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Get weather for NYC"}],
            response_format=WeatherInfo
        )
        
        assert isinstance(response, WeatherInfo)
        assert response.city == "New York"
        assert response.temperature == 72.5
        assert response.condition == "Sunny"
        assert response.humidity == 45
    
    def test_structured_output_with_full_response(self, mock_anthropic_client):
        """Test structured output with full_response=True."""
        from tests.conftest import create_anthropic_structured_response
        client = from_anthropic(mock_anthropic_client)
        
        json_data = {"name": "Jane", "age": 25}
        mock_response = create_anthropic_structured_response(json_data)
        mock_anthropic_client.messages.create.return_value = mock_response
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Create a person"}],
            response_format=SimpleModel,
            full_response=True
        )
        
        # Should return full response object when full_response=True
        assert hasattr(response, 'content')
        # With structured output, the parsed result should be available through the response
        # The exact structure may depend on how full_response handles structured output


class TestStructuredOutputsWithTools:
    """Test structured outputs combined with tool execution."""
    
    def test_structured_output_after_tool_execution_openai(self, mock_openai_client):
        """Test structured output after tool execution with OpenAI."""
        from tests.conftest import simple_math_tool, create_openai_tool_call, create_openai_structured_response
        
        client = from_openai(mock_openai_client)
        
        # First response: tool call
        tool_call = create_openai_tool_call("call_123", "simple_math_tool", {"a": 5, "b": 3})
        mock_response_1 = create_openai_response(tool_calls=[tool_call])
        
        # Second response: structured output
        json_data = {"name": "Calculator Result", "age": 8}
        mock_response_2 = create_openai_structured_response(json_data)
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 5+3 and format as person"}],
            tools=[simple_math_tool],
            response_format=SimpleModel
        )
        
        assert isinstance(response, SimpleModel)
        assert response.name == "Calculator Result"
        assert response.age == 8
    
    def test_structured_output_after_tool_execution_anthropic(self, mock_anthropic_client):
        """Test structured output after tool execution with Anthropic."""
        from tests.conftest import simple_math_tool, create_anthropic_tool_call, create_anthropic_structured_response
        
        client = from_anthropic(mock_anthropic_client)
        
        # First response: tool call
        tool_call = create_anthropic_tool_call("toolu_123", "simple_math_tool", {"a": 5, "b": 3})
        mock_response_1 = create_anthropic_response(tool_calls=[tool_call])
        
        # Second response: structured output
        json_data = {"city": "Math City", "temperature": 8.0, "condition": "Calculated", "humidity": 100}
        mock_response_2 = create_anthropic_structured_response(json_data)
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Calculate 5+3 and format as weather"}],
            tools=[simple_math_tool],
            response_format=WeatherInfo
        )
        
        assert isinstance(response, WeatherInfo)
        assert response.city == "Math City"
        assert response.temperature == 8.0


class TestStructuredOutputEdgeCases:
    """Test edge cases for structured outputs."""
    
    def test_empty_response_content(self, mock_openai_client):
        """Test handling of empty response content."""
        client = from_openai(mock_openai_client)
        
        mock_response = create_openai_response(content="")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises((ValidationError, ValueError)):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Create a person"}],
                response_format=SimpleModel
            )
    
    def test_non_json_response_content(self, mock_openai_client):
        """Test handling of non-JSON response content."""
        client = from_openai(mock_openai_client)
        
        mock_response = create_openai_response(content="This is just plain text, not JSON")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises((ValidationError, ValueError)):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Create a person"}],
                response_format=SimpleModel
            )
    
    def test_response_format_none(self, mock_openai_client):
        """Test that response_format=None works normally."""
        client = from_openai(mock_openai_client)
        
        mock_response = create_openai_response(content="Regular text response")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            response_format=None
        )
        
        # Should return regular text response
        assert response == "Regular text response"
    
    def test_optional_fields_in_model(self, mock_openai_client):
        """Test structured output with optional fields."""
        from tests.conftest import create_openai_structured_response
        client = from_openai(mock_openai_client)
        
        # JSON with optional field missing
        json_data = {"users": [], "total_count": 0}
        mock_response = create_openai_structured_response(json_data)
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Create empty user list"}],
            response_format=NestedModel
        )
        
        assert isinstance(response, NestedModel)
        assert response.users == []
        assert response.total_count == 0
        assert response.metadata is None  # Optional field should be None
    
    def test_schema_generation_for_response_format(self, mock_openai_client):
        """Test that schema is properly generated for response_format."""
        from tests.conftest import create_openai_structured_response
        client = from_openai(mock_openai_client)
        
        json_data = {"name": "Test", "age": 25}
        mock_response = create_openai_structured_response(json_data)
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # This should work without errors - schema generation should be automatic
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Create a person"}],
            response_format=SimpleModel
        )
        
        assert isinstance(response, SimpleModel)
        
        # Verify that the call was made (schema generation happens internally)
        mock_openai_client.chat.completions.create.assert_called_once() 