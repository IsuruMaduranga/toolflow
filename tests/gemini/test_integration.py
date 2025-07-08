"""
Integration tests for Gemini provider with real tool execution.
"""
import pytest
from unittest.mock import Mock, patch
import time
import asyncio
from typing import List
from pydantic import BaseModel

from toolflow.providers.gemini import from_gemini
from toolflow import tool
from toolflow.core.exceptions import MaxToolCallsError

# Test fixtures
def create_gemini_response(text: str = None, function_calls: list = None):
    """Create a mock Gemini response."""
    mock_response = Mock()
    mock_response.text = text or ""
    mock_response.candidates = []
    
    if text or function_calls:
        candidate = Mock()
        candidate.content = Mock()
        candidate.content.parts = []
        
        if text:
            text_part = Mock()
            text_part.text = text
            if hasattr(text_part, 'function_call'):
                delattr(text_part, 'function_call')
            candidate.content.parts.append(text_part)
        
        if function_calls:
            for func_call in function_calls:
                func_part = Mock()
                func_part.function_call = Mock()
                func_part.function_call.name = func_call["name"]
                func_part.function_call.args = func_call["args"]
                if hasattr(func_part, 'text'):
                    delattr(func_part, 'text')
                candidate.content.parts.append(func_part)
        
        candidate.finish_reason = "STOP"
        mock_response.candidates.append(candidate)
    
    return mock_response

@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client."""
    mock = Mock()
    mock.generate_content = Mock()
    return mock

@pytest.fixture
def client(mock_gemini_client):
    """Create toolflow Gemini client."""
    return from_gemini(mock_gemini_client)

# Test tools
@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic math operations."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")

@tool
def get_weather(city: str, units: str = "fahrenheit") -> str:
    """Get weather information for a city."""
    # Mock weather data
    weather_data = {
        "new york": {"f": "72°F, sunny", "c": "22°C, sunny"},
        "london": {"f": "59°F, cloudy", "c": "15°C, cloudy"},
        "tokyo": {"f": "68°F, rainy", "c": "20°C, rainy"},
        "paris": {"f": "65°F, partly cloudy", "c": "18°C, partly cloudy"}
    }
    
    city_key = city.lower()
    unit_key = "f" if units.lower() == "fahrenheit" else "c"
    
    if city_key in weather_data:
        return f"Weather in {city}: {weather_data[city_key][unit_key]}"
    else:
        return f"Weather data not available for {city}"

@tool
def get_time(timezone: str = "UTC") -> str:
    """Get current time in specified timezone."""
    import datetime
    
    # Mock timezone data
    if timezone.upper() == "UTC":
        return datetime.datetime.utcnow().isoformat() + "Z"
    elif timezone.upper() == "EST":
        return "2024-01-15T10:30:00-05:00"
    elif timezone.upper() == "PST":
        return "2024-01-15T07:30:00-08:00"
    else:
        return f"Current time in {timezone}: 2024-01-15T15:30:00"

@tool
def search_database(query: str, limit: int = 5) -> List[str]:
    """Search a mock database."""
    # Mock search results
    all_results = [
        "Document about artificial intelligence",
        "Article on machine learning",
        "Research paper on neural networks",
        "Guide to deep learning",
        "Tutorial on computer vision",
        "Study on natural language processing",
        "Book on data science",
        "Paper on reinforcement learning"
    ]
    
    # Simple mock search - return results that contain query words
    query_words = query.lower().split()
    matching_results = []
    
    for result in all_results:
        if any(word in result.lower() for word in query_words):
            matching_results.append(result)
    
    return matching_results[:limit]

@tool
def slow_operation(duration: float = 1.0, task_name: str = "operation") -> str:
    """Simulate a slow operation."""
    time.sleep(duration)
    return f"Completed {task_name} after {duration} seconds"

@tool
def failing_operation(should_fail: bool = True, error_message: str = "Operation failed") -> str:
    """Operation that can fail."""
    if should_fail:
        raise ValueError(error_message)
    return "Operation succeeded"

@tool
def complex_calculation(numbers: List[float], operation: str = "sum") -> dict:
    """Perform complex calculations on a list of numbers."""
    if not numbers:
        return {"error": "No numbers provided"}
    
    if operation == "sum":
        result = sum(numbers)
    elif operation == "average":
        result = sum(numbers) / len(numbers)
    elif operation == "max":
        result = max(numbers)
    elif operation == "min":
        result = min(numbers)
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    return {
        "operation": operation,
        "numbers": numbers,
        "result": result,
        "count": len(numbers)
    }

class TestBasicIntegration:
    """Test basic integration scenarios."""
    
    def test_single_tool_execution(self, client, mock_gemini_client):
        """Test execution of a single tool."""
        # Model decides to call calculator
        function_calls = [{"name": "calculator", "args": {"operation": "add", "a": 15, "b": 25}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        
        # Model responds with result
        mock_response_2 = create_gemini_response(text="The sum of 15 and 25 is 40.")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = client.generate_content(
            "What is 15 + 25?",
            tools=[calculator]
        )
        
        assert response == "The sum of 15 and 25 is 40."
        assert mock_gemini_client.generate_content.call_count == 2
    
    def test_multiple_tool_execution(self, client, mock_gemini_client):
        """Test execution of multiple tools in sequence."""
        # Model calls multiple tools
        function_calls = [
            {"name": "calculator", "args": {"operation": "multiply", "a": 6, "b": 7}},
            {"name": "get_weather", "args": {"city": "New York"}}
        ]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        
        # Model responds with both results
        mock_response_2 = create_gemini_response(
            text="6 × 7 = 42. Weather in New York: 72°F, sunny"
        )
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = client.generate_content(
            "Calculate 6 × 7 and get weather for New York",
            tools=[calculator, get_weather]
        )
        
        assert "42" in response
        assert "New York" in response
        assert "72°F, sunny" in response
    
    def test_multi_round_conversation(self, client, mock_gemini_client):
        """Test multi-round conversation with tools."""
        # Round 1: Model calls calculator
        function_calls_1 = [{"name": "calculator", "args": {"operation": "add", "a": 10, "b": 5}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls_1)
        
        # Round 2: Model calls another tool
        function_calls_2 = [{"name": "calculator", "args": {"operation": "multiply", "a": 15, "b": 2}}]
        mock_response_2 = create_gemini_response(function_calls=function_calls_2)
        
        # Round 3: Final response
        mock_response_3 = create_gemini_response(text="First: 15, then multiplied by 2: 30")
        
        mock_gemini_client.generate_content.side_effect = [
            mock_response_1, mock_response_2, mock_response_3
        ]
        
        response = client.generate_content(
            "Add 10 and 5, then multiply the result by 2",
            tools=[calculator]
        )
        
        assert "30" in response
        assert mock_gemini_client.generate_content.call_count == 3
    
    def test_no_tools_needed(self, client, mock_gemini_client):
        """Test when model doesn't need tools."""
        mock_response = create_gemini_response(text="I can answer this directly: The sky is blue.")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = client.generate_content(
            "What color is the sky?",
            tools=[calculator, get_weather]
        )
        
        assert response == "I can answer this directly: The sky is blue."
        assert mock_gemini_client.generate_content.call_count == 1

class TestComplexScenarios:
    """Test complex integration scenarios."""
    
    def test_tool_with_complex_parameters(self, client, mock_gemini_client):
        """Test tool with complex parameter types."""
        function_calls = [{
            "name": "complex_calculation",
            "args": {"numbers": [1.5, 2.3, 3.7, 4.1], "operation": "average"}
        }]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(text="The average is 2.9")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = client.generate_content(
            "Calculate the average of 1.5, 2.3, 3.7, and 4.1",
            tools=[complex_calculation]
        )
        
        assert "2.9" in response
    
    def test_tool_returning_structured_data(self, client, mock_gemini_client):
        """Test tool that returns structured data."""
        function_calls = [{
            "name": "search_database",
            "args": {"query": "machine learning", "limit": 3}
        }]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(text="Found articles about ML")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = client.generate_content(
            "Search for machine learning articles",
            tools=[search_database]
        )
        
        assert "Found articles about ML" in response
    
    def test_parallel_tool_execution(self, client, mock_gemini_client):
        """Test parallel execution of multiple tools."""
        function_calls = [
            {"name": "get_weather", "args": {"city": "London"}},
            {"name": "get_weather", "args": {"city": "Paris"}},
            {"name": "get_time", "args": {"timezone": "UTC"}}
        ]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(
            text="London: cloudy, Paris: partly cloudy, UTC time provided"
        )
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = client.generate_content(
            "Get weather for London and Paris, plus current UTC time",
            tools=[get_weather, get_time],
            parallel_tool_execution=True
        )
        execution_time = time.time() - start_time
        
        assert "London" in response
        assert "Paris" in response
        # Parallel execution should be faster than sequential
        assert execution_time < 0.5  # Should be very fast for mock tools

class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_tool_execution_error(self, client, mock_gemini_client):
        """Test handling of tool execution errors."""
        # Model calls failing tool
        function_calls = [{"name": "failing_operation", "args": {"should_fail": True}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        
        # Model handles the error
        mock_response_2 = create_gemini_response(text="The operation failed as expected")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = client.generate_content(
            "Try the failing operation",
            tools=[failing_operation],
            graceful_error_handling=True
        )
        
        assert "failed" in response
    
    def test_tool_execution_error_strict(self, client, mock_gemini_client):
        """Test strict error handling."""
        function_calls = [{"name": "failing_operation", "args": {"should_fail": True}}]
        mock_response = create_gemini_response(function_calls=function_calls)
        mock_gemini_client.generate_content.return_value = mock_response
        
        with pytest.raises(ValueError, match="Operation failed"):
            client.generate_content(
                "Try the failing operation",
                tools=[failing_operation],
                graceful_error_handling=False
            )
    
    def test_divide_by_zero_error(self, client, mock_gemini_client):
        """Test specific mathematical error."""
        function_calls = [{"name": "calculator", "args": {"operation": "divide", "a": 10, "b": 0}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(text="Cannot divide by zero")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = client.generate_content(
            "What is 10 divided by 0?",
            tools=[calculator],
            graceful_error_handling=True
        )
        
        assert "Cannot divide by zero" in response
    
    def test_max_tool_calls_exceeded(self, client, mock_gemini_client):
        """Test max tool calls limit."""
        # Always return tool calls to exceed limit
        function_calls = [{"name": "calculator", "args": {"operation": "add", "a": 1, "b": 1}}]
        mock_response = create_gemini_response(function_calls=function_calls)
        mock_gemini_client.generate_content.return_value = mock_response
        
        with pytest.raises(MaxToolCallsError):
            client.generate_content(
                "Keep calculating",
                tools=[calculator],
                max_tool_call_rounds=3
            )

class TestPerformanceAndLimits:
    """Test performance and limit scenarios."""
    
    def test_slow_tool_execution(self, client, mock_gemini_client):
        """Test execution with slow tools."""
        function_calls = [{"name": "slow_operation", "args": {"duration": 0.1, "task_name": "test"}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(text="Task completed")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        start_time = time.time()
        response = client.generate_content(
            "Run slow operation",
            tools=[slow_operation]
        )
        execution_time = time.time() - start_time
        
        assert "Task completed" in response
        assert execution_time >= 0.1  # Should take at least the sleep time
    
    def test_many_tools_registration(self, client, mock_gemini_client):
        """Test registration of many tools."""
        all_tools = [
            calculator, get_weather, get_time, search_database,
            slow_operation, failing_operation, complex_calculation
        ]
        
        mock_response = create_gemini_response(text="All tools available")
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = client.generate_content(
            "I have many tools available",
            tools=all_tools
        )
        
        assert response == "All tools available"
        # Check that tools were properly registered
        call_args = mock_gemini_client.generate_content.call_args[1]
        assert "tools" in call_args
        assert len(call_args["tools"]) == len(all_tools)

class TestStructuredOutputs:
    """Test structured output functionality."""
    
    def test_structured_output_simple(self, client, mock_gemini_client):
        """Test simple structured output."""
        from pydantic import BaseModel
        
        class WeatherReport(BaseModel):
            city: str
            temperature: float
            condition: str
        
        # Mock response that calls structured output tool
        from toolflow.core.constants import RESPONSE_FORMAT_TOOL_NAME
        function_calls = [{
            "name": RESPONSE_FORMAT_TOOL_NAME,
            "args": {
                "response": {
                    "city": "New York",
                    "temperature": 72.0,
                    "condition": "sunny"
                }
            }
        }]
        mock_response = create_gemini_response(function_calls=function_calls)
        mock_gemini_client.generate_content.return_value = mock_response
        
        response = client.generate_content(
            "Get weather report for New York",
            response_format=WeatherReport
        )
        
        assert isinstance(response, WeatherReport)
        assert response.city == "New York"
        assert response.temperature == 72.0
        assert response.condition == "sunny"
    
    def test_structured_output_with_tools(self, client, mock_gemini_client):
        """Test structured output combined with regular tools."""
        from pydantic import BaseModel
        
        class CalculationResult(BaseModel):
            operation: str
            operands: List[float]
            result: float
        
        # First call: use calculator tool
        function_calls_1 = [{"name": "calculator", "args": {"operation": "multiply", "a": 6, "b": 9}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls_1)
        
        # Second call: return structured output
        from toolflow.core.constants import RESPONSE_FORMAT_TOOL_NAME
        function_calls_2 = [{
            "name": RESPONSE_FORMAT_TOOL_NAME,
            "args": {
                "response": {
                    "operation": "multiply",
                    "operands": [6.0, 9.0],
                    "result": 54.0
                }
            }
        }]
        mock_response_2 = create_gemini_response(function_calls=function_calls_2)
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = client.generate_content(
            "Calculate 6 × 9 and return structured result",
            tools=[calculator],
            response_format=CalculationResult
        )
        
        assert isinstance(response, CalculationResult)
        assert response.operation == "multiply"
        assert response.result == 54.0

class TestConversationFlow:
    """Test conversation flow and context maintenance."""
    
    def test_context_preservation(self, client, mock_gemini_client):
        """Test that conversation context is preserved across tool calls."""
        # This tests that the conversation history is properly maintained
        function_calls = [{"name": "calculator", "args": {"operation": "add", "a": 5, "b": 3}}]
        mock_response_1 = create_gemini_response(function_calls=function_calls)
        mock_response_2 = create_gemini_response(text="The sum is 8, as calculated")
        
        mock_gemini_client.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        response = client.generate_content(
            "Please calculate 5 + 3 for me",
            tools=[calculator]
        )
        
        assert "8" in response
        
        # Verify that the second call includes the conversation history
        second_call_args = mock_gemini_client.generate_content.call_args_list[1][1]
        assert "contents" in second_call_args
        
        # The contents should include the original message, the tool call, and the tool result
        contents = second_call_args["contents"]
        assert len(contents) >= 3  # User message, assistant message with tool call, function response

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
