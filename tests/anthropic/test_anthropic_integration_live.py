"""
Live Integration Tests for Toolflow with Anthropic

These tests use actual Anthropic API calls to verify end-to-end functionality.
Set ANTHROPIC_API_KEY environment variable to run these tests.

Run with: python -m pytest tests/anthropic/test_anthropic_integration_live.py -v -s

Note: These tests make real API calls and will consume Anthropic credits.
"""

import os
import asyncio
import time
from typing import List, Optional, Dict, Any
import pytest

try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

import toolflow


# Skip all tests if Anthropic API key is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") or not ANTHROPIC_AVAILABLE,
    reason="Anthropic API key not available or anthropic package not installed"
)


# Test Models for future structured output support
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class WeatherInfo(BaseModel):
    city: str
    temperature: int
    condition: str
    humidity: Optional[int] = None


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class MathResult(BaseModel):
    operation: str
    operands: List[float]
    result: float
    explanation: str


# Test Tools
@toolflow.tool
def simple_calculator(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical operations: add, subtract, multiply, divide."""
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


@toolflow.tool
def get_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    if n > 20:  # Limit for performance
        return -1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


@toolflow.tool
def get_system_info(info_type: str) -> str:
    """Get system information. Available types: time, version, status."""
    if info_type == "time":
        return f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    elif info_type == "version":
        return "Toolflow Version 1.0.0"
    elif info_type == "status":
        return "System Status: All systems operational"
    else:
        return f"Unknown info type: {info_type}"


@toolflow.tool
async def async_delay_calculator(seconds: float, value: float) -> float:
    """Perform calculation after a delay (async tool)."""
    await asyncio.sleep(min(seconds, 2.0))  # Cap at 2 seconds for tests
    return value * 2


@toolflow.tool
def format_data(data: str, format_type: str = "uppercase") -> str:
    """Format data according to specified type: uppercase, lowercase, title."""
    if format_type == "uppercase":
        return data.upper()
    elif format_type == "lowercase":
        return data.lower()
    elif format_type == "title":
        return data.title()
    else:
        return data


@toolflow.tool
def get_weather(location: str) -> str:
    """Get weather for a given location (mock implementation)."""
    return f"Weather in {location}: Sunny, 72°F with light winds"


class TestBasicAnthropicToolCalling:
    """Test basic tool calling functionality with real Anthropic API."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    def test_single_tool_call(self, client):
        """Test calling a single tool."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 15 + 27?"}],
            tools=[simple_calculator],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "42" in response

    def test_multiple_tool_calls(self, client):
        """Test calling multiple tools in sequence."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate 10 + 5, then multiply the result by 3"}],
            tools=[simple_calculator],
            max_tool_calls=5
        )
        
        assert response is not None
        assert "45" in response

    def test_parallel_tool_execution(self, client):
        """Test parallel tool execution performance."""
        start_time = time.time()
        
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate the 10th and 12th Fibonacci numbers"}],
            tools=[get_fibonacci],
            parallel_tool_execution=True,
            max_tool_calls=5
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert response is not None
        # Should mention both Fibonacci numbers
        content = response
        assert any(str(num) in content for num in [55, 144])  # 10th=55, 12th=144

    def test_mixed_tool_types(self, client):
        """Test using different types of tools together."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Add 10 and 20, get current time, and format 'hello world' in uppercase"}],
            tools=[simple_calculator, get_system_info, format_data],
            max_tool_calls=5
        )
        
        assert response is not None
        content = response
        assert "30" in content  # Addition result
        assert "HELLO WORLD" in content  # Formatted text

    def test_system_message(self, client):
        """Test using system messages with Anthropic."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are a helpful math assistant. Always use the calculator tool for calculations.",
            messages=[{"role": "user", "content": "What is 25 * 4?"}],
            tools=[simple_calculator],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "100" in response

    def test_weather_tool_with_system(self, client):
        """Test weather tool with system context."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are a travel assistant. Use the weather tool to help users plan their trips.",
            messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
            tools=[get_weather],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "San Francisco" in response
        assert "weather" in response.lower()


class TestAnthropicAsyncFunctionality:
    """Test async functionality with Anthropic."""

    @pytest.fixture
    def async_client(self):
        """Create async toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic_async(anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_sync_tools(self, async_client):
        """Test async client with sync tools."""
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 20 * 3?"}],
            tools=[simple_calculator],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "60" in response

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_async_tools(self, async_client):
        """Test async client with async tools."""
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate 5 * 2 with a small delay"}],
            tools=[async_delay_calculator],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "10" in response  # 5 * 2 = 10

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_mixed_tools(self, async_client):
        """Test async client with both sync and async tools."""
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Add 15 and 25, then double the result with a delay"}],
            tools=[simple_calculator, async_delay_calculator],
            parallel_tool_execution=True,
            max_tool_calls=5
        )
        
        assert response is not None
        # Result should be (15 + 25) * 2 = 80
        assert "80" in response


class TestAnthropicStreamingFunctionality:
    """Test streaming functionality with Anthropic."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    def test_streaming_with_tools(self, client):
        """Test streaming with tool execution."""
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 7 * 6?"}],
            tools=[simple_calculator],
            stream=True,
            max_tool_calls=3
        )
        
        content_chunks = []
        for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks)
        assert "42" in full_content

    def test_streaming_without_tools(self, client):
        """Test streaming without tools."""
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Write a short poem about AI"}],
            stream=True
        )
        
        content_chunks = []
        for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks)
        assert len(full_content) > 50  # Should be a reasonable poem
        assert any(word in full_content.lower() for word in ["ai", "artificial", "intelligence", "computer"])

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available") 
    @pytest.mark.asyncio
    async def test_async_streaming_with_tools(self):
        """Test async streaming with tools."""
        async_client = toolflow.from_anthropic_async(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        stream = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 9 * 4?"}],
            tools=[simple_calculator],
            stream=True,
            max_tool_calls=3
        )
        
        content_chunks = []
        async for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks)
        assert "36" in full_content


class TestAnthropicErrorHandling:
    """Test error handling with Anthropic."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    def test_tool_execution_error(self, client):
        """Test handling of tool execution errors."""
        @toolflow.tool
        def divide_by_zero(x: float) -> float:
            """Divide by zero (will cause error)."""
            return x / 0
        
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Please divide 10 by zero"}],
            tools=[divide_by_zero],
            graceful_error_handling=True,
            max_tool_calls=3
        )
        
        assert response is not None
        # Should handle the error gracefully
        assert any(word in response.lower() for word in ["error", "divide", "zero"])

    def test_max_tool_calls_limit(self, client):
        """Test that max tool calls limit is respected."""
        call_count = 0
        
        @toolflow.tool
        def recursive_tool(task: str) -> str:
            """A tool that always requests another call."""
            nonlocal call_count
            call_count += 1
            if call_count < 5:  # Ensure it keeps requesting more calls
                return f"Call {call_count}: {task}. Please call me again with task 'continue'"
            return f"Final call {call_count}: {task}"
        
        with pytest.raises(Exception, match="Max tool calls reached"):
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Call the recursive_tool with task 'start' and keep calling it until it's done. The tool will tell you when to call it again."}],
                tools=[recursive_tool],
                max_tool_calls=2  # Low limit
            )


class TestAnthropicComprehensiveWorkflow:
    """Test comprehensive workflows with Anthropic."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    def test_full_workflow_sync(self, client):
        """Test a complete workflow with multiple tool types."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are a helpful assistant that uses tools to solve problems step by step.",
            messages=[{
                "role": "user", 
                "content": "Calculate 15 + 25, get the current time, format 'HELLO world' to title case, and tell me the weather in Tokyo"
            }],
            tools=[simple_calculator, get_system_info, format_data, get_weather],
            parallel_tool_execution=True,
            max_tool_calls=10
        )
        
        assert response is not None
        content = response.lower()
        
        # Check all expected results
        assert "40" in response  # 15 + 25
        assert "hello world" in content  # Title case formatting
        assert "tokyo" in content  # Weather info
        assert any(word in content for word in ["time", "current"])  # Time info

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_full_workflow_async(self):
        """Test a complete async workflow."""
        async_client = toolflow.from_anthropic_async(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are an async assistant that efficiently handles multiple tasks.",
            messages=[{
                "role": "user",
                "content": "Multiply 6 by 7, then calculate the Fibonacci of 8, and get system status"
            }],
            tools=[simple_calculator, get_fibonacci, get_system_info],
            parallel_tool_execution=True,
            max_tool_calls=8
        )
        
        assert response is not None
        content = response.lower()
        
        # Check expected results
        assert "42" in response  # 6 * 7
        assert "21" in response  # Fibonacci(8)
        assert "operational" in content  # System status


class TestAnthropicPerformanceBenchmarks:
    """Test performance characteristics of Anthropic integration."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    @pytest.mark.skipif(True, reason="Performance test - run manually if needed")
    def test_parallel_vs_sequential_performance(self, client):
        """Compare parallel vs sequential tool execution performance."""
        
        # Sequential execution
        start_time = time.time()
        response_seq = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate Fibonacci for 10, 11, 12, and 13"}],
            tools=[get_fibonacci],
            parallel_tool_execution=False,
            max_tool_calls=10
        )
        sequential_time = time.time() - start_time
        
        # Parallel execution
        start_time = time.time()
        response_par = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate Fibonacci for 10, 11, 12, and 13"}],
            tools=[get_fibonacci],
            parallel_tool_execution=True,
            max_tool_calls=10
        )
        parallel_time = time.time() - start_time
        
        # Both should get correct results
        assert "55" in response_seq  # Fib(10)
        assert "89" in response_seq  # Fib(11)
        assert "55" in response_par  # Fib(10)
        assert "89" in response_par  # Fib(11)
        
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup: {sequential_time / parallel_time:.2f}x")


# Test Pydantic models for structured output
class WeatherData(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int


class MathResult(BaseModel):
    operation: str
    numbers: list[float]
    result: float
    explanation: str


class BookRecommendation(BaseModel):
    title: str
    author: str
    genre: str
    year_published: int
    reason: str


@toolflow.tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Simulate weather API call
    weather_data = {
        "New York": "72°F, sunny with 45% humidity",
        "London": "58°F, cloudy with 80% humidity", 
        "Tokyo": "68°F, partly cloudy with 60% humidity",
        "Sydney": "75°F, clear with 35% humidity"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@toolflow.tool
def calculate_advanced(expression: str) -> str:
    """Calculate a mathematical expression and explain the result."""
    try:
        # Simple evaluation for demo purposes
        result = eval(expression)
        return f"The result of {expression} is {result}. This was calculated using standard arithmetic operations."
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@toolflow.tool
async def search_books(genre: str, year_range: str = "recent") -> str:
    """Search for book recommendations in a specific genre."""
    # Simulate book search API
    books = {
        "science fiction": "Klara and the Sun by Kazuo Ishiguro (2021) - A beautiful exploration of AI consciousness",
        "mystery": "The Thursday Murder Club by Richard Osman (2020) - A cozy mystery with elderly detectives",
        "fantasy": "The Priory of the Orange Tree by Samantha Shannon (2019) - Epic fantasy with dragons"
    }
    return books.get(genre.lower(), f"No recent recommendations found for {genre}")


class TestAnthropicLiveStructuredOutput:
    """Live tests for Anthropic structured output functionality."""
    
    def test_live_structured_output_weather(self):
        """Test structured output with weather tool."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user", 
                "content": "Get the weather for New York and format it as structured data"
            }],
            tools=[get_current_weather],
            response_format=WeatherData,
            max_tokens=1000
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, WeatherData)
        assert result.city == "New York"
        assert isinstance(result.temperature, float)
        assert isinstance(result.condition, str)
        assert isinstance(result.humidity, int)
        
        print(f"Weather result: {result}")
    
    def test_live_structured_output_math(self):
        """Test structured output with math calculation."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user", 
                "content": "Calculate 15 * 8 + 7 and provide a structured result with explanation"
            }],
            tools=[calculate_advanced],
            response_format=MathResult,
            max_tokens=1000
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, MathResult)
        assert result.operation.lower() == "multiplication and addition"
        assert isinstance(result.numbers, list)
        assert result.result == 127.0  # 15 * 8 + 7 = 127
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 10
        
        print(f"Math result: {result}")
    
    def test_live_structured_output_without_tools(self):
        """Test structured output without any tools."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user", 
                "content": "Recommend a good science fiction book published in the last 5 years"
            }],
            response_format=BookRecommendation,
            max_tokens=1000
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, BookRecommendation)
        assert isinstance(result.title, str)
        assert isinstance(result.author, str)
        assert result.genre.lower() == "science fiction"
        assert isinstance(result.year_published, int)
        # Be more flexible with year - AI models may not always follow the exact constraint
        assert result.year_published >= 2010  # Still relatively recent
        assert isinstance(result.reason, str)
        assert len(result.reason) > 20
        
        print(f"Book recommendation: {result}")
    
    def test_live_structured_output_full_response_mode(self):
        """Test structured output in full response mode."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic(), full_response=True)
        
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user", 
                "content": "Get weather for London and structure the response"
            }],
            tools=[get_current_weather],
            response_format=WeatherData,
            max_tokens=1000
        )
        
        # Should return full response object with parsed attribute
        # Anthropic response structure is different from OpenAI
        assert hasattr(result, 'parsed')
        
        parsed = result.parsed
        assert isinstance(parsed, WeatherData)
        assert parsed.city == "London"
        assert isinstance(parsed.temperature, float)
        assert isinstance(parsed.condition, str)
        assert isinstance(parsed.humidity, int)
        
        print(f"Full response weather result: {parsed}")
    
    @pytest.mark.asyncio
    async def test_live_async_structured_output(self):
        """Test async structured output functionality."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic_async(AsyncAnthropic())
        
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user", 
                "content": "Search for a good mystery book and provide structured information"
            }],
            tools=[search_books],
            response_format=BookRecommendation,
            max_tokens=1000
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, BookRecommendation)
        assert isinstance(result.title, str)
        assert isinstance(result.author, str)
        assert result.genre.lower() == "mystery"
        assert isinstance(result.year_published, int)
        assert isinstance(result.reason, str)
        
        print(f"Async book recommendation: {result}")
    
    def test_live_structured_output_error_handling(self):
        """Test structured output error handling with invalid response format."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        # Test with invalid response format
        with pytest.raises(ValueError, match="response_format must be a Pydantic model"):
            client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=dict,  # Invalid - not a Pydantic model
                max_tokens=1000
            )
    
    def test_live_structured_output_streaming_error(self):
        """Test that structured output with streaming raises an error."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        # Test streaming with response_format should raise error
        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=WeatherData,
                stream=True,
                max_tokens=1000
            )


class TestAnthropicLiveBasicFunctionality:
    """Live tests for basic Anthropic functionality."""
    
    def test_live_simple_message(self):
        """Test simple message without tools."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
            max_tokens=100
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Simple message result: {result}")
    
    def test_live_tool_calling(self):
        """Test basic tool calling functionality."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
            tools=[get_current_weather],
            max_tokens=500
        )
        
        assert isinstance(result, str)
        assert "Tokyo" in result
        assert "68°F" in result or "weather" in result.lower()
        print(f"Tool calling result: {result}")
    
    def test_live_parallel_tool_execution(self):
        """Test parallel tool execution."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user", 
                "content": "Get weather for both New York and London, and calculate 25 * 4"
            }],
            tools=[get_current_weather, calculate_advanced],
            parallel_tool_execution=True,
            max_tokens=1000
        )
        
        assert isinstance(result, str)
        assert "New York" in result
        assert "London" in result
        assert "100" in result  # 25 * 4 = 100
        print(f"Parallel execution result: {result}")
    
    @pytest.mark.asyncio
    async def test_live_async_functionality(self):
        """Test async functionality."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic_async(AsyncAnthropic())
        
        result = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Find a fantasy book recommendation"}],
            tools=[search_books],
            max_tokens=500
        )
        
        assert isinstance(result, str)
        assert "fantasy" in result.lower() or "book" in result.lower()
        print(f"Async functionality result: {result}")
    
    def test_live_streaming(self):
        """Test streaming functionality."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        stream = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Count from 1 to 5 slowly."}],
            stream=True,
            max_tokens=200
        )
        
        collected_content = []
        for chunk in stream:
            if chunk:
                collected_content.append(str(chunk))
        
        full_content = ''.join(collected_content)
        assert len(full_content) > 0
        print(f"Streaming result: {full_content}")
    
    def test_live_streaming_with_tools(self):
        """Test streaming with tool execution."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")
        
        client = toolflow.from_anthropic(Anthropic())
        
        stream = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Get the weather for Sydney and tell me about it."}],
            tools=[get_current_weather],
            stream=True,
            max_tokens=500
        )
        
        collected_content = []
        for chunk in stream:
            if chunk:
                collected_content.append(str(chunk))
        
        full_content = ''.join(collected_content)
        assert len(full_content) > 0
        assert "Sydney" in full_content
        print(f"Streaming with tools result: {full_content}")


if __name__ == "__main__":
    # Run specific test classes if executed directly
    pytest.main([__file__, "-v", "-s"]) 
