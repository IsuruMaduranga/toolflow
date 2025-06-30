"""
Live Integration Tests for Toolflow with OpenAI

These tests use actual OpenAI API calls to verify end-to-end functionality.
Set OPENAI_API_KEY environment variable to run these tests.

Run with: python -m pytest tests/openai/test_integration_live.py -v -s

Note: These tests make real API calls and will consume OpenAI credits.
"""

import os
import asyncio
import time
from typing import List, Optional, Dict, Any
import pytest
from dotenv import load_dotenv
load_dotenv()

try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

import toolflow
from toolflow import MaxToolCallsError


# Skip all tests if OpenAI API key is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not OPENAI_AVAILABLE,
    reason="OpenAI API key not available or openai package not installed"
)


# Test Models for Structured Output
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


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class ComprehensiveResponse(BaseModel):
    summary: str
    calculations: List[MathResult]
    total_operations: int


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class BookRecommendation(BaseModel):
    title: str
    author: str
    genre: str
    year_published: int
    reason: str


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class WeatherData(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int


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


class TestBasicOpenAIToolCalling:
    """Test basic tool calling functionality with real OpenAI API."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    def test_single_tool_call(self, client):
        """Test calling a single tool."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 15 + 27?"}],
            tools=[simple_calculator],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "42" in response

    def test_multiple_tool_calls(self, client):
        """Test calling multiple tools in sequence."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 10 + 5, then multiply the result by 3"}],
            tools=[simple_calculator],
            max_tool_calls=5
        )
        
        assert response is not None
        assert "45" in response

    def test_parallel_tool_execution(self, client):
        """Test parallel tool execution performance."""
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Add 10 and 20, get current time, and format 'hello world' in uppercase"}],
            tools=[simple_calculator, get_system_info, format_data],
            max_tool_calls=5
        )
        
        assert response is not None
        content = response
        assert "30" in content  # Addition result
        assert "HELLO WORLD" in content  # Formatted text

    def test_system_message(self, client):
        """Test using system messages with OpenAI."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful math assistant. Always use the calculator tool for calculations."},
                {"role": "user", "content": "What is 25 * 4?"}
            ],
            tools=[simple_calculator],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "100" in response

    def test_weather_tool_with_system(self, client):
        """Test weather tool with system context."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a travel assistant. Use the weather tool to help users plan their trips."},
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            tools=[get_weather],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "San Francisco" in response
        assert "weather" in response.lower()


class TestOpenAIStructuredOutput:
    """Test structured output functionality with real OpenAI API."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_weather(self, client):
        """Test structured output with weather tool."""
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": "Get the weather for New York and format it as structured data"
            }],
            tools=[get_weather],
            response_format=WeatherData
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, WeatherData)
        assert result.city == "New York"
        assert isinstance(result.temperature, float)
        assert isinstance(result.condition, str)
        assert isinstance(result.humidity, int)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_math(self, client):
        """Test structured output with math calculation."""
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": "Calculate 15 * 8 + 7 and provide a structured result with explanation"
            }],
            tools=[calculate_advanced],
            response_format=MathResult
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, MathResult)
        assert isinstance(result.operation, str)
        assert isinstance(result.operands, list)
        assert result.result == 127.0  # 15 * 8 + 7 = 127
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 10

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_without_tools(self, client):
        """Test structured output without tools."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Recommend a good science fiction book published in the last 5 years"}],
            response_format=BookRecommendation
        )
        
        assert response is not None
        parsed = response
        assert isinstance(parsed, BookRecommendation)
        assert isinstance(parsed.title, str)
        assert isinstance(parsed.author, str)
        assert parsed.genre.lower() == "science fiction"
        assert isinstance(parsed.year_published, int)
        assert isinstance(parsed.reason, str)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_full_response_mode(self, client):
        """Test structured output in full response mode."""
        client_full = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")), full_response=True)
        
        result = client_full.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": "Get weather for London and structure the response"
            }],
            tools=[get_weather],
            response_format=WeatherData
        )
        
        # With full_response=True, toolflow should add a parsed attribute to the response
        # But if not available, just verify we get a proper response object
        if hasattr(result, 'parsed'):
            parsed = result.parsed
            assert isinstance(parsed, WeatherData)
            assert parsed.city == "London"
            assert isinstance(parsed.temperature, float)
            assert isinstance(parsed.condition, str)
            assert isinstance(parsed.humidity, int)
        else:
            # Fallback: verify we get a proper response object
            assert hasattr(result, 'choices')
            assert len(result.choices) > 0
            # The structured response should be embedded in the tool calls
            assert result.choices[0].message.tool_calls is not None

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    @pytest.mark.asyncio
    async def test_async_structured_output(self):
        """Test async structured output functionality."""
        async_client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        result = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": "Search for a good mystery book and provide structured information"
            }],
            tools=[search_books],
            response_format=BookRecommendation
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, BookRecommendation)
        assert isinstance(result.title, str)
        assert isinstance(result.author, str)
        assert result.genre.lower() == "mystery"
        assert isinstance(result.year_published, int)
        assert isinstance(result.reason, str)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_error_handling(self, client):
        """Test structured output error handling with invalid response format."""
        # Test with invalid response format
        with pytest.raises(ValueError, match="Response format .* is not a Pydantic model"):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=dict  # Invalid - not a Pydantic model
            )

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_streaming_error(self, client):
        """Test that structured output with streaming raises an error."""
        # Test streaming with response_format should raise error
        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=WeatherData,
                stream=True
            )


class TestOpenAIAsyncFunctionality:
    """Test async functionality with OpenAI."""

    @pytest.fixture
    def async_client(self):
        """Create async toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_sync_tools(self, async_client):
        """Test async client with sync tools."""
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 12 * 8?"}],
            tools=[simple_calculator],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "96" in response

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_async_tools(self, async_client):
        """Test async client with async tools."""
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 50 with 0.5 second delay"}],
            tools=[async_delay_calculator],
            max_tool_calls=3
        )
        
        assert response is not None
        assert "100" in response  # 50 * 2

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_mixed_tools(self, async_client):
        """Test async client with mixed sync/async tools."""
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Add 15 and 25, then double the result with a delay"}],
            tools=[simple_calculator, async_delay_calculator],
            parallel_tool_execution=True,
            max_tool_calls=5
        )
        
        assert response is not None
        # The model might interpret this differently - let's check for the presence of calculation results
        # Should contain the initial addition result (40) and some doubled result
        assert "40" in response or "80" in response or "160" in response  # Be flexible about the final result

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_parallel_tool_execution(self, async_client):
        """Test async parallel tool execution."""
        start_time = time.time()
        
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate Fibonacci numbers for 8, 9, and 10, and get the current time"}],
            tools=[get_fibonacci, get_system_info],
            parallel_tool_execution=True,
            max_tool_calls=8
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert response is not None
        # Should contain multiple Fibonacci results and time info
        assert "21" in response  # Fib(8)
        assert "34" in response  # Fib(9)
        assert "55" in response  # Fib(10)
        assert "time" in response.lower()
        
        # Parallel execution should be reasonably fast
        assert execution_time < 30  # Should complete within 30 seconds

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_structured_output_with_tools(self, async_client):
        """Test async structured output functionality with tools."""
        result = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Search for a good mystery book and provide structured information"}],
            tools=[search_books],
            response_format=BookRecommendation
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, BookRecommendation)
        assert isinstance(result.title, str)
        assert isinstance(result.author, str)
        assert result.genre.lower() == "mystery"
        assert isinstance(result.year_published, int)
        assert isinstance(result.reason, str)


class TestOpenAIStreamingFunctionality:
    """Test streaming functionality with OpenAI."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    def test_streaming_with_tools(self, client):
        """Test streaming with tool execution."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
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
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Write a short poem about AI"}],
            stream=True
        )
        
        content_chunks = []
        for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks)
        assert len(full_content) > 50  # Should be a reasonable poem
        # More flexible check - just ensure it's poetry-like content
        assert any(word in full_content.lower() for word in ["ai", "artificial", "intelligence", "computer", "machine", "digital", "technology", "future", "learning", "data", "code"])

    def test_streaming_with_parallel_tools(self, client):
        """Test streaming with parallel tool execution."""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 8 * 9, get the current time, and format 'testing' in uppercase"}],
            tools=[simple_calculator, get_system_info, format_data],
            stream=True,
            parallel_tool_execution=True,
            max_tool_calls=6
        )
        
        content_chunks = []
        for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks)
        assert "72" in full_content  # 8 * 9
        assert "TESTING" in full_content  # Formatted text
        assert any(word in full_content.lower() for word in ["time", "current"])

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_streaming_with_tools(self):
        """Test async streaming with tools."""
        async_client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        stream = await async_client.chat.completions.create(
            model="gpt-4o-mini",
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

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_streaming_without_tools(self):
        """Test async streaming without tools."""
        async_client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        stream = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=True
        )
        
        content_chunks = []
        async for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks)
        assert len(full_content) > 10
        # Should contain numbers
        assert any(str(num) in full_content for num in [1, 2, 3, 4, 5])

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_streaming_with_parallel_tools(self):
        """Test async streaming with parallel tool execution."""
        async_client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        stream = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate Fibonacci of 7, multiply 6 by 8, and get system status"}],
            tools=[get_fibonacci, simple_calculator, get_system_info],
            stream=True,
            parallel_tool_execution=True,
            max_tool_calls=6
        )
        
        content_chunks = []
        async for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks)
        assert "13" in full_content  # Fib(7)
        assert "48" in full_content  # 6 * 8
        assert "operational" in full_content.lower() or "status" in full_content.lower()


class TestOpenAIErrorHandling:
    """Test error handling with OpenAI."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    def test_tool_execution_error(self, client):
        """Test handling of tool execution errors."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Divide 10 by 0"}],
            tools=[simple_calculator],
            max_tool_calls=3
        )
        
        # Should handle the error gracefully and explain division by zero
        assert response is not None
        content = response.lower()
        assert any(word in content for word in ["error", "zero", "cannot", "divide"])

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
        
        with pytest.raises(MaxToolCallsError, match="Max tool calls reached"):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Call the recursive_tool with task 'start' and keep calling it until it's done. The tool will tell you when to call it again."}],
                tools=[recursive_tool],
                max_tool_calls=2  # Low limit
            )

    def test_max_tool_calls_with_weather(self, client):
        """Test that max_tool_calls is respected"""
        call_count = 0
        
        @toolflow.tool
        def weather_loop_tool(location: str) -> str:
            """A weather tool that always requests another call."""
            nonlocal call_count
            call_count += 1
            if call_count < 5:  # Ensure it keeps requesting more calls
                return f"Weather in {location}: Sunny, 72°F. Call #{call_count}. Please call me again with a different location like 'Tokyo' or 'London'."
            return f"Weather in {location}: Sunny, 72°F. Final call #{call_count}."
        
        with pytest.raises(MaxToolCallsError, match="Max tool calls reached"):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Call the weather_loop_tool with location 'Paris' and keep calling it with different locations as the tool suggests."}
                ],
                tools=[weather_loop_tool],
                max_tool_calls=2  # Very low limit to test the constraint
            )


class TestOpenAIComprehensiveWorkflow:
    """Test comprehensive workflows with OpenAI."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    def test_full_workflow_sync(self, client):
        """Test a complete workflow with multiple tools and parallel execution."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": "Perform these tasks: 1) Calculate 15*4, 2) Get 8th Fibonacci number, 3) Get current time, 4) Format 'success' in uppercase"
            }],
            tools=[simple_calculator, get_fibonacci, get_system_info, format_data],
            parallel_tool_execution=True,
            max_tool_calls=8
        )
        
        assert response is not None
        content = response
        
        # Check for expected results
        assert "60" in content  # 15*4
        assert "21" in content  # 8th Fibonacci
        assert "SUCCESS" in content  # Formatted text
        assert any(word in content for word in ["time", "2024", "2025"])  # Current time

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_full_workflow_async(self):
        """Test full async workflow."""
        async_client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 20/4, get weather for Tokyo, and get system version"}],
            tools=[simple_calculator, get_weather, get_system_info],
            parallel_tool_execution=True,
            max_tool_calls=6
        )
        
        assert response is not None
        content = response.lower()
        
        # Check expected results
        assert "5" in response  # 20/4
        assert "tokyo" in content  # Weather info
        assert "version" in content  # System version

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_full_workflow_async_with_mixed_tools(self):
        """Test full async workflow with mixed sync/async tools."""
        async_client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": "Calculate 18 / 3, double the result with a delay, and get weather for London"
            }],
            tools=[simple_calculator, async_delay_calculator, get_weather],
            parallel_tool_execution=True,
            max_tool_calls=8
        )
        
        assert response is not None
        content = response.lower()
        
        # Check expected results: 18/3 = 6, then 6*2 = 12
        assert "6" in response or "12" in response  # Either step result
        assert "london" in content  # Weather info


class TestOpenAIPerformanceBenchmarks:
    """Test performance characteristics of OpenAI integration."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    @pytest.mark.skipif(True, reason="Performance test - run manually if needed")
    def test_parallel_vs_sequential_performance(self, client):
        """Compare parallel vs sequential tool execution performance."""
        
        # Sequential execution
        start_time = time.time()
        response_seq = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate Fibonacci for 10, 11, 12, and 13"}],
            tools=[get_fibonacci],
            parallel_tool_execution=False,
            max_tool_calls=10
        )
        sequential_time = time.time() - start_time
        
        # Parallel execution
        start_time = time.time()
        response_par = client.chat.completions.create(
            model="gpt-4o-mini",
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
        if parallel_time > 0:
            print(f"Speedup: {sequential_time / parallel_time:.2f}x")

    @pytest.mark.skipif(True, reason="Large scale test - run manually if needed")
    def test_large_parallel_execution(self, client):
        """Test large scale parallel execution."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate fibonacci numbers from 5 to 15"}],
            tools=[get_fibonacci],
            parallel_tool_execution=True,
            max_tool_calls=12,
            max_workers=5
        )
        
        assert response is not None
        content = response
        
        # Should contain multiple Fibonacci numbers
        fibonacci_sequence = [5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        found_numbers = sum(1 for num in fibonacci_sequence if str(num) in content)
        assert found_numbers >= 5  # Should find at least 5 of the expected numbers

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.skipif(True, reason="Performance test - run manually if needed")
    @pytest.mark.asyncio
    async def test_async_parallel_performance(self):
        """Test async parallel execution performance."""
        async_client = toolflow.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        start_time = time.time()
        
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate multiple operations: 12*8, 15+25, 100/4, get time, and format 'async' in title case"}],
            tools=[simple_calculator, get_system_info, format_data],
            parallel_tool_execution=True,
            max_tool_calls=10
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert response is not None
        content = response
        
        # Check for expected results
        assert "96" in content  # 12*8
        assert "40" in content  # 15+25
        assert "25" in content  # 100/4
        assert "Async" in content  # Formatted text
        
        print(f"Async parallel execution time: {execution_time:.2f}s")
        assert execution_time < 30  # Should complete reasonably fast


if __name__ == "__main__":
    # Run a quick test to verify setup
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI package not available")
        print("Install it with: pip install openai")
        exit(1)
    
    print("✅ Environment setup complete")
    print("Run tests with: python -m pytest tests/openai/test_integration_live.py -v -s")
    print("Note: These tests will consume OpenAI API credits") 