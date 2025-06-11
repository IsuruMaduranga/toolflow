"""
Live Integration Tests for Toolflow

These tests use actual OpenAI API calls to verify end-to-end functionality.
Set OPENAI_API_KEY environment variable to run these tests.

Run with: python -m pytest tests/test_integration_live.py -v -s

Note: These tests make real API calls and will consume OpenAI credits.
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


class TestBasicToolCalling:
    """Test basic tool calling functionality with real API."""

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


class TestStructuredOutput:
    """Test structured output functionality with real API."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_parse_method_basic(self, client):
        """Test basic parse method functionality."""
        response = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 25 + 17 and explain the operation"}],
            tools=[simple_calculator],
            response_format=MathResult
        )
        
        assert response is not None
        parsed = response
        assert isinstance(parsed, MathResult)
        assert parsed.result == 42.0
        assert parsed.operation == "add"
        assert parsed.operands == [25.0, 17.0]

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_parse_method_complex(self, client):
        """Test parse method with complex structured output."""
        response = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Perform these calculations: 10+5, 20*3, 50/2. Provide a summary."}],
            tools=[simple_calculator],
            response_format=ComprehensiveResponse
        )
        
        assert response is not None
        parsed = response
        assert isinstance(parsed, ComprehensiveResponse)
        assert len(parsed.calculations) == 3
        assert parsed.total_operations == 3
        assert parsed.summary is not None

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_beta_api_structured_output(self, client):
        """Test beta API with structured output."""
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 8 * 7 and provide details"}],
            tools=[simple_calculator],
            response_format=MathResult
        )
        
        assert response is not None
        parsed = response
        assert isinstance(parsed, MathResult)
        assert parsed.result == 56.0


class TestAsyncFunctionality:
    """Test async functionality with real API."""

    @pytest.fixture
    def async_client(self):
        """Create async toolflow wrapped OpenAI client."""
        return toolflow.from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

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
            messages=[{"role": "user", "content": "Add 10+5, then double it with a delay"}],
            tools=[simple_calculator, async_delay_calculator],
            max_tool_calls=5
        )
        
        assert response is not None

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_structured_output(self, async_client):
        """Test async client with structured output."""
        response = await async_client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 6 * 9 and explain"}],
            tools=[simple_calculator],
            response_format=MathResult
        )
        
        assert response is not None
        parsed = response
        assert parsed.result == 54.0


class TestStreamingFunctionality:
    """Test streaming functionality with real API."""

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

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_streaming_with_tools(self):
        """Test async streaming with tools."""
        async_client = toolflow.from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
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


class TestErrorHandling:
    """Test error handling with real API."""

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
        """Test max tool calls limit enforcement."""
        # This should work within the limit
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 5+5, then 10*2"}],
            tools=[simple_calculator],
            max_tool_calls=3
        )
        
        assert response is not None

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_streaming_with_structured_output_error(self, client):
        """Test that streaming with structured output raises appropriate error."""
        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            client.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test"}],
                tools=[simple_calculator],
                response_format=MathResult,
                stream=True
            )


class TestComprehensiveWorkflow:
    """Test comprehensive workflows combining multiple features."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    def test_full_workflow_sync(self, client):
        """Test a full workflow with multiple tools and parallel execution."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": "Perform these tasks: 1) Calculate 15*4, 2) Get 8th Fibonacci number, 3) Get current time, 4) Format 'success' in uppercase"
            }],
            tools=[simple_calculator, get_fibonacci, get_system_info, format_data],
            parallel_tool_execution=True,
            max_tool_calls=8,
            max_workers=4
        )
        
        assert response is not None
        content = response
        
        # Check for expected results
        assert "60" in content  # 15*4
        assert "21" in content  # 8th Fibonacci
        assert "SUCCESS" in content  # Formatted text
        assert any(word in content for word in ["time", "2024", "2025"])  # Current time

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_full_workflow_async_structured(self):
        """Test full async workflow with structured output."""
        async_client = toolflow.from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        
        response = await async_client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 20/4 and explain the division operation"}],
            tools=[simple_calculator],
            response_format=MathResult,
            parallel_tool_execution=True,
            max_workers=2
        )
        
        assert response is not None
        parsed = response
        assert isinstance(parsed, MathResult)
        assert parsed.result == 5.0
        assert parsed.operation == "divide"
        assert parsed.operands == [20.0, 4.0]

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_beta_vs_main_api_comparison(self, client):
        """Test comparing beta API vs main API structured output."""
        message = [{"role": "user", "content": "Multiply 11 by 12 and explain the calculation"}]
        
        # Main API
        response_main = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=message,
            tools=[simple_calculator],
            response_format=MathResult
        )
        
        # Beta API
        response_beta = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=message,
            tools=[simple_calculator],
            response_format=MathResult
        )
        
        # Both should work and give same result
        assert response_main is not None
        assert response_beta is not None
        
        parsed_main = response_main
        parsed_beta = response_beta
        
        assert parsed_main.result == 132.0
        assert parsed_beta.result == 132.0
        assert parsed_main.operation == parsed_beta.operation == "multiply"


# Performance benchmark (optional, can be skipped for regular testing)
class TestPerformanceBenchmarks:
    """Optional performance benchmarks."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped OpenAI client."""
        return toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

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
    print("Run tests with: python -m pytest tests/test_integration_live.py -v -s")
    print("Note: These tests will consume OpenAI API credits")
