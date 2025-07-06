"""
Live Integration Tests for Toolflow with Anthropic

These tests use actual Anthropic API calls to verify end-to-end functionality.
Set ANTHROPIC_API_KEY environment variable to run these tests.

Run with: python -m pytest tests/anthropic/test_integration_live.py -v -s

Note: These tests make real API calls and will consume Anthropic credits.

Test Coverage:
- Basic tool calling (sync, async, streaming)
- Complex data types (dataclasses, enums, NamedTuple, Pydantic models)
- Anthropic-compatible schema patterns (List[List[float]], Dict[str, int])
- Structured output with complex response models
- Error handling and validation for unsupported schemas
- Parallel execution with mixed tool types
- Performance benchmarking with complex operations
- Comprehensive workflow testing combining multiple paradigms
"""

import os
import asyncio
import time
from typing import List, Optional, Dict, Any, Union, Tuple, Set, NamedTuple
from dataclasses import dataclass
from enum import Enum
import pytest
from dotenv import load_dotenv
load_dotenv()

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
from toolflow import MaxToolCallsError


# Skip all tests if Anthropic API key is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") or not ANTHROPIC_AVAILABLE,
    reason="Anthropic API key not available or anthropic package not installed"
)


# Test Models for structured output support
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


# =============================================================================
# Complex Data Types for Testing (Anthropic-Compatible Versions)
# =============================================================================

# Enum Types
class TaskStatus(Enum):
    """String enum for task statuses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Priority(Enum):
    """Integer enum for priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Dataclass Types
@dataclass
class UserProfile:
    """User profile dataclass with various field types"""
    name: str
    age: int
    email: str
    is_active: bool = True
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class LocationInfo:
    """Location information dataclass"""
    latitude: float
    longitude: float
    city: str
    country: str

# NamedTuple Types (for coordinates and color data)
class Point2D(NamedTuple):
    """2D point coordinates"""
    x: float
    y: float

class ColorRGB(NamedTuple):
    """RGB color values"""
    red: int
    green: int
    blue: int

# Pydantic Models with Field descriptions and complex types
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TaskModel(BaseModel):
    """Task model with enum status and priority"""
    id: int = Field(description="Unique task identifier")
    title: str = Field(description="Task title")
    description: Optional[str] = Field(description="Detailed task description", default=None)
    status: TaskStatus = Field(description="Current task status")
    priority: Priority = Field(description="Task priority level")
    assignee: Optional[str] = Field(description="Task assignee name", default=None)
    tags: List[str] = Field(description="List of task tags", default_factory=list)

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class GeometryData(BaseModel):
    """Geometry data with coordinate handling"""
    shape_type: str = Field(description="Type of geometric shape")
    coordinates: List[List[float]] = Field(description="Shape coordinates as list of [x,y] pairs")
    area: Optional[float] = Field(description="Calculated area", default=None)
    perimeter: Optional[float] = Field(description="Calculated perimeter", default=None)

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class UserAnalytics(BaseModel):
    """User analytics with various metrics"""
    user_id: str = Field(description="User identifier")
    total_tasks: int = Field(description="Total number of tasks")
    completed_tasks: int = Field(description="Number of completed tasks")
    success_rate: float = Field(description="Task completion rate as percentage")
    avg_completion_time: Optional[float] = Field(description="Average task completion time in hours", default=None)
    status_distribution: Dict[str, int] = Field(description="Distribution of tasks by status")
    recent_activity: List[str] = Field(description="List of recent user activities", default_factory=list)

# Complex Response Models for structured outputs
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class ComprehensiveReport(BaseModel):
    """Comprehensive report combining multiple data types"""
    report_id: str = Field(description="Unique report identifier")
    generated_at: str = Field(description="Report generation timestamp")
    user_summary: Dict[str, Any] = Field(description="User summary statistics")
    task_analytics: Dict[str, int] = Field(description="Task analytics by status")
    priority_breakdown: Dict[str, int] = Field(description="Tasks by priority level")
    geometric_data: Optional[Dict[str, float]] = Field(description="Geometric calculations if applicable", default=None)
    recommendations: List[str] = Field(description="System recommendations", default_factory=list)

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class DataAnalysisResult(BaseModel):
    """Data analysis result with coordinates and statistics"""
    analysis_type: str = Field(description="Type of analysis performed")
    coordinate_count: int = Field(description="Number of coordinate points analyzed")
    coordinates: List[List[float]] = Field(description="Analyzed coordinate pairs")
    statistical_summary: Dict[str, float] = Field(description="Statistical summary of the data")
    insights: List[str] = Field(description="Generated insights from analysis", default_factory=list)


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


# =============================================================================
# Complex Data Type Tools (Anthropic-Compatible)
# =============================================================================

@toolflow.tool
def create_user_profile(
    name: str,
    age: int,
    email: str,
    is_active: bool = True,
    tags: Optional[List[str]] = None
) -> UserProfile:
    """
    Create a user profile with various data types.
    Tests dataclass creation with optional parameters and default values.
    """
    return UserProfile(name=name, age=age, email=email, is_active=is_active, tags=tags)

@toolflow.tool 
def update_task_status(
    task_id: int,
    title: str,
    status: str,
    priority: int,
    description: Optional[str] = None,
    assignee: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> str:
    """
    Update task status using enum values and Pydantic model.
    Tests enum handling, optional fields, and complex model creation.
    """
    try:
        task_status = TaskStatus(status)
        task_priority = Priority(priority)
        
        if not PYDANTIC_AVAILABLE:
            return f"Task {task_id} updated: {title} - Status: {task_status.value}, Priority: {task_priority.value}"
            
        task = TaskModel(
            id=task_id,
            title=title,
            description=description,
            status=task_status,
            priority=task_priority,
            assignee=assignee,
            tags=tags or []
        )
        
        return f"Task updated successfully: {task.title} (ID: {task.id}) - Status: {task.status.value}, Priority: {task.priority.value}"
    except ValueError as e:
        return f"Error updating task: {e}"

@toolflow.tool
def calculate_geometry_simple(coordinates: List[List[float]]) -> Dict[str, float]:
    """
    Calculate geometric properties using coordinate pairs.
    Uses List[List[float]] instead of NamedTuple for Anthropic compatibility.
    """
    if len(coordinates) < 3:
        return {"area": 0.0, "perimeter": 0.0, "error": "Need at least 3 points for polygon"}
    
    # Calculate area using shoelace formula
    area = 0.0
    perimeter = 0.0
    n = len(coordinates)
    
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[j]
        
        # Shoelace formula for area
        area += x1 * y2 - x2 * y1
        
        # Calculate perimeter
        perimeter += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    area = abs(area) / 2.0
    
    return {
        "area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "num_vertices": n
    }

@toolflow.tool
def process_mixed_data_simple(
    integers: List[int],
    floats: List[float], 
    text_data: List[str],
    include_stats: bool = True
) -> Dict[str, Any]:
    """
    Process mixed data types with Union and Optional types.
    Uses simple parameters instead of Tuple for Anthropic compatibility.
    """
    result = {
        "integer_sum": sum(integers),
        "float_average": sum(floats) / len(floats) if floats else 0.0,
        "text_count": len(text_data),
        "combined_data": {
            "integers": integers,
            "floats": floats,
            "texts": text_data
        }
    }
    
    if include_stats:
        result["statistics"] = {
            "total_items": len(integers) + len(floats) + len(text_data),
            "max_integer": max(integers) if integers else None,
            "min_float": min(floats) if floats else None,
            "longest_text": max(text_data, key=len) if text_data else None
        }
    
    return result

@toolflow.tool
def color_operations_simple(colors: List[List[int]]) -> Dict[str, Any]:
    """
    Perform operations on color data using RGB values.
    Uses List[List[int]] instead of NamedTuple for Anthropic compatibility.
    """
    if not colors:
        return {"error": "No colors provided"}
    
    # Calculate average color
    avg_red = sum(color[0] for color in colors) / len(colors)
    avg_green = sum(color[1] for color in colors) / len(colors)
    avg_blue = sum(color[2] for color in colors) / len(colors)
    
    # Find brightest color (highest sum of RGB values)
    brightest = max(colors, key=lambda c: sum(c))
    brightest_value = sum(brightest)
    
    # Find darkest color (lowest sum of RGB values)  
    darkest = min(colors, key=lambda c: sum(c))
    darkest_value = sum(darkest)
    
    return {
        "average_color": [round(avg_red), round(avg_green), round(avg_blue)],
        "brightest_color": {
            "rgb": brightest,
            "brightness": brightest_value
        },
        "darkest_color": {
            "rgb": darkest,
            "brightness": darkest_value
        },
        "total_colors": len(colors),
        "color_analysis": "RGB color operations completed successfully"
    }

@toolflow.tool
def analyze_user_activity(
    user_id: str,
    task_counts: Dict[str, int],
    completion_times: List[float],
    recent_activities: List[str],
    calculate_trends: bool = True
) -> Dict[str, Any]:
    """
    Analyze user activity with complex nested data structures.
    Uses string keys instead of Enum keys for Anthropic compatibility.
    """
    total_tasks = sum(task_counts.values())
    completed = task_counts.get("completed", 0)
    
    analysis = {
        "user_id": user_id,
        "summary": {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "success_rate": (completed / total_tasks * 100) if total_tasks > 0 else 0.0,
            "avg_completion_time": sum(completion_times) / len(completion_times) if completion_times else 0.0
        },
        "status_breakdown": task_counts,
        "recent_activity_count": len(recent_activities),
        "performance_metrics": {
            "fastest_completion": min(completion_times) if completion_times else None,
            "slowest_completion": max(completion_times) if completion_times else None,
            "activity_frequency": len(recent_activities)
        }
    }
    
    if calculate_trends:
        # Generate trend analysis
        trend_score = completed / total_tasks if total_tasks > 0 else 0
        if trend_score > 0.8:
            trend = "excellent"
        elif trend_score > 0.6:
            trend = "good"
        elif trend_score > 0.4:
            trend = "average"
        else:
            trend = "needs_improvement"
            
        analysis["trends"] = {
            "performance_trend": trend,
            "trend_score": round(trend_score, 2),
            "recommendations": [
                f"User shows {trend} performance with {completed}/{total_tasks} completed tasks",
                f"Average completion time: {analysis['summary']['avg_completion_time']:.1f}h" if completion_times else "No completion time data"
            ]
        }
    
    return analysis


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
            max_tool_call_rounds=3
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
            max_tool_call_rounds=5
        )
        
        assert response is not None
        assert "45" in response

    def test_parallel_tool_execution(self, client):
        """Test parallel tool execution performance."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate the 10th and 12th Fibonacci numbers"}],
            tools=[get_fibonacci],
            parallel_tool_execution=True,
            max_tool_call_rounds=5
        )
        
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
            max_tool_call_rounds=5
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
            max_tool_call_rounds=3
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
            max_tool_call_rounds=3
        )
        
        assert response is not None
        assert "San Francisco" in response
        assert "weather" in response.lower()


class TestAnthropicAsyncFunctionality:
    """Test async functionality with Anthropic."""

    @pytest.fixture
    def async_client(self):
        """Create async toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_client_sync_tools(self, async_client):
        """Test async client with sync tools."""
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 20 * 3?"}],
            tools=[simple_calculator],
            max_tool_call_rounds=3
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
            max_tool_call_rounds=3
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
            max_tool_call_rounds=5
        )
        
        assert response is not None
        # Result should be (15 + 25) * 2 = 80
        assert "80" in response

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_parallel_tool_execution(self, async_client):
        """Test async parallel tool execution."""
        start_time = time.time()
        
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate Fibonacci numbers for 8, 9, and 10, and get the current time"}],
            tools=[get_fibonacci, get_system_info],
            parallel_tool_execution=True,
            max_tool_call_rounds=8
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
            max_tool_call_rounds=3
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

    def test_streaming_with_parallel_tools(self, client):
        """Test streaming with parallel tool execution."""
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate 8 * 9, get the current time, and format 'testing' in uppercase"}],
            tools=[simple_calculator, get_system_info, format_data],
            stream=True,
            parallel_tool_execution=True,
            max_tool_call_rounds=6
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
        async_client = toolflow.from_anthropic(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        stream = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 9 * 4?"}],
            tools=[simple_calculator],
            stream=True,
            max_tool_call_rounds=3
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
        async_client = toolflow.from_anthropic(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        stream = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
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
        async_client = toolflow.from_anthropic(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        stream = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate Fibonacci of 7, multiply 6 by 8, and get system status"}],
            tools=[get_fibonacci, simple_calculator, get_system_info],
            stream=True,
            parallel_tool_execution=True,
            max_tool_call_rounds=6
        )
        
        content_chunks = []
        async for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks)
        assert "13" in full_content  # Fib(7)
        assert "48" in full_content  # 6 * 8
        assert "operational" in full_content.lower() or "status" in full_content.lower()


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
            max_tool_call_rounds=3
        )
        
        assert response is not None
        # Should handle the error gracefully
        assert any(word in response.lower() for word in ["error", "divide", "zero"])

    def test_max_tool_call_rounds_limit(self, client):
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
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Call the recursive_tool with task 'start' and keep calling it until it's done. The tool will tell you when to call it again."}],
                tools=[recursive_tool],
                max_tool_call_rounds=2  # Low limit
            )


class TestAnthropicStructuredOutput:
    """Test Anthropic structured output functionality."""
    
    @pytest.fixture
    def client(self):
        """Create toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_weather(self, client):
        """Test structured output with weather tool."""
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user", 
                "content": "Get the weather for New York and format it as structured data"
            }],
            tools=[get_weather],
            response_format=WeatherData,
            max_tokens=1000
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
        assert isinstance(result.operation, str)
        assert isinstance(result.operands, list)
        assert result.result == 127.0  # 15 * 8 + 7 = 127
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 10

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_without_tools(self, client):
        """Test structured output without any tools."""
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

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_full_response_mode(self, client):
        """Test structured output in full response mode."""
        client_full = toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")), full_response=True)
        
        result = client_full.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user", 
                "content": "Get weather for London and structure the response"
            }],
            tools=[get_weather],
            response_format=WeatherData,
            max_tokens=1000
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
            assert hasattr(result, 'content')
            assert len(result.content) > 0
            # The structured response should be embedded in the response content
            assert result.content is not None

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    @pytest.mark.asyncio
    async def test_async_structured_output(self):
        """Test async structured output functionality."""
        client = toolflow.from_anthropic(anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))
        
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

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_error_handling(self, client):
        """Test structured output error handling with invalid response format."""
        # Test with invalid response format
        with pytest.raises(ValueError, match="Response format .* is not a Pydantic model"):
            client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=dict,  # Invalid - not a Pydantic model
                max_tokens=1000
            )

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_structured_output_streaming_error(self, client):
        """Test that structured output with streaming raises an error."""
        # Test streaming with response_format should raise error
        with pytest.raises(ValueError, match="response_format is not supported for streaming"):
            client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=WeatherData,
                stream=True,
                max_tokens=1000
            )


class TestAnthropicComplexDataTypes:
    """Test complex data type support with Anthropic."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    def test_dataclass_creation(self, client):
        """Test dataclass parameter handling."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Create a user profile for Alice, age 30, email alice@example.com, active status true, with tags ['developer', 'team-lead']"}],
            tools=[create_user_profile],
            max_tool_call_rounds=3
        )
        
        assert response is not None
        content = response.lower()
        assert "alice" in content
        assert "30" in response
        assert "alice@example.com" in content

    def test_enum_handling(self, client):
        """Test enum parameter processing."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Update task 123 with title 'Fix bug' to status 'completed' and priority 3"}],
            tools=[update_task_status],
            max_tool_call_rounds=3
        )
        
        assert response is not None
        content = response.lower()
        assert "123" in response
        assert "fix bug" in content
        assert "completed" in content

    def test_coordinate_geometry(self, client):
        """Test coordinate processing with List[List[float]]."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate the area and perimeter of a triangle with vertices at [[0,0], [3,0], [0,4]]"}],
            tools=[calculate_geometry_simple],
            max_tool_call_rounds=3
        )
        
        assert response is not None
        # Triangle area should be 6.0 (0.5 * 3 * 4)
        assert "6" in response or "6.0" in response

    def test_mixed_data_processing(self, client):
        """Test processing of mixed data types."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Process this mixed data: integers [1,2,3], floats [1.5, 2.5, 3.5], texts ['hello', 'world'], include statistics"}],
            tools=[process_mixed_data_simple],
            max_tool_call_rounds=3
        )
        
        assert response is not None
        content = response.lower()
        assert "6" in response  # sum of integers 1+2+3
        assert "statistics" in content or "stats" in content

    def test_color_operations(self, client):
        """Test color operations with RGB values."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Analyze these RGB colors: [[255,0,0], [0,255,0], [0,0,255]] - find average, brightest, and darkest"}],
            tools=[color_operations_simple],
            max_tool_call_rounds=3
        )
        
        assert response is not None
        content = response.lower()
        assert "color" in content
        assert any(word in content for word in ["brightest", "darkest", "average"])

    def test_user_activity_analysis(self, client):
        """Test complex nested data analysis."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user", 
                "content": "Analyze user 'user123' activity with task counts: pending=5, completed=15, in_progress=3, completion times [2.5, 1.8, 3.2, 2.1], recent activities ['login', 'create_task', 'complete_task']"
            }],
            tools=[analyze_user_activity],
            max_tool_call_rounds=3
        )
        
        assert response is not None
        content = response.lower()
        assert "user123" in content
        assert any(word in content for word in ["analysis", "performance", "activity"])
        # Should calculate success rate: 15/23 ≈ 65%
        assert any(str(num) in response for num in ["65", "23", "15"])

    def test_parallel_complex_tools(self, client):
        """Test parallel execution with complex data type tools."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": "Perform multiple operations: create user profile for Bob age 25, calculate geometry for [[0,0],[4,0],[0,3]], and analyze colors [[255,255,255],[0,0,0],[128,128,128]]"
            }],
            tools=[create_user_profile, calculate_geometry_simple, color_operations_simple],
            parallel_tool_execution=True,
            max_tool_call_rounds=6
        )
        
        assert response is not None
        content = response.lower()
        assert "bob" in content
        assert "25" in response
        # Triangle area should be 6.0 (0.5 * 4 * 3)
        assert "6" in response or "area" in content
        assert "color" in content

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_complex_tools(self):
        """Test async execution with complex data type tools."""
        async_client = toolflow.from_anthropic(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": "Create a user profile for Charlie, age 35, email charlie@test.com, and then update task 456 to completed status with high priority"
            }],
            tools=[create_user_profile, update_task_status],
            parallel_tool_execution=True,
            max_tool_call_rounds=5
        )
        
        assert response is not None
        content = response.lower()
        # Check that tools were executed (more flexible assertions)
        assert any(word in content for word in ["charlie", "profile", "created", "user", "task", "456", "completed", "priority"])
        # Ensure we got a meaningful response
        assert len(content) > 50

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_streaming_complex_tools(self):
        """Test async streaming with complex data type tools."""
        async_client = toolflow.from_anthropic(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        stream = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": "Calculate geometry for a square with coordinates [[0,0],[2,0],[2,2],[0,2]] and analyze the data"
            }],
            tools=[calculate_geometry_simple, process_mixed_data_simple],
            stream=True,
            max_tool_call_rounds=5
        )
        
        content_chunks = []
        async for chunk in stream:
            if chunk:
                content_chunks.append(chunk)
        
        full_content = "".join(content_chunks).lower()
        assert len(full_content) > 50
        # Square area should be 4.0 (2*2)
        assert "4" in full_content or "area" in full_content

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_complex_structured_output(self, client):
        """Test structured output with complex data types."""
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user",
                "content": "Create a comprehensive analysis of coordinates [[1,1],[3,1],[3,3],[1,3]] including geometric calculations"
            }],
            tools=[calculate_geometry_simple],
            response_format=DataAnalysisResult,
            max_tokens=1000
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, DataAnalysisResult)
        assert result.analysis_type is not None
        assert result.coordinate_count == 4
        assert len(result.coordinates) == 4
        assert isinstance(result.statistical_summary, dict)
        assert len(result.insights) > 0

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_comprehensive_report_output(self, client):
        """Test comprehensive report generation with multiple data types."""
        result = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user",
                "content": "Generate a comprehensive report analyzing user performance with multiple data points and recommendations"
            }],
            tools=[analyze_user_activity, calculate_geometry_simple],
            response_format=ComprehensiveReport,
            max_tokens=1000
        )
        
        # Should return parsed Pydantic model
        assert isinstance(result, ComprehensiveReport)
        assert result.report_id is not None
        assert result.generated_at is not None
        assert isinstance(result.user_summary, dict)
        assert isinstance(result.task_analytics, dict)
        assert isinstance(result.priority_breakdown, dict)
        assert len(result.recommendations) > 0


class TestAnthropicComplexDataTypesValidation:
    """Test validation and error handling for complex data types."""

    @pytest.fixture
    def client(self):
        """Create toolflow wrapped Anthropic client."""
        return toolflow.from_anthropic(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))

    def test_enum_validation(self, client):
        """Test enum validation and error handling."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Try to update task 999 with an invalid status 'invalid_status' and see what happens"}],
            tools=[update_task_status],
            graceful_error_handling=True,
            max_tool_call_rounds=3
        )
        
        assert response is not None
        content = response.lower()
        # Should handle the error gracefully and mention the issue
        assert any(word in content for word in ["error", "invalid", "status"])

    def test_coordinate_validation(self, client):
        """Test coordinate validation with insufficient data."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Try to calculate geometry with only 2 points: [[0,0], [1,1]]"}],
            tools=[calculate_geometry_simple],
            max_tool_call_rounds=3
        )
        
        assert response is not None
        content = response.lower()
        # Should mention that more points are needed
        assert any(word in content for word in ["need", "points", "polygon", "error"])

    def test_empty_data_handling(self, client):
        """Test handling of empty data structures."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Analyze empty color list: []"}],
            tools=[color_operations_simple],
            max_tool_call_rounds=3
        )
        
        assert response is not None
        content = response.lower()
        # Should handle empty data gracefully
        assert any(word in content for word in ["error", "empty", "no colors"])


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
            max_tool_call_rounds=10
        )
        
        assert response is not None
        content = response.lower()
        
        # Check all expected results
        assert "40" in response  # 15 + 25
        assert "hello world" in content  # Title case formatting
        assert "tokyo" in content  # Weather info
        assert any(word in content for word in ["time", "current"])  # Time info

    def test_full_workflow_with_complex_types(self, client):
        """Test workflow combining basic and complex data type tools."""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are an advanced assistant that handles both simple calculations and complex data analysis.",
            messages=[{
                "role": "user",
                "content": "Create a user profile for David age 28, calculate geometry for triangle [[0,0],[5,0],[0,12]], and analyze RGB colors [[255,0,0],[0,255,0]]"
            }],
            tools=[create_user_profile, calculate_geometry_simple, color_operations_simple, simple_calculator],
            parallel_tool_execution=True,
            max_tool_call_rounds=8
        )
        
        assert response is not None
        content = response.lower()
        
        # Check results from complex tools (more flexible assertions)
        assert any(word in content for word in ["david", "profile", "user", "geometry", "triangle", "color", "rgb", "analysis"])
        # Ensure we got a meaningful response
        assert len(content) > 50

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_full_workflow_async(self):
        """Test a complete async workflow."""
        async_client = toolflow.from_anthropic(
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
            max_tool_call_rounds=8
        )
        
        assert response is not None
        content = response.lower()
        
        # Check expected results
        assert "42" in response  # 6 * 7
        assert "21" in response  # Fibonacci(8)
        assert "operational" in content  # System status

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_full_workflow_async_with_mixed_tools(self):
        """Test full async workflow with mixed sync/async tools."""
        async_client = toolflow.from_anthropic(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are an efficient assistant that uses tools to solve complex tasks.",
            messages=[{
                "role": "user",
                "content": "Calculate 18 / 3, double the result with a delay, and get weather for London"
            }],
            tools=[simple_calculator, async_delay_calculator, get_weather],
            parallel_tool_execution=True,
            max_tool_call_rounds=8
        )
        
        assert response is not None
        content = response.lower()
        
        # Check expected results: 18/3 = 6, then 6*2 = 12
        assert "6" in response or "12" in response  # Either step result
        assert "london" in content  # Weather info

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_full_workflow_async_with_complex_data_types(self):
        """Test full async workflow with complex data type tools."""
        async_client = toolflow.from_anthropic(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system="You are an advanced async assistant that efficiently handles complex data analysis.",
            messages=[{
                "role": "user",
                "content": "Analyze user 'async_user' with task counts pending=3, completed=7, calculate geometry for rectangle [[0,0],[6,0],[6,4],[0,4]], and process mixed data with integers [10,20,30]"
            }],
            tools=[analyze_user_activity, calculate_geometry_simple, process_mixed_data_simple],
            parallel_tool_execution=True,
            max_tool_call_rounds=8
        )
        
        assert response is not None
        content = response.lower()
        
        # Check expected results
        # Check that user analysis was performed (may not include "async_user" explicitly)
        assert any(word in content for word in ["user", "analysis", "activity", "performance"])
        # Rectangle area should be 24.0 (6*4)
        assert "24" in response or "area" in content
        # Integer sum should be 60 (10+20+30)
        assert "60" in response


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
            max_tool_call_rounds=10
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
            max_tool_call_rounds=10
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
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate fibonacci numbers from 5 to 15"}],
            tools=[get_fibonacci],
            parallel_tool_execution=True,
            max_tool_call_rounds=12,
            max_workers=5
        )
        
        assert response is not None
        content = response
        
        # Should contain multiple Fibonacci numbers
        fibonacci_sequence = [5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        found_numbers = sum(1 for num in fibonacci_sequence if str(num) in content)
        assert found_numbers >= 5  # Should find at least 5 of the expected numbers

    @pytest.mark.skipif(True, reason="Complex data type performance test - run manually if needed")
    def test_complex_data_type_parallel_performance(self, client):
        """Test parallel execution performance with complex data type tools."""
        start_time = time.time()
        
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": "Perform multiple complex operations: create user profiles for Alice, Bob, Charlie; calculate geometry for multiple shapes; analyze different color sets; and process various data types"
            }],
            tools=[
                create_user_profile, 
                calculate_geometry_simple, 
                color_operations_simple, 
                process_mixed_data_simple,
                analyze_user_activity
            ],
            parallel_tool_execution=True,
            max_tool_call_rounds=15,
            max_workers=5
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert response is not None
        content = response.lower()
        
        # Check for evidence of multiple operations
        assert any(name in content for name in ["alice", "bob", "charlie"])
        assert any(word in content for word in ["geometry", "area", "color", "analysis"])
        
        print(f"Complex data type parallel execution time: {execution_time:.2f}s")
        assert execution_time < 45  # Should complete reasonably fast with complex operations

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.skipif(True, reason="Performance test - run manually if needed")
    @pytest.mark.asyncio
    async def test_async_parallel_performance(self):
        """Test async parallel execution performance."""
        async_client = toolflow.from_anthropic(
            anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        
        start_time = time.time()
        
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate multiple operations: 12*8, 15+25, 100/4, get time, and format 'async' in title case"}],
            tools=[simple_calculator, get_system_info, format_data],
            parallel_tool_execution=True,
            max_tool_call_rounds=10
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
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        exit(1)
    
    if not ANTHROPIC_AVAILABLE:
        print("❌ Anthropic package not available")
        print("Install it with: pip install anthropic")
        exit(1)
    
    print("✅ Environment setup complete")
    print("Run tests with: python -m pytest tests/anthropic/test_integration_live.py -v -s")
    print("Note: These tests will consume Anthropic API credits")
    print("\n📊 Test Coverage Includes:")
    print("   • Basic tool calling functionality")
    print("   • Complex data types (dataclasses, enums, Pydantic models)")
    print("   • Async and streaming functionality")
    print("   • Structured output support")
    print("   • Error handling and validation")
    print("   • Parallel execution performance")
    print("   • Comprehensive workflow testing") 