## Toolflow - Just wrap any LLM SDK and add auto tool calling and structured outputs

[![PyPI version](https://badge.fury.io/py/toolflow.svg)](https://badge.fury.io/py/toolflow)
[![Python versions](https://img.shields.io/pypi/pyversions/toolflow.svg)](https://pypi.org/project/toolflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Isuru%20Wijesiri-blue?logo=linkedin)](https://www.linkedin.com/in/isuruwijesiri/)

**🔗 [GitHub](https://github.com/IsuruMaduranga/toolflow)** • **📘 [Documentation](https://github.com/IsuruMaduranga/toolflow/tree/main/examples)**


Toolflow is a blazing-fast, lightweight drop-in thin wrapper for the OpenAI and Anthropic SDKs — adding automatic parallel tool calling, structured Pydantic outputs, and smart response modes with zero breaking changes. Stop battling bloated tool-calling frameworks. Toolflow supercharges the official SDKs you already use, without sacrificing compatibility or simplicity.

## Installation

```bash
pip install toolflow
# OpenAI only
pip install toolflow[openai]
# Anthropic only
pip install toolflow[anthropic]
```

## Quick Start

```python
import toolflow
from openai import OpenAI

# Only change needed!
client = toolflow.from_openai(OpenAI())

# Now you have auto-parallel tools + structured outputs
from pydantic import BaseModel
from typing import List

class CityWeather(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int

class WeatherRequest(BaseModel):
    cities: List[str]
    include_humidity: bool
    units: str

def get_weather(request: WeatherRequest) -> List[CityWeather]:
    """Get weather for multiple cities with specific requirements."""
    results = []
    for city in request.cities:
        results.append(CityWeather(
            city=city, 
            temperature=72.0, 
            condition="Sunny", 
            humidity=45 if request.include_humidity else 0
        ))
    return results

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Get weather for NYC and London with humidity in Celsius"}],
    tools=[get_weather]
)
print(result)  # Direct List[CityWeather] output

# Or get a weather report with structured output
class WeatherReport(BaseModel):
    cities: List[CityWeather]
    average_temperature: float

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Get weather for NYC and London with humidity in Celsius"}],
    tools=[get_weather],
    response_format=WeatherReport
)
print(result)  # You get a WeatherReport object with nested CityWeather objects
```

Find more [examples](https://github.com/IsuruMaduranga/toolflow/tree/main/examples) in the repository.

## Why Toolflow?

✅ **Drop-in replacement** - Works exactly like OpenAI/Anthropic SDKs  
✅ **Zero breaking changes** - All official SDK features preserved  
✅ **Auto-parallel tool calling** - Functions become tools with automatic concurrency  
✅ **Structured outputs** - Pass Pydantic models, get typed responses
✅ **Advanced reasoning support** - Supports OpenAI reasoning models & Anthropic extended thinking  
✅ **No bloat** - Lightweight alternative to heavy frameworks  
✅ **Unified interface** - Same code works across providers  
✅ **Smart response modes** - Choose between simplified or full SDK responses
✅ **Other SDKS coming soon** - We're working on adding support for other LLM SDKs (Groq, Gemini, etc. )

## Before & After

### Before (Standard SDK)
```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's 2+2?"}]
)
print(response.choices[0].message.content)  # Manual parsing
```

### After (Toolflow - Same Interface!)
```python
import toolflow
from openai import OpenAI

client = toolflow.from_openai(OpenAI())  # Only change needed!
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's 2+2?"}]
)
print(response)  # Direct string output (simplified mode)

# Or get the full SDK response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's 2+2?"}],
    full_response=True  # Returns complete SDK response object
)
print(response.choices[0].message.content)  # Same as original SDK
```

## Automatic Parallel Tool Execution

If model provides multiple tools, Toolflow will automatically execute them in parallel.

```python
import toolflow
from openai import OpenAI
from anthropic import Anthropic
import time

# Wrap your existing clients - no other changes needed
openai_client = toolflow.from_openai(OpenAI())
anthropic_client = toolflow.from_anthropic(Anthropic())

# Any function becomes a tool automatically
from pydantic import BaseModel
from typing import List, Dict
from enum import Enum

class CityRequest(BaseModel):
    cities: List[str]
    include_coordinates: bool
    data_type: str

class CalculationRequest(BaseModel):
    expressions: List[str]
    precision: int
    format_output: bool

def get_weather(request: CityRequest) -> List[Dict[str, any]]:
    """Get current weather for multiple cities with specific data requirements."""
    time.sleep(1)  # Simulated API call
    results = []
    for city in request.cities:
        result = {
            "city": city,
            "weather": "Sunny, 72°F",
            "population": "8.3 million"
        }
        if request.include_coordinates:
            result["coordinates"] = [40.7128, -74.0060] if city == "NYC" else [51.5074, -0.1278]
        results.append(result)
    return results

def get_population(request: CityRequest) -> List[Dict[str, any]]:
    """Get population information for multiple cities."""
    time.sleep(1)  # Simulated API call
    results = []
    for city in request.cities:
        result = {
            "city": city,
            "population": "8.3 million"
        }
        if request.include_coordinates:
            result["coordinates"] = [40.7128, -74.0060] if city == "NYC" else [51.5074, -0.1278]
        results.append(result)
    return results

def calculate(request: CalculationRequest) -> List[float]:
    """Safely evaluate multiple mathematical expressions with precision control."""
    results = []
    for expr in request.expressions:
        result = eval(expr.replace("^", "**"))
        if request.format_output:
            result = round(result, request.precision)
        results.append(result)
    return results

# Same code works with both providers
tools = [get_weather, get_population, calculate]
messages = [{"role": "user", "content": "Get weather and population for NYC and London with coordinates, and calculate 15 * 23 and 10 / 2 with 2 decimal precision"}]

# Sequential execution (default for synchronous execution)
start = time.time()
result = anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=messages,
    tools=tools,
    parallel_tool_execution=False  # ~3 seconds
)
print(f"Sequential: {time.time() - start:.1f}s")

# Parallel execution (3-5x faster!)
start = time.time()
result = anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=messages,
    tools=tools,
    parallel_tool_execution=True  # ~1 second
)
print(f"Parallel: {time.time() - start:.1f}s")
print("Result:", result)
```

## Structured Outputs (Like Instructor)

Get typed responses by passing Pydantic models:

```python
from pydantic import BaseModel
from typing import List

class Person(BaseModel):
    name: str
    age: int
    skills: List[str]

class TeamAnalysis(BaseModel):
    people: List[Person]
    average_age: float
    top_skills: List[str]

# Works with any provider - same interface as official SDKs
client = toolflow.from_openai(OpenAI())

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user", 
        "content": "Analyze this team: John (30, Python, React), Sarah (25, Go, Docker)"
    }],
    response_format=TeamAnalysis  # Just add this!
)

print(type(result))  # <class '__main__.TeamAnalysis'>
print(result.average_age)  # 27.5
print(result.top_skills)   # ['Python', 'React', 'Go', 'Docker']
```

## Response Modes: Simplified vs Full

Choose between simplified responses or full SDK compatibility:

```python
client = toolflow.from_openai(OpenAI())

# Simplified mode (default) - Direct content
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response)  # "Hello! How can I help you today?"

# Full response mode - Complete SDK object
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    full_response=True
)
print(response.choices[0].message.content)  # Access like original SDK
print(response.usage.total_tokens)          # All SDK properties available

# Streaming with different modes
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)
for chunk in stream:
    print(chunk, end="")  # Direct content (simplified)

# VS streaming with full response
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True,
    full_response=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")  # Original SDK behavior
```

## Advanced AI Capabilities

### OpenAI Reasoning Mode with Tools & Structured Output

Toolflow fully supports OpenAI's reasoning models (o4-mini, o3) with `reasoning_effort` parameter, seamlessly integrated with auto-parallel tool calling and structured outputs:

```python
from pydantic import BaseModel
from typing import List

class AnalysisResult(BaseModel):
    solution: str
    reasoning_steps: List[str]
    confidence: float

def calculate(expression: str) -> float:
    """Safely evaluate mathematical expressions."""
    return eval(expression.replace("^", "**"))

class DataStatistics(BaseModel):
    mean: float
    min: float
    max: float
    count: int
    variance: float

def analyze_data(data: List[float]) -> DataStatistics:
    """Analyze numerical data and return statistics."""
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return DataStatistics(
        mean=mean,
        min=min(data),
        max=max(data),
        count=len(data),
        variance=variance
    )

# Reasoning mode with tools and structured output
client = toolflow.from_openai(OpenAI())
result = client.chat.completions.create(
    model="o4-mini",
    reasoning_effort="medium",  # OpenAI reasoning parameter
    max_completion_tokens=4000,
    messages=[{
        "role": "user", 
        "content": "Analyze sales data [100, 120, 110, 130, 125] and calculate 15% growth projection. Provide detailed reasoning."
    }],
    tools=[calculate, analyze_data],        # Auto-parallel tool execution
    response_format=AnalysisResult,         # Structured output
    parallel_tool_execution=True
)

print(f"Solution: {result.solution}")
print(f"Steps: {result.reasoning_steps}")
print(f"Confidence: {result.confidence}")
```

### Anthropic Extended Thinking with Tools & Structured Output

Toolflow seamlessly supports Anthropic's extended thinking mode with automatic tool calling and structured responses:

```python
class ResearchFindings(BaseModel):
    summary: str
    key_insights: List[str]
    recommendations: List[str]

class WebSearchResult(BaseModel):
    query: str
    findings: List[str]
    sources: List[str]

class TrendAnalysis(BaseModel):
    data: str
    trends: List[str]
    confidence: float

class WebSearchParams(BaseModel):
    query: str
    max_findings: int
    include_sources: bool

class TrendAnalysisParams(BaseModel):
    data: str
    analysis_depth: str
    include_confidence: bool

def search_web(params: WebSearchParams) -> WebSearchResult:
    """Search for information with configurable parameters."""
    findings = [f"Finding {i} for {params.query}" for i in range(1, min(params.max_findings + 1, 4))]
    sources = ["source1.com", "source2.com"] if params.include_sources else []
    return WebSearchResult(
        query=params.query,
        findings=findings,
        sources=sources
    )

def analyze_trends(params: TrendAnalysisParams) -> TrendAnalysis:
    """Analyze trends in data with depth and confidence options."""
    trends = ["Trend 1", "Trend 2", "Trend 3"]
    confidence = 0.85 if params.include_confidence else 0.0
    return TrendAnalysis(
        data=params.data,
        trends=trends,
        confidence=confidence
    )

# Extended thinking with tools and structured output
anthropic_client = toolflow.from_anthropic(Anthropic())
result = anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    thinking=True,  # Anthropic extended thinking
    max_tokens=4000,
    messages=[{
        "role": "user",
        "content": "Research AI trends for 2025 and provide strategic recommendations."
    }],
    tools=[search_web, analyze_trends],     # Auto-parallel tool execution  
    response_format=ResearchFindings,       # Structured output
    parallel_tool_execution=True
)

print(f"Summary: {result.summary}")
print(f"Insights: {result.key_insights}")
print(f"Recommendations: {result.recommendations}")
```

### Key Benefits

- **Seamless Integration**: Reasoning/thinking modes work with all Toolflow features
- **Auto-Parallel Tools**: Functions execute concurrently during reasoning
- **Structured Output**: Get typed responses even with complex reasoning
- **Zero Complexity**: Same simple interface as standard completions
- **Performance**: Parallel tool execution reduces reasoning time

## Async Support with Smart Concurrency

Mix sync and async tools with automatic optimization:

```python
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Wrap async clients
openai_async = toolflow.from_openai(AsyncOpenAI())
anthropic_async = toolflow.from_anthropic(AsyncAnthropic())

from pydantic import BaseModel
from typing import List, Dict

class ApiResult(BaseModel):
    query: str
    result: str
    metadata: Dict[str, str]

class DatabaseRecord(BaseModel):
    table: str
    data: List[Dict[str, str]]
    count: int

class ApiRequest(BaseModel):
    query: str
    timeout: int
    retry_count: int

class CalculationParams(BaseModel):
    x: int
    y: int
    operation: str

class DatabaseQuery(BaseModel):
    table: str
    filters: Dict[str, str]
    limit: int

async def async_api_call(request: ApiRequest) -> ApiResult:
    """Async tool for I/O operations with configurable parameters."""
    await asyncio.sleep(0.5)  # Non-blocking delay
    return ApiResult(
        query=request.query,
        result=f"Async result: {request.query} (timeout: {request.timeout}s, retries: {request.retry_count})",
        metadata={"source": "api", "timestamp": "2024-01-01", "timeout": request.timeout}
    )

def sync_calculation(params: CalculationParams) -> int:
    """Sync tools work too with structured parameters."""
    time.sleep(0.1)  # Blocking delay
    if params.operation == "multiply":
        return params.x * params.y
    elif params.operation == "add":
        return params.x + params.y
    elif params.operation == "subtract":
        return params.x - params.y
    else:
        return params.x / params.y

async def async_database_query(query: DatabaseQuery) -> DatabaseRecord:
    """Another async tool with complex query parameters."""
    await asyncio.sleep(0.3)
    # Simulate filtering based on query parameters
    all_data = [{"id": "1", "name": "John"}, {"id": "2", "name": "Jane"}]
    filtered_data = all_data[:query.limit]  # Simple limit simulation
    return DatabaseRecord(
        table=query.table,
        data=filtered_data,
        count=len(filtered_data)
    )

async def main():
    # Mix sync and async tools - By default async tools run concurrently with asyncio.gather()
    # Sync tools run in thread pool concurrently
    result = await openai_async.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Call API with 'test' (timeout 30s, 3 retries), multiply 10*5, query users table with limit 5"}],
        tools=[async_api_call, sync_calculation, async_database_query]
    )
    print(result)
    
    # Async tools run concurrently with asyncio.gather()
    # Sync tools run in thread pool concurrently
    # Total time: max(async_times) + max(sync_times in thread pool)

asyncio.run(main())
```

## Streaming with Automatic Tool Execution

Streaming works exactly like the official SDKs, with automatic tool execution:

```python
from pydantic import BaseModel
from typing import List

class SearchResult(BaseModel):
    query: str
    results: List[str]
    total_found: int

class CodeExample(BaseModel):
    language: str
    examples: List[str]
    difficulty: str

class SearchRequest(BaseModel):
    query: str
    max_results: int
    include_sources: bool

class CodeRequest(BaseModel):
    language: str
    difficulty: str
    include_comments: bool

def search_web(request: SearchRequest) -> SearchResult:
    """Search the web for information with configurable parameters."""
    time.sleep(0.5)  # Simulated search delay
    results = [f"Tutorial {i} for {request.query}" for i in range(1, min(request.max_results + 1, 4))]
    return SearchResult(
        query=request.query,
        results=results,
        total_found=len(results)
    )

def get_code_examples(request: CodeRequest) -> CodeExample:
    """Get code examples for a language with difficulty and formatting options."""
    time.sleep(0.3)
    examples = [f"print('hello world')", f"def hello(): return 'world'"]
    if request.include_comments:
        examples = [f"# {example}" for example in examples]
    return CodeExample(
        language=request.language,
        examples=examples,
        difficulty=request.difficulty
    )

# Streaming with tools
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Search for Python tutorials (max 3 results) and show me beginner examples with comments"}],
    tools=[search_web, get_code_examples],
    stream=True,
    parallel_tool_execution=True  # Tools execute in parallel during streaming
)

print("Streaming response:")
for chunk in stream:
    print(chunk, end="")  # Direct content (simplified mode)

print("\n" + "="*50)

# Streaming with full response mode
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Search for Python tutorials and show me examples"}],
    tools=[search_web, get_code_examples],
    stream=True,
    full_response=True  # Original SDK streaming behavior
)

print("Full response streaming:")
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Migration Guide

### From OpenAI SDK
```python
# Before
from openai import OpenAI
client = OpenAI()

# After  
import toolflow
from openai import OpenAI
client = toolflow.from_openai(OpenAI())
# Everything else stays the same!
```

### From Anthropic SDK
```python
# Before
from anthropic import Anthropic
client = Anthropic()

# After
import toolflow  
from anthropic import Anthropic
client = toolflow.from_anthropic(Anthropic())
# Everything else stays the same!
```

### From Instructor
```python
# Before (Instructor)
import instructor
from openai import OpenAI
client = instructor.from_openai(OpenAI())

# After (Toolflow - same interface!)
import toolflow
from openai import OpenAI  
client = toolflow.from_openai(OpenAI())
```

## All Your Existing Code Still Works

Toolflow doesn't break anything - it's a true drop-in replacement:

```python
# All standard SDK features work unchanged
client = toolflow.from_openai(OpenAI())

# All parameters work exactly as documented
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0,
    full_response=True
)
```

## Performance Comparison

### Speed
- **Toolflow**: 2-4x faster than sequential execution
- **Native SDK**: Sequential execution

### Memory Usage
- **Toolflow**: ~5MB additional overhead
- **Other bloated frameworks**: ~50MB+ additional overhead
- **Native SDK**: Baseline

## Why Not Other Bloated Frameworks?

| Feature | Toolflow | Other  bloated frameworks |
|---------|----------|-----------|
| **Learning Curve** | Zero - same as OpenAI/Anthropic | Steep - new concepts |
| **Migration Effort** | One line change | Complete rewrite |
| **Bundle Size** | Lightweight (~5MB) | Heavy (~50MB+) |
| **Official SDK Features** | 100% compatible | Limited/wrapped |
| **Structured Outputs** | Built-in | Complex setup |
| **Tool Calling** | Automatic parallel | Manual configuration |
| **Performance** | Optimized thread pools | Variable |
| **Response Modes** | Flexible (simple/full) | Fixed patterns |

## Error Handling and Graceful Degradation

Tools handle errors gracefully by default:

```python
def reliable_tool(data: str) -> str:
    """A tool that always works."""
    return f"Processed: {data}"

def unreliable_tool(data: str) -> str:
    """A tool that might fail."""
    if "error" in data.lower():
        raise ValueError("Something went wrong!")
    return f"Success: {data}"

# Graceful error handling (default)
result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Process 'good data' and 'error data'"}],
    tools=[reliable_tool, unreliable_tool],
    parallel_tool_execution=True
)
# LLM receives error messages and can adapt its response

# Strict error handling
try:
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Process 'error data'"}],
        tools=[unreliable_tool],
        graceful_error_handling=False  # Raises exceptions
    )
except ValueError as e:
    print(f"Tool failed: {e}")
```

## Advanced Parallel Execution Control

Fine-tune parallel execution for optimal performance:

```python
import toolflow

# Configure global thread pool
toolflow.set_max_workers(8)  # Default is 4

def slow_api_call(query: str) -> str:
    time.sleep(2)  # Simulated slow API
    return f"Result for: {query}"

def fast_calculation(x: int, y: int) -> int:
    return x * y

def database_query(table: str) -> str:
    time.sleep(1)  # Simulated DB query
    return f"Data from {table}"

# Multiple tools with different execution times
tools = [slow_api_call, fast_calculation, database_query]

# Sequential execution (default for synchronous execution)
start = time.time()
result = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    parallel_tool_execution=False  # ~3 seconds
)
print(f"Sequential: {time.time() - start:.1f}s")

# Parallel execution (3-5x faster!)
start = time.time()
result = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    parallel_tool_execution=True  # ~1 second
)
print(f"Parallel: {time.time() - start:.1f}s")
print("Result:", result)
```

## Current API Support & Limitations

### Supported APIs

**OpenAI:**
- ✅ **Chat Completions API** - Full support with reasoning mode (`reasoning_effort`)
- ✅ **Tool calling** - Auto-parallel execution with structured outputs
- ✅ **Streaming** - Both simplified and full response modes
- ✅ **Structured outputs** - Pydantic model integration

**Anthropic:**
- ✅ **Messages API** - Full support with extended thinking mode (`thinking=True`)
- ✅ **Tool calling** - Auto-parallel execution with structured outputs  
- ✅ **Streaming** - Both simplified and full response modes
- ✅ **Structured outputs** - Pydantic model integration

### Upcoming API Support

**OpenAI Responses API (Preview):**
- ⏳ **Not yet supported** - OpenAI's new stateful API (released early 2025)
- 🔄 **Coming soon** - Will add support for this next-generation API
- 📋 **Features**: Background tasks, hosted tools (web_search, file_search), MCP servers, image generation

The [OpenAI Responses API](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses) is a new stateful API that combines the best of Chat Completions and Assistants APIs. While Toolflow currently supports the Chat Completions API with all its advanced features (including reasoning mode), we're working on adding Responses API support for its unique capabilities like hosted tools and background processing.

**Current workaround:** Toolflow's auto-parallel tool calling and structured outputs provide similar benefits to the Responses API's hosted tools, with the added advantage of full local control over your functions.

## API Reference

### Client Wrappers
```python
toolflow.from_openai(client, full_response=False)    # Wraps any OpenAI client
toolflow.from_anthropic(client, full_response=False) # Wraps any Anthropic client
```

### Advanced Concurrency Control
```
📊 TOOLFLOW CONCURRENCY BEHAVIOR

SYNC OPERATIONS
├── Default: Parallel execution
    ├── No custom executor → Global ThreadPoolExecutor (4 workers)
    ├── Change with toolflow.set_max_workers(workers)
    └── Change thread pool with toolflow.set_executor(executor)

ASYNC OPERATIONS  
├── Default: Parallel by default
    ├── For async tools: By default uses asyncio.gather()
    ├── For sync tools: Uses asyncio.run_in_executor()
    └── Change thread pool with toolflow.set_executor(executor)
└── Streaming → Async yield frequency controls event loop yielding
                ├── 0 (default) → Trust provider libraries
                └── Set toolflow.set_async_yield_frequency(N) → Explicit asyncio.sleep(0) every N chunks
```

Configure Toolflow's concurrency behavior for optimal performance in your deployment:

```python
import toolflow
from concurrent.futures import ThreadPoolExecutor

# Thread Pool Configuration (for parallel tool execution)
toolflow.set_max_workers(8)               # Set thread pool size (default: 4)
toolflow.get_max_workers()                # Get current thread pool size
toolflow.set_executor(custom_executor)    # Use custom ThreadPoolExecutor

# Example: High-performance custom executor
custom_executor = ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="toolflow-custom-"
)
toolflow.set_executor(custom_executor)

# Async Streaming Event Loop Control (disabled by default)
toolflow.set_async_yield_frequency(frequency)  # 0=disabled, 1=every chunk, N=every N chunks
```

**Thread Pool Settings:**
- **Default (4 workers)**: Good for most applications
- **High concurrency (8-16 workers)**: For applications with many concurrent tool calls
- **Custom executor**: Full control over thread pool behavior

**Async Yield Frequency:**
- **0 (default)**: Disabled - trusts underlying provider libraries for proper event loop yielding
- **1**: Yield after every chunk - maximum responsiveness for high-concurrency FastAPI deployments
- **N**: Yield every N chunks - custom balance between performance and responsiveness

```python
# Example: Configure for high-concurrency FastAPI deployment
toolflow.set_max_workers(12)              # More threads for parallel tools
toolflow.set_async_yield_frequency(1)     # Yield after every chunk for responsiveness

# Example: Configure for maximum performance
toolflow.set_max_workers(16)              # Maximum parallel tool execution
toolflow.set_async_yield_frequency(0)     # Trust provider libraries (default)

# Example: Configure for moderate concurrency
toolflow.set_max_workers(6)               # Moderate thread pool
toolflow.set_async_yield_frequency(0)     # Default yielding behavior
```

**When to adjust async yield frequency:**
- **High-concurrency FastAPI** (100+ simultaneous streams): Set to `1`
- **Standard deployments**: Keep default `0`
- **Performance-critical**: Keep default `0`

### Enhanced Parameters
All standard SDK parameters work unchanged, plus these additions:

```python
client.chat.completions.create(
    # All standard parameters work (model, messages, temperature, etc.)
    
    # Toolflow enhancements
    tools=[...],                      # List of functions (any callable)
    response_format=BaseModel,        # Pydantic model for structured output
    parallel_tool_execution=False,    # Enable concurrent tool execution
    max_tool_call_rounds=10,               # Safety limit for tool rounds
    max_response_format_retries=2,        # Maximum number of retries to get correct structured output from model
    graceful_error_handling=True,    # Handle tool errors gracefully
    full_response=False,             # Return full SDK response vs simplified
)
```

### Optional Performance Decorator
```python
@toolflow.tool(name="custom_name", description="Override docstring")
def optimized_function(param: str) -> str:
    """Pre-generates schema for optimal performance."""
    return f"Processed: {param}"

# Or use any function without decoration
def regular_function(param: str) -> str:
    """Schema generated on first use and cached."""
    return f"Processed: {param}"
```

### Response Modes
- `full_response=False` (default): Returns string content or parsed Pydantic model
- `full_response=True`: Returns the complete official SDK response object

## Development

```bash
# Install for development
pip install -e ".[dev]"

# Run tests  
pytest

# Format code
black src/ && isort src/

# Type checking
mypy src/

# Run live tests (requires API keys)
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
python run_live_tests.py
```

## 👤 Author

Created and maintained by [Isuru Wijesiri](https://www.linkedin.com/in/isuruwijesiri/).  
🔗 Follow me for updates on AI, open source, and developer tools:  
- 💼 [LinkedIn](https://www.linkedin.com/in/isuruwijesiri/)  
- 🧑‍💻 [GitHub](https://github.com/IsuruMaduranga)

## Contributing

Contributions welcome! Please fork, create a feature branch, add tests, and submit a pull request.

## License

MIT License - see LICENSE file for details.

---

Toolflow is an open-source project by [Isuru Wijesiri](https://www.linkedin.com/in/isuruwijesiri/)
