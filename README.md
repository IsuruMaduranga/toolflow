# Toolflow

A No bullshit minimalistic Python library that makes LLM tool calling as simple as decorating a function. Just wrap your AI clients and pass decorated functions directly to the `tools` parameter - no complex setup required.

```python
import toolflow
from openai import OpenAI

client = toolflow.from_openai(OpenAI())

@toolflow.tool  
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72°F"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=[get_weather],  # Just pass your function! We'll handle the rest.
    max_tool_calls=5, # Maximum number of tool calls to execute as a safety measure.
)

print(response.choices[0].message.content)
```

## Features

- 🎯 **Simple decorator-based tool registration** - Just use `@tool` to register functions
- 🔧 **Multiple LLM provider support**(Coming soon) - OpenAI, Anthropic (Claude), and extensible for others
- 📝 **Automatic schema generation** - Function signatures are automatically converted to JSON schemas
- ⚡ **Automatic execution** - Tools can be automatically executed when called by the LLM
- 🔄 **Asynchronous support** - Full async/await support for both sync and async tools
- ⚡ **Parallel execution** - Multiple tool calls execute in parallel for significant performance gains
- 🔄 **Streaming support**(Coming soon) - Tools can be streamed

## Installation

```bash
pip install toolflow
```

## Quick Start

```python
import os
import toolflow
from openai import OpenAI

# Create a toolflow wrapped client
client = toolflow.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Define tools using the @tool decorator
@toolflow.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@toolflow.tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

# Use the familiar OpenAI API with your functions as tools
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco? Also, what's 15 + 27?"}
    ],
    tools=[get_weather, add_numbers]
)

print("Response:", response.choices[0].message.content)
```

## Async Support

Toolflow fully supports OpenAI's `AsyncOpenAI` client. You can use both sync and async tools in the same call:

> NOTE: Avoid using blocking calls in sync tools when using async client as it will block the event loop.

```python
import asyncio
import os
import toolflow
from openai import AsyncOpenAI

# Create an async toolflow wrapped client
async_client = toolflow.from_openai_async(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Define both sync and async tools
@toolflow.tool
def sync_calculator(a: int, b: int) -> int:
    """Add two numbers (sync version)."""
    return a + b

@toolflow.tool
async def async_database_query(query: str) -> str:
    """Query database asynchronously."""
    # Simulate async database operation
    await asyncio.sleep(0.1)
    return f"Database result for: {query}"

async def main():
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Add 15 + 27 and query the users table"}
        ],
        tools=[sync_calculator, async_database_query]  # Mix sync and async tools!
    )
    
    print("Response:", response.choices[0].message.content)

# Run the async function
asyncio.run(main())
```

### Key Async Features

- **Mixed sync/async tools**: Use both synchronous and asynchronous tools in the same call
- **Automatic detection**: The library automatically detects and properly handles async functions
- **Non-blocking execution**: Async tools don't block the event loop
- **Parallel execution**: Multiple tools execute concurrently for improved performance
- **Same API**: Async client follows the same patterns as the sync client

## Core Concepts

### 1. Decorating Functions with `@tool`

The `@tool` decorator adds metadata to functions so they can be used as LLM tools:

```python
import toolflow

@toolflow.tool
def search_database(query: str, limit: int = 10) -> list:
    """Search the database for matching records."""
    # Your implementation here
    return ["result1", "result2"]

# The function can now be passed directly to the tools parameter
```

### 2. Custom Tool Names and Descriptions

```python
@toolflow.tool(name="db_search", description="Search our product database")
def search_products(query: str) -> list:
    """Original docstring."""
    return search_results
```

### 3. Multiple LLM Providers

```python
# OpenAI
openai_client = toolflow.from_openai(OpenAI(api_key="your-openai-key"))

# Anthropic Claude (Coming soon)
claude_client = toolflow.from_anthropic(anthropic.Anthropic(api_key="your-anthropic-key"))
```

### 4. Async Tools support

```python
import asyncio
import aiohttp

@toolflow.tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

@toolflow.tool  
async def query_database(query: str) -> list:
    """Execute a database query asynchronously."""
    # Simulate async database operation
    await asyncio.sleep(0.1)
    return [{"id": 1, "name": "result"}]
```

### 5. Direct Function Calls

```python
# You can still call your functions directly
@toolflow.tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

@toolflow.tool
async def async_weather(city: str) -> str:
    await asyncio.sleep(0.1)
    return f"Async weather in {city}: Sunny"

# Direct calls
sync_result = get_weather("New York")
async_result = await async_weather("New York")
```


## API Reference

### `@tool` decorator

```python
@toolflow.tool(name=None, description=None)
```

- `name`: Custom tool name (defaults to function name)
- `description`: Tool description (defaults to docstring)

### `from_openai(client)`

Wraps a synchronous OpenAI client to support toolflow functions:

```python
import openai
import toolflow

client = toolflow.from_openai(openai.OpenAI())
```

### `from_openai_async(client)`

Wraps an asynchronous OpenAI client to support toolflow functions:

```python
import openai
import toolflow

async_client = toolflow.from_openai_async(openai.AsyncOpenAI())
```

### Enhanced `chat.completions.create()`

When using a wrapped client (sync or async), the `create` method gains additional parameters:

- `tools`: List of toolflow decorated functions (sync or async) or regular tool dicts
- `parallel_tool_execution`: Whether to execute multiple tool calls in parallel (default: `False`)
- `max_tool_calls`: Maximum number of tool calls to execute (default: 5)
- `max_workers`: Maximum number of worker threads to use for parallel execution of sync tools (default: 10)

**Sync usage:**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    tools=[tool1, tool2, tool3],
    parallel_tool_execution=True,  # Enable parallel execution
    max_tool_calls=10,
    max_workers=10
)
```

**Async usage:**
```python
response = await async_client.chat.completions.create(
    model="gpt-4o-mini", 
    tools=[sync_tool, async_tool],
    parallel_tool_execution=True,  # Enable parallel execution
    max_tool_calls=10,
    max_workers=10
)
```

### Tool Function Methods

Every `@tool` decorated function gets these methods:

- `_tool_metadata` - JSON schema for the tool

## Parallel Tool Execution Support

Toolflow can execute multiple tool calls in parallel when enabled, providing significant performance improvements:

```python
# Enable parallel execution with parallel_tool_execution=True
response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "Calculate 5+3, fetch user data, and query database"}],
    tools=[math_tool, fetch_user_tool, database_tool],
    parallel_tool_execution=True,  # Enable parallel execution
    max_tool_calls=10,
    max_workers=10
)

# Sequential execution (default behavior)
response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "Calculate 5+3, fetch user data, and query database"}],
    tools=[math_tool, fetch_user_tool, database_tool]
    # parallel_tool_execution=False (default) - tools execute sequentially
)
```

### Performance Benefits

- **Sync client**: Uses `ThreadPoolExecutor` for parallel execution of sync tools
- **Async client**: Separates sync and async tools for optimal execution:
  - Sync tools run in `ThreadPoolExecutor` 
  - Async tools run with `asyncio.gather`
  - Both groups execute in parallel with each other
- **Typical speedup**: 2-10x faster depending on tool execution time and I/O operations
- **Order preservation**: Results maintain the original tool call order
- **Default behavior**: Sequential execution (`parallel_tool_execution=False`) for backward compatibility

### Execution Strategies

```python
# Async client with mixed tools (optimal strategy)
response = await async_client.chat.completions.create(
    model="gpt-4o-mini",
    tools=[
        sync_database_query,    # Runs in thread pool
        async_api_call,         # Runs with asyncio.gather  
        sync_file_processing,   # Runs in thread pool
        async_network_request   # Runs with asyncio.gather
    ],
    parallel_tool_execution=True,
    max_tool_calls=10,
    max_workers=10 # Thread pool size for sync tools
)
# sync_database_query + sync_file_processing execute in ThreadPoolExecutor
# async_api_call + async_network_request execute with asyncio.gather  
# Both groups run in parallel with each other
```

## Best Practices

### When to Use Async Tools

Use async tools for I/O-bound operations:

```python
# ✅ Good: I/O operations should be async
@toolflow.tool
async def fetch_user_data(user_id: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/api/users/{user_id}") as resp:
            return await resp.json()

@toolflow.tool  
async def query_database(sql: str) -> list:
    async with asyncpg.connect() as conn:
        return await conn.fetch(sql)

# ✅ Good: CPU-bound operations can stay sync
@toolflow.tool
def calculate_fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# ❌ Avoid: Blocking I/O in sync tools when using async client
@toolflow.tool
def bad_fetch(url: str) -> str:
    import requests  # This blocks the event loop!
    return requests.get(url).text
```

### Error Handling

```python
@toolflow.tool
async def safe_api_call(endpoint: str) -> str:
    """Make a safe API call with proper error handling."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint) as response:
                response.raise_for_status()
                return await response.text()
    except aiohttp.ClientError as e:
        return f"API call failed: {str(e)}"
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests (sync and async)
pytest

# Run only sync tests
pytest tests/test.py

# Run only async tests  
pytest tests/test_async.py

# Run only parallel execution tests
pytest tests/test_parallel.py

# Format code
black toolflow/
isort toolflow/

# Type checking
mypy toolflow/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
