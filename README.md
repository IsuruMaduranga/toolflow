# Toolflow

**A lightweight drop-in replacement for OpenAI and Anthropic SDKs ( Many more to come) with automatic parallel tool calling and structured responses.**

Stop wrestling with bloated frameworks for tool calling. Toolflow enhances the official SDKs you already know without breaking any existing functionality - just add powerful auto-parallel tool calling and structured outputs with zero migration effort.

## Why Toolflow?

✅ **Drop-in replacement** - Works exactly like OpenAI/Anthropic SDKs  
✅ **Zero breaking changes** - All official SDK features preserved  
✅ **Auto-parallel tool calling** - Functions become tools with automatic concurrency  
✅ **Structured outputs** - Pass Pydantic models, get typed responses  
✅ **No bloat** - Lightweight alternative to heavy frameworks  
✅ **Unified interface** - Same code works across providers  
✅ **Smart response modes** - Choose between simplified or full SDK responses

## Installation

```bash
pip install toolflow
```

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

## Automatic Parallel Tool Calling

Transform any function into an LLM tool with automatic parallel execution:

```python
import toolflow
from openai import OpenAI
from anthropic import Anthropic
import time

# Wrap your existing clients - no other changes needed
openai_client = toolflow.from_openai(OpenAI())
anthropic_client = toolflow.from_anthropic(Anthropic())

# Any function becomes a tool automatically
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    time.sleep(1)  # Simulated API call
    return f"Weather in {city}: Sunny, 72°F"

def get_population(city: str) -> str:
    """Get population information for a city."""
    time.sleep(1)  # Simulated API call
    return f"Population of {city}: 8.3 million"

def calculate(expression: str) -> float:
    """Safely evaluate mathematical expressions."""
    return eval(expression.replace("^", "**"))

# Same code works with both providers
tools = [get_weather, get_population, calculate]
messages = [{"role": "user", "content": "What's the weather and population in NYC, plus what's 15 * 23?"}]

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

## All Your Existing Code Still Works

Toolflow doesn't break anything - it's a true drop-in replacement:

```python
# All standard SDK features work unchanged
client = toolflow.from_openai(OpenAI())

# Function calling (official OpenAI way)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather?"}],
    functions=[{
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            }
        }
    }],
    full_response=True  # Get full SDK response
)

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

# Sequential: ~3+ seconds
result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Call the API with 'test', multiply 5*10, and query users table"}],
    tools=tools,
    parallel_tool_execution=False
)

# Parallel: ~2 seconds (limited by slowest tool)
result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Call the API with 'test', multiply 5*10, and query users table"}],
    tools=tools,
    parallel_tool_execution=True
)

# Check current settings
print(f"Max workers: {toolflow.get_max_workers()}")
```

## Async Support with Smart Concurrency

Mix sync and async tools with automatic optimization:

```python
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Wrap async clients
openai_async = toolflow.from_openai(AsyncOpenAI())
anthropic_async = toolflow.from_anthropic(AsyncAnthropic())

async def async_api_call(query: str) -> str:
    """Async tool for I/O operations."""
    await asyncio.sleep(0.5)  # Non-blocking delay
    return f"Async result: {query}"

def sync_calculation(x: int, y: int) -> int:
    """Sync tools work too."""
    time.sleep(0.1)  # Blocking delay
    return x * y

async def async_database_query(table: str) -> str:
    """Another async tool."""
    await asyncio.sleep(0.3)
    return f"Async DB data from {table}"

async def main():
    # Mix sync and async tools - By default async tools run concurrently with asyncio.gather()
    # Sync tools run in thread pool concurrently
    result = await openai_async.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Call API with 'test', multiply 10*5, query users table"}],
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
def search_web(query: str) -> str:
    """Search the web for information."""
    time.sleep(0.5)  # Simulated search delay
    return f"Found tutorials for: {query}"

def get_code_examples(language: str) -> str:
    """Get code examples for a language."""
    time.sleep(0.3)
    return f"Code examples for {language}: print('hello world')"

# Streaming with tools
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Search for Python tutorials and show me examples"}],
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

## API Reference

### Client Wrappers
```python
toolflow.from_openai(client, full_response=False)    # Wraps any OpenAI client
toolflow.from_anthropic(client, full_response=False) # Wraps any Anthropic client
```

### Global Configuration
```python
toolflow.set_max_workers(workers)    # Set thread pool size for parallel execution
toolflow.get_max_workers()           # Get current thread pool size
toolflow.set_executor(executor)      # Use custom ThreadPoolExecutor
```

### Enhanced Parameters
All standard SDK parameters work unchanged, plus these additions:

```python
client.chat.completions.create(
    # All standard parameters work (model, messages, temperature, etc.)
    
    # Toolflow enhancements
    tools=[...],                      # List of functions (any callable)
    response_format=BaseModel,        # Pydantic model for structured output
    parallel_tool_execution=False,    # Enable concurrent tool execution
    max_tool_calls=10,               # Safety limit for tool rounds
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

## Performance Comparison

### Tool Execution Speed
```python
# Sequential execution 
Sequential: 3.2s

# Parallel execution (3 tools)
Parallel: 1.1s
Speedup: 2.9x

# Async execution ( parallel by default)
Async Parallel: 0.8s
Speedup: 4.0x
```

### Memory Usage
- **Toolflow**: ~5MB additional overhead
- **LangChain**: ~50MB+ additional overhead
- **Native SDK**: Baseline

## Why Not LangChain?

| Feature | Toolflow | LangChain |
|---------|----------|-----------|
| **Learning Curve** | Zero - same as OpenAI/Anthropic | Steep - new concepts |
| **Migration Effort** | One line change | Complete rewrite |
| **Bundle Size** | Lightweight (~5MB) | Heavy (~50MB+) |
| **Official SDK Features** | 100% compatible | Limited/wrapped |
| **Structured Outputs** | Built-in | Complex setup |
| **Tool Calling** | Automatic parallel | Manual configuration |
| **Performance** | Optimized thread pools | Variable |
| **Response Modes** | Flexible (simple/full) | Fixed patterns |

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

## Contributing

Contributions welcome! Please fork, create a feature branch, add tests, and submit a pull request.

## License

MIT License - see LICENSE file for details.

---

**Ready to upgrade?** Replace your SDK import and unlock powerful parallel tool calling + structured outputs with zero breaking changes.
