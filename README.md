# Toolflow

A minimal Python library for seamless LLM usage across multiple providers.
- Just decorate your functions and pass them directly to a wrapped AI client for auto tool calling.
- Just pass a Pydantic model to the API for structured outputs.
- Drop in replacement for OpenAI and Anthropic SDKs with unified interface.

```python
import toolflow
from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel

# 1. Wrap your client (OpenAI or Anthropic)
openai_client = toolflow.from_openai(OpenAI())
anthropic_client = toolflow.from_anthropic(Anthropic())

# 2. Decorate your function
@toolflow.tool  
def get_weather(city: str) -> str:
  """Gets the current weather for a given city."""
  return f"Weather in {city}: Sunny, 72°F"

class WeatherReport(BaseModel):
    city: str
    weather: str
    temperature: float

# 3. Use with OpenAI
openai_content = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[get_weather],
)
print(openai_content)  # Direct string output: "Weather in NYC: Sunny, 72°F"

# 4. Use with Anthropic (same interface!)
anthropic_content = anthropic_client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[get_weather],
)
print(anthropic_content)  # Direct string output: "Weather in NYC: Sunny, 72°F"

# 5. For structured outputs with both providers
openai_parsed = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[get_weather],
    response_format=WeatherReport # Pydantic model for structured output
)

anthropic_parsed = anthropic_client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[get_weather],
    response_format=WeatherReport # Same Pydantic model works!
)

print(openai_parsed)    # Direct WeatherReport object
print(anthropic_parsed) # Direct WeatherReport object
```

## Key Features
🎯 **Multi-Provider**: Unified interface for OpenAI and Anthropic with identical APIs.

🔧 **Decorator-Based**: Use @toolflow.tool to make any function an LLM tool.

🔄 **Automatic Handling**: Automatically generates JSON schemas, executes tool calls, and sends results back to the LLM.

⚡ **Parallel Execution**: **UNIQUE FEATURE** - Runs multiple tools concurrently for significant speed improvements (3-5x faster).

📡 **Seamless Streaming**: Full streaming support with automatic, non-blocking tool execution.

🤝 **Async Support**: First-class support for async functions and clients, with automatic handling of mixed sync/async tools.

🔧 **Structured Outputs**: Full support for structured outputs with tool execution across all providers.

## Installation

```bash
pip install toolflow
```

## Usage

### Basic Example

Define multiple tools and pass them in a list. Toolflow manages the conversation flow for both providers.

```python
import toolflow
from openai import OpenAI
from anthropic import Anthropic

# Works with both providers using the same interface
openai_client = toolflow.from_openai(OpenAI())
anthropic_client = toolflow.from_anthropic(Anthropic())

@toolflow.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@toolflow.tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

# OpenAI usage
openai_content = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What's the weather in SF? Also, what's 15 + 27?"}
    ],
    tools=[get_weather, add_numbers],
    parallel_tool_execution=True, # Tools will run in parallel
)

# Anthropic usage (same tools, same interface!)
anthropic_content = anthropic_client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What's the weather in SF? Also, what's 15 + 27?"}
    ],
    tools=[get_weather, add_numbers],
    parallel_tool_execution=True, # Tools will run in parallel
)

print("OpenAI:", openai_content)      # Direct string output
print("Anthropic:", anthropic_content) # Direct string output
```

### Streaming with Tools

Simply add `stream=True` as you would with the native SDKs. Toolflow detects tool calls from the stream, executes them, and continues the conversation seamlessly.

```python
# OpenAI Streaming
openai_stream = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[get_weather],
    stream=True
)

for content in openai_stream:
    print(content, end="")  # Direct content strings

# Anthropic Streaming (same interface!)
anthropic_stream = anthropic_client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[get_weather],
    stream=True
)

for content in anthropic_stream:
    print(content, end="")  # Direct content strings
```

### ⚡ Parallel Tool Execution Support

**Toolflow's parallel execution can speed up your LLM applications by 3-5x when multiple tools are called.**

#### The Problem
Traditional LLM tool calling executes tools sequentially:
```
Tool 1 → Wait → Tool 2 → Wait → Tool 3 → Wait → Response
Total time: 3-5 seconds
```

With `parallel_tool_execution=True`, tools run concurrently:
```
Tool 1 ┐
Tool 2 ├─ All run simultaneously → Response  
Tool 3 ┘
Total time: 1-2 seconds
```

#### Example

```python
import toolflow
import time
from openai import OpenAI

client = toolflow.from_openai(OpenAI())

@toolflow.tool
def get_weather(city: str) -> str:
    # Implementation here

@toolflow.tool
def get_population(city: str) -> str:
    # Implementation here

@toolflow.tool
def get_timezone(city: str) -> str:
    # Implementation here

# Sequential execution (default) - get_weather -> get_population -> get_timezone
sequential_result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me about NYC: weather, population, and timezone"}],
    tools=[get_weather, get_population, get_timezone],
    parallel_tool_execution=False  # Default behavior
)

# Parallel execution - get_weather, get_population, get_timezone run concurrently
parallel_result = await client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "Tell me about NYC: weather, population, and timezone"}],
    tools=[get_weather, get_population, get_timezone],
    parallel_tool_execution=True, 
    max_workers=3,  # Set the number of concurrent threads
    max_tool_calls=15  # Safety limit for tool rounds
)
```

- **Sync Clients**: Uses a `ThreadPoolExecutor` to run sync tools in parallel.
- **Async Clients**: Runs sync tools in a `ThreadPoolExecutor` and async tools with `asyncio.gather`, executing both groups concurrently for optimal performance.
- **Mixed Tools**: Automatically handles combinations of sync and async tools efficiently. ( See Async Operations section )

## Async Operations

Wrap async clients and use `async/await`. You can mix sync and async tools in the same call with both providers.

```python
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

openai_async_client = toolflow.from_openai_async(AsyncOpenAI())
anthropic_async_client = toolflow.from_anthropic_async(AsyncAnthropic())

@toolflow.tool
def sync_calculator(a: int, b: int) -> int:
    """A regular synchronous tool."""
    return a + b

@toolflow.tool
async def async_db_query(query: str) -> str:
    """An async tool for I/O-bound tasks."""
    await asyncio.sleep(0.1) # Represents a non-blocking DB call
    return f"Result for: {query}"

async def main():
    # OpenAI async
    openai_content = await openai_async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Add 15 + 27 and query the users table."}],
        tools=[sync_calculator, async_db_query] # Mix and match!
    )
    
    # Anthropic async (same interface!)
    anthropic_content = await anthropic_async_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Add 15 + 27 and query the users table."}],
        tools=[sync_calculator, async_db_query] # Same tools work!
    )
    
    print("OpenAI:", openai_content)      # Direct string output
    print("Anthropic:", anthropic_content) # Direct string output

asyncio.run(main())
```

> Note: When using an async client, avoid blocking I/O (like the requests library) in your synchronous tools to prevent blocking the event loop. Or enable `parallel_tool_execution=True` to run sync tools in a thread pool.

### Structured Outputs

Toolflow supports structured outputs with tool execution across both OpenAI and Anthropic providers.

```python
@toolflow.tool
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Fib(BaseModel):
    n: int
    value_of_n_th_fibonacci: int

class FibonacciResponse(BaseModel):
    fibonacci_numbers: list[Fib]

openai_client = toolflow.from_openai(OpenAI())
anthropic_client = toolflow.from_anthropic(Anthropic())

# OpenAI structured output
openai_parsed = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
    tools=[fibonacci], # Toolcalling works seamlessly
    response_format=FibonacciResponse # Pydantic model for structured output
)

# Anthropic structured output (same interface!)
anthropic_parsed = anthropic_client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
    tools=[fibonacci], # Same tools work!
    response_format=FibonacciResponse # Same Pydantic model works!
)

print("OpenAI:", openai_parsed)      # Direct FibonacciResponse object
print("Anthropic:", anthropic_parsed) # Direct FibonacciResponse object

# OpenAI also supports the Beta API for structured parsing
beta_parsed_data = openai_client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
    tools=[fibonacci],
    response_format=FibonacciResponse
)
print("OpenAI Beta:", beta_parsed_data)  # Direct FibonacciResponse object
```

## API Reference

### `@toolflow.tool` decorator

```python
@toolflow.tool(name="custom_name", description="Override the docstring.")
def my_function(param: str) -> str:
  """This is the default description."""
  return "result"
```
- `name` (optional): A custom name for the tool. Defaults to the function's name.
- `description` (optional): A custom description. Defaults to the function's docstring.

### Client Wrappers

#### OpenAI
- `toolflow.from_openai(client, full_response=False)`: Wraps a synchronous openai.OpenAI client.
- `toolflow.from_openai_async(client, full_response=False)`: Wraps an asynchronous openai.AsyncOpenAI client.

#### Anthropic
- `toolflow.from_anthropic(client, full_response=False)`: Wraps a synchronous anthropic.Anthropic client.
- `toolflow.from_anthropic_async(client, full_response=False)`: Wraps an asynchronous anthropic.AsyncAnthropic client.

#### `full_response` Parameter
By default, toolflow returns only the content or parsed data from responses for a simplified API. You can control this behavior:

- `full_response=False` (default): Returns simplified content or parsed data
  - OpenAI: `response.choices[0].message.content` or `response.choices[0].message.parsed`
  - Anthropic: `response.content[0].text` or parsed data
- `full_response=True`: Returns the complete provider response object

```python
# Default behavior (simplified API) - returns only content
openai_client = toolflow.from_openai(OpenAI())  # full_response=False by default
anthropic_client = toolflow.from_anthropic(Anthropic())  # full_response=False by default

openai_content = openai_client.chat.completions.create(...)  # Returns string directly
anthropic_content = anthropic_client.messages.create(...)   # Returns string directly

# Full response mode (traditional provider APIs)
openai_full = toolflow.from_openai(OpenAI(), full_response=True) 
anthropic_full = toolflow.from_anthropic(Anthropic(), full_response=True)

openai_response = openai_full.chat.completions.create(...)  # Returns full OpenAI response
anthropic_response = anthropic_full.messages.create(...)    # Returns full Anthropic response
```

### Enhanced Parameters

Both OpenAI and Anthropic wrapped clients support enhanced parameters for better control:

#### Enhanced Parameters ( For any provider )
```python
openai_client.chat.completions.create(
    # Standard OpenAI parameters...
    model="gpt-4o-mini",
    messages=[...],
    stream=False,

    # Toolflow parameters
    tools: list,
    response_format: Pydantic model,
    parallel_tool_execution: bool = False,
    max_tool_calls: int = 10,
    max_workers: int = 10,
    graceful_error_handling: bool = True,
    full_response: bool = False,
)
```

#### Parameter Descriptions
- `tools`: A list of functions decorated with @toolflow.tool.
- `response_format`: A Pydantic model for structured output.
- `parallel_tool_execution`: If True, executes multiple tool calls concurrently.
- `max_tool_calls`: A safety limit for the number of tool call rounds in a single turn.
- `max_workers`: The maximum number of threads for running synchronous tools in parallel.
- `graceful_error_handling`: If True (default), tool execution errors are passed to the model as error messages instead of raising exceptions.

## Development

```bash
# Install for development (includes test dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Format code
black toolflow/
isort toolflow/

# Type checking
mypy toolflow/

```

## Live Integration Testing

Test your toolflow installation with real OpenAI API calls:

```bash
# Set your API key
export OPENAI_API_KEY='your-api-key-here'

# Run interactive test suite
python run_live_tests.py

# Or run directly with pytest
python -m pytest tests/test_integration_live.py -v -s
```

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, add tests, and submit a pull request.

## License

MIT License - see LICENSE file for details.
