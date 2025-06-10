# Toolflow

A minimal Python library for seamless LLM usage.
- Just decorate your functions and pass them directly to a wrapped AI client for auto tool calling.
- Just pass a Pydantic model to the API for structured outputs.
- Drop in replacement for OpenAI sdk. ( Other providers coming soon )

```python
import toolflow
from openai import OpenAI
from pydantic import BaseModel

# 1. Wrap your client
client = toolflow.from_openai(OpenAI())

# 2. Decorate your function
@toolflow.tool  
def get_weather(city: str) -> str:
  """Gets the current weather for a given city."""
  return f"Weather in {city}: Sunny, 72°F"

class WeatherReport(BaseModel):
    city: str
    weather: str
    temperature: float

# 4. Call the API, passing the function directly
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[get_weather],
)

# 5. If you want structured outputs, pass a Pydantic model to the API
response_structured = client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[get_weather],
    response_format=WeatherReport # Pydantic model for structured output
)

# 6. Read the outputs
print(response.choices[0].message.content)
print(response_structured.choices[0].message.parsed)

# Toolflow handles the schema generation, execution, and response automatically
# Output: Weather in NYC: Sunny, 72°F
```

## Key Features
🎯 Decorator-Based: Use @toolflow.tool to make any function an LLM tool.

🔄 Automatic Handling: Automatically generates JSON schemas, executes tool calls, and sends results back to the LLM.

⚡ Parallel Execution: Runs multiple tools concurrently for significant speed improvements.

📡 Seamless Streaming: Full streaming support with automatic, non-blocking tool execution.

🤝 Async Support: First-class support for async functions and clients, with automatic handling of mixed sync/async tools.

🔧 Structured Outputs: Full support for structured outputs with tool execution ( Using OpenAI's beta API or the toolflow enhanced API)

## Installation

```bash
pip install toolflow
```

## Usage

### Basic Example

Define multiple tools and pass them in a list. Toolflow manages the conversation flow.

```python
import toolflow
from openai import OpenAI

client = toolflow.from_openai(OpenAI())

@toolflow.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@toolflow.tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What's the weather in SF? Also, what's 15 + 27?"}
    ],
    tools=[get_weather, add_numbers],
    parallel_tool_execution=True, # Tools will run in parallel
)

print(response.choices[0].message.content)
```

### Streaming with Tools

Simply add `stream=True` as you would with OpenAI. Toolflow detects tool calls from the stream, executes them, and continues the conversation seamlessly.

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[get_weather],
    stream=True
)

for chunk in stream:
    if content := chunk.choices[0].delta.content:
        print(content, end="")
```

## Async Operations

Wrap `AsyncOpenAI` and use `async/await`. You can mix sync and async tools in the same call.

```python
import asyncio
from openai import AsyncOpenAI

async_client = toolflow.from_openai_async(AsyncOpenAI())

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
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Add 15 + 27 and query the users table."}],
        tools=[sync_calculator, async_db_query] # Mix and match!
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

> Note: When using an async client, avoid blocking I/O (like the requests library) in your synchronous tools to prevent blocking the event loop. Or enable `parallel_tool_execution=True` to run sync tools in a thread pool.

### Structured Outputs

Toolflow supports structured outputs with tool execution using OpenAI's beta API or the toolflow enhanced API ( both Sync and Async )

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

client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Toolflow enhanced API
response = client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
    tools=[fibonacci], # Toolcalling works seamlessly
    response_format=FibonacciResponse # Pydantic model for structured output
)

# OpenAI Beta API
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are 10th, 11th and 12th Fibonacci numbers."}],
    tools=[fibonacci],
    response_format=FibonacciResponse
)

print(response.choices[0].message.parsed)
print(response.choices[0].message.content)
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
- `toolflow.from_openai(client)`: Wraps a synchronous openai.OpenAI client.
- `toolflow.from_openai_async(client)`: Wraps an asynchronous openai.AsyncOpenAI client.

### Enhanced chat.completions.create()

The `create` method on a wrapped client is enhanced with the following parameters:

```python
client.chat.completions.create(
    # Standard OpenAI parameters...
    model="gpt-4o-mini",
    messages=[...],
    stream=False,

    # Toolflow parameters
    tools: list,
    parallel_tool_execution: bool = False,
    max_tool_calls: int = 10,
    max_workers: int = 10,
)
```
- `tools`: A list of functions decorated with @toolflow.tool.
- `parallel_tool_execution`: If True, executes multiple tool calls concurrently.
- `max_tool_calls`: A safety limit for the number of tool call rounds in a single turn.
- `max_workers`: The maximum number of threads for running synchronous tools in parallel.

### Parallel Execution
Set `parallel_tool_execution=True` to significantly improve performance when an LLM calls multiple tools.

- Sync Client: Uses a `ThreadPoolExecutor` to run sync tools in parallel.
- Async Client: Runs sync tools in a `ThreadPoolExecutor` and async tools with `asyncio.gather`, executing both groups concurrently for optimal performance.

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
