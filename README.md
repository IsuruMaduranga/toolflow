# Toolflow

A Python library that makes LLM tool calling as simple as decorating a function. Just wrap your OpenAI client and pass decorated functions directly to the `tools` parameter - no complex setup required.

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
- 🔄 **Asynchronous support**(Coming soon) - Tools can be called asynchronously
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

# Anthropic Claude
claude_client = toolflow.from_anthropic(anthropic.Anthropic(api_key="your-anthropic-key"))
```

### 4. Direct Function Calls

```python
# You can still call your functions directly
@toolflow.tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

# Direct call
result = get_weather("New York")
print(result)  # "Weather in New York: Sunny"
```


## API Reference

### `@tool` decorator

```python
@toolflow.tool(name=None, description=None)
```

- `name`: Custom tool name (defaults to function name)
- `description`: Tool description (defaults to docstring)

### `from_openai(client)`

Wraps an OpenAI client to support toolflow functions:

```python
import openai
import toolflow

client = toolflow.from_openai(openai.OpenAI())
```

### Enhanced `chat.completions.create()`

When using a wrapped client, the `create` method gains additional parameters:

- `tools`: List of toolflow decorated functions or regular tool dicts
- `max_tool_calls`: Maximum number of tool calls to execute (default: 5)

### Tool Function Methods

Every `@tool` decorated function gets these methods:

- `_tool_metadata` - JSON schema for the tool

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

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
