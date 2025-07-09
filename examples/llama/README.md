# Llama Provider Examples

This directory contains examples demonstrating how to use Llama models with toolflow through OpenAI-compatible APIs like OpenRouter.

## Setup

### 1. Install Dependencies

```bash
pip install openai toolflow
```

### 2. Get API Access

The Llama provider works with any OpenAI-compatible API that provides access to Llama models. Popular options include:

- **OpenRouter** (https://openrouter.ai/) - Recommended
- **Together AI** (https://together.ai/)
- **Anyscale** (https://anyscale.com/)
- **Fireworks AI** (https://fireworks.ai/)

### 3. Configure Environment

Set your API key:

```bash
export LLAMA_API_KEY="your-api-key-here"
```

For OpenRouter specifically:
```bash
export LLAMA_API_KEY="sk-or-v1-your-openrouter-key"
```

## Available Examples

### Basic Usage (`sync_basic.py`)
Demonstrates fundamental Llama model capabilities:
- Basic text generation
- Tool calling with complex parameters
- Parallel tool execution
- Async operations
- Performance comparisons

```bash
python examples/llama/sync_basic.py
```

## Configuration

### OpenRouter Setup

```python
import openai
import toolflow

# Configure for OpenRouter
client = openai.OpenAI(
    api_key="your-openrouter-key",
    base_url="https://openrouter.ai/api/v1"
)

# Enhance with toolflow
enhanced_client = toolflow.from_llama(client)

# Use with Llama models
response = enhanced_client.chat.completions.create(
    model="meta-llama/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Other Providers

For other providers, just change the `base_url`:

```python
# Together AI
client = openai.OpenAI(
    api_key="your-together-key",
    base_url="https://api.together.xyz/v1"
)

# Anyscale
client = openai.OpenAI(
    api_key="your-anyscale-key", 
    base_url="https://api.anyscale.com/v1"
)
```

## Available Llama Models

Popular Llama models available through OpenRouter:

| Model | Context | Description |
|-------|---------|-------------|
| `meta-llama/llama-3.1-70b-instruct` | 131K | Best balance of performance and speed |
| `meta-llama/llama-3.1-405b-instruct` | 131K | Most capable, slower |
| `meta-llama/llama-3.2-90b-instruct` | 131K | Latest version, good performance |
| `meta-llama/llama-2-70b-chat` | 4K | Older but reliable |

## Features

### Tool Calling
```python
@toolflow.tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

response = enhanced_client.chat.completions.create(
    model="meta-llama/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[get_weather]
)
```

### Parallel Execution
```python
response = enhanced_client.chat.completions.create(
    model="meta-llama/llama-3.1-70b-instruct", 
    messages=[{"role": "user", "content": "Get weather for NYC, LA, and Chicago"}],
    tools=[get_weather],
    parallel_tool_calls=True  # Execute tools in parallel
)
```

### Structured Outputs
```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

response = enhanced_client.chat.completions.create(
    model="meta-llama/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Summarize this article..."}],
    response_format=Summary
)
```

### Async Operations
```python
import asyncio

async def main():
    async_client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    enhanced_async_client = toolflow.from_llama(async_client)
    
    response = await enhanced_async_client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": "Hello async world!"}]
    )

asyncio.run(main())
```

## Tips

### Model Selection
- **For most applications**: `meta-llama/llama-3.1-70b-instruct`
- **For complex reasoning**: `meta-llama/llama-3.1-405b-instruct`
- **For latest features**: `meta-llama/llama-3.2-90b-instruct`
- **For cost efficiency**: `meta-llama/llama-2-70b-chat`

### Performance Optimization
- Use `parallel_tool_calls=True` for multiple tool calls
- Adjust `max_tokens` based on your needs (default: 2048)
- Consider `temperature` settings (default: 0.7 for creativity)

### Error Handling
Common issues and solutions:

1. **Model not found**: Check model name and provider availability
2. **Rate limits**: Verify your API quota and rate limits
3. **Context length**: Use models with appropriate context windows
4. **API key issues**: Ensure correct key format for your provider

## Troubleshooting

### API Key Issues
```bash
# Check your API key is set
echo $LLAMA_API_KEY

# For OpenRouter, key should start with "sk-or-v1-"
# For Together AI, key format may differ
```

### Model Availability
Different providers offer different Llama models. Check your provider's documentation for:
- Available model names
- Context length limits
- Rate limits and pricing
- Special features support

### Debugging
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

- **toolflow issues**: [GitHub Issues](https://github.com/AdhiDevX369/toolflow/issues)
- **OpenRouter support**: [OpenRouter Discord](https://discord.gg/openrouter)
- **Model-specific questions**: Check your provider's documentation
