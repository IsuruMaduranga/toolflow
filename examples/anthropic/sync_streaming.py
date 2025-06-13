import toolflow
from anthropic import Anthropic
from openai import OpenAI

@toolflow.tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location"""
    return f"Weather in {location}: Sunny, 72°F"

client = toolflow.from_anthropic(Anthropic())

# Simple streaming (yields content strings)
stream = client.messages.create(
    model="claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    max_tokens=100,
    tools=[get_weather],
    stream=True
)

for content in stream:
    print(content, end="", flush=True)

# Full response streaming (yields complete chunks)
stream = client.messages.create(
    model="claude-3-5-haiku-latest", 
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    max_tokens=100,
    tools=[get_weather],
    stream=True,
    full_response=True
)

for chunk in stream:
    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
        print(chunk.delta.text, end="", flush=True)
