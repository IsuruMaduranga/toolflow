import toolflow
from anthropic import Anthropic

@toolflow.tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location"""
    return f"Weather in {location}: Sunny, 72°F"

client = toolflow.from_anthropic(Anthropic())

# Works exactly like the original Anthropic SDK with added tool support
response = client.messages.create(
    system="You are a helpful assistant.",  # System message as separate parameter
    model="claude-3-5-haiku-latest",
    max_tokens=1024,
    tools=[get_weather],  # Pass toolflow decorated functions directly
    parallel_tool_execution=True,  # Enable parallel execution
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
)

print(response)  # Returns just the content string (simplified API)
