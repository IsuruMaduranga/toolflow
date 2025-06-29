import asyncio
import toolflow
from anthropic import AsyncAnthropic

@toolflow.tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location"""
    return f"Weather in {location}: Sunny, 72°F"

client = toolflow.from_anthropic(AsyncAnthropic())

async def main():
    
    stream = await client.messages.create(
        model="claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "What's the weather in NYC?"}],
        max_tokens=100,
        tools=[get_weather],
        stream=True
    )

    async for content in stream:
        print(content, end="", flush=True)

    client_full = toolflow.from_anthropic(AsyncAnthropic(), full_response=True)
    stream = await client_full.messages.create(
        model="claude-3-5-haiku-latest", 
        messages=[{"role": "user", "content": "What's the weather in NYC?"}],
        max_tokens=100,
        tools=[get_weather],
        stream=True
    )

    async for chunk in stream:
        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
            print(chunk.delta.text, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
