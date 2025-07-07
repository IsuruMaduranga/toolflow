from toolflow.errors import MaxToolCallsError
import os
from openai import AsyncOpenAI
from enum import Enum
from typing import List, Union
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import asyncio
from toolflow import from_openai

client = from_openai(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

def thinking_tool(chain_of_thought: str) -> str:
    """
    Use this tool to think about the question.

    Args:
        chain_of_thought (str): This is the chain of thought for the question.

    Returns:
        str: The same chain of thought (or processed result).
    """
    print("Thinking tool called")
    print(chain_of_thought)
    print("Thinking tool finished")
    return chain_of_thought

async def main():
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Give me a plan to build a rocket"}],
            tools=[thinking_tool],
            max_tool_call_rounds=20
        )

        print(response)

    except MaxToolCallsError as e:
        print(e)
        print(e.tool_calls)
        print(e.tool_calls[0].function.arguments)
        print(e.tool_calls[1].function.arguments)

if __name__ == "__main__":
    asyncio.run(main())
