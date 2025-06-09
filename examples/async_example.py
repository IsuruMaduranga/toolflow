"""
Example demonstrating async client usage with toolflow.

This example shows how to use the async OpenAI client wrapper
with both synchronous and asynchronous tool functions.
"""

import asyncio
import openai
import toolflow
import os

@toolflow.tool
def sync_calculator(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical operations (sync version)."""
    if operation == "add":
        print(f"Tool: Adding {a} and {b}")
        return a + b
    elif operation == "subtract":
        print(f"Tool: Subtracting {a} and {b}")
        return a - b
    elif operation == "multiply":
        print(f"Tool: Multiplying {a} and {b}")
        return a * b
    elif operation == "divide":
        print(f"Tool: Dividing {a} by {b}")
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


@toolflow.tool
async def async_database_query(query: str) -> str:
    """Give your SQL query to the database and get the result.
    Two tables are available: users and orders.
    users table has the following columns: id, name, email, age
    orders table has the following columns: id, user_id, amount, date
    """
    # Simulate async database operation
    await asyncio.sleep(0.1)
    
    if "users" in query.lower():
        return "Found 42 users in the database"
    elif "orders" in query.lower():
        return "Found 128 orders in the database"
    else:
        return f"Executed query: {query}"


@toolflow.tool
async def async_api_call(endpoint: str) -> str:
    """Simulate an async API call."""
    # Simulate async API call
    await asyncio.sleep(0.2)
    
    return f"API response from {endpoint}: Status 200 OK"


async def main():
    """Main async function demonstrating the async client."""
    
    # Create async OpenAI client (you'll need to set your API key)
    client = toolflow.from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    # Using async client with a sync tool
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 3.145 divided by 2?"}],
        tools=[sync_calculator],
        max_tool_calls=5,
    )
    print(response.choices[0].message.content)

    # Using async client with an async tool
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "How many users are there in the database?"}],
        tools=[async_database_query],
        max_tool_calls=5,
    )
    print(response.choices[0].message.content)

    # Using async client with a sync tool and an async tool
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Multiply orders by number of users"}],
        tools=[sync_calculator, async_database_query],
        max_tool_calls=5,
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main()) 
