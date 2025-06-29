"""
Example demonstrating async client usage with toolflow.

This example shows how to use the async OpenAI client wrapper
with both synchronous and asynchronous tool functions.
"""

import asyncio
import openai
import toolflow
import os

def sync_calculator(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical operations add, subtract, multiply, divide."""
    print("Executing sync calculator: ", operation, a, b)
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

async def async_database_query(query: str) -> str:
    """Execute SQL queries on the database and get results.
    Available tables: 
    - users (columns: id, name, email, age) - contains 42 total users
    - orders (columns: id, user_id, amount, date) - contains 128 total orders
    
    You can query for counts, specific data, or any SQL operations on these tables.
    """
    print("Executing async database query: ", query)
    # Simulate async database operation
    await asyncio.sleep(0.1)
    
    if "users" in query.lower():
        if "count" in query.lower() or "total" in query.lower():
            return "42"  # Return just the number for easier calculation
        else:
            return "Found 42 users in the database"
    elif "orders" in query.lower():
        if "count" in query.lower() or "total" in query.lower():
            return "128"  # Return just the number for easier calculation
        else:
            return "Found 128 orders in the database"
    else:
        return f"Executed query: {query}. Database contains 42 users and 128 orders."


async def async_api_call(endpoint: str) -> str:
    """Simulate an async API call."""
    # Simulate async API call
    await asyncio.sleep(0.2)
    
    return f"API response from {endpoint}: Status 200 OK"


async def main():
    """Main async function demonstrating the async client."""
    
    # Create async OpenAI client (you'll need to set your API key)
    # Default behavior: simplified API (returns content directly)
    client = toolflow.from_openai_async(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    # Using async client with a sync tool
    content = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 3.145 divided by 2?"}],
        tools=[sync_calculator],
        max_tool_calls=5,
    )
    print(content)  # Direct string output

    # Using async client with an async tool
    content = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "How many users are there in the database?"}],
        tools=[async_database_query],
        max_tool_calls=5,
    )
    print(content)  # Direct string output

    # Using async client with a sync tool and an async tool
    content = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Query the database to get the total number of users and orders, then multiply users by orders to get the result."}],
        tools=[sync_calculator, async_database_query],
        max_tool_calls=10,
    )
    print(content)  # Direct string output

if __name__ == "__main__":
    asyncio.run(main()) 
