import openai
import toolflow
import os
import time

@toolflow.tool(name="sync_calculator")
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
def sync_database_query(query: str) -> str:
    """Give your SQL query to the database and get the result.
    Two tables are available: users and orders.
    users table has the following columns: id, name, email, age
    orders table has the following columns: id, user_id, amount, date
    """
    # Simulate async database operation
    time.sleep(0.1)
    
    if "users" in query.lower():
        return "Found 42 users in the database"
    elif "orders" in query.lower():
        return "Found 128 orders in the database"
    else:
        return f"Executed query: {query}"


@toolflow.tool
def sync_api_call(endpoint: str) -> str:
    """Simulate an async API call."""
    # Simulate async API call
    time.sleep(0.2)
    
    return f"API response from {endpoint}: Status 200 OK"


def main():
    """Main async function demonstrating the async client."""
    
    # Create async OpenAI client (you'll need to set your API key)
    # Default behavior: simplified API (returns content directly)
    client = toolflow.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    
    # Using async client with a sync tool
    content = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 3.145 divided by 2?"}],
        tools=[sync_calculator],
        max_tool_calls=5,
    )

    client.chat.completions.create()
    print(content)  # Direct string output


if __name__ == "__main__":
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 3.145 divided by 2?"}],
        tools=[sync_calculator],
        max_tool_calls=5,
    )
    main()
