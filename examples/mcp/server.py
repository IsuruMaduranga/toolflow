# server.py
from fastmcp import FastMCP
from typing import List

mcp = FastMCP("Demo 🚀")

@mcp.tool
def add(numbers: List[float]) -> float:
    """Add two numbers"""
    print(f"Adding {numbers}")
    return sum(numbers)

@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    print(f"Dividing {a} by {b}")
    return a / b

@mcp.tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers"""
    print(f"Subtracting {a} from {b}")
    return a - b

@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    print(f"Multiplying {a} and {b}")
    return a * b

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=9000)
