"""
ToolKit Example - Demonstrating organized tool collections in Toolflow.

This example shows how to use the new ToolKit concept to organize related
tool functions into classes for better modularity and code organization.
"""
import toolflow
from openai import OpenAI
from typing import List, Dict
from pydantic import BaseModel


class MathToolKit:
    """A ToolKit containing mathematical operations."""
    
    def __init__(self, precision: int = 2):
        """Initialize with configurable precision."""
        self.precision = precision
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return round(a + b, self.precision)
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return round(a - b, self.precision)
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return round(a * b, self.precision)
    
    def divide(self, dividend: float, divisor: float) -> float:
        """Divide dividend by divisor."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return round(dividend / divisor, self.precision)
    
    def power(self, base: float, exponent: int) -> float:
        """Calculate base raised to exponent."""
        return round(base ** exponent, self.precision)


class DataAnalysisToolKit:
    """A ToolKit for data analysis operations."""
    
    def calculate_statistics(self, numbers: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of numbers."""
        if not numbers:
            return {"error": "Empty list provided"}
        
        return {
            "count": len(numbers),
            "sum": sum(numbers),
            "average": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers)
        }
    
    def filter_outliers(self, numbers: List[float], threshold: float = 2.0) -> List[float]:
        """Remove outliers using standard deviation threshold."""
        if len(numbers) < 2:
            return numbers
        
        mean = sum(numbers) / len(numbers)
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        std_dev = variance ** 0.5
        
        return [x for x in numbers if abs(x - mean) <= threshold * std_dev]
    
    def find_percentile(self, numbers: List[float], percentile: int) -> float:
        """Find the value at a given percentile."""
        if not numbers or percentile < 0 or percentile > 100:
            raise ValueError("Invalid input for percentile calculation")
        
        sorted_numbers = sorted(numbers)
        index = (percentile / 100) * (len(sorted_numbers) - 1)
        
        if index.is_integer():
            return sorted_numbers[int(index)]
        else:
            lower = sorted_numbers[int(index)]
            upper = sorted_numbers[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class StringToolKit:
    """A ToolKit for string manipulation operations."""
    
    def reverse_string(self, text: str) -> str:
        """Reverse a string."""
        return text[::-1]
    
    def count_words(self, text: str) -> int:
        """Count the number of words in text."""
        return len(text.split())
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text (simple pattern)."""
        import re
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    def format_title(self, text: str) -> str:
        """Convert text to title case."""
        return text.title()


# Standalone function (traditional approach)
def get_current_time() -> str:
    """Get the current time as a string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Demonstrate ToolKit usage with Toolflow."""
    
    # Initialize OpenAI client with Toolflow wrapper
    client = toolflow.from_openai(OpenAI())
    
    # Create ToolKit instances
    math_tools = MathToolKit(precision=2)
    data_tools = DataAnalysisToolKit()
    string_tools = StringToolKit()
    
    print("=== ToolKit Example: Mathematical Operations ===")
    
    # Example 1: Using MathToolKit
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user", 
            "content": "Calculate (15.7 + 8.3) * 2, then divide by 4, and finally raise to the power of 2"
        }],
        tools=[math_tools],  # Pass the entire ToolKit
        parallel_tool_execution=True
    )
    print(f"Math calculation result: {response}")
    print()
    
    print("=== ToolKit Example: Data Analysis ===")
    
    # Example 2: Using DataAnalysisToolKit
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": "Analyze this dataset: [10, 15, 20, 25, 30, 35, 40, 100]. Calculate statistics and remove outliers."
        }],
        tools=[data_tools],
        parallel_tool_execution=True
    )
    print(f"Data analysis result: {response}")
    print()
    
    print("=== ToolKit Example: Mixed ToolKites and Functions ===")
    
    # Example 3: Multiple ToolKites + standalone function
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": "What's the current time? Also, analyze the text 'Hello World! Contact us at info@example.com or support@test.org' - count words and extract emails."
        }],
        tools=[string_tools, get_current_time],  # Mix ToolKit and function
        parallel_tool_execution=True
    )
    print(f"Mixed tools result: {response}")
    print()
    
    print("=== ToolKit Example: Structured Output ===")
    
    # Example 4: ToolKit with structured output
    class AnalysisResult(BaseModel):
        statistics: Dict[str, float]
        outliers_removed: List[float]
        percentile_50: float
        summary: str
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": "Analyze the dataset [1, 2, 3, 4, 5, 100, 7, 8, 9, 10] and provide comprehensive analysis"
        }],
        tools=[data_tools],
        response_format=AnalysisResult,
        parallel_tool_execution=True
    )
    print(f"Structured analysis result: {response}")
    print(f"Statistics: {response.statistics}")
    print(f"Clean data: {response.outliers_removed}")
    print()


if __name__ == "__main__":
    # Note: You'll need to set your OpenAI API key
    # export OPENAI_API_KEY="your-api-key-here"
    
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set your OPENAI_API_KEY environment variable.")
        print("You can also replace OpenAI with a mock client for testing the ToolKit concept.")
