"""
Example demonstrating structured outputs with Google Gemini and toolflow.

This example shows how to use Pydantic models to get structured responses
from Gemini using toolflow's response_format feature.
"""

import os
import toolflow
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
client = toolflow.from_gemini(model)

# Pydantic models for structured outputs
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class TextAnalysis(BaseModel):
    """Analysis of a text passage."""
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    key_topics: List[str] = Field(description="Main topics or themes in the text")
    word_count: int = Field(description="Approximate number of words")
    summary: str = Field(description="Brief summary of the text")

class WeatherInfo(BaseModel):
    """Weather information for a location."""
    location: str
    temperature: str
    condition: str
    humidity: Optional[str] = None
    wind_speed: Optional[str] = None

class TravelPlan(BaseModel):
    """Travel plan with destinations and activities."""
    destination: str
    duration_days: int
    activities: List[str]
    estimated_budget: str
    best_season: str
    weather_info: Optional[WeatherInfo] = None

class MathResult(BaseModel):
    """Result of a mathematical calculation."""
    expression: str
    result: float
    steps: List[str] = Field(description="Step-by-step calculation process")

class ProductReview(BaseModel):
    """Product review analysis."""
    product_name: str
    overall_rating: float = Field(ge=1.0, le=5.0)
    pros: List[str]
    cons: List[str]
    recommendation: str
    target_audience: str

# Tools for structured output examples
@toolflow.tool
def get_weather_data(city: str) -> WeatherInfo:
    """Get detailed weather information for a city."""
    # Mock weather data
    weather_data = {
        "New York": WeatherInfo(
            location="New York, NY",
            temperature="72°F (22°C)",
            condition="Sunny",
            humidity="45%",
            wind_speed="8 mph"
        ),
        "London": WeatherInfo(
            location="London, UK",
            temperature="59°F (15°C)",
            condition="Cloudy",
            humidity="70%",
            wind_speed="12 mph"
        ),
        "Tokyo": WeatherInfo(
            location="Tokyo, Japan",
            temperature="68°F (20°C)",
            condition="Rainy",
            humidity="85%",
            wind_speed="6 mph"
        )
    }
    
    return weather_data.get(city, WeatherInfo(
        location=city,
        temperature="Data not available",
        condition="Unknown"
    ))

def main():
    print("=== Gemini Structured Outputs Example ===\n")
    
    # Example 1: Text analysis with structured output
    print("1. Text Analysis with Structured Output:")
    text_to_analyze = """
    I absolutely love this new smartphone! The camera quality is incredible, 
    and the battery life lasts all day. The design is sleek and modern. 
    However, it's quite expensive and the learning curve for some features 
    is steep. Overall, I'm very satisfied with this purchase.
    """
    
    response = client.generate_content(
        f"Analyze this text: {text_to_analyze}",
        response_format=TextAnalysis
    )
    
    print(f"Analysis Result:")
    print(f"- Sentiment: {response.sentiment}")
    print(f"- Confidence: {response.confidence}")
    print(f"- Key Topics: {', '.join(response.key_topics)}")
    print(f"- Word Count: {response.word_count}")
    print(f"- Summary: {response.summary}")
    print("\n" + "="*60 + "\n")
    
    # Example 2: Travel planning with structured output
    print("2. Travel Planning with Structured Output:")
    
    response = client.generate_content(
        "Create a 5-day travel plan for visiting Paris, France. Include activities, budget estimate, and best season to visit.",
        response_format=TravelPlan
    )
    
    print(f"Travel Plan:")
    print(f"- Destination: {response.destination}")
    print(f"- Duration: {response.duration_days} days")
    print(f"- Activities: {', '.join(response.activities)}")
    print(f"- Estimated Budget: {response.estimated_budget}")
    print(f"- Best Season: {response.best_season}")
    if response.weather_info:
        print(f"- Weather: {response.weather_info.condition}, {response.weather_info.temperature}")
    print("\n" + "="*60 + "\n")
    
    # Example 3: Mathematical calculation with structured steps
    print("3. Math Calculation with Structured Steps:")
    
    response = client.generate_content(
        "Calculate the compound interest for $1000 invested at 5% annual rate for 3 years, compounded annually. Show all steps.",
        response_format=MathResult
    )
    
    print(f"Math Result:")
    print(f"- Expression: {response.expression}")
    print(f"- Result: {response.result}")
    print(f"- Steps:")
    for i, step in enumerate(response.steps, 1):
        print(f"  {i}. {step}")
    print("\n" + "="*60 + "\n")
    
    # Example 4: Product review analysis
    print("4. Product Review Analysis:")
    
    review_text = """
    This laptop is amazing for productivity work. The 16-inch screen is perfect 
    for multitasking, and the M2 chip handles everything I throw at it. 
    Battery life easily gets me through a full workday. The build quality 
    feels premium and solid. The keyboard is comfortable for long typing sessions.
    
    However, it's quite heavy to carry around, and the price point is definitely 
    on the higher side. Some ports I need aren't available, so I need dongles. 
    The fan can get loud during intensive tasks.
    
    Overall, I'd recommend it for professionals who need a powerful machine 
    and don't mind paying premium prices.
    """
    
    response = client.generate_content(
        f"Analyze this product review and provide a structured analysis: {review_text}",
        response_format=ProductReview
    )
    
    print(f"Product Review Analysis:")
    print(f"- Product: {response.product_name}")
    print(f"- Rating: {response.overall_rating}/5.0")
    print(f"- Pros: {', '.join(response.pros)}")
    print(f"- Cons: {', '.join(response.cons)}")
    print(f"- Recommendation: {response.recommendation}")
    print(f"- Target Audience: {response.target_audience}")
    print("\n" + "="*60 + "\n")
    
    # Example 5: Combining tools with structured output
    print("5. Combining Tools with Structured Output:")
    
    response = client.generate_content(
        "Get weather information for Tokyo and create a travel plan based on that weather data.",
        tools=[get_weather_data],
        response_format=TravelPlan,
        max_tool_call_rounds=3
    )
    
    print(f"Weather-Based Travel Plan:")
    print(f"- Destination: {response.destination}")
    print(f"- Duration: {response.duration_days} days")
    print(f"- Activities: {', '.join(response.activities)}")
    print(f"- Budget: {response.estimated_budget}")
    if response.weather_info:
        print(f"- Current Weather: {response.weather_info.condition}, {response.weather_info.temperature}")

if __name__ == "__main__":
    main()
