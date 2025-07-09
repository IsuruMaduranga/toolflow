#!/usr/bin/env python3
"""
Llama structured outputs example using toolflow.

Demonstrates how to use Pydantic models for structured responses
with Llama models through OpenRouter or similar providers.
"""

import os
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import openai
import toolflow

# Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLAMA_MODEL = "meta-llama/llama-3.1-70b-instruct"

# Pydantic models for structured outputs
class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class TextAnalysis(BaseModel):
    sentiment: Sentiment = Field(description="Sentiment classification")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    key_phrases: List[str] = Field(description="Important phrases from the text")
    summary: str = Field(description="Brief summary of the text")

class PersonInfo(BaseModel):
    name: str = Field(description="Full name")
    age: Optional[int] = Field(description="Age if mentioned", default=None)
    occupation: Optional[str] = Field(description="Job or profession", default=None)
    location: Optional[str] = Field(description="City or country", default=None)
    interests: List[str] = Field(description="Hobbies or interests", default_factory=list)

class TaskPlan(BaseModel):
    title: str = Field(description="Task title")
    steps: List[str] = Field(description="List of steps to complete the task")
    estimated_time: str = Field(description="Estimated time to completion")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard")
    resources: List[str] = Field(description="Required resources or tools")

def setup_llama_client() -> openai.OpenAI:
    """Setup OpenAI client configured for Llama models."""
    api_key = os.getenv('LLAMA_API_KEY')
    if not api_key:
        raise ValueError(
            "LLAMA_API_KEY environment variable is required. "
            "Get your key from OpenRouter or your Llama model provider."
        )
    
    return openai.OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

def demonstrate_text_analysis():
    """Demonstrate text analysis with structured output."""
    print("📊 Text Analysis Example")
    print("=" * 50)
    
    client = setup_llama_client()
    enhanced_client = toolflow.from_llama(client)
    
    texts_to_analyze = [
        "I absolutely love this new restaurant! The food is amazing and the service is outstanding.",
        "The product arrived damaged and the customer service was unhelpful. Very disappointed.",
        "The weather today is partly cloudy with a chance of rain. Temperature is around 70 degrees."
    ]
    
    for i, text in enumerate(texts_to_analyze, 1):
        print(f"\n{i}. Analyzing: '{text}'")
        
        response = enhanced_client.chat.completions.create(
            model=DEFAULT_LLAMA_MODEL,
            messages=[
                {"role": "user", "content": f"Analyze this text for sentiment, confidence, key phrases, and provide a summary: {text}"}
            ],
            response_format=TextAnalysis,
            max_tokens=500
        )
        
        analysis = response.parsed
        print(f"   Sentiment: {analysis.sentiment.value}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Key phrases: {', '.join(analysis.key_phrases)}")
        print(f"   Summary: {analysis.summary}")

def demonstrate_person_extraction():
    """Demonstrate extracting person information from text."""
    print("\n👤 Person Information Extraction")
    print("=" * 50)
    
    client = setup_llama_client()
    enhanced_client = toolflow.from_llama(client)
    
    bio_texts = [
        "Sarah Johnson is a 28-year-old software engineer from Seattle. She enjoys hiking, photography, and playing guitar in her free time.",
        "Dr. Martinez works as a pediatrician in Chicago. He's passionate about community health and volunteers at local clinics.",
        "Emily Chen, age 35, is a marketing director living in San Francisco. She loves cooking, traveling, and reading mystery novels."
    ]
    
    for i, bio in enumerate(bio_texts, 1):
        print(f"\n{i}. Extracting from: '{bio}'")
        
        response = enhanced_client.chat.completions.create(
            model=DEFAULT_LLAMA_MODEL,
            messages=[
                {"role": "user", "content": f"Extract person information from this text: {bio}"}
            ],
            response_format=PersonInfo,
            max_tokens=400
        )
        
        person = response.parsed
        print(f"   Name: {person.name}")
        print(f"   Age: {person.age or 'Not specified'}")
        print(f"   Occupation: {person.occupation or 'Not specified'}")
        print(f"   Location: {person.location or 'Not specified'}")
        print(f"   Interests: {', '.join(person.interests) if person.interests else 'None specified'}")

def demonstrate_task_planning():
    """Demonstrate task planning with structured output."""
    print("\n📋 Task Planning Example")
    print("=" * 50)
    
    client = setup_llama_client()
    enhanced_client = toolflow.from_llama(client)
    
    tasks = [
        "Plan a birthday party for a 10-year-old",
        "Learn to play guitar",
        "Organize a home office space"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. Planning: '{task}'")
        
        response = enhanced_client.chat.completions.create(
            model=DEFAULT_LLAMA_MODEL,
            messages=[
                {"role": "user", "content": f"Create a detailed plan for this task: {task}"}
            ],
            response_format=TaskPlan,
            max_tokens=600
        )
        
        plan = response.parsed
        print(f"   Title: {plan.title}")
        print(f"   Estimated time: {plan.estimated_time}")
        print(f"   Difficulty: {plan.difficulty}")
        print(f"   Steps:")
        for step_num, step in enumerate(plan.steps, 1):
            print(f"     {step_num}. {step}")
        print(f"   Resources: {', '.join(plan.resources)}")

def main():
    """Run all structured output demonstrations."""
    print("🦙 Llama Structured Outputs Examples")
    print("=" * 60)
    print("Using OpenRouter for Llama model access")
    print("Model:", DEFAULT_LLAMA_MODEL)
    print("=" * 60)
    
    try:
        demonstrate_text_analysis()
        demonstrate_person_extraction()
        demonstrate_task_planning()
        
        print("\n✅ All structured output examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure LLAMA_API_KEY is set in your environment")
        print("2. Verify your API key is valid for OpenRouter or your provider")
        print("3. Check that the model supports structured outputs")
        print("4. Ensure you have sufficient API credits/quota")

if __name__ == "__main__":
    main()
