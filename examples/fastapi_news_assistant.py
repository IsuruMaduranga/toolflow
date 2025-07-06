from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="News Assistant", description="AI-powered news assistant using Tavily API")

# Initialize ToolFlow with OpenAI client
from openai import AsyncOpenAI
import toolflow

# Tavily API configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your-tavily-api-key")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

client = toolflow.from_openai(AsyncOpenAI(api_key=OPENAI_API_KEY))


# Pydantic models for request/response
class NewsQuery(BaseModel):
    query: str
    max_results: Optional[int] = 3

class ContentExtractRequest(BaseModel):
    urls: List[str]
    max_content_length: Optional[int] = 2000

class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float

class NewsResponse(BaseModel):
    query: str
    results: List[SearchResult]
    summary: str
    topics: List[str]

class ContentExtractResponse(BaseModel):
    url: str
    content: str
    summary: str

# Tool functions
async def search_news(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Search for news articles using Tavily API.
    
    Args:
        query: Search query for finding news articles
        max_results: Maximum number of results to return
    
    Returns:
        List of search results with title, URL, content and relevance score
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TAVILY_API_KEY}"
    }
    
    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TAVILY_SEARCH_URL,
            headers=headers,
            json=payload,
            timeout=30.0
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Tavily API error: {response.text}"
            )
        
        data = response.json()
        results = []
        for result in data.get("results", []):
            results.append(SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0)
            ))
        return results

async def extract_content(urls: List[str]) -> List[ContentExtractResponse]:
    """
    Extract detailed content from URLs using Tavily API.
    
    Args:
        urls: List of URLs to extract content from
    
    Returns:
        List of extracted content with URL, raw content and summary
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TAVILY_API_KEY}"
    }
    
    payload = {
        "urls": urls
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TAVILY_EXTRACT_URL,
            headers=headers,
            json=payload,
            timeout=60.0
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Tavily API error: {response.text}"
            )
        
        data = response.json()
        results = []
        for result in data.get("results", []):
            results.append(ContentExtractResponse(
                url=result.get("url", ""),
                content=result.get("content", ""),
                summary=result.get("summary", "")
            ))
        return results

@app.post("/news", response_model=NewsResponse)
async def get_news(request: NewsQuery):
    """
    Get news articles and analysis based on the query.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    """
    try:
        # Use Toolflow to orchestrate the news search and analysis
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": f"Search for news about user query:'{request.query}' then extract content from the first 3 results and provide a summary with key topics."
            }],
            tools=[search_news, extract_content],
            response_format=NewsResponse,
            parallel_tool_execution=True
        )
        
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing news request: {str(e)}"
        )
    
    # test with curl
    # curl -X POST "http://localhost:8000/news" -H "Content-Type: application/json" -d '{"query": "latest news about AI", "max_results": 3}'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


