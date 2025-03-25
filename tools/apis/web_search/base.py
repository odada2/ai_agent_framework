"""
Base Classes for Web Search Module

Defines:
- SearchResult: Standardized result format
- SearchProvider: Abstract base interface
"""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, validator

class SearchResult(BaseModel):
    """Standardized search result model"""
    title: str
    url: str
    snippet: Optional[str]
    source: Literal["serper", "google", "bing"]
    timestamp: datetime = Field(default_factory=datetime.now)
    score: float = Field(0.0, ge=0.0, le=1.0)

    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Invalid URL format")
        return v

class SearchProvider:
    """Abstract base class for search providers"""
    
    async def search(self, query: str, num_results: int) -> dict:
        """
        Args:
            query: Search term
            num_results: Max results to return
            
        Returns:
            Raw API response data
        """
        raise NotImplementedError
        
    def parse_results(self, raw_data: dict, query: str) -> list[SearchResult]:
        """
        Args:
            raw_data: Provider-specific API response
            query: Original search query
            
        Returns:
            List of standardized SearchResult objects
        """
        raise NotImplementedError
        
    async def close(self):
        """Clean up resources"""
        pass