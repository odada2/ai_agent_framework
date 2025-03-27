"""
Base Classes for Web Search Module

This module defines the foundation for the web search functionality, including:
- SearchResult: Standardized result format for all search providers
- SearchProvider: Abstract base interface for all search providers
- SearchError: Exceptions for search-related errors
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, validator


class SearchError(Exception):
    """Base exception for search-related errors"""
    pass


class RateLimitError(SearchError):
    """Raised when API rate limits are exceeded"""
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        self.provider = provider
        self.retry_after = retry_after
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(message)


class ApiKeyError(SearchError):
    """Raised when there are issues with API keys"""
    pass


class SearchResult(BaseModel):
    """Standardized search result model for unified result handling"""
    title: str
    url: str
    snippet: Optional[str] = None
    source: Literal["serper", "google", "bing"]
    timestamp: datetime = Field(default_factory=datetime.now)
    score: float = Field(0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('url')
    def validate_url(cls, v):
        """Validate that URLs have proper format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Invalid URL format")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime handling"""
        result = self.dict()
        result["timestamp"] = self.timestamp.isoformat()
        return result


class SearchProvider(ABC):
    """
    Abstract base class for all search providers.
    
    All provider implementations must inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute a search query and return raw results from the provider.
        
        Args:
            query: The search query
            num_results: Max number of results to return
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Raw provider-specific response data
            
        Raises:
            SearchError: For general search errors
            RateLimitError: When provider rate limits are exceeded
            ApiKeyError: When there are issues with API keys
        """
        pass
        
    @abstractmethod
    def parse_results(self, raw_data: Dict[str, Any], query: str) -> List[SearchResult]:
        """
        Parse raw provider data into standardized SearchResult objects.
        
        Args:
            raw_data: Raw data from the provider's API
            query: The original search query
            
        Returns:
            List of standardized SearchResult objects
        """
        pass
    
    @abstractmethod
    async def check_status(self) -> Dict[str, Any]:
        """
        Check the status of the provider, including:
        - API key validity
        - Rate limit status
        - Service availability
        
        Returns:
            Dictionary with status information
        """
        pass

    async def close(self):
        """Clean up resources (e.g., HTTP sessions)"""
        pass