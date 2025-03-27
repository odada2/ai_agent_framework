"""
Bing Search API Provider Implementation

This module provides a robust implementation of the Microsoft Bing Search API with:
- Support for multiple search types (web, news, images)
- Response parsing and normalization
- Error handling and rate limiting
- Result scoring
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union, Literal
from urllib.parse import quote_plus

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import SearchProvider, SearchResult, RateLimitError, ApiKeyError, SearchError
from ..api_key_manager import ApiKeyManager
from ..utils import calculate_score

logger = logging.getLogger(__name__)


class BingSearchProvider(SearchProvider):
    """
    Bing Web Search API provider implementation.
    
    Features:
    - Support for multiple search types (web, news, images)
    - API key rotation and management
    - Rate limit handling
    - Result scoring
    """
    
    # API endpoints for different search types
    ENDPOINTS = {
        "web": "https://api.bing.microsoft.com/v7.0/search",
        "news": "https://api.bing.microsoft.com/v7.0/news/search",
        "images": "https://api.bing.microsoft.com/v7.0/images/search"
    }
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        search_type: Literal["web", "news", "images"] = "web",
        timeout: int = 10
    ):
        """
        Initialize the Bing Search provider.
        
        Args:
            api_keys: List of API keys (will check environment variables if not provided)
            search_type: Type of search to perform
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.search_type = search_type
        
        if search_type not in self.ENDPOINTS:
            logger.warning(f"Invalid search type '{search_type}'. Defaulting to 'web'.")
            self.search_type = "web"
            
        self.base_url = self.ENDPOINTS[self.search_type]
        self.session = None
        
        # Initialize API key manager
        self.key_manager = ApiKeyManager(
            provider="bing",
            keys=api_keys,
            auto_rotate=True
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session.
        
        Returns:
            aiohttp ClientSession
        """
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search(self, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute a search using the Bing Search API.
        
        Args:
            query: Search query
            num_results: Number of results to request (1-50)
            **kwargs: Additional parameters:
                - mkt: Market code (e.g., 'en-US')
                - safe_search: Safe search level ('off', 'moderate', 'strict')
                - freshness: Time period ('day', 'week', 'month')
                - site_restrict: Restrict to specific site
                
        Returns:
            Raw API response
            
        Raises:
            RateLimitError: When rate limit is exceeded
            ApiKeyError: When API key is invalid
            SearchError: For other search-related errors
        """
        session = await self._get_session()
        
        # Get an API key
        try:
            api_key = self.key_manager.get_key()
        except ApiKeyError as e:
            logger.error(f"API key error: {str(e)}")
            raise
        
        # Build URL parameters
        params = {
            "q": query,
            "count": min(num_results, 50)  # Bing allows up to 50 results per request
        }
        
        # Add site restriction if specified
        if site_restrict := kwargs.get("site_restrict"):
            params["q"] = f"{params['q']} site:{site_restrict}"
        
        # Add market parameter
        if mkt := kwargs.get("mkt"):
            params["mkt"] = mkt
        
        # Add safe search parameter
        if safe_search := kwargs.get("safe_search"):
            params["safeSearch"] = safe_search
        
        # Add freshness parameter
        if freshness := kwargs.get("freshness"):
            params["freshness"] = freshness
        
        # Add proper headers for Bing API
        headers = {
            "Ocp-Apim-Subscription-Key": api_key
        }
        
        try:
            logger.debug(f"Initiating Bing {self.search_type} search: {query}")
            start_time = time.time()
            
            async with session.get(
                self.base_url,
                params=params,
                headers=headers
            ) as response:
                # Handle rate limit response
                if response.status == 429:
                    # Get rate limit headers
                    retry_after = response.headers.get("Retry-After")
                    reset_time = None
                    
                    if retry_after:
                        try:
                            reset_time = time.time() + int(retry_after)
                        except ValueError:
                            reset_time = time.time() + 60  # Default: 1 minute
                    
                    # Report rate limit error to key manager
                    self.key_manager.report_error(
                        key=api_key,
                        is_rate_limit=True,
                        reset_time=reset_time
                    )
                    
                    retry_after_seconds = int(reset_time - time.time()) if reset_time else 60
                    raise RateLimitError(
                        provider="bing",
                        retry_after=retry_after_seconds
                    )
                
                # Handle authentication error
                if response.status == 401:
                    self.key_manager.report_error(
                        key=api_key,
                        is_authentication_error=True
                    )
                    raise ApiKeyError(f"Invalid API key for Bing Search")
                
                # Handle other errors
                if response.status != 200:
                    error_text = await response.text()
                    self.key_manager.report_error(key=api_key)
                    logger.error(f"Bing Search API error: {response.status} - {error_text}")
                    raise SearchError(f"API request failed: {error_text}")
                
                # Process successful response
                data = await response.json()
                
                # Report success to key manager
                self.key_manager.report_success(key=api_key)
                
                # Extract quota information if available
                if "X-MSEdge-ClientID" in response.headers:
                    client_id = response.headers["X-MSEdge-ClientID"]
                    if client_id:
                        # Store client ID for future requests
                        pass
                
                logger.debug(f"Bing search completed in {time.time() - start_time:.2f}s")
                return data
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Network-related errors
            self.key_manager.report_error(key=api_key)
            logger.error(f"Bing search request error: {str(e)}")
            raise SearchError(f"Request failed: {str(e)}")
            
        except Exception as e:
            # Unexpected errors
            if not isinstance(e, (RateLimitError, ApiKeyError, SearchError)):
                self.key_manager.report_error(key=api_key)
                logger.error(f"Bing search error: {str(e)}")
                raise SearchError(f"Search failed: {str(e)}")
            raise

    def parse_results(self, raw_data: Dict[str, Any], query: str) -> List[SearchResult]:
        """
        Parse Bing search results into standardized SearchResult objects.
        
        Args:
            raw_data: Raw API response
            query: Original search query
            
        Returns:
            List of SearchResult objects
        """
        results = []
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        
        # Process web search results
        if self.search_type == "web" and "webPages" in raw_data:
            for item in raw_data["webPages"].get("value", []):
                result = SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="bing",
                    metadata={
                        "display_url": item.get("displayUrl"),
                        "type": "web"
                    }
                )
                
                # Score and add result
                result = calculate_score(result, keywords)
                results.append(result)
        
        # Process news search results
        elif self.search_type == "news" and "value" in raw_data:
            for item in raw_data["value"]:
                result = SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    source="bing",
                    metadata={
                        "type": "news",
                        "published": item.get("datePublished"),
                        "provider": item.get("provider", [{}])[0].get("name", "")
                    }
                )
                
                # News is time-sensitive, so we boost fresh results
                result = calculate_score(result, keywords)
                
                # Add thumbnail if available
                if "image" in item and "thumbnail" in item["image"]:
                    result.metadata["image_url"] = item["image"]["thumbnail"].get("contentUrl")
                
                results.append(result)
        
        # Process image search results
        elif self.search_type == "images" and "value" in raw_data:
            for item in raw_data["value"]:
                result = SearchResult(
                    title=item.get("name", ""),
                    url=item.get("contentUrl", ""),
                    snippet=item.get("name", ""),  # Images don't have snippets, use name
                    source="bing",
                    metadata={
                        "type": "image",
                        "thumbnail_url": item.get("thumbnailUrl"),
                        "width": item.get("width"),
                        "height": item.get("height"),
                        "content_size": item.get("contentSize"),
                        "media_source": item.get("hostPageDisplayUrl")
                    }
                )
                
                result = calculate_score(result, keywords)
                results.append(result)
        
        # Sort by score (descending)
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    async def check_status(self) -> Dict[str, Any]:
        """
        Check the status of the Bing search provider.
        
        Returns:
            Status information
        """
        status = {
            "provider": "bing",
            "search_type": self.search_type,
            "operational": False,
            "active_keys": 0,
            "total_keys": 0
        }
        
        # Check if we have any keys
        if not hasattr(self.key_manager, 'keys') or not self.key_manager.keys:
            status["error"] = "No API keys configured"
            return status
        
        # Get key statistics
        status["total_keys"] = len(self.key_manager.keys)
        status["active_keys"] = len(self.key_manager.active_keys)
        
        if status["active_keys"] == 0:
            status["error"] = "No active API keys"
            return status
        
        # Test with a simple query
        try:
            await self.search("test query", num_results=1)
            status["operational"] = True
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()