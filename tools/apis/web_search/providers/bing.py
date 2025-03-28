"""
Bing Search API Provider Implementation

This module provides a robust implementation of the Microsoft Bing Search API with:
- Support for multiple search types (web, news, images)
- Comprehensive response parsing and normalization
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
from datetime import datetime

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
    - Comprehensive result parsing
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
        
        # Retry with client ID if available
        if client_id := kwargs.get("client_id"):
            headers["X-MSEdge-ClientID"] = client_id
        
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
                
                # Handle quota exceeded - could come with a 403 status
                if response.status == 403:
                    error_text = await response.text()
                    if "quota" in error_text.lower() or "limit" in error_text.lower():
                        self.key_manager.report_error(
                            key=api_key,
                            is_rate_limit=True,
                            reset_time=time.time() + 3600  # Default to 1 hour for quota limits
                        )
                        raise RateLimitError(
                            provider="bing",
                            retry_after=3600
                        )
                    else:
                        self.key_manager.report_error(key=api_key)
                        raise SearchError(f"API access forbidden: {error_text}")
                
                # Handle other errors
                if response.status != 200:
                    error_text = await response.text()
                    self.key_manager.report_error(key=api_key)
                    logger.error(f"Bing Search API error: {response.status} - {error_text}")
                    raise SearchError(f"API request failed: {error_text}")
                
                # Process successful response
                data = await response.json()
                
                # Check for errors in response body (even with 200 status)
                if "errors" in data or "error" in data:
                    error_info = data.get("errors", []) or [data.get("error", {})]
                    error_msg = "; ".join(str(e.get("message", "Unknown error")) for e in error_info)
                    
                    # Check for quota or rate limit errors
                    if any("quota" in str(e).lower() for e in error_info) or any("rate" in str(e).lower() for e in error_info):
                        self.key_manager.report_error(
                            key=api_key,
                            is_rate_limit=True
                        )
                        raise RateLimitError(provider="bing", retry_after=3600)
                    
                    self.key_manager.report_error(key=api_key)
                    raise SearchError(f"API error: {error_msg}")
                
                # Extract client ID for future requests
                client_id = response.headers.get("X-MSEdge-ClientID")
                if client_id:
                    # Store client ID for future requests
                    data["_client_id"] = client_id
                
                # Report success to key manager
                self.key_manager.report_success(key=api_key)
                
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
                        "type": "web",
                        "id": item.get("id")
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
                        "provider": item.get("provider", [{}])[0].get("name", ""),
                        "category": item.get("category", ""),
                        "id": item.get("id")
                    }
                )
                
                # News is time-sensitive, so we boost fresh results
                result = calculate_score(result, keywords)
                
                # Boost score for more recent news
                if "datePublished" in item:
                    try:
                        pub_date = datetime.fromisoformat(item["datePublished"].replace('Z', '+00:00'))
                        days_old = (datetime.now().replace(tzinfo=None) - pub_date.replace(tzinfo=None)).days
                        if days_old < 1:
                            result.score += 0.3  # Today's news
                        elif days_old < 2:
                            result.score += 0.2  # Yesterday's news
                        elif days_old < 7:
                            result.score += 0.1  # This week's news
                    except (ValueError, AttributeError):
                        # Ignore date parsing errors
                        pass
                
                # Add thumbnail if available
                if "image" in item:
                    if "thumbnail" in item["image"]:
                        result.metadata["image_url"] = item["image"]["thumbnail"].get("contentUrl")
                    elif "contentUrl" in item["image"]:
                        result.metadata["image_url"] = item["image"].get("contentUrl")
                
                results.append(result)
        
        # Process image search results
        elif self.search_type == "images" and "value" in raw_data:
            for item in raw_data["value"]:
                # For images, we need both the image URL and hosting page URL
                image_url = item.get("contentUrl", "")
                hosting_url = item.get("hostPageUrl", "")
                
                # Prefer the hosting URL for primary result.url if available
                url = hosting_url if hosting_url else image_url
                
                result = SearchResult(
                    title=item.get("name", ""),
                    url=url,  # Use hosting page URL as primary
                    snippet=item.get("name", ""),  # Images don't have snippets, use name
                    source="bing",
                    metadata={
                        "type": "image",
                        "image_url": image_url,  # Actual image URL
                        "thumbnail_url": item.get("thumbnailUrl"),
                        "width": item.get("width"),
                        "height": item.get("height"),
                        "content_size": item.get("contentSize"),
                        "media_source": item.get("hostPageDisplayUrl"),
                        "encoding_format": item.get("encodingFormat", ""),
                        "accentColor": item.get("accentColor")
                    }
                )
                
                # Score with emphasis on image specifics
                result = calculate_score(result, keywords)
                
                # Boost high-resolution images
                if item.get("width", 0) > 1000 and item.get("height", 0) > 1000:
                    result.score += 0.1
                
                results.append(result)
                
        # Process related searches (can be present in any search type)
        if "relatedSearches" in raw_data and raw_data["relatedSearches"].get("value"):
            # Add some related searches as results with lower scores
            for item in raw_data["relatedSearches"].get("value", [])[:2]:  # Limit to 2 related searches
                related_query = item.get("text", "")
                url = item.get("webSearchUrl", "")
                
                if related_query and url:
                    result = SearchResult(
                        title=f"Related: {related_query}",
                        url=url,
                        snippet=f"Related search: {related_query}",
                        source="bing",
                        metadata={
                            "type": "related_search",
                            "original_query": query
                        }
                    )
                    
                    # Lower score for related searches
                    result.score = 0.3
                    results.append(result)
                    
        # Handle empty results with helpful metadata
        if not results:
            # Check if we have specific info on why no results were found
            if "rankingResponse" in raw_data:
                ranking = raw_data["rankingResponse"]
                if "mainline" in ranking and not ranking["mainline"].get("items", []):
                    logger.info(f"No results found for query: {query}")
            
            logger.debug(f"No parseable results for query: {query}")
        
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