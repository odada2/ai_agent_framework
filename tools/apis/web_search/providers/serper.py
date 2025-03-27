"""
Serper.dev Search Provider Implementation

This module provides a robust implementation of the Serper.dev search API with:
- Proper rate limit handling
- Response parsing
- Error handling
- Result scoring
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import SearchProvider, SearchResult, RateLimitError, ApiKeyError
from ..api_key_manager import ApiKeyManager
from ..utils import calculate_score

logger = logging.getLogger(__name__)


class SerperProvider(SearchProvider):
    """
    Serper.dev search provider implementation.
    
    Features:
    - API key rotation
    - Rate limit handling
    - Result scoring and normalization
    - Structured error handling
    """
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        timeout: int = 10,
        base_url: str = "https://google.serper.dev/search"
    ):
        """
        Initialize the Serper provider.
        
        Args:
            api_keys: List of API keys (will check environment variables if not provided)
            timeout: Request timeout in seconds
            base_url: API endpoint URL
        """
        self.timeout = timeout
        self.base_url = base_url
        self.session = None
        
        # Initialize API key manager
        self.key_manager = ApiKeyManager(
            provider="serper",
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
        Execute a search using the Serper API.
        
        Args:
            query: Search query
            num_results: Number of results to request (1-10)
            **kwargs: Additional parameters:
                - gl: Country code (e.g., 'us')
                - hl: Language code (e.g., 'en')
                
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
        
        # Prepare request
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": min(num_results, 10)
        }
        
        # Add optional parameters
        if gl := kwargs.get("gl"):
            payload["gl"] = gl
        
        if hl := kwargs.get("hl"):
            payload["hl"] = hl
        
        try:
            logger.debug(f"Initiating Serper search: {query}")
            start_time = time.time()
            
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload
            ) as response:
                # Handle rate limit response
                if response.status == 429:
                    # Parse rate limit headers if available
                    reset_after = response.headers.get("X-RateLimit-Reset")
                    reset_time = None
                    if reset_after:
                        try:
                            reset_time = time.time() + int(reset_after)
                        except ValueError:
                            reset_time = time.time() + 60  # Default: 1 minute
                    
                    # Report rate limit error to key manager
                    self.key_manager.report_error(
                        key=api_key,
                        is_rate_limit=True,
                        reset_time=reset_time
                    )
                    
                    raise RateLimitError(
                        provider="serper",
                        retry_after=int(reset_time - time.time()) if reset_time else 60
                    )
                
                # Handle authentication error
                if response.status == 401 or response.status == 403:
                    self.key_manager.report_error(
                        key=api_key,
                        is_authentication_error=True
                    )
                    raise ApiKeyError(f"Invalid API key for Serper.dev")
                
                # Handle other errors
                if response.status != 200:
                    error_text = await response.text()
                    self.key_manager.report_error(key=api_key)
                    logger.error(f"Serper API error: {response.status} - {error_text}")
                    raise ValueError(f"API request failed: {error_text}")
                
                # Process successful response
                data = await response.json()
                
                # Extract rate limit information if available
                calls_remaining = None
                if "X-RateLimit-Remaining" in response.headers:
                    try:
                        calls_remaining = int(response.headers["X-RateLimit-Remaining"])
                    except ValueError:
                        pass
                
                # Report success to key manager
                self.key_manager.report_success(
                    key=api_key,
                    calls_remaining=calls_remaining
                )
                
                logger.debug(f"Serper search completed in {time.time() - start_time:.2f}s")
                return data
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Network-related errors
            self.key_manager.report_error(key=api_key)
            logger.error(f"Serper request error: {str(e)}")
            raise ValueError(f"Request failed: {str(e)}")
            
        except Exception as e:
            # Unexpected errors
            if not isinstance(e, (RateLimitError, ApiKeyError)):
                self.key_manager.report_error(key=api_key)
            logger.error(f"Serper search error: {str(e)}")
            raise

    def parse_results(self, raw_data: Dict[str, Any], query: str) -> List[SearchResult]:
        """
        Parse Serper results into standardized SearchResult objects.
        
        Args:
            raw_data: Raw API response
            query: Original search query
            
        Returns:
            List of SearchResult objects
        """
        results = []
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        
        # Process organic results
        if "organic" in raw_data:
            for item in raw_data["organic"]:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="serper",
                    metadata={
                        "position": item.get("position"),
                        "type": "organic"
                    }
                )
                
                # Score and add result
                result = calculate_score(result, keywords)
                results.append(result)
        
        # Process knowledge graph results if available
        if "knowledgeGraph" in raw_data:
            kg = raw_data["knowledgeGraph"]
            if "title" in kg and "link" in kg:
                result = SearchResult(
                    title=kg.get("title", ""),
                    url=kg.get("link", ""),
                    snippet=kg.get("description", ""),
                    source="serper",
                    metadata={
                        "type": "knowledge_graph",
                        "attributes": kg.get("attributes", {})
                    }
                )
                
                # Knowledge graph results get a scoring boost
                result = calculate_score(result, keywords)
                result.score += 0.2  # Boost knowledge graph results
                results.append(result)
        
        # Sort by score (descending)
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    async def check_status(self) -> Dict[str, Any]:
        """
        Check the status of the Serper provider.
        
        Returns:
            Status information
        """
        status = {
            "provider": "serper",
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