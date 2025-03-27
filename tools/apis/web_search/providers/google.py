"""
Google Custom Search API Provider Implementation

This module provides a robust implementation of the Google Custom Search API with:
- Support for Google Programmable Search Engine
- Response parsing and normalization
- Error handling and rate limiting
- Result scoring
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote_plus

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import SearchProvider, SearchResult, RateLimitError, ApiKeyError, SearchError
from ..api_key_manager import ApiKeyManager
from ..utils import calculate_score

logger = logging.getLogger(__name__)


class GoogleSearchProvider(SearchProvider):
    """
    Google Custom Search API provider implementation.
    
    Features:
    - Support for Google Programmable Search Engine (CSE)
    - API key rotation and management
    - Rate limit handling
    - Result scoring
    """
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        engine_id: Optional[str] = None,
        timeout: int = 10
    ):
        """
        Initialize the Google Search provider.
        
        Args:
            api_keys: List of API keys (will check environment variables if not provided)
            engine_id: Google Custom Search Engine ID (will check GOOGLE_CSE_ID env var if not provided)
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = None
        
        # Get engine ID
        self.engine_id = engine_id or os.environ.get("GOOGLE_CSE_ID") or os.environ.get("GOOGLE_ENGINE_ID")
        if not self.engine_id:
            logger.warning("No Google Custom Search Engine ID provided. Set GOOGLE_CSE_ID environment variable.")
        
        # Initialize API key manager
        self.key_manager = ApiKeyManager(
            provider="google",
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
        Execute a search using the Google Custom Search API.
        
        Args:
            query: Search query
            num_results: Number of results to request (1-10)
            **kwargs: Additional parameters:
                - gl: Country code (e.g., 'us')
                - lr: Language restriction (e.g., 'lang_en')
                - safe: Safe search ('active' or 'off')
                - site_restrict: Restrict to specific site
                
        Returns:
            Raw API response
            
        Raises:
            RateLimitError: When rate limit is exceeded
            ApiKeyError: When API key is invalid
            SearchError: For other search-related errors
        """
        if not self.engine_id:
            raise SearchError("Google Custom Search Engine ID is required")
        
        session = await self._get_session()
        
        # Get an API key
        try:
            api_key = self.key_manager.get_key()
        except ApiKeyError as e:
            logger.error(f"API key error: {str(e)}")
            raise
        
        # Build URL parameters
        params = {
            "key": api_key,
            "cx": self.engine_id,
            "q": query,
            "num": min(num_results, 10)
        }
        
        # Add optional parameters
        if gl := kwargs.get("gl"):
            params["gl"] = gl
        
        if lr := kwargs.get("lr"):
            params["lr"] = lr
        
        if safe := kwargs.get("safe"):
            params["safe"] = safe
            
        if site_restrict := kwargs.get("site_restrict"):
            if "q" in params:
                params["q"] = f"{params['q']} site:{site_restrict}"
        
        try:
            logger.debug(f"Initiating Google search: {query}")
            start_time = time.time()
            
            async with session.get(
                self.base_url,
                params=params
            ) as response:
                # Handle rate limit response
                if response.status == 429:
                    # Parse response to get quota details
                    error_data = await response.json()
                    error_info = error_data.get("error", {})
                    
                    # Get reset time if available, default to 100 seconds
                    reset_time = time.time() + 100
                    if 'details' in error_info:
                        for detail in error_info['details']:
                            if detail.get('reason') == 'quotaExceeded':
                                # Google usually resets quotas at midnight PST
                                # But we'll use a conservative 100-second default
                                pass
                    
                    # Report rate limit error to key manager
                    self.key_manager.report_error(
                        key=api_key,
                        is_rate_limit=True,
                        reset_time=reset_time
                    )
                    
                    retry_after = max(1, int(reset_time - time.time()))
                    raise RateLimitError(
                        provider="google",
                        retry_after=retry_after
                    )
                
                # Handle authentication error
                if response.status == 400:
                    error_data = await response.json()
                    error_info = error_data.get("error", {})
                    if error_info.get("status") == "INVALID_ARGUMENT":
                        if "API key not valid" in error_info.get("message", ""):
                            self.key_manager.report_error(
                                key=api_key,
                                is_authentication_error=True
                            )
                            raise ApiKeyError(f"Invalid API key for Google Custom Search")
                
                # Handle authentication error (different status)
                if response.status == 403:
                    error_data = await response.json()
                    self.key_manager.report_error(
                        key=api_key,
                        is_authentication_error=True
                    )
                    raise ApiKeyError(f"Authentication failed for Google Custom Search: {error_data.get('error', {}).get('message')}")
                
                # Handle other errors
                if response.status != 200:
                    error_text = await response.text()
                    self.key_manager.report_error(key=api_key)
                    logger.error(f"Google Search API error: {response.status} - {error_text}")
                    raise SearchError(f"API request failed: {error_text}")
                
                # Process successful response
                data = await response.json()
                
                # Report success to key manager
                self.key_manager.report_success(key=api_key)
                
                logger.debug(f"Google search completed in {time.time() - start_time:.2f}s")
                return data
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Network-related errors
            self.key_manager.report_error(key=api_key)
            logger.error(f"Google search request error: {str(e)}")
            raise SearchError(f"Request failed: {str(e)}")
            
        except Exception as e:
            # Unexpected errors
            if not isinstance(e, (RateLimitError, ApiKeyError, SearchError)):
                self.key_manager.report_error(key=api_key)
                logger.error(f"Google search error: {str(e)}")
                raise SearchError(f"Search failed: {str(e)}")
            raise

    def parse_results(self, raw_data: Dict[str, Any], query: str) -> List[SearchResult]:
        """
        Parse Google search results into standardized SearchResult objects.
        
        Args:
            raw_data: Raw API response
            query: Original search query
            
        Returns:
            List of SearchResult objects
        """
        results = []
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        
        # Process search items
        if "items" in raw_data:
            for item in raw_data["items"]:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google",
                    metadata={
                        "display_link": item.get("displayLink"),
                        "type": "organic"
                    }
                )
                
                # Add image data if available
                if "pagemap" in item and "cse_image" in item["pagemap"]:
                    if item["pagemap"]["cse_image"]:
                        result.metadata["image_url"] = item["pagemap"]["cse_image"][0].get("src")
                
                # Score and add result
                result = calculate_score(result, keywords)
                results.append(result)
        
        # Sort by score (descending)
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    async def check_status(self) -> Dict[str, Any]:
        """
        Check the status of the Google search provider.
        
        Returns:
            Status information
        """
        status = {
            "provider": "google",
            "operational": False,
            "active_keys": 0,
            "total_keys": 0,
            "has_engine_id": bool(self.engine_id)
        }
        
        # Check if we have engine ID
        if not self.engine_id:
            status["error"] = "No Custom Search Engine ID configured"
            return status
        
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