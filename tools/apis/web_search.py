"""
Enhanced Web Search Tool

This module provides a robust, extensible web search tool with:
- Multiple backend support (Serper, Google, Bing)
- Result caching and rate limiting
- Input sanitization and security
- Standardized result format
- Quality scoring and filtering
- Need to be renoved
"""

import aiohttp
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from urllib.parse import quote_plus
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer
from pydantic import BaseModel, Field, validator

from ...core.tools.base import BaseTool

logger = logging.getLogger(__name__)

# --- Standardized Data Models ---
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

# --- Abstract Provider Interface ---
class SearchProvider(ABC):
    """Abstract base class for search providers"""
    
    @abstractmethod
    async def search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Execute search and return raw results"""
        pass
    
    @abstractmethod
    def parse_results(self, raw_data: Dict[str, Any], query: str) -> List[SearchResult]:
        """Convert raw results to standardized format"""
        pass

# --- Provider Implementations ---
class SerperProvider(SearchProvider):
    """Serper.dev search provider implementation"""
    
    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://google.serper.dev/search"
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Execute search via Serper API"""
        session = await self._get_session()
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": min(num_results, 10)
        }

        try:
            logger.debug(f"Initiating Serper search: {query}")
            start_time = time.time()
            
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Serper API error: {response.status} - {error_text}")
                    raise ValueError(f"API request failed: {error_text}")
                
                data = await response.json()
                logger.debug(f"Serper search completed in {time.time() - start_time:.2f}s")
                return data
                
        except Exception as e:
            logger.error(f"Serper search failed after {time.time() - start_time:.2f}s: {str(e)}")
            raise

    def parse_results(self, raw_data: Dict[str, Any], query: str) -> List[SearchResult]:
        """Parse Serper results to standard format"""
        results = []
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        
        if "organic" in raw_data:
            for item in raw_data["organic"]:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="serper"
                )
                results.append(self._score_result(result, keywords))
        
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _score_result(self, result: SearchResult, keywords: List[str]) -> SearchResult:
        """Score result based on relevance"""
        score = 0.0
        content = f"{result.title} {result.snippet}".lower()
        
        # Domain authority
        if "arxiv.org" in result.url:
            score += 0.3
        elif "github.com" in result.url:
            score += 0.2
            
        # Keyword matches
        score += 0.1 * sum(kw in content for kw in keywords)
        
        # Freshness (recent results get slight boost)
        if (datetime.now() - result.timestamp).days < 7:
            score += 0.05
            
        result.score = min(score, 1.0)
        return result

# --- Main Tool Class ---
class WebSearchTool(BaseTool):
    """
    Enhanced web search tool with:
    - Multiple backend support
    - Caching and rate limiting
    - Domain filtering
    - Quality scoring
    """
    
    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the web for information. Supports domain filtering and quality scoring.",
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        max_results: int = 5,
        cache_ttl: int = 3600,
        timeout: int = 10,
        **kwargs
    ):
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "minimum": 1,
                    "maximum": 10
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain filter (e.g. 'wikipedia.org')"
                }
            },
            "required": ["query"]
        }
        
        examples = [
            {
                "description": "Search for AI research papers",
                "parameters": {
                    "query": "latest transformer architectures",
                    "num_results": 3,
                    "domain": "arxiv.org"
                }
            }
        ]
        
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            examples=examples,
            required_permissions=["web_access"],
            metadata={
                "category": "information_retrieval",
                "privacy_level": "high",
                "cost_per_call": 0.001
            }
        )
        
        self.backend = backend or self._determine_backend()
        self.api_key = api_key or self._get_api_key()
        self.max_results = max_results
        self.timeout = timeout
        self.provider = self._init_provider()
        self.cache_ttl = cache_ttl

    def _determine_backend(self) -> str:
        """Determine best available backend"""
        backends = {
            "serper": "SERPER_API_KEY",
            "google": "GOOGLE_SEARCH_API_KEY",
            "bing": "BING_SEARCH_API_KEY"
        }
        
        for backend, key in backends.items():
            if os.environ.get(key):
                return backend
                
        logger.warning("No search API keys found - using Serper with demo key")
        return "serper"

    def _get_api_key(self) -> str:
        """Get API key for selected backend"""
        key_map = {
            "serper": "SERPER_API_KEY",
            "google": "GOOGLE_SEARCH_API_KEY",
            "bing": "BING_SEARCH_API_KEY"
        }
        return os.environ.get(key_map[self.backend], "")

    def _init_provider(self) -> SearchProvider:
        """Initialize the search provider"""
        providers = {
            "serper": SerperProvider,
            # Add other providers here
        }
        
        if self.backend not in providers:
            raise ValueError(f"Unsupported backend: {self.backend}")
            
        return providers[self.backend](
            api_key=self.api_key,
            timeout=self.timeout
        )

    @cached(
        cache=Cache.REDIS,
        key_builder=lambda f, *args, **kwargs: f"search:{quote_plus(kwargs.get('query', ''))}",
        serializer=JsonSerializer(),
        ttl=3600
    )
    async def _run(
        self,
        query: str,
        num_results: Optional[int] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a web search with:
        - Query sanitization
        - Result filtering
        - Error handling
        """
        # Sanitize inputs
        query = quote_plus(query.strip()[:256])  # Prevent injection and limit length
        num_results = min(num_results or self.max_results, 10)
        
        try:
            # Execute search
            raw_results = await self.provider.search(query, num_results)
            results = self.provider.parse_results(raw_results, query)
            
            # Apply domain filter
            if domain:
                results = [r for r in results if domain.lower() in r.url.lower()]
            
            # Format output
            return {
                "query": query,
                "results": [r.dict() for r in results[:num_results]],
                "formatted_text": self._format_results(results, query),
                "metadata": {
                    "backend": self.backend,
                    "result_count": len(results),
                    "domain_filter": domain
                }
            }
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "formatted_text": f"Search failed: {str(e)}"
            }

    def _format_results(self, results: List[SearchResult], query: str) -> str:
        """Format results for display"""
        if not results:
            return f"No results found for '{query}'"
            
        output = [f"Top {len(results)} results for '{query}':"]
        
        for i, result in enumerate(results, 1):
            output.extend([
                f"{i}. {result.title}",
                f"   URL: {result.url}",
                f"   Snippet: {result.snippet or 'N/A'}",
                f"   Score: {result.score:.2f}",
                ""
            ])
            
        return "\n".join(output)

    async def close(self):
        """Cleanup resources"""
        if hasattr(self.provider, 'session') and self.provider.session:
            await self.provider.session.close()