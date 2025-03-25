"""
Enhanced Web Search Tool (Modular Version)

Key Features:
- Standardized interface for multiple search providers
- Built-in caching and rate limiting
- Domain filtering and result scoring
- Async/await support
"""

import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus
from datetime import datetime

from aiocache import cached, Cache
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import SearchResult, SearchProvider
from .caching import cache_config
from .utils import sanitize_query, calculate_score

logger = logging.getLogger(__name__)

class WebSearchTool:
    """
    Main web search tool class that coordinates:
    - Query processing
    - Provider selection
    - Result formatting
    """

    def __init__(
        self,
        provider: SearchProvider,
        cache_ttl: int = 3600,
        max_results: int = 5,
        default_domain: Optional[str] = None
    ):
        """
        Args:
            provider: Initialized search provider instance
            cache_ttl: Cache lifetime in seconds
            max_results: Default number of results
            default_domain: Optional domain filter
        """
        self.provider = provider
        self.cache_ttl = cache_ttl
        self.max_results = min(max_results, 10)
        self.default_domain = default_domain

    @cached(
        **cache_config,
        ttl=cache_ttl,
        key_builder=lambda f, *args, **kwargs: f"search:{quote_plus(kwargs.get('query', ''))}"
    )
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a web search with:
        - Automatic query sanitization
        - Domain filtering
        - Result scoring
        
        Returns:
            {
                "query": str,
                "results": List[SearchResult],
                "metadata": {
                    "provider": str,
                    "cache_hit": bool,
                    "domain_filter": Optional[str]
                }
            }
        """
        # Sanitize and validate inputs
        query = sanitize_query(query)
        num_results = min(num_results or self.max_results, 10)
        domain = domain or self.default_domain

        try:
            # Execute search via provider
            raw_results = await self.provider.search(query, num_results)
            results = self.provider.parse_results(raw_results, query)
            
            # Apply domain filter if specified
            if domain:
                results = [r for r in results if domain.lower() in r.url.lower()]
            
            # Score and sort results
            scored_results = sorted(
                [calculate_score(r) for r in results],
                key=lambda x: x.score,
                reverse=True
            )[:num_results]

            return {
                "query": query,
                "results": scored_results,
                "metadata": {
                    "provider": self.provider.__class__.__name__,
                    "domain_filter": domain,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "metadata": {
                    "provider": self.provider.__class__.__name__,
                    "failed_at": datetime.utcnow().isoformat()
                }
            }

    async def close(self):
        """Clean up provider resources"""
        if hasattr(self.provider, 'close'):
            await self.provider.close()

    def __repr__(self):
        return f"WebSearchTool(provider={self.provider.__class__.__name__}, cache_ttl={self.cache_ttl})"