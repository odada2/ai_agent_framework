"""
Enhanced Web Search Tool

This module provides a robust web search tool that:
- Supports multiple search providers (Serper, Google, Bing)
- Implements caching, rate limiting, and fallbacks
- Provides standardized results across providers
- Includes domain filtering and result scoring
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any, Literal, Union, Set
from urllib.parse import urlparse

from ...core.tools.base import BaseTool
from ..web_search.base import SearchProvider, SearchResult, SearchError, RateLimitError, ApiKeyError
from ..web_search.providers.serper import SerperProvider
from ..web_search.providers.google import GoogleSearchProvider
from ..web_search.providers.bing import BingSearchProvider
from ..web_search.caching import get_cache, cached_search
from ..web_search.utils import sanitize_query, is_valid_domain, format_search_results

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Enhanced web search tool with multiple provider support.
    
    Features:
    - Multiple backend providers
    - Automatic fallback between providers
    - Result caching
    - Domain filtering
    - Result scoring and formatting
    """
    
    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the web for information. Supports filtering by domain and multiple search providers.",
        providers: Optional[List[str]] = None,
        default_provider: str = "serper",
        cache_ttl: int = 3600,
        max_results: int = 5,
        default_domain: Optional[str] = None,
        safe_search: bool = True,
        api_keys: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        """
        Initialize the web search tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            providers: List of providers to use ('serper', 'google', 'bing')
            default_provider: Default provider to use
            cache_ttl: Time-to-live for cache entries (seconds)
            max_results: Default number of results to return
            default_domain: Optional default domain to filter results
            safe_search: Whether to enable safe search
            api_keys: Optional dictionary of API keys by provider
            **kwargs: Additional provider-specific parameters
        """
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
                "provider": {
                    "type": "string",
                    "description": "Search provider to use ('serper', 'google', 'bing')"
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
                "description": "Basic web search",
                "parameters": {
                    "query": "latest advancements in quantum computing"
                }
            },
            {
                "description": "Search with domain restriction",
                "parameters": {
                    "query": "climate change research",
                    "domain": "science.gov",
                    "num_results": 3
                }
            },
            {
                "description": "Search with specific provider",
                "parameters": {
                    "query": "python programming tutorial",
                    "provider": "google",
                    "num_results": 5
                }
            }
        ]
        
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            examples=examples,
            required_permissions=["web_access"],
            **kwargs
        )
        
        # Initialize settings
        self.default_provider = default_provider
        self.cache_ttl = cache_ttl
        self.max_results = max_results
        self.default_domain = default_domain
        self.safe_search = safe_search
        
        # Initialize API keys
        self.api_keys = api_keys or {}
        
        # Initialize providers
        available_providers = set(['serper', 'google', 'bing'])
        self.provider_list = providers or ['serper', 'google', 'bing']
        
        # Validate providers
        self.provider_list = [p for p in self.provider_list if p in available_providers]
        if not self.provider_list:
            logger.warning("No valid search providers specified. Using default: serper")
            self.provider_list = ['serper']
        
        # Validate default provider
        if self.default_provider not in self.provider_list:
            self.default_provider = self.provider_list[0]
            
        # Initialize search providers
        self.providers: Dict[str, SearchProvider] = {}
        self._initialize_providers()
        
        # Initialize cache
        self.cache = get_cache({
            "ttl": self.cache_ttl,
            "namespace": "web_search"
        })
    
    def _initialize_providers(self) -> None:
        """Initialize configured search providers"""
        provider_classes = {
            "serper": SerperProvider,
            "google": GoogleSearchProvider,
            "bing": BingSearchProvider
        }
        
        for provider_name in self.provider_list:
            if provider_name not in provider_classes:
                logger.warning(f"Unknown provider: {provider_name}")
                continue
                
            provider_class = provider_classes[provider_name]
            
            try:
                # Get API keys for this provider
                api_keys = self.api_keys.get(provider_name)
                
                # Create provider instance
                if provider_name == "serper":
                    self.providers[provider_name] = provider_class(api_keys=api_keys)
                    
                elif provider_name == "google":
                    engine_id = os.environ.get("GOOGLE_CSE_ID")
                    self.providers[provider_name] = provider_class(
                        api_keys=api_keys,
                        engine_id=engine_id
                    )
                    
                elif provider_name == "bing":
                    self.providers[provider_name] = provider_class(
                        api_keys=api_keys,
                        search_type="web"
                    )
                
                logger.info(f"Initialized {provider_name} search provider")
                
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} provider: {str(e)}")
    
    async def _search_with_provider(
        self,
        provider_name: str,
        query: str,
        num_results: int,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute search with a specific provider.
        
        Args:
            provider_name: Name of the provider to use
            query: Search query
            num_results: Number of results to return
            domain: Optional domain filter
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Search results
            
        Raises:
            SearchError: If the search fails
        """
        if provider_name not in self.providers:
            raise SearchError(f"Provider '{provider_name}' not available")
            
        provider = self.providers[provider_name]
        
        # Add safe search parameter if enabled
        if self.safe_search:
            if provider_name == "google":
                kwargs["safe"] = "active"
            elif provider_name == "bing":
                kwargs["safe_search"] = "strict"
        
        # Add domain restriction if specified
        if domain:
            if provider_name in ["google", "bing"]:
                kwargs["site_restrict"] = domain
        
        # Execute search
        try:
            raw_results = await provider.search(query, num_results, **kwargs)
            results = provider.parse_results(raw_results, query)
            
            # Apply domain filter manually for providers that don't support it directly
            if domain and provider_name == "serper":
                results = [r for r in results if domain.lower() in r.url.lower()]
            
            return {
                "query": query,
                "provider": provider_name,
                "results": [r.to_dict() for r in results[:num_results]],
                "total_results": len(results),
                "domain_filter": domain,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error with {provider_name} provider: {str(e)}")
            raise
    
    @cached_search()
    async def _run(
        self,
        query: str,
        num_results: int = None,
        provider: str = None,
        domain: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a web search using available providers.
        
        Args:
            query: Search query
            num_results: Number of results to return
            provider: Specific provider to use
            domain: Optional domain filter
            **kwargs: Additional search parameters
            
        Returns:
            Search results dict
        """
        # Set up parameters
        sanitized_query = sanitize_query(query)
        num_results = min(num_results or self.max_results, 10)
        domain = domain or self.default_domain
        
        # Validate domain if provided
        if domain and not is_valid_domain(domain):
            return {
                "error": f"Invalid domain: {domain}",
                "query": query,
                "results": []
            }
        
        # Start with requested provider or default
        provider_name = provider if provider in self.providers else self.default_provider
        
        # Try initial provider
        try:
            return await self._search_with_provider(
                provider_name=provider_name,
                query=sanitized_query,
                num_results=num_results,
                domain=domain,
                **kwargs
            )
        except (ApiKeyError, RateLimitError) as e:
            # If specific provider was requested, fail immediately
            if provider:
                return {
                    "error": f"Error with provider '{provider}': {str(e)}",
                    "query": query,
                    "results": []
                }
            
            # Try fallback providers
            fallback_providers = [p for p in self.provider_list if p != provider_name]
            
            for fallback in fallback_providers:
                try:
                    result = await self._search_with_provider(
                        provider_name=fallback,
                        query=sanitized_query,
                        num_results=num_results,
                        domain=domain,
                        **kwargs
                    )
                    
                    # Mark as fallback
                    result["provider_fallback"] = True
                    result["original_provider"] = provider_name
                    return result
                    
                except Exception:
                    # Continue to next fallback provider
                    continue
            
            # If all providers failed
            return {
                "error": "All search providers failed",
                "query": query,
                "results": []
            }
            
        except Exception as e:
            # Log the error and return an error response
            logger.error(f"Search error: {str(e)}")
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": []
            }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Override the BaseTool execute method to provide additional formatting.
        
        Args:
            **kwargs: Search parameters
            
        Returns:
            Formatted search results
        """
        result = await super().execute(**kwargs)
        
        # Check if there was an error
        if "error" in result:
            return {
                "error": result["error"],
                "results": [],
                "formatted_results": f"Search error: {result['error']}"
            }
        
        # Convert back to SearchResult objects
        search_results = []
        
        for item in result.get("results", []):
            try:
                search_results.append(SearchResult(**item))
            except Exception as e:
                logger.warning(f"Failed to parse search result: {str(e)}")
        
        # Format the results
        formatted_results = format_search_results(
            search_results,
            result.get("query", ""),
            highlight=True
        )
        
        # Return enriched results
        return {
            **result,
            "formatted_results": formatted_results
        }
    
    async def check_status(self) -> Dict[str, Any]:
        """Check the status of all configured providers"""
        status = {
            "operational": False,
            "providers": {}
        }
        
        # Check each provider
        for name, provider in self.providers.items():
            try:
                provider_status = await provider.check_status()
                status["providers"][name] = provider_status
            except Exception as e:
                status["providers"][name] = {
                    "operational": False,
                    "error": str(e)
                }
        
        # Overall status is operational if at least one provider is working
        status["operational"] = any(
            provider.get("operational", False)
            for provider in status["providers"].values()
        )
        
        return status
    
    async def close(self) -> None:
        """Clean up resources"""
        tasks = []
        
        for provider in self.providers.values():
            tasks.append(provider.close())
            
        if tasks:
            await asyncio.gather(*tasks)