"""
Search Provider Implementations

This package contains implementations for various search providers used by the 
WebSearchTool. Each provider connects to a different search API and implements
the common SearchProvider interface.

Available providers:
- SerperProvider: Interface to Serper.dev Google Search API
  * Provides Google search results via a simple REST API
  * Handles rate limits and result parsing
  * Performs well for standard web searches

- GoogleSearchProvider: Interface to Google Custom Search API
  * Requires a Google API key and Custom Search Engine ID
  * Provides more customization options than Serper
  * Great for specific domain searching

- BingSearchProvider: Interface to Bing Web Search API
  * Supports multiple search types (web, news, images)
  * Often provides different results than Google-based providers
  * Useful as a fallback or for diverse results

Usage:
    from ai_agent_framework.tools.apis.web_search.providers import (
        SerperProvider,
        GoogleSearchProvider,
        BingSearchProvider
    )
    
    # Initialize a provider
    provider = SerperProvider(api_keys=["your_key_here"])
    
    # Perform a search
    results = await provider.search("your query", num_results=5)
    
    # Parse the results
    parsed_results = provider.parse_results(results, "your query")
"""

from .serper import SerperProvider
from .google import GoogleSearchProvider
from .bing import BingSearchProvider

__all__ = [
    'SerperProvider',
    'GoogleSearchProvider',
    'BingSearchProvider'
]