# ai_agent_framework/tools/apis/web_search/__init__.py

"""
Web Search Module Entry Point

Exports key components for web searching, including the tool, providers,
base classes, exceptions, API key manager, and caching utilities.
"""

# Import the main tool class
from .web_search import WebSearchTool

# Import base classes and exceptions
from .base import SearchResult, SearchProvider, SearchError, RateLimitError, ApiKeyError

# Import concrete provider implementations
from .providers import SerperProvider, GoogleSearchProvider, BingSearchProvider

# Import the API key manager
from .api_key_manager import ApiKeyManager

# Import caching utilities
from .caching import configure_aiocache, get_cache

# Define public exports for the package
__all__ = [
    'WebSearchTool',
    'SearchResult',
    'SearchProvider',
    'SearchError',
    'RateLimitError',
    'ApiKeyError',
    'SerperProvider',
    'GoogleSearchProvider',
    'BingSearchProvider',
    'ApiKeyManager',
    'configure_aiocache', # Export configuration function
    'get_cache',        # Export function to get cache instance
]