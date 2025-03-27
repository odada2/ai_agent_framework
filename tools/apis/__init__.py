"""
Web Search Package

This package provides a comprehensive web search capability with:
- Multiple search providers (Serper, Google, Bing)
- Result caching and rate limiting
- API key management and rotation
- Standardized result format across providers
"""

from .web_search import WebSearchTool
from .base import SearchResult, SearchProvider, SearchError, RateLimitError, ApiKeyError
from .providers.serper import SerperProvider
from .providers.google import GoogleSearchProvider
from .providers.bing import BingSearchProvider
from .caching import get_cache
from .api_key_manager import ApiKeyManager

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
    'get_cache'
]