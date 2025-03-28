# ai_agent_framework/tools/apis/__init__.py

"""
APIs Tools Package

This package provides tools for interacting with external APIs.
"""

# Import from the web_search subdirectory structure
from .web_search import (
    WebSearchTool,
    SearchResult,
    SearchProvider,
    SearchError,
    RateLimitError,
    ApiKeyError,
    SerperProvider,
    GoogleSearchProvider,
    BingSearchProvider,
    ApiKeyManager,
    get_cache
)

# Import the new APIConnectorTool
from .connector import APIConnectorTool

__all__ = [
    # Web Search related
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
    'get_cache',

    # General API Connector
    'APIConnectorTool',
]