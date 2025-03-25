from .web_search import WebSearchTool
"""
Web Search Module Entry Point

Exports:
- WebSearchTool: Main interface
- SearchError: Exception class
"""

from .web_search import WebSearchTool
from .providers import SerperProvider, GoogleProvider, BingProvider
from .base import SearchResult

__all__ = [
    'WebSearchTool',
    'SearchResult',
    'SerperProvider',
    'GoogleProvider',
    'BingProvider'
]

class SearchError(Exception):
    """Base exception for search-related errors"""
    pass