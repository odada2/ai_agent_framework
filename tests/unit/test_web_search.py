"""
Unit Tests for Web Search Implementation

This module provides tests for the WebSearchTool components, including
providers, caching, and API key management.
"""

import pytest
import os
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from ai_agent_framework.tools.apis.web_search.base import SearchResult, SearchError, RateLimitError, ApiKeyError
from ai_agent_framework.tools.apis.web_search.api_key_manager import ApiKeyStatus, ApiKeyManager
from ai_agent_framework.tools.apis.web_search.caching import SearchCache
from ai_agent_framework.tools.apis.web_search.utils import (
    sanitize_query, is_valid_domain, calculate_score, 
    format_search_results, normalize_url
)

# Mock aiohttp responses for testing providers
try:
    from ai_agent_framework.tools.apis.web_search.providers.serper import SerperProvider
    from ai_agent_framework.tools.apis.web_search.providers.google import GoogleSearchProvider
    from ai_agent_framework.tools.apis.web_search.providers.bing import BingSearchProvider
    
    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False

try:
    from ai_agent_framework.tools.apis.web_search.web_search import WebSearchTool
    WEBSEARCH_TOOL_AVAILABLE = True
except ImportError:
    WEBSEARCH_TOOL_AVAILABLE = False


# Example search results for testing
EXAMPLE_SEARCH_RESULTS = [
    SearchResult(
        title="Python Programming Language",
        url="https://www.python.org/",
        snippet="Python is a programming language that lets you work quickly and integrate systems more effectively.",
        source="serper",
        metadata={"type": "organic"}
    ),
    SearchResult(
        title="Learn Python - Free Interactive Python Tutorial",
        url="https://www.learnpython.org/",
        snippet="Learn Python, a powerful programming language used for web development, AI, data analysis, and more.",
        source="serper",
        metadata={"type": "organic"}
    )
]


# Tests for base functionality
class TestSearchBase:
    def test_search_result_creation(self):
        """Test creation of SearchResult objects"""
        result = SearchResult(
            title="Test Result",
            url="https://example.com",
            snippet="This is a test snippet",
            source="test"
        )
        
        assert result.title == "Test Result"
        assert result.url == "https://example.com"
        assert result.snippet == "This is a test snippet"
        assert result.source == "test"
        assert isinstance(result.timestamp, datetime)
        assert result.score == 0.0
    
    def test_search_result_validation(self):
        """Test URL validation in SearchResult"""
        # Valid URL should work
        valid_result = SearchResult(
            title="Valid",
            url="https://example.com",
            snippet="Valid URL",
            source="test"
        )
        assert valid_result.url == "https://example.com"
        
        # Invalid URL should raise ValueError
        with pytest.raises(ValueError):
            SearchResult(
                title="Invalid",
                url="not-a-url",
                snippet="Invalid URL",
                source="test"
            )


# Tests for API key management
class TestApiKeyManager:
    def test_api_key_status(self):
        """Test ApiKeyStatus class"""
        status = ApiKeyStatus(key="test-key", provider="test")
        
        assert status.key == "test-key"
        assert status.provider == "test"
        assert status.is_valid
        assert status.is_active
        assert status.error_count == 0
        
        # Test error tracking
        status.mark_error()
        assert status.error_count == 1
        assert status.is_active  # Still active after 1 error
        
        # Test rate limit tracking
        status.mark_error(is_rate_limit=True, reset_time=time.time() + 60)
        assert status.is_rate_limited()
        assert status.time_to_reset() <= 60
        
        # Reset errors
        status.reset_errors()
        assert status.error_count == 0
    
    @pytest.mark.asyncio
    async def test_api_key_rotation(self):
        """Test API key rotation"""
        manager = ApiKeyManager(
            provider="test",
            keys=["key1", "key2", "key3"]
        )
        
        # Initial key should be the first one
        assert manager.get_key() == "key1"
        
        # Report error and rate limit for key1
        manager.report_error(
            key="key1",
            is_rate_limit=True,
            reset_time=time.time() + 60
        )
        
        # Should switch to key2
        assert manager.get_key() == "key2"
        
        # Report authentication error for key2
        manager.report_error(
            key="key2",
            is_authentication_error=True
        )
        
        # Should switch to key3
        assert manager.get_key() == "key3"
        
        # Verify key statuses
        assert not manager.keys["key2"].is_valid  # Authentication error
        assert manager.keys["key1"].is_rate_limited()  # Rate limited


# Tests for utility functions
class TestWebSearchUtils:
    def test_sanitize_query(self):
        """Test query sanitization"""
        # Basic sanitization
        assert sanitize_query("normal query") == "normal query"
        
        # Strip dangerous characters
        assert sanitize_query("<script>alert('xss')</script>") == "scriptalertxssscript"
        
        # Limit length
        long_query = "a" * 300
        assert len(sanitize_query(long_query)) <= 256
    
    def test_is_valid_domain(self):
        """Test domain validation"""
        # Valid domains
        assert is_valid_domain("example.com")
        assert is_valid_domain("sub.example.co.uk")
        
        # Invalid domains
        assert not is_valid_domain("")
        assert not is_valid_domain("not a domain")
        assert not is_valid_domain("missing-tld")
    
    def test_calculate_score(self):
        """Test result scoring"""
        result = SearchResult(
            title="Python Programming",
            url="https://www.python.org/",
            snippet="Python is a programming language.",
            source="test"
        )
        
        # Score with relevant keywords
        scored = calculate_score(result, ["python", "programming"])
        assert scored.score > 0
        
        # Educational domains get a boost
        edu_result = SearchResult(
            title="Python Course",
            url="https://cs.stanford.edu/python",
            snippet="Learn Python programming",
            source="test"
        )
        edu_scored = calculate_score(edu_result, ["python"])
        assert edu_scored.score > scored.score
    
    def test_normalize_url(self):
        """Test URL normalization"""
        # Remove tracking parameters
        url = "https://example.com/page?utm_source=google&id=123&utm_medium=cpc"
        normalized = normalize_url(url)
        assert "utm_source" not in normalized
        assert "utm_medium" not in normalized
        assert "id=123" in normalized
        
        # Ensure protocol
        assert normalize_url("example.com").startswith("https://")


# Tests for search providers (if available)
@pytest.mark.skipif(not PROVIDERS_AVAILABLE, reason="Search providers not available")
class TestSearchProviders:
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_serper_provider(self, mock_post):
        """Test Serper provider with mocked response"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Python.org",
                    "link": "https://www.python.org/",
                    "snippet": "The official home of Python"
                }
            ]
        }
        mock_response.headers = {}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create provider with mocked key manager
        provider = SerperProvider()
        provider.key_manager = MagicMock()
        provider.key_manager.get_key.return_value = "test-key"
        
        # Perform search
        results = await provider.search("python", 1)
        assert "organic" in results
        
        # Parse results
        parsed = provider.parse_results(results, "python")
        assert len(parsed) == 1
        assert parsed[0].title == "Python.org"
        assert parsed[0].source == "serper"
    
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_serper_rate_limit(self, mock_post):
        """Test Serper provider rate limit handling"""
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"X-RateLimit-Reset": "60"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create provider with mocked key manager
        provider = SerperProvider()
        provider.key_manager = MagicMock()
        provider.key_manager.get_key.return_value = "test-key"
        
        # Perform search - should raise RateLimitError
        with pytest.raises(RateLimitError):
            await provider.search("python", 1)
        
        # Verify key manager was called to report error
        provider.key_manager.report_error.assert_called_once()


# Tests for WebSearchTool (if available)
@pytest.mark.skipif(not WEBSEARCH_TOOL_AVAILABLE, reason="WebSearchTool not available")
class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_websearch_tool_initialization(self):
        """Test WebSearchTool initialization"""
        # Create a simple WebSearchTool
        tool = WebSearchTool(name="test_search")
        
        assert tool.name == "test_search"
        assert tool.default_provider in tool.providers
    
    @pytest.mark.asyncio
    @patch("ai_agent_framework.tools.apis.web_search.web_search.WebSearchTool._search_with_provider")
    async def test_websearch_tool_execute(self, mock_search):
        """Test WebSearchTool execute method"""
        # Mock search response
        mock_search.return_value = {
            "query": "python",
            "provider": "test",
            "results": [
                {
                    "content": "Python is a programming language",
                    "url": "https://www.python.org/",
                    "score": 0.9
                }
            ]
        }
        
        # Create tool
        tool = WebSearchTool()
        
        # Execute search
        result = await tool.execute(query="python")
        
        # Check result
        assert "query" in result
        assert "results" in result
        assert "formatted_results" in result
        assert "python" in result["formatted_results"].lower()


@pytest.mark.skipif(not PROVIDERS_AVAILABLE, reason="Search providers not available")
class TestCaching:
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client"""
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None  # Default to cache miss
        redis_mock.setex.return_value = True
        return redis_mock
    
    def test_search_cache_initialization(self, mock_redis):
        """Test SearchCache initialization"""
        with patch("redis.Redis", return_value=mock_redis):
            cache = SearchCache(enabled=True)
            assert cache.enabled
            assert cache.redis is not None
    
    @pytest.mark.asyncio
    async def test_cache_hit_miss(self, mock_redis):
        """Test cache hits and misses"""
        with patch("redis.Redis", return_value=mock_redis):
            cache = SearchCache(enabled=True)
            
            # Test cache miss
            assert await cache.get("test_query") is None
            assert cache._stats["misses"] == 1
            
            # Setup cache hit
            cached_data = {"results": ["test"]}
            mock_redis.get.return_value = json.dumps(cached_data).encode()
            
            # Test cache hit
            result = await cache.get("test_query")
            assert result is not None
            assert "results" in result
            assert cache._stats["hits"] == 1