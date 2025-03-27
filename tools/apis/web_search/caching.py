"""
Caching Configuration for Web Searches

This module provides a Redis-based caching system for search results with:
- Configurable TTL (Time-To-Live)
- JSON serialization for complex objects
- Custom key generation based on query parameters
- Cache statistics tracking
"""

import json
import logging
import hashlib
from typing import Any, Callable, Dict, Optional
from functools import wraps
from datetime import datetime

import redis
from aiocache import Cache, RedisCache
from aiocache.serializers import JsonSerializer
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Default Redis configuration
DEFAULT_REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None,
    "ttl": 3600,  # 1 hour default TTL
    "namespace": "search_cache",
    "key_builder": lambda f, *args, **kwargs: (
        f"search:{kwargs.get('query', '').lower()}"
        f":{kwargs.get('num_results', '5')}"
        f":{kwargs.get('domain', '')}"
    )
}


class SearchCache:
    """
    Manages caching for web search results with Redis backend.
    
    Features:
    - Automatic serialization/deserialization of complex objects
    - TTL-based expiration
    - Namespace support to avoid key collisions
    - Cache statistics tracking
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        host: str = DEFAULT_REDIS_CONFIG["host"],
        port: int = DEFAULT_REDIS_CONFIG["port"],
        db: int = DEFAULT_REDIS_CONFIG["db"],
        password: Optional[str] = DEFAULT_REDIS_CONFIG["password"],
        ttl: int = DEFAULT_REDIS_CONFIG["ttl"],
        namespace: str = DEFAULT_REDIS_CONFIG["namespace"],
        serializer: Any = JsonSerializer(),
        enabled: bool = True
    ):
        """
        Initialize the search cache.
        
        Args:
            redis_url: Redis connection string (overrides other connection params if provided)
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            ttl: Default TTL in seconds
            namespace: Namespace prefix for keys
            serializer: Serializer for cache values
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.ttl = ttl
        self.namespace = namespace
        self.serializer = serializer
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "stores": 0
        }
        
        if not enabled:
            logger.info("Search caching is disabled")
            return
        
        try:
            # Setup Redis client
            if redis_url:
                self.redis = redis.from_url(redis_url)
            else:
                self.redis = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False  # We handle serialization ourselves
                )
            
            # Setup aiocache for decorator usage
            self.aiocache = RedisCache(
                endpoint=host,
                port=port,
                db=db,
                password=password,
                namespace=namespace,
                serializer=serializer,
                ttl=ttl
            )
            
            # Test connection
            self.redis.ping()
            logger.info(f"Connected to Redis cache at {host}:{port}/{db}")
        except (RedisError, ConnectionError) as e:
            logger.warning(f"Failed to connect to Redis: {str(e)}. Caching will be disabled.")
            self.enabled = False
            self.redis = None
            self.aiocache = None
    
    def _build_key(self, query: str, **kwargs) -> str:
        """
        Build a cache key from the query and other parameters.
        
        Args:
            query: Search query
            **kwargs: Additional parameters to include in the key
            
        Returns:
            Formatted cache key
        """
        # Create a deterministic string from all parameters
        params = {
            "q": query.lower().strip(),
            **{k: v for k, v in kwargs.items() if v is not None}
        }
        
        # Create a sorted, stable representation
        param_str = json.dumps(params, sort_keys=True)
        
        # Use a hash for potentially long queries
        key_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{self.namespace}:{key_hash}"
    
    async def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get a value from the cache.
        
        Args:
            query: Search query
            **kwargs: Additional parameters that were part of the original cache key
            
        Returns:
            Cached result or None if not found
        """
        if not self.enabled or not self.redis:
            return None
            
        key = self._build_key(query, **kwargs)
        
        try:
            raw_data = self.redis.get(key)
            if raw_data:
                self._stats["hits"] += 1
                result = self.serializer.loads(raw_data)
                
                # Add cache metadata
                if isinstance(result, dict):
                    if "metadata" not in result:
                        result["metadata"] = {}
                    result["metadata"]["cache_hit"] = True
                    result["metadata"]["cache_time"] = datetime.now().isoformat()
                
                return result
            else:
                self._stats["misses"] += 1
                return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set(self, query: str, data: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """
        Store a value in the cache.
        
        Args:
            query: Search query
            data: Data to cache
            ttl: Optional custom TTL (overrides default)
            **kwargs: Additional parameters to include in the cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis:
            return False
            
        key = self._build_key(query, **kwargs)
        ttl = ttl or self.ttl
        
        try:
            # Add cache metadata
            if isinstance(data, dict):
                if "metadata" not in data:
                    data["metadata"] = {}
                data["metadata"]["cached_at"] = datetime.now().isoformat()
            
            # Serialize and store
            serialized = self.serializer.dumps(data)
            result = self.redis.setex(key, ttl, serialized)
            
            if result:
                self._stats["stores"] += 1
            
            return bool(result)
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache set error: {str(e)}")
            return False
    
    async def invalidate(self, query: str, **kwargs) -> bool:
        """
        Remove a value from the cache.
        
        Args:
            query: Search query
            **kwargs: Additional parameters that were part of the original cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis:
            return False
            
        key = self._build_key(query, **kwargs)
        
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Cache invalidate error: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache usage statistics"""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset cache statistics counters"""
        for key in self._stats:
            self._stats[key] = 0
    
    async def clear(self, pattern: str = "*") -> int:
        """
        Clear cache entries matching the pattern.
        
        Args:
            pattern: Key pattern to match (default: all keys in namespace)
            
        Returns:
            Number of keys removed
        """
        if not self.enabled or not self.redis:
            return 0
            
        try:
            keys = self.redis.keys(f"{self.namespace}:{pattern}")
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return 0
    
    def cache_decorator(self, ttl: Optional[int] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Optional TTL override
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract query from kwargs or first positional arg
                query = kwargs.get("query")
                if query is None and len(args) > 1:
                    query = args[1]  # Assuming self is args[0]
                
                if not query or not self.enabled:
                    return await func(*args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(query, **kwargs)
                if cached_result is not None:
                    return cached_result
                
                # Call the function
                result = await func(*args, **kwargs)
                
                # Store in cache
                await self.set(query, result, ttl=ttl, **kwargs)
                
                return result
            return wrapper
        return decorator


# Initialize a global cache instance
search_cache = None


def get_cache(config: Optional[Dict[str, Any]] = None) -> SearchCache:
    """
    Get or create the global search cache instance.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured SearchCache instance
    """
    global search_cache
    
    if search_cache is None:
        # Apply config overrides
        settings = DEFAULT_REDIS_CONFIG.copy()
        if config:
            settings.update(config)
        
        # Create cache instance
        search_cache = SearchCache(**settings)
    
    return search_cache


def cached_search(ttl: Optional[int] = None):
    """
    Decorator for caching search results.
    
    Args:
        ttl: Optional TTL override
        
    Returns:
        Decorated function
    """
    cache = get_cache()
    return cache.cache_decorator(ttl=ttl)