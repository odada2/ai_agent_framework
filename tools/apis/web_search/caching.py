"""
Caching Configuration for Web Searches

Features:
- Redis cache backend
- JSON serialization
- Custom key generation
"""

from aiocache import Cache
from aiocache.serializers import JsonSerializer

cache_config = {
    "cache": Cache.REDIS,
    "serializer": JsonSerializer(),
    "key_builder": lambda f, *args, **kwargs: (
        f"search:{kwargs.get('query', '').lower()}"
        f":{kwargs.get('domain', '').lower()}"
    )
}

def get_cache_stats() -> dict:
    """Returns cache utilization metrics"""
    # Implementation would connect to Redis
    return {
        "hits": 0,
        "misses": 0,
        "size": 0
    }