# ai_agent_framework/tools/apis/web_search/caching.py

"""
Web Search Caching Configuration using aiocache

This module configures the caching mechanism (using the aiocache library)
for the WebSearchTool based on application settings. It allows switching
between different cache backends like memory or Redis.
"""

import logging
from typing import Optional, Dict, Any

from aiocache import caches, Cache
from aiocache.serializers import JsonSerializer, PickleSerializer

# Assuming Settings class is available for loading configuration
# Adjust import path as necessary for your project structure
from ....config.settings import Settings

logger = logging.getLogger(__name__)

# Flag to ensure configuration happens only once
_cache_configured = False

def configure_aiocache(settings: Optional[Settings] = None) -> None:
    """
    Configures the global aiocache settings based on application configuration.

    Reads settings for cache type, TTL, Redis connection details (if applicable),
    and serializer.

    Args:
        settings: An optional Settings instance. If None, a new one is created.
    """
    global _cache_configured
    if _cache_configured:
        logger.debug("aiocache already configured. Skipping.")
        return

    if settings is None:
        try:
            settings = Settings() # Load settings if not provided
        except Exception as e:
            logger.error(f"Failed to load settings for cache configuration: {e}. Using defaults.")
            settings = None # Proceed with defaults

    # Determine cache configuration from settings or use defaults
    cache_config: Dict[str, Any] = {
        'default': {
            'cache': "aiocache.SimpleMemoryCache", # Default to in-memory
            'serializer': {
                'class': "aiocache.serializers.PickleSerializer"
            },
            'ttl': 3600 # Default TTL: 1 hour
        }
    }

    if settings:
        cache_type = settings.get("cache.web_search.type", "memory").lower()
        default_ttl = settings.get("cache.web_search.ttl_seconds", 3600)
        serializer_type = settings.get("cache.web_search.serializer", "pickle").lower()

        # Choose serializer
        if serializer_type == "json":
            serializer_class = "aiocache.serializers.JsonSerializer"
        else: # Default to pickle
            serializer_class = "aiocache.serializers.PickleSerializer"

        cache_settings = {
             'ttl': default_ttl,
             'serializer': {'class': serializer_class}
        }

        if cache_type == "redis":
            redis_host = settings.get("cache.redis.host", "localhost")
            redis_port = settings.get("cache.redis.port", 6379)
            redis_db = settings.get("cache.redis.db", 0)
            redis_password = settings.get("cache.redis.password", None)

            cache_settings.update({
                'cache': "aiocache.RedisCache",
                'endpoint': redis_host,
                'port': redis_port,
                'db': redis_db,
                'password': redis_password,
            })
            logger.info(f"Configuring aiocache for Redis at {redis_host}:{redis_port} (DB: {redis_db})")
        elif cache_type == "memory":
             cache_settings['cache'] = "aiocache.SimpleMemoryCache"
             logger.info("Configuring aiocache for in-memory cache.")
        else:
            logger.warning(f"Unsupported cache type '{cache_type}' in settings. Defaulting to memory cache.")
            cache_settings['cache'] = "aiocache.SimpleMemoryCache"

        cache_config['default'] = cache_settings

    try:
        caches.set_config(cache_config)
        _cache_configured = True
        logger.info(f"aiocache configured successfully with default settings: {cache_config['default']}")
    except Exception as e:
        logger.error(f"Failed to configure aiocache: {e}", exc_info=True)


def get_cache(alias="default") -> Cache:
    """
    Retrieves the configured cache instance.

    Ensures that the cache is configured before returning an instance.

    Args:
        alias: The cache alias to retrieve (usually 'default').

    Returns:
        An aiocache Cache instance.
    """
    if not _cache_configured:
        logger.warning("Cache accessed before configuration. Attempting default configuration.")
        configure_aiocache() # Attempt configuration if not done yet

    return caches.get(alias)

# Example usage (optional, mainly for testing this module directly)
# async def example_main():
#     configure_aiocache() # Configure based on settings or defaults
#     cache = get_cache()
#     await cache.set("my_key", "my_value", ttl=60)
#     value = await cache.get("my_key")
#     print(f"Retrieved value: {value}")
#     await cache.close() # Close connections if needed (e.g., for Redis)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(example_main())