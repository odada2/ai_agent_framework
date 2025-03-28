async def _run(
        self,
        query: str,
        num_results: Optional[int] = None,
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
        
        # Track errors for reporting
        provider_errors = {}
        
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
            # Record error
            provider_errors[provider_name] = str(e)
            
            # If specific provider was requested and failed, check if we should try fallbacks
            if provider and not kwargs.get("allow_fallback", True):
                return {
                    "error": f"Error with provider '{provider}': {str(e)}",
                    "query": query,
                    "results": [],
                    "provider_errors": provider_errors
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
                    result["provider_errors"] = provider_errors
                    return result
                    
                except Exception as e:
                    # Record fallback error
                    provider_errors[fallback] = str(e)
                    # Continue to next fallback provider
                    continue
            
            # If all providers failed
            return {
                "error": "All search providers failed",
                "query": query,
                "results": [],
                "provider_errors": provider_errors
            }
            
        except Exception as e:
            # Record primary provider error
            provider_errors[provider_name] = str(e)
            
            # Log the error and check if we should try fallbacks
            logger.error(f"Search error with provider {provider_name}: {str(e)}")
            
            if kwargs.get("allow_fallback", True):
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
                        result["provider_errors"] = provider_errors
                        return result
                    except Exception as fallback_e:
                        # Record fallback error
                        provider_errors[fallback] = str(fallback_e)
                        # Continue to next fallback provider
                        continue
            
            # If all providers failed or no fallbacks attempted
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": [],
                "provider_errors": provider_errors
            }# ai_agent_framework/tools/apis/web_search/web_search.py

"""
WebSearchTool Implementation

Provides a tool for searching the web using multiple backend providers,
with support for caching, API key management, and result normalization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type

# Framework components (adjust paths based on final project structure)
from ....core.tools.base import BaseTool
from ....config.settings import Settings
# Web search specific components
from .base import SearchProvider, SearchResult, ApiKeyError, RateLimitError, SearchError
from .providers import SerperProvider, GoogleSearchProvider, BingSearchProvider
from .api_key_manager import ApiKeyManager
from .utils import sanitize_query, is_valid_domain, format_search_results, normalize_url
# Caching configuration and decorator
from .caching import configure_aiocache, get_cache
from aiocache import cached
from aiocache.serializers import JsonSerializer

logger = logging.getLogger(__name__)

# Configure cache on module load or first use
# Calling it here ensures it's configured before any @cached method is potentially called.
# The configure_aiocache function itself should be idempotent.
try:
    configure_aiocache()
except Exception as e:
     logger.error(f"Initial cache configuration failed: {e}. Caching might not work.", exc_info=True)

class WebSearchTool(BaseTool):
    """
    Tool for performing web searches using various providers like Google, Bing, Serper.

    Features include:
    - Multiple backend support with automatic fallback.
    - API key management and rotation.
    - Result caching via aiocache.
    - Domain filtering and result normalization.
    """

    DEFAULT_PROVIDER_ORDER = ["serper", "google", "bing"]
    PROVIDER_MAP: Dict[str, Type[SearchProvider]] = {
        "serper": SerperProvider,
        "google": GoogleSearchProvider,
        "bing": BingSearchProvider,
    }

    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the web for information. Use 'domain' parameter to filter results (e.g., 'wikipedia.org').",
        settings: Optional[Settings] = None,
        max_results: int = 5,
        default_domain: Optional[str] = None,
        provider_order: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the WebSearchTool.

        Args:
            name: Name of the tool.
            description: Description of the tool.
            settings: Optional Settings instance. If None, loads default settings.
            max_results: Default maximum number of results to return.
            default_domain: Default domain to filter results by.
            provider_order: List of provider names in preferred order of use.
            **kwargs: Additional arguments for BaseTool.
        """
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string."
                },
                "num_results": {
                    "type": "integer",
                    "description": f"Number of results to return (1-{max_results}).",
                    "minimum": 1,
                    "maximum": max_results, # Use configured max_results
                    "default": max_results
                },
                "domain": {
                    "type": "string",
                    "description": "Optional: Restrict search to this domain (e.g., 'wikipedia.org')."
                },
                "provider": {
                    "type": "string",
                    "description": f"Optional: Specify a provider ({', '.join(self.PROVIDER_MAP.keys())})."
                }
                # Add other common params like 'gl' (country), 'hl' (language) if desired
            },
            "required": ["query"]
        }

        examples = [
            {"description": "Find recent AI news", "parameters": {"query": "latest AI advancements news"}},
            {"description": "Search Wikipedia for Python", "parameters": {"query": "Python programming language", "domain": "wikipedia.org", "num_results": 3}},
            {"description": "Search using Bing specifically", "parameters": {"query": "Microsoft Copilot features", "provider": "bing"}}
        ]

        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            examples=examples,
            required_permissions=["web_access"], # Assuming a permission system
            **kwargs
        )

        self.settings = settings or Settings()
        self.max_results = max_results
        self.default_domain = default_domain
        self.provider_order = provider_order or self.DEFAULT_PROVIDER_ORDER

        # Initialize providers based on settings and order
        self.providers: Dict[str, SearchProvider] = self._initialize_providers()
        self.provider_list = [p for p in self.provider_order if p in self.providers] # Actual available providers in order

        if not self.providers:
            logger.warning(f"No web search providers could be initialized for tool '{name}'. Web search will not function.")


    def _initialize_providers(self) -> Dict[str, SearchProvider]:
        """Initialize configured search providers."""
        initialized_providers = {}
        provider_configs = self.settings.get("web_search.providers", {}) # Expecting config like: {'serper': {'api_keys': ['key1']}, 'google': {...}}

        for provider_name in self.provider_order:
            if provider_name not in self.PROVIDER_MAP:
                logger.warning(f"Unknown provider '{provider_name}' specified in provider order.")
                continue

            config = provider_configs.get(provider_name, {}) # Get specific config for this provider
            ProviderClass = self.PROVIDER_MAP[provider_name]

            try:
                # Pass relevant config directly to the provider's init
                # Assumes provider __init__ methods accept kwargs like 'api_keys', 'engine_id', etc.
                provider_instance = ProviderClass(**config)
                initialized_providers[provider_name] = provider_instance
                logger.info(f"Initialized web search provider: {provider_name}")
            except (ImportError, ValueError, Exception) as e:
                # Log specific errors, e.g., missing API keys or dependencies
                logger.error(f"Failed to initialize provider '{provider_name}': {e}. This provider will be unavailable.")

        return initialized_providers


    # Configure caching using aiocache decorator
    # The key includes query, num_results, domain for uniqueness
    # Uses the globally configured cache (Memory or Redis via configure_aiocache)
    @cached(
        # ttl=Optional, can be specified here or rely on global default from configure_aiocache
        serializer=JsonSerializer(), # Use JSON for broader compatibility
        key_builder=lambda f, self, query, num_results=None, domain=None, **kwargs:
             f"websearch:{sanitize_query(query)}:{num_results or self.max_results}:{domain or 'any'}"
    )
    async def _run(
        self,
        query: str,
        num_results: Optional[int] = None,
        provider: Optional[str] = None, # Allow specifying a provider directly
        domain: Optional[str] = None,
        **kwargs # Allow passing extra args like gl, hl, etc.
    ) -> Dict[str, Any]:
        """
        Execute the web search asynchronously, handling provider selection, execution, fallback, and caching.

        Args:
            query: Search query string.
            num_results: Max number of results.
            provider: Specific provider to use (optional).
            domain: Domain to filter by (optional).
            **kwargs: Additional parameters for search providers (e.g., gl, hl).

        Returns:
            Dictionary containing search results or an error.
        """
        if not self.providers:
             return {"error": "No web search providers are configured.", "query": query, "results": []}

        # --- Input Processing ---
        sanitized_query = sanitize_query(query)
        num_results = min(num_results or self.max_results, self.max_results) # Ensure respects max_results
        domain = domain or self.default_domain
        effective_kwargs = kwargs.copy() # Start with extra args
        if domain:
            effective_kwargs["site_restrict"] = domain # Add domain restriction if specified

        # Validate domain format if provided
        if domain and not is_valid_domain(domain):
            return {"error": f"Invalid domain format: {domain}", "query": query, "results": []}

        # --- Provider Selection ---
        provider_errors: Dict[str, str] = {}
        providers_to_try: List[str] = []

        if provider and provider in self.providers:
            providers_to_try.append(provider)
            # Optionally add fallbacks if the specified provider fails
            if kwargs.get("allow_fallback", True):
                 providers_to_try.extend([p for p in self.provider_list if p != provider])
        else:
             providers_to_try = self.provider_list # Try configured providers in order

        if not providers_to_try:
             return {"error": "No suitable web search providers available.", "query": query, "results": []}


        # --- Execution Loop with Fallback ---
        final_result: Optional[Dict[str, Any]] = None
        used_provider: Optional[str] = None

        for provider_name in providers_to_try:
            logger.info(f"Attempting web search for '{query[:30]}...' using provider: {provider_name}")
            try:
                search_result_dict = await self._search_with_provider(
                    provider_name=provider_name,
                    query=sanitized_query,
                    num_results=num_results,
                    **effective_kwargs # Pass domain filter and other kwargs
                )
                # Add provider info to the successful result
                search_result_dict["provider_used"] = provider_name
                final_result = search_result_dict
                used_provider = provider_name
                break # Success, exit loop

            except (ApiKeyError, RateLimitError) as e:
                logger.warning(f"Provider '{provider_name}' failed with recoverable error: {e}. Trying next provider.")
                provider_errors[provider_name] = str(e)
                # Key/RateLimit errors often mean trying another provider is worthwhile
            except SearchError as e:
                 logger.error(f"Provider '{provider_name}' encountered a search error: {e}.")
                 provider_errors[provider_name] = str(e)
                 # Depending on the error, might not be worth trying others, but we will for robustness
            except Exception as e:
                # Catch unexpected errors during search
                logger.exception(f"Unexpected error during search with provider '{provider_name}': {e}")
                provider_errors[provider_name] = f"Unexpected error: {e}"

        # --- Result Formatting ---
        if final_result:
             if provider_errors: # Add errors from failed providers if any occurred before success
                  final_result["provider_errors"] = provider_errors
             # Add formatted text results for convenience
             final_result["formatted_text"] = format_search_results(final_result.get("results", []), query)
             return final_result
        else:
             # All providers failed
             error_summary = "All web search providers failed."
             if provider_errors:
                  error_summary += " Errors: " + json.dumps(provider_errors)
             return {
                  "error": error_summary,
                  "query": query,
                  "results": [],
                  "provider_errors": provider_errors
             }


    async def _search_with_provider(
        self,
        provider_name: str,
        query: str,
        num_results: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Helper method to perform search using a specific provider.

        Args:
            provider_name: Name of the provider instance to use.
            query: Sanitized query string.
            num_results: Number of results requested.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Dictionary with results from the provider.

        Raises:
            ApiKeyError, RateLimitError, SearchError, Exception: If search fails.
        """
        provider = self.providers[provider_name]
        start_time = time.monotonic()

        # Execute the provider's search method
        # Providers are expected to raise specific errors on failure
        raw_results = await provider.search(query, num_results, **kwargs)

        # Parse the raw results into standardized SearchResult objects
        parsed_results = provider.parse_results(raw_results, query) # Should return List[SearchResult]

        duration = time.monotonic() - start_time
        logger.info(f"Search with {provider_name} completed in {duration:.2f}s, found {len(parsed_results)} results.")

        # Return structured results
        return {
            "query": query,
            "results": [res.to_dict() for res in parsed_results[:num_results]], # Ensure only num_results are returned
            "metadata": {
                "provider": provider_name,
                "duration_seconds": duration,
                "result_count_parsed": len(parsed_results),
                # Add raw_results if needed for debugging, but can be large
                # "raw_data_preview": str(raw_results)[:500] + "..."
            }
        }


    async def close(self):
        """Clean up resources, like closing provider sessions."""
        logger.info(f"Closing WebSearchTool '{self.name}' resources...")
        for provider_name, provider in self.providers.items():
            if hasattr(provider, 'close') and callable(provider.close):
                try:
                    await provider.close()
                    logger.info(f"Closed provider: {provider_name}")
                except Exception as e:
                    logger.error(f"Error closing provider {provider_name}: {e}")
        # Close global cache connections if necessary (aiocache handles this partly)
        # cache = get_cache()
        # if hasattr(cache, 'close') and callable(cache.close):
        #     await cache.close()