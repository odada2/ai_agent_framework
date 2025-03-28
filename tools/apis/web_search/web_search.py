# ai_agent_framework/tools/apis/web_search/web_search.py

"""
WebSearchTool Implementation

Provides a tool for searching the web using multiple backend providers,
with support for caching, API key management, and result normalization.
"""

import asyncio
import logging
import time
import json # Added for error summary serialization
from typing import Dict, List, Optional, Any, Type

# Framework components (adjust paths based on final project structure)
from ....core.tools.base import BaseTool
# Assuming Settings can be imported or is passed appropriately if needed
# from ....config.settings import Settings # Optional: If tool needs direct settings access
# Web search specific components
from .base import SearchProvider, SearchResult, ApiKeyError, RateLimitError, SearchError
from .providers import SerperProvider, GoogleSearchProvider, BingSearchProvider
# Removed direct import of ApiKeyManager - assumed handled by providers
# from .api_key_manager import ApiKeyManager
from .utils import sanitize_query, is_valid_domain, format_search_results, normalize_url
# Caching configuration and decorator
from .caching import configure_aiocache, get_cache
from aiocache import cached, Cache # Import Cache explicitly if needed
from aiocache.serializers import JsonSerializer

logger = logging.getLogger(__name__)

# Configure cache on module load or first use
# Calling it here ensures it's configured before any @cached method is potentially called.
# The configure_aiocache function itself should be idempotent.
try:
    # Pass Settings instance if configure_aiocache requires it
    # from ....config.settings import Settings
    # configure_aiocache(Settings())
    configure_aiocache() # Assuming configure_aiocache can load settings itself
except Exception as e:
     logger.error(f"Initial cache configuration failed: {e}. Caching might not work.", exc_info=True)

class WebSearchTool(BaseTool):
    """
    Tool for performing web searches using various providers like Google, Bing, Serper.

    Features include:
    - Multiple backend support with automatic fallback.
    - API key management and rotation (handled within providers).
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
        settings: Optional[Any] = None, # Changed type hint for flexibility if Settings class path is complex
        max_results: int = 5,
        default_domain: Optional[str] = None,
        provider_order: Optional[List[str]] = None,
        # Add provider_configs if explicit configuration is needed here
        provider_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize the WebSearchTool.

        Args:
            name: Name of the tool.
            description: Description of the tool.
            settings: Optional Settings instance or similar config object.
            max_results: Default maximum number of results to return.
            default_domain: Default domain to filter results by.
            provider_order: List of provider names in preferred order of use.
            provider_configs: Explicit configurations for providers (e.g., API keys).
                              If None, providers might try to load from environment.
            **kwargs: Additional arguments for BaseTool.
        """
        # Define parameters schema before calling super().__init__
        parameters_schema = {
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
            parameters=parameters_schema, # Pass the defined schema
            examples=examples,
            required_permissions=["web_access"], # Assuming a permission system
            **kwargs
        )

        self.settings = settings # Store settings object if provided
        self.max_results = max_results
        self.default_domain = default_domain
        self.provider_order = provider_order or self.DEFAULT_PROVIDER_ORDER
        # Use provided configs or load from settings/env via provider init
        self.provider_configs = provider_configs

        # Initialize providers based on settings and order
        self.providers: Dict[str, SearchProvider] = self._initialize_providers()
        # Actual available providers in order
        self.provider_list = [p for p in self.provider_order if p in self.providers]

        if not self.providers:
            logger.warning(f"No web search providers could be initialized for tool '{name}'. Web search will not function.")


    def _initialize_providers(self) -> Dict[str, SearchProvider]:
        """Initialize configured search providers."""
        initialized_providers = {}
        # Use explicit configs if passed, otherwise try loading from settings or env vars via provider constructors
        provider_configs_source = self.provider_configs

        # If no explicit configs, try getting from settings object
        if provider_configs_source is None and self.settings and hasattr(self.settings, 'get'):
            provider_configs_source = self.settings.get("web_search.providers", {})

        # Fallback to empty dict if still no configs
        provider_configs_source = provider_configs_source or {}

        for provider_name in self.provider_order:
            if provider_name not in self.PROVIDER_MAP:
                logger.warning(f"Unknown provider '{provider_name}' specified in provider order.")
                continue

            # Get specific config for this provider from the source
            config = provider_configs_source.get(provider_name, {})
            ProviderClass = self.PROVIDER_MAP[provider_name]

            try:
                # Pass relevant config directly to the provider's init
                # Assumes provider __init__ methods accept kwargs like 'api_keys', 'engine_id', etc.,
                # and can handle loading from environment if config values (e.g., api_keys) are missing.
                provider_instance = ProviderClass(**config)
                initialized_providers[provider_name] = provider_instance
                logger.info(f"Initialized web search provider: {provider_name}")
            except (ImportError, ValueError, Exception) as e:
                # Log specific errors, e.g., missing API keys or dependencies
                logger.error(f"Failed to initialize provider '{provider_name}': {e}. This provider will be unavailable.", exc_info=self.verbose) # Add traceback if verbose

        return initialized_providers


    # Configure caching using aiocache decorator
    # The key includes query, num_results, domain for uniqueness
    # Uses the globally configured cache (Memory or Redis via configure_aiocache)
    # Using the _run method as the target for caching
    @cached(
        # ttl=Optional, can be specified here or rely on global default from configure_aiocache
        cache=Cache.MEMORY, # Default cache type can be specified or rely on global config
        serializer=JsonSerializer(), # Use JSON for broader compatibility
        key_builder=lambda _f, self, query, num_results=None, domain=None, **_kwargs:
             f"websearch:{sanitize_query(query)}:{num_results or self.max_results}:{domain or 'any'}"
    )
    async def _run( # This is the correct method definition within the class
        self,
        query: str,
        num_results: Optional[int] = None,
        provider: Optional[str] = None, # Allow specifying a provider directly
        domain: Optional[str] = None,
        **kwargs # Allow passing extra args like gl, hl, etc.
    ) -> Dict[str, Any]:
        """
        Execute the web search asynchronously, handling provider selection, execution, fallback, and caching.
        This method is decorated with @cached.

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
        # Use self.max_results as the upper limit for num_results
        num_results = min(num_results or self.max_results, self.max_results)
        domain_filter = domain or self.default_domain # Use the argument `domain` if provided
        effective_kwargs = kwargs.copy() # Start with extra args
        if domain_filter:
            # Pass domain filter correctly based on provider needs (some use 'site_restrict')
            effective_kwargs["site_restrict"] = domain_filter # Example key, adjust if providers differ

        # Validate domain format if provided
        if domain_filter and not is_valid_domain(domain_filter):
            return {"error": f"Invalid domain format: {domain_filter}", "query": query, "results": []}

        # --- Provider Selection ---
        provider_errors: Dict[str, str] = {}
        providers_to_try: List[str] = []

        # Determine the primary provider to try
        primary_provider = provider if provider and provider in self.providers else (self.provider_list[0] if self.provider_list else None)

        if primary_provider:
             providers_to_try.append(primary_provider)
             # Add fallbacks if allowed (and if primary wasn't explicitly specified, or if specified but fallback is ok)
             allow_fallback = kwargs.get("allow_fallback", True)
             if allow_fallback:
                 providers_to_try.extend([p for p in self.provider_list if p != primary_provider])
        else:
             # If no primary could be determined (e.g., invalid specified provider), try all available
             providers_to_try = self.provider_list

        if not providers_to_try:
             # This case should ideally be caught by the initial self.providers check
             return {"error": "No web search providers available to attempt search.", "query": query, "results": []}


        # --- Execution Loop with Fallback ---
        final_result_dict: Optional[Dict[str, Any]] = None

        for provider_name in providers_to_try:
            logger.info(f"Attempting web search for '{sanitized_query[:30]}...' using provider: {provider_name}")
            try:
                # Call the helper method to perform search with this specific provider
                search_result_dict = await self._search_with_provider(
                    provider_name=provider_name,
                    query=sanitized_query,
                    num_results=num_results,
                    **effective_kwargs # Pass domain filter and other kwargs
                )
                # Add provider info to the successful result
                search_result_dict["provider_used"] = provider_name
                final_result_dict = search_result_dict
                # If this was a fallback, note it
                if provider_name != primary_provider and primary_provider is not None:
                     final_result_dict["provider_fallback"] = True
                     final_result_dict["original_provider"] = primary_provider
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
        if final_result_dict:
             if provider_errors: # Add errors from failed providers if any occurred before success
                  final_result_dict["provider_errors"] = provider_errors
             # Add formatted text results for convenience
             # Ensure results key exists and is a list before formatting
             results_list = final_result_dict.get("results", [])
             if isinstance(results_list, list):
                   # Convert result dicts back to SearchResult objects for formatting if needed
                   search_results_obj = [SearchResult(**res_dict) for res_dict in results_list]
                   final_result_dict["formatted_text"] = format_search_results(search_results_obj, query)
             else:
                  final_result_dict["formatted_text"] = "Error: Results format unexpected."

             return final_result_dict
        else:
             # All providers failed
             error_summary = "All available web search providers failed."
             if provider_errors:
                  try:
                       error_details = json.dumps(provider_errors)
                  except TypeError: # Handle non-serializable errors if necessary
                       error_details = str(provider_errors)
                  error_summary += " Errors: " + error_details
             return {
                  "error": error_summary,
                  "query": query, # Return original query
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
            **kwargs: Additional provider-specific arguments (e.g., site_restrict).

        Returns:
            Dictionary with results from the provider.

        Raises:
            ApiKeyError, RateLimitError, SearchError, Exception: If search fails.
        """
        provider = self.providers[provider_name]
        start_time = time.monotonic()

        # Execute the provider's search method
        # Providers are expected to raise specific errors on failure
        raw_results_data = await provider.search(query, num_results, **kwargs)

        # Parse the raw results into standardized SearchResult objects
        # Provider's parse_results should handle different result structures
        parsed_results: List[SearchResult] = provider.parse_results(raw_results_data, query)

        duration = time.monotonic() - start_time
        logger.info(f"Search with {provider_name} completed in {duration:.2f}s, parsed {len(parsed_results)} results.")

        # Normalize URLs and potentially filter/score further if needed centrally
        final_results_list = []
        for res in parsed_results[:num_results]: # Ensure only num_results are returned
            try:
                # Example: Normalize URL here if not done by provider
                res.url = normalize_url(res.url)
                # Convert SearchResult object to dict for the final return structure
                final_results_list.append(res.to_dict())
            except Exception as e:
                 logger.warning(f"Failed to process/normalize result item: {e}. Result: {res}")
                 # Optionally skip this result or add it as is

        # Return structured results
        return {
            "query": query,
            "results": final_results_list,
            "metadata": {
                "provider": provider_name,
                "duration_seconds": duration,
                "result_count_parsed": len(parsed_results),
                "result_count_returned": len(final_results_list),
                # Add raw_results preview if needed for debugging, but can be large
                # "raw_data_preview": str(raw_results_data)[:500] + "..."
            }
        }


    async def close(self):
        """Clean up resources, like closing provider sessions."""
        logger.info(f"Closing WebSearchTool '{self.name}' resources...")
        close_tasks = []
        for provider_name, provider in self.providers.items():
            if hasattr(provider, 'close') and callable(provider.close):
                # Ensure close is awaitable if it's async
                if asyncio.iscoroutinefunction(provider.close):
                    close_tasks.append(provider.close())
                else:
                    try:
                         provider.close() # Try sync close
                         logger.info(f"Closed provider sync: {provider_name}")
                    except Exception as e:
                         logger.error(f"Error closing provider sync {provider_name}: {e}")

        # Await all async close tasks
        if close_tasks:
             results = await asyncio.gather(*close_tasks, return_exceptions=True)
             for i, res in enumerate(results):
                  # Attempt to map back to provider name if needed for logging errors
                  if isinstance(res, Exception):
                       logger.error(f"Error during async close for provider: {res}") # Name mapping difficult here

        # Close global cache connections if necessary (aiocache can manage this)
        # cache = get_cache()
        # if hasattr(cache, 'close') and callable(cache.close):
        #     try:
        #         await cache.close()
        #         logger.info("Closed web search cache connection.")
        #     except Exception as e:
        #         logger.error(f"Error closing web search cache: {e}")