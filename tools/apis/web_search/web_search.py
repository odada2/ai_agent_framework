# ai_agent_framework/tools/apis/web_search/web_search.py

"""
WebSearchTool Implementation

Provides a tool for searching the web using multiple backend providers,
with support for caching, API key management, and result normalization.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Type

# Framework components
from ....core.tools.base import BaseTool
# Assuming Settings is available or passed if needed
# from ....config.settings import Settings
# Web search specific components
from .base import SearchProvider, SearchResult, ApiKeyError, RateLimitError, SearchError
from .providers import SerperProvider, GoogleSearchProvider, BingSearchProvider
# Import the custom serializer default function from utils
from .utils import sanitize_query, is_valid_domain, format_search_results, normalize_url, json_serializer_default
# Caching configuration and decorator
from .caching import configure_aiocache, get_cache
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer # Keep JsonSerializer

logger = logging.getLogger(__name__)

# Configure cache on module load or first use
try:
    configure_aiocache()
except Exception as e:
     logger.error(f"Initial cache configuration failed: {e}. Caching might not work.", exc_info=True)

class WebSearchTool(BaseTool):
    """
    Tool for performing web searches using various providers like Google, Bing, Serper.
    (Docstring remains the same)
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
        settings: Optional[Any] = None,
        max_results: int = 5,
        default_domain: Optional[str] = None,
        provider_order: Optional[List[str]] = None,
        provider_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize the WebSearchTool.
        (Implementation remains the same as provided in the previous fix)
        """
        parameters_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query string."},
                "num_results": {"type": "integer", "description": f"Number of results (1-{max_results}).", "minimum": 1, "maximum": max_results, "default": max_results},
                "domain": {"type": "string", "description": "Optional: Restrict to domain (e.g., 'wikipedia.org')."},
                "provider": {"type": "string", "description": f"Optional: Specify provider ({', '.join(self.PROVIDER_MAP.keys())})."}
            },
            "required": ["query"]
        }
        examples = [
            {"description": "Find recent AI news", "parameters": {"query": "latest AI advancements news"}},
            {"description": "Search Wikipedia", "parameters": {"query": "Python programming language", "domain": "wikipedia.org", "num_results": 3}},
        ]
        super().__init__(name=name, description=description, parameters=parameters_schema, examples=examples, required_permissions=["web_access"], **kwargs)
        self.settings = settings
        self.max_results = max_results
        self.default_domain = default_domain
        self.provider_order = provider_order or self.DEFAULT_PROVIDER_ORDER
        self.provider_configs = provider_configs
        self.providers: Dict[str, SearchProvider] = self._initialize_providers()
        self.provider_list = [p for p in self.provider_order if p in self.providers]
        if not self.providers: logger.warning(f"No web search providers initialized for tool '{name}'.")

    def _initialize_providers(self) -> Dict[str, SearchProvider]:
        """Initialize configured search providers."""
        # (Implementation remains the same as provided in the previous fix)
        initialized_providers = {}
        provider_configs_source = self.provider_configs
        if provider_configs_source is None and self.settings and hasattr(self.settings, 'get'):
            provider_configs_source = self.settings.get("web_search.providers", {})
        provider_configs_source = provider_configs_source or {}
        for provider_name in self.provider_order:
            if provider_name not in self.PROVIDER_MAP: continue
            config = provider_configs_source.get(provider_name, {})
            ProviderClass = self.PROVIDER_MAP[provider_name]
            try:
                provider_instance = ProviderClass(**config)
                initialized_providers[provider_name] = provider_instance
                logger.info(f"Initialized web search provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider '{provider_name}': {e}.", exc_info=self.verbose)
        return initialized_providers


    # --- Updated Cache Decorator ---
    @cached(
        # ttl=Optional, configure via aiocache global settings or add here
        cache=Cache.MEMORY, # Or Cache.REDIS depending on global config
        # Use JsonSerializer WITH the custom default handler
        serializer=JsonSerializer(default=json_serializer_default),
        key_builder=lambda _f, self, query, num_results=None, domain=None, **_kwargs:
             f"websearch:{sanitize_query(query)}:{num_results or self.max_results}:{domain or 'any'}"
    )
    async def _run(
        self,
        query: str,
        num_results: Optional[int] = None,
        provider: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the web search asynchronously, handling provider selection,
        execution, fallback, and caching with robust JSON serialization.
        (Docstring updated slightly)
        """
        # (Implementation of the _run method remains the same as provided in the previous fix)
        if not self.providers: return {"error": "No web search providers configured.", "query": query, "results": []}
        sanitized_query = sanitize_query(query)
        num_results = min(num_results or self.max_results, self.max_results)
        domain_filter = domain or self.default_domain
        effective_kwargs = kwargs.copy()
        if domain_filter: effective_kwargs["site_restrict"] = domain_filter
        if domain_filter and not is_valid_domain(domain_filter): return {"error": f"Invalid domain format: {domain_filter}", "query": query, "results": []}

        provider_errors: Dict[str, str] = {}
        providers_to_try: List[str] = []
        primary_provider = provider if provider and provider in self.providers else (self.provider_list[0] if self.provider_list else None)
        if primary_provider:
             providers_to_try.append(primary_provider)
             allow_fallback = kwargs.get("allow_fallback", True)
             if allow_fallback: providers_to_try.extend([p for p in self.provider_list if p != primary_provider])
        else: providers_to_try = self.provider_list
        if not providers_to_try: return {"error": "No web search providers available.", "query": query, "results": []}

        final_result_dict: Optional[Dict[str, Any]] = None
        for provider_name in providers_to_try:
            logger.info(f"Attempting search for '{sanitized_query[:30]}...' using: {provider_name}")
            try:
                search_result_dict = await self._search_with_provider(provider_name, sanitized_query, num_results, **effective_kwargs)
                search_result_dict["provider_used"] = provider_name
                final_result_dict = search_result_dict
                if provider_name != primary_provider and primary_provider is not None:
                     final_result_dict["provider_fallback"] = True
                     final_result_dict["original_provider"] = primary_provider
                break # Success
            except (ApiKeyError, RateLimitError) as e: logger.warning(f"Provider '{provider_name}' failed (recoverable): {e}. Trying next."); provider_errors[provider_name] = str(e)
            except SearchError as e: logger.error(f"Provider '{provider_name}' search error: {e}."); provider_errors[provider_name] = str(e)
            except Exception as e: logger.exception(f"Unexpected error with provider '{provider_name}': {e}"); provider_errors[provider_name] = f"Unexpected: {e}"

        if final_result_dict:
             if provider_errors: final_result_dict["provider_errors"] = provider_errors
             results_list = final_result_dict.get("results", [])
             if isinstance(results_list, list):
                   try: # Use list comprehension with conditional creation
                        search_results_obj = [SearchResult(**res_dict) for res_dict in results_list if isinstance(res_dict, dict)]
                        final_result_dict["formatted_text"] = format_search_results(search_results_obj, query)
                   except Exception as format_e: # Catch potential errors during SearchResult creation or formatting
                        logger.error(f"Error formatting search results: {format_e}")
                        final_result_dict["formatted_text"] = "Error: Could not format results."
             else: final_result_dict["formatted_text"] = "Error: Results format unexpected."
             return final_result_dict
        else:
             error_summary = "All available web search providers failed."
             if provider_errors:
                  try: error_details = json.dumps(provider_errors)
                  except TypeError: error_details = str(provider_errors)
                  error_summary += " Errors: " + error_details
             return {"error": error_summary, "query": query, "results": [], "provider_errors": provider_errors}


    async def _search_with_provider(
        self, provider_name: str, query: str, num_results: int, **kwargs
    ) -> Dict[str, Any]:
        """Helper method to perform search using a specific provider."""
        # (Implementation remains the same as provided in the previous fix)
        provider = self.providers[provider_name]
        start_time = time.monotonic()
        raw_results_data = await provider.search(query, num_results, **kwargs)
        parsed_results: List[SearchResult] = provider.parse_results(raw_results_data, query)
        duration = time.monotonic() - start_time
        logger.info(f"Search with {provider_name} completed in {duration:.2f}s, parsed {len(parsed_results)} results.")
        final_results_list = []
        for res in parsed_results[:num_results]:
            try:
                res.url = normalize_url(res.url)
                final_results_list.append(res.to_dict()) # Use existing to_dict()
            except Exception as e:
                 logger.warning(f"Failed to process/normalize result item: {e}. Result: {res}")
        return {
            "query": query, "results": final_results_list,
            "metadata": { "provider": provider_name, "duration_seconds": duration,
                          "result_count_parsed": len(parsed_results), "result_count_returned": len(final_results_list) }
        }

    async def close(self):
        """Clean up resources, like closing provider sessions."""
        # (Implementation remains the same as provided in the previous fix)
        logger.info(f"Closing WebSearchTool '{self.name}' resources...")
        close_tasks = []
        for provider_name, provider in self.providers.items():
            if hasattr(provider, 'close') and callable(provider.close):
                if asyncio.iscoroutinefunction(provider.close): close_tasks.append(provider.close())
                else:
                    try: provider.close(); logger.info(f"Closed provider sync: {provider_name}")
                    except Exception as e: logger.error(f"Error closing provider sync {provider_name}: {e}")
        if close_tasks:
             results = await asyncio.gather(*close_tasks, return_exceptions=True)
             for res in results:
                  if isinstance(res, Exception): logger.error(f"Error during async close for provider: {res}")