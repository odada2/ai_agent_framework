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
            }