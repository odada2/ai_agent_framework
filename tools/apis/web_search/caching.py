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
                try:
                    result = self.serializer.loads(raw_data)
                    self._stats["hits"] += 1
                    
                    # Add cache metadata
                    if isinstance(result, dict):
                        if "metadata" not in result:
                            result["metadata"] = {}
                        result["metadata"]["cache_hit"] = True
                        result["metadata"]["cache_time"] = datetime.now().isoformat()
                    
                    return result
                except Exception as serialization_error:
                    # Handle deserialization errors
                    logger.warning(f"Cache deserialization error: {str(serialization_error)}")
                    self._stats["errors"] += 1
                    
                    # Attempt to invalidate corrupt cache entry
                    try:
                        self.redis.delete(key)
                        logger.info(f"Invalidated corrupt cache entry for key: {key}")
                    except Exception:
                        pass
                    
                    return None
            else:
                self._stats["misses"] += 1
                return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache get error: {str(e)}")
            return None