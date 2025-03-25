"""
LLM Factory

This module provides a factory pattern for creating LLM instances.
"""

import logging
import os
from typing import Dict, Optional, Type

from .base import BaseLLM
from .claude import ClaudeLLM

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory class for creating and managing LLM instances.
    """
    
    _llm_registry: Dict[str, Type[BaseLLM]] = {}
    
    @classmethod
    def register_llm(cls, name: str, llm_class: Type[BaseLLM]) -> None:
        """
        Register an LLM class with the factory.
        
        Args:
            name: Name to register the LLM under
            llm_class: The LLM class to register
        """
        cls._llm_registry[name.lower()] = llm_class
        logger.debug(f"Registered LLM provider: {name}")
    
    @classmethod
    def create_llm(
        cls,
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance based on the provider.
        
        Args:
            provider: The LLM provider name (e.g., 'claude', 'openai')
            model_name: The specific model to use
            api_key: Optional API key (will look for env var if not provided)
            **kwargs: Additional parameters for the LLM constructor
            
        Returns:
            An instance of the specified LLM
            
        Raises:
            ValueError: If the provider is not registered
        """
        provider = provider.lower()
        
        if provider not in cls._llm_registry:
            raise ValueError(f"Unknown LLM provider: {provider}. "
                            f"Available providers: {list(cls._llm_registry.keys())}")
        
        llm_class = cls._llm_registry[provider]
        
        # Set model name specific to provider if not specified
        if model_name is None:
            if provider == "claude":
                model_name = "claude-3-7-sonnet-20250219"
            else:
                # Default model for other providers could be set here
                pass
        
        # Look for API key in environment variables if not provided
        if api_key is None:
            if provider == "claude":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
            # Add more providers as needed
        
        # Create and return the LLM instance
        return llm_class(model_name=model_name, api_key=api_key, **kwargs)


# Register available LLM providers
LLMFactory.register_llm("claude", ClaudeLLM)
# Register other LLM providers as they're implemented