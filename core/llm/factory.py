# ai_agent_framework/core/llm/factory.py (Updated for Org ID)

import logging
import os
from typing import Dict, Optional, Type

# Assuming BaseLLM, ClaudeLLM, OpenAILLM are importable
try:
    from .base import BaseLLM
    from .claude import ClaudeLLM
    from .openai import OpenAILLM
except ImportError:
    # Fallback for different execution contexts
    from core.llm.base import BaseLLM
    from core.llm.claude import ClaudeLLM
    from core.llm.openai import OpenAILLM


logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory class for creating and managing LLM instances.
    Handles API Key and Organization ID loading.
    """

    _llm_registry: Dict[str, Type[BaseLLM]] = {
        "claude": ClaudeLLM,
        "openai": OpenAILLM,
        # Add other registered providers here
    }

    # Environment variable mapping
    # Keys are lower-case provider names
    _ENV_VAR_MAP = {
        "claude": {"api_key": "ANTHROPIC_API_KEY"},
        "openai": {
            "api_key": "OPENAI_API_KEY",
            "organization_id": "OPENAI_ORGANIZATION_ID" # <-- Org ID env var name
            },
        # Add other providers here
    }

    @classmethod
    def register_llm(cls, name: str, llm_class: Type[BaseLLM]) -> None:
        """Register an LLM class with the factory."""
        provider_name = name.lower()
        if provider_name in cls._llm_registry: logger.warning(f"LLM provider '{provider_name}' is already registered. Overwriting.")
        cls._llm_registry[provider_name] = llm_class
        logger.debug(f"Registered LLM provider: {provider_name}")

    @classmethod
    def create_llm(
        cls,
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None, # <-- Accept Org ID parameter
        **kwargs # Catch other kwargs for the specific LLM class
    ) -> BaseLLM:
        """
        Create an LLM instance based on the provider.

        Args:
            provider: LLM provider name (case-insensitive).
            model_name: Specific model name.
            api_key: Optional API key (checks env var if None).
            organization_id: Optional Organization ID (checks env var if None, primarily for OpenAI).
            **kwargs: Additional parameters for the LLM constructor.

        Returns:
            An instance of the specified LLM.
        """
        provider_key = provider.lower()

        if provider_key not in cls._llm_registry:
            raise ValueError(f"Unknown LLM provider: '{provider}'. Available: {list(cls._llm_registry.keys())}")

        llm_class = cls._llm_registry[provider_key]

        # --- Load credentials/IDs from environment if not provided ---
        provider_env_vars = cls._ENV_VAR_MAP.get(provider_key, {})

        # Load API Key if not passed directly
        if api_key is None:
            api_key_env_var = provider_env_vars.get("api_key")
            if api_key_env_var:
                api_key = os.environ.get(api_key_env_var)
                # Let constructor handle warning/error for missing required key

        # Load Organization ID if not passed directly (specifically for openai)
        if provider_key == "openai" and organization_id is None:
             org_id_env_var = provider_env_vars.get("organization_id")
             if org_id_env_var:
                  organization_id = os.environ.get(org_id_env_var)
                  # Log if Org ID is found via environment variable
                  # if organization_id:
                  #    logger.debug(f"Loaded OpenAI Organization ID from env var {org_id_env_var}.")
        # -----------------------------------------------------------

        try:
            logger.info(f"Creating LLM instance for provider: '{provider}', model: '{model_name or 'default'}'")
            # --- Pass parameters specifically to the constructor ---
            constructor_args = kwargs.copy() # Start with extra kwargs
            constructor_args['api_key'] = api_key # Pass potentially loaded key

            # Add organization_id only if provider is openai
            if provider_key == "openai":
                 constructor_args['organization_id'] = organization_id

            # Add model_name if provided (it's usually a required or key param)
            if model_name:
                 constructor_args['model_name'] = model_name
            # -------------------------------------------------------------

            # Instantiate the class
            # The LLM constructor (e.g., OpenAILLM.__init__) handles missing required args like api_key
            return llm_class(**constructor_args)

        except ImportError as e: logger.error(f"Missing library for '{provider}': {e}"); raise
        except ValueError as e: logger.error(f"Config error for '{provider}': {e}"); raise # Catch missing key errors etc.
        except Exception as e:
            logger.error(f"Failed to instantiate LLM class for '{provider}': {e}", exc_info=True)
            raise ValueError(f"Failed to create LLM for provider '{provider}': {e}")