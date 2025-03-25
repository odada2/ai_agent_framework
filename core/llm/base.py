"""
Base LLM Interface

This module defines the abstract base class for all LLM integrations in the framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseLLM(ABC):
    """
    Abstract base class for all Language Model integrations.
    
    This class defines the interface that all concrete LLM implementations must follow,
    ensuring consistent behavior regardless of the underlying model or provider.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        context_window: int = 8192,
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: The name/identifier of the specific model to use
            temperature: Controls randomness of outputs (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate in the response
            context_window: Maximum token capacity of the model's context window
            timeout: Timeout in seconds for API requests
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.timeout = timeout
        self.config = kwargs
        
        # Track usage for monitoring/billing
        self.usage_stats = {
            "total_prompts": 0,
            "total_tokens": 0,
            "total_completion_tokens": 0,
        }

    @abstractmethod
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM based on the provided prompt.
        
        Args:
            prompt: The primary user prompt or query
            system_prompt: Optional system prompt for model context/instruction
            temperature: Override the default temperature if provided
            max_tokens: Override the default max_tokens if provided
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing the generated response and metadata
        """
        pass
    
    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response that may include tool calls.
        
        Args:
            prompt: The primary user prompt or query
            tools: List of tool definitions available to the model
            system_prompt: Optional system prompt for model context
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing the response, potential tool calls, and metadata
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize the input text according to this model's tokenizer.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of token IDs
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the input text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Token count as an integer
        """
        return len(self.tokenize(text))
    
    def update_usage_stats(self, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Update the usage statistics for this LLM instance.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        self.usage_stats["total_prompts"] += 1
        self.usage_stats["total_tokens"] += prompt_tokens + completion_tokens
        self.usage_stats["total_completion_tokens"] += completion_tokens
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get the current usage statistics for this LLM instance.
        
        Returns:
            Dictionary containing usage statistics
        """
        return self.usage_stats