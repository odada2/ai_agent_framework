"""
Claude LLM Integration

This module provides an implementation of the BaseLLM interface for Anthropic's Claude models.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

import anthropic

from .base import BaseLLM

logger = logging.getLogger(__name__)


class ClaudeLLM(BaseLLM):
    """
    Implementation of BaseLLM for Anthropic's Claude models.
    """

    def __init__(
        self,
        model_name: str = "claude-3-7-sonnet-20250219",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize the Claude LLM integration.
        
        Args:
            model_name: Claude model to use (default: claude-3-7-sonnet-20250219)
            api_key: Anthropic API key, if not provided will look for ANTHROPIC_API_KEY env var
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Claude-specific parameters
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Initialize the Claude client
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Map of Claude model names to their context window sizes
        self.model_context_windows = {
            "claude-3-7-sonnet-20250219": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-5-haiku-20240307": 48000,
            "claude-3-opus-20240229": 200000,
        }
        
        # Set the context window based on the model
        if model_name in self.model_context_windows:
            self.context_window = self.model_context_windows[model_name]
        else:
            logger.warning(f"Unknown model: {model_name}. Using default context window size.")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using Claude.
        
        Args:
            prompt: The user prompt to send to Claude
            system_prompt: Optional system prompt
            temperature: Override default temperature if provided
            max_tokens: Override default max_tokens if provided
            **kwargs: Additional parameters to pass to the Claude API
            
        Returns:
            Dict containing the response and metadata
        """
        temp = temperature if temperature is not None else self.temperature
        max_out_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                temperature=temp,
                max_tokens=max_out_tokens,
                **kwargs
            )
            
            # Update usage statistics
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            self.update_usage_stats(prompt_tokens, completion_tokens)
            
            result = {
                "content": response.content[0].text,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "finish_reason": response.stop_reason
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from Claude: {str(e)}")
            return {
                "error": str(e),
                "content": "I encountered an error while processing your request."
            }

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response that may include tool calls using Claude.
        
        Args:
            prompt: The user prompt
            tools: List of tool definitions in Claude-compatible format
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the Claude API
            
        Returns:
            Dict containing the response and any tool calls
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Convert tools to Claude's expected format if needed
        claude_tools = []
        for tool in tools:
            claude_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {})
            }
            claude_tools.append(claude_tool)
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                tools=claude_tools,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **kwargs
            )
            
            # Process the response to extract tool calls
            result = {
                "content": response.content[0].text if response.content else "",
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "finish_reason": response.stop_reason,
                "tool_calls": []
            }
            
            # Extract any tool calls from the response
            for content_block in response.content:
                if content_block.type == "tool_use":
                    result["tool_calls"].append({
                        "name": content_block.name,
                        "parameters": content_block.input
                    })
            
            self.update_usage_stats(
                response.usage.input_tokens,
                response.usage.output_tokens
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with tools from Claude: {str(e)}")
            return {
                "error": str(e),
                "content": "I encountered an error while processing your tool-based request.",
                "tool_calls": []
            }
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using Claude's tokenizer.
        
        Note: This is an estimate as Anthropic doesn't provide a public tokenizer.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of token IDs (estimated)
        """
        # Anthropic doesn't provide a public tokenizer, so we estimate
        # This is a very rough approximation
        tokens = []
        # In a production system, you would use a proper tokenizer
        # For now, we'll just return a placeholder
        return [0] * (len(text) // 4)  # Very rough approximation
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text for Claude.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
        """
        # For Claude, we can use their utility if available
        try:
            count = self.client.count_tokens(text)
            return count
        except:
            # Fallback to rough estimation
            return len(text) // 4  # Very rough approximation