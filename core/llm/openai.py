# ai_agent_framework/core/llm/openai.py
# Updated with string forward reference for type hint

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

try:
    import openai
    import tiktoken
    # Import specific openai errors for retry logic
    # No need to import ChatCompletion type here if using string hint
    from openai import RateLimitError, APIStatusError, APIConnectionError, APITimeoutError, AuthenticationError, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Define dummy classes/exceptions if openai is not installed
    class OpenAI: pass
    class AsyncOpenAI: pass
    class RateLimitError(Exception): pass
    class APIStatusError(Exception): pass
    class APIConnectionError(Exception): pass
    class APITimeoutError(Exception): pass
    class AuthenticationError(Exception): pass
    class APIError(Exception): pass
    # Define dummy ChatCompletion for type hint reference if needed, but string is better
    # class ChatCompletion: pass
    class TiktokenEncoding:
         def encode(self, text: str) -> list: return list(range(len(text)//4))
         def decode(self, tokens: list) -> str: return "Token decoding unavailable."
    class tiktoken:
         @staticmethod
         def encoding_for_model(model_name:str) -> TiktokenEncoding: return TiktokenEncoding()
         @staticmethod
         def get_encoding(encoding_name:str) -> TiktokenEncoding: return TiktokenEncoding()

# Import tenacity for retry logic
try:
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError
except ImportError:
     logging.getLogger(__name__).warning("Tenacity library not found. Retries disabled.")
     def retry(*args, **kwargs): return lambda func: func
     class RetryError(Exception): pass
     RateLimitError = RateLimitError or Exception; APIConnectionError = APIConnectionError or Exception
     APITimeoutError = APITimeoutError or Exception; APIStatusError = APIStatusError or Exception

# Assuming BaseLLM is correctly located relative to this file
try:
    from .base import BaseLLM
except ImportError:
    from core.llm.base import BaseLLM


logger = logging.getLogger(__name__)

# (Retry decorators remain the same)
retry_on_rate_limit = retry(...)
retry_on_transient_error = retry(...)

class OpenAILLM(BaseLLM):
    """
    Implementation of BaseLLM for OpenAI's models (GPT-3.5, GPT-4, etc.).
    Uses the Chat Completions API with retry logic and optional Organization ID.
    """
    # (Class attributes like DEFAULT_MODEL etc. remain the same)
    DEFAULT_MODEL = "gpt-3.5-turbo"
    MODEL_CONTEXT_WINDOWS = { # Example values
        "gpt-4": 8192, "gpt-4-32k": 32768, "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000, "gpt-4o": 128000, "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385, "gpt-3.5-turbo-instruct": 4096,
    }
    MODEL_ALIAS_TO_TOKENIZER = { "gpt-4-turbo-preview": "gpt-4", "gpt-4o": "gpt-4o", "gpt-3.5-turbo-16k": "gpt-3.5-turbo" }


    def __init__(
        self, model_name: Optional[str] = None, api_key: Optional[str] = None,
        organization_id: Optional[str] = None, temperature: float = 0.7,
        max_tokens: Optional[int] = 1024, timeout: int = 60, **kwargs
    ):
        # (Initialization remains the same as previous version, including Org ID)
        if not OPENAI_AVAILABLE: raise ImportError("OpenAI library not installed. `pip install openai tiktoken`.")
        model_name = model_name or self.DEFAULT_MODEL
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens, timeout=timeout, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.organization_id = organization_id or os.environ.get("OPENAI_ORGANIZATION_ID")
        if not self.api_key: raise ValueError("OpenAI API key is required.")
        client_args = {"api_key": self.api_key, "timeout": timeout}
        if self.organization_id: client_args["organization"] = self.organization_id; logger.info(f"Using OpenAI Org ID: {self.organization_id[:7]}...")
        try:
            if not hasattr(openai, "AsyncOpenAI"): raise ImportError("Installed `openai` library version might be too old.")
            self.client = openai.AsyncOpenAI(**client_args)
        except Exception as e: logger.error(f"Failed to init OpenAI client: {e}", exc_info=True); raise ValueError(f"Failed init OpenAI client: {e}")
        self.context_window = self.MODEL_CONTEXT_WINDOWS.get(model_name, 8192)
        if model_name not in self.MODEL_CONTEXT_WINDOWS: logger.warning(f"Unknown OpenAI model: {model_name}. Default context: {self.context_window}.")
        tokenizer_model_name = self.MODEL_ALIAS_TO_TOKENIZER.get(self.model_name, self.model_name)
        try: self.tokenizer = tiktoken.encoding_for_model(tokenizer_model_name)
        except KeyError: logger.warning(f"No tiktoken encoding for '{tokenizer_model_name}'. Using cl100k_base."); self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e: logger.error(f"Failed load tiktoken tokenizer: {e}."); self.tokenizer = None
        logger.info(f"Initialized OpenAILLM with model {self.model_name}")


    # --- Helper method with retry logic ---
    @retry_on_transient_error
    @retry_on_rate_limit
    async def _create_chat_completion(self, **kwargs) -> 'openai.types.chat.ChatCompletion': # <-- CORRECTED TYPE HINT
        """Internal helper to create chat completion with retry logic."""
        if not hasattr(self, 'client') or not self.client: raise RuntimeError("OpenAI client not initialized.")
        # Type hint uses string forward reference
        response: openai.types.chat.ChatCompletion = await self.client.chat.completions.create(**kwargs)
        return response

    # --- generate method (unchanged logic, uses helper) ---
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # (Implementation remains the same - calls _create_chat_completion)
        temp = temperature if temperature is not None else self.temperature
        max_out_tokens = max_tokens if max_tokens is not None else self.max_tokens
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = await self._create_chat_completion(model=self.model_name, messages=messages, temperature=temp, max_tokens=max_out_tokens, **kwargs)
            completion_content = response.choices[0].message.content; usage = response.usage; usage_dict = None
            if usage:
                prompt_tokens = usage.prompt_tokens; completion_tokens = usage.completion_tokens
                self.update_usage_stats(prompt_tokens, completion_tokens)
                usage_dict = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": usage.total_tokens}
            return {"content": completion_content or "", "model": self.model_name, "usage": usage_dict, "finish_reason": response.choices[0].finish_reason}
        except RetryError as e: logger.error(f"OpenAI API call failed after retries: {e}"); original_error = e.cause if hasattr(e, 'cause') else e; return {"error": f"OpenAI API call failed after retries: {original_error}", "content": None}
        except AuthenticationError as e: logger.error(f"OpenAI Authentication Error: {e}"); return {"error": f"OpenAI Authentication Error: Invalid API Key or credentials.", "content": None}
        except APIError as e:
            logger.error(f"OpenAI API Error: {e.status_code} - {e.message}"); error_msg = e.message
            error_code = getattr(e, 'code', None)
            if error_code == 'insufficient_quota': error_msg = "OpenAI Error: Insufficient quota. Please check plan and billing details."
            elif error_code == 'invalid_api_key': error_msg = "OpenAI Error: Invalid API Key provided."
            return {"error": f"OpenAI API Error ({e.status_code}): {error_msg}", "content": None}
        except Exception as e: logger.error(f"Error generating response from OpenAI: {e}", exc_info=True); return {"error": str(e), "content": None}


    # --- generate_with_tools method (unchanged logic, uses helper) ---
    async def generate_with_tools(
        self, prompt: str, tools: List[Dict[str, Any]], system_prompt: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict]] = "auto", **kwargs
    ) -> Dict[str, Any]:
        # (Implementation remains the same - calls _create_chat_completion)
        temp = temperature if temperature is not None else self.temperature
        max_out_tokens = max_tokens if max_tokens is not None else self.max_tokens
        messages = []; openai_tools = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if tools:
            for tool_def in tools:
                if not isinstance(tool_def, dict) or "name" not in tool_def or "parameters" not in tool_def: logger.warning(f"Skipping invalid tool: {tool_def}"); continue
                params = tool_def.get("parameters", {"type": "object", "properties": {}})
                if not isinstance(params, dict) or params.get("type") != "object": params = {"type": "object", "properties": {}}
                openai_tools.append({"type": "function", "function": {"name": tool_def["name"], "description": tool_def.get("description", ""), "parameters": params}})
        try:
            response = await self._create_chat_completion(model=self.model_name, messages=messages, tools=openai_tools or None, tool_choice=tool_choice if openai_tools else None, temperature=temp, max_tokens=max_out_tokens, **kwargs)
            response_message = response.choices[0].message; usage = response.usage
            result: Dict[str, Any] = {"content": response_message.content or "", "tool_calls": [], "model": self.model_name, "usage": None, "finish_reason": response.choices[0].finish_reason}
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if tool_call.type == "function":
                        try: arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError: logger.error(f"Failed parse args JSON for '{tool_call.function.name}': {tool_call.function.arguments}"); arguments = {"error": "Failed to parse arguments JSON"}
                        result["tool_calls"].append({"name": tool_call.function.name, "parameters": arguments, "id": tool_call.id})
            if usage:
                prompt_tokens = usage.prompt_tokens; completion_tokens = usage.completion_tokens
                self.update_usage_stats(prompt_tokens, completion_tokens)
                result["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": usage.total_tokens}
            return result
        except RetryError as e: logger.error(f"OpenAI API call (tools) failed after retries: {e}"); original_error = e.cause if hasattr(e, 'cause') else e; return {"error": f"OpenAI API call failed after retries: {original_error}", "content": None, "tool_calls": []}
        except AuthenticationError as e: logger.error(f"OpenAI Authentication Error (tools): {e}"); return {"error": f"OpenAI Authentication Error: Invalid API Key or credentials.", "content": None, "tool_calls": []}
        except APIError as e:
            logger.error(f"OpenAI API Error (tools): {e.status_code} - {e.message}"); error_msg = e.message
            error_code = getattr(e, 'code', None)
            if error_code == 'insufficient_quota': error_msg = "OpenAI Error: Insufficient quota. Check plan/billing."
            elif error_code == 'invalid_api_key': error_msg = "OpenAI Error: Invalid API Key."
            return {"error": f"OpenAI API Error ({e.status_code}): {error_msg}", "content": None, "tool_calls": []}
        except Exception as e: logger.error(f"Error generating response with tools (OpenAI): {e}", exc_info=True); return {"error": str(e), "content": None, "tool_calls": []}

    # --- tokenize method (unchanged) ---
    def tokenize(self, text: str) -> List[int]:
        if not OPENAI_AVAILABLE or self.tokenizer is None: logger.warning("Tiktoken unavailable/failed init. Estimating token count."); return list(range(len(text or "") // 4))
        try: return self.tokenizer.encode(text or "")
        except Exception as e: logger.error(f"Tiktoken encoding failed: {e}. Estimating."); return list(range(len(text or "") // 4))

    # get_stats method (unchanged)
    def get_stats(self) -> Dict[str, Any]: return self.usage_stats.copy()