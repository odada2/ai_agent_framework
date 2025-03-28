# ai_agent_framework/tools/apis/web_search/api_key_manager.py

"""
API Key Manager

This module provides a utility class for managing and rotating API keys
for external services used by search providers. It handles key status tracking,
including rate limits and validity.
"""

import os
import time
import logging
import threading
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field

from .base import ApiKeyError  # Assuming ApiKeyError is defined in base.py

logger = logging.getLogger(__name__)

@dataclass
class ApiKeyStatus:
    """Holds the status information for a single API key."""
    key: str
    provider: str
    is_valid: bool = True
    is_rate_limited: bool = False
    rate_limit_reset_time: Optional[float] = None
    error_count: int = 0
    last_used: float = 0.0
    last_error_time: Optional[float] = None
    calls_remaining: Optional[int] = None # Optional: For providers that return remaining calls

    def is_active(self) -> bool:
        """Check if the key is currently usable."""
        if not self.is_valid:
            return False
        if self.is_rate_limited:
            # Use monotonic time for checking intervals
            if self.rate_limit_reset_time and time.monotonic() < self.rate_limit_reset_time:
                return False
            # Reset rate limit status if time has passed
            self.is_rate_limited = False
            self.rate_limit_reset_time = None
        return True

    def time_to_reset(self) -> float:
        """Calculate remaining time until rate limit reset in seconds."""
        if self.is_rate_limited and self.rate_limit_reset_time:
            return max(0.0, self.rate_limit_reset_time - time.monotonic())
        return 0.0

    def mark_error(self, is_rate_limit: bool = False, is_authentication_error: bool = False, reset_time: Optional[float] = None):
        """Mark an error for this key."""
        self.error_count += 1
        self.last_error_time = time.monotonic()
        if is_authentication_error:
            self.is_valid = False
            logger.warning(f"API key for {self.provider} marked as invalid: {self.key[:4]}...{self.key[-4:]}")
        elif is_rate_limit:
            self.is_rate_limited = True
            # Use monotonic time for reset calculation
            self.rate_limit_reset_time = reset_time if reset_time else (time.monotonic() + 60) # Default 60s cooldown
            logger.warning(f"API key for {self.provider} marked as rate-limited: {self.key[:4]}...{self.key[-4:]}. Reset in {self.time_to_reset():.1f}s.")

    def mark_success(self, calls_remaining: Optional[int] = None):
         """Mark successful usage of the key."""
         # Reset error count on success, potentially? Or keep it for longer-term health?
         # self.error_count = 0 # Optional: Reset errors on success
         self.is_rate_limited = False # Assume success means not rate limited for now
         self.rate_limit_reset_time = None
         self.last_used = time.monotonic()
         if calls_remaining is not None:
             self.calls_remaining = calls_remaining


class ApiKeyManager:
    """
    Manages a pool of API keys for a specific provider.

    Handles key rotation, tracks key status (valid, rate-limited), and provides
    methods for providers to report key usage success or failure.
    """
    MAX_ERRORS_BEFORE_INVALID = 5 # Mark as invalid after this many consecutive errors (excluding rate limits)

    def __init__(
        self,
        provider: str,
        keys: Optional[List[str]] = None,
        env_var: Optional[str] = None,
        auto_rotate: bool = True
    ):
        """
        Initialize the ApiKeyManager.

        Args:
            provider: Name of the provider (e.g., 'serper', 'google', 'bing').
            keys: An explicit list of API keys.
            env_var: Environment variable name to load keys from (if keys list is None).
                     Keys can be comma-separated in the environment variable.
            auto_rotate: Automatically rotate to the next key on failure.
        """
        self.provider = provider
        self.auto_rotate = auto_rotate
        self.keys: Dict[str, ApiKeyStatus] = {}
        self._key_list: List[str] = [] # Maintain order for rotation
        self._current_key_index: int = 0
        self._lock = threading.RLock() # For thread safety during key access/rotation

        loaded_keys = keys or []
        if not loaded_keys and env_var:
            env_val = os.environ.get(env_var)
            if env_val:
                # Split by comma and strip whitespace
                loaded_keys = [key.strip() for key in env_val.split(',') if key.strip()]

        if not loaded_keys:
            logger.warning(f"No API keys provided or found in env var '{env_var}' for provider '{provider}'.")
            # Raise error? Or allow empty manager? Allow empty for now.
            # raise ValueError(f"No API keys found for provider '{provider}'")
        else:
            for key in loaded_keys:
                if key not in self.keys:
                    self.keys[key] = ApiKeyStatus(key=key, provider=provider)
                    self._key_list.append(key)
            logger.info(f"Initialized ApiKeyManager for '{provider}' with {len(self._key_list)} key(s).")


    def get_key(self) -> str:
        """
        Get an active API key. Rotates keys if necessary and possible.

        Returns:
            An active API key string.

        Raises:
            ApiKeyError: If no active keys are available.
        """
        with self._lock:
            if not self._key_list:
                raise ApiKeyError(f"No API keys configured for provider '{self.provider}'.")

            initial_index = self._current_key_index
            attempts = 0

            while attempts < len(self._key_list):
                current_key_str = self._key_list[self._current_key_index]
                key_status = self.keys.get(current_key_str)

                if key_status and key_status.is_active():
                    key_status.last_used = time.monotonic()
                    # logger.debug(f"Providing key index {self._current_key_index} for {self.provider}")
                    return current_key_str

                # If current key is not active, move to the next one
                self._current_key_index = (self._current_key_index + 1) % len(self._key_list)
                attempts += 1

                # If we've checked all keys and returned to the start, break
                if self._current_key_index == initial_index and attempts >= len(self._key_list):
                     break

            # If we exit the loop, no active key was found
            # Find the key with the minimum reset time if any are rate-limited
            min_reset_time = float('inf')
            rate_limited_key_found = False
            for key_status in self.keys.values():
                if key_status.is_rate_limited:
                    rate_limited_key_found = True
                    min_reset_time = min(min_reset_time, key_status.time_to_reset())

            if rate_limited_key_found:
                 raise ApiKeyError(f"All keys for '{self.provider}' are currently rate-limited. Try again in {min_reset_time:.1f} seconds.")
            else:
                 raise ApiKeyError(f"No valid/active API keys available for provider '{self.provider}'. Check key validity.")

    def report_error(
        self,
        key: str,
        is_rate_limit: bool = False,
        is_authentication_error: bool = False,
        reset_time: Optional[float] = None # Absolute monotonic time for reset
    ):
        """
        Report an error encountered while using a specific key.

        Args:
            key: The API key that encountered the error.
            is_rate_limit: True if the error was a rate limit.
            is_authentication_error: True if the error indicates an invalid key.
            reset_time: The absolute monotonic time when a rate limit should reset.
        """
        with self._lock:
            if key in self.keys:
                key_status = self.keys[key]
                key_status.mark_error(
                    is_rate_limit=is_rate_limit,
                    is_authentication_error=is_authentication_error,
                    reset_time=reset_time
                )

                # Optionally mark as invalid after too many generic errors
                if not is_rate_limit and not is_authentication_error and key_status.error_count >= self.MAX_ERRORS_BEFORE_INVALID:
                    key_status.is_valid = False
                    logger.warning(f"API key for {self.provider} marked as invalid after {key_status.error_count} errors: {key[:4]}...{key[-4:]}")


                # Rotate if auto-rotate is enabled and the current key had the error
                current_key_str = self._key_list[self._current_key_index]
                if self.auto_rotate and key == current_key_str and not key_status.is_active():
                    self._current_key_index = (self._current_key_index + 1) % len(self._key_list)
                    logger.info(f"Auto-rotated to next key for {self.provider} due to error on key {key[:4]}...{key[-4:]}")
            else:
                logger.warning(f"Attempted to report error for unknown key: {key[:4]}...{key[-4:]}")

    def report_success(self, key: str, calls_remaining: Optional[int] = None):
        """
        Report successful usage of a specific key.

        Args:
            key: The API key that was used successfully.
            calls_remaining: Optional information about remaining calls from the provider.
        """
        with self._lock:
            if key in self.keys:
                self.keys[key].mark_success(calls_remaining=calls_remaining)
            else:
                logger.warning(f"Attempted to report success for unknown key: {key[:4]}...{key[-4:]}")

    @property
    def active_keys(self) -> List[str]:
         """Return a list of currently active keys."""
         with self._lock:
             return [key for key, status in self.keys.items() if status.is_active()]

    def get_status(self) -> Dict[str, Any]:
        """Return the status of all managed keys."""
        with self._lock:
            return {
                key: {
                    "is_valid": status.is_valid,
                    "is_active": status.is_active(), # Check current active status
                    "is_rate_limited": status.is_rate_limited,
                    "time_to_reset": status.time_to_reset(),
                    "error_count": status.error_count,
                    "last_used_ago": time.monotonic() - status.last_used if status.last_used > 0 else None,
                    "calls_remaining": status.calls_remaining
                }
                for key, status in self.keys.items()
            }