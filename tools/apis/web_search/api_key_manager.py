"""
API Key Management for Web Search Providers

This module provides a robust system for managing API keys with:
- Key rotation to prevent rate limiting issues
- Health tracking for keys
- Support for multiple keys per provider
- Automatic fallback if a key becomes invalid or rate limited
"""

import logging
import os
import time
import json
import random
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

from filelock import FileLock

from ...core.tools.base import ApiKeyError

logger = logging.getLogger(__name__)


@dataclass
class ApiKeyStatus:
    """Status tracking for an individual API key"""
    key: str  # The API key (may be partially masked for logging)
    provider: str  # The provider this key is for
    is_valid: bool = True  # Whether the key is currently valid
    is_active: bool = True  # Whether the key is currently in rotation
    error_count: int = 0  # Number of consecutive errors
    last_used: Optional[float] = None  # Last time the key was used (timestamp)
    rate_limit_reset: Optional[float] = None  # When rate limit resets (timestamp)
    calls_remaining: Optional[int] = None  # Rate limit info if available
    total_calls: int = 0  # Total number of calls made with this key
    daily_calls: int = 0  # Calls made today
    last_reset_day: Optional[str] = None  # Last day the daily counter was reset (YYYY-MM-DD)
    
    def update_usage(self) -> None:
        """Update usage statistics when key is used"""
        now = time.time()
        today = datetime.now().strftime("%Y-%m-%d")
        
        self.last_used = now
        self.total_calls += 1
        
        # Reset daily counter if day changed
        if self.last_reset_day != today:
            self.daily_calls = 1
            self.last_reset_day = today
        else:
            self.daily_calls += 1
    
    def mark_error(self, is_rate_limit: bool = False, reset_time: Optional[float] = None) -> None:
        """
        Mark an error for this key.
        
        Args:
            is_rate_limit: Whether this is a rate limit error
            reset_time: When the rate limit resets (if applicable)
        """
        self.error_count += 1
        
        if is_rate_limit:
            self.rate_limit_reset = reset_time or (time.time() + 60)  # Default 1 minute if unknown
            self.calls_remaining = 0
        
        # Deactivate key if too many errors
        if self.error_count >= 5:
            self.is_active = False
    
    def reset_errors(self) -> None:
        """Reset error count after successful use"""
        self.error_count = 0
    
    def is_rate_limited(self) -> bool:
        """Check if key is currently rate limited"""
        if not self.rate_limit_reset:
            return False
        return time.time() < self.rate_limit_reset
    
    def time_to_reset(self) -> Optional[int]:
        """Get seconds until rate limit reset"""
        if not self.rate_limit_reset:
            return None
        return max(0, int(self.rate_limit_reset - time.time()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiKeyStatus':
        """Create from dictionary"""
        return cls(**data)
    
    def get_masked_key(self) -> str:
        """Get a masked version of the key for logging"""
        if not self.key or len(self.key) < 8:
            return "***"
        return f"{self.key[:4]}...{self.key[-4:]}"


class ApiKeyManager:
    """
    Manager for API keys with rotation and health tracking.
    
    Features:
    - Key rotation based on usage and health
    - Persistent storage of key status
    - Rate limit awareness
    - Automatic fallback to backup keys
    """
    
    def __init__(
        self,
        provider: str,
        keys: Optional[List[str]] = None,
        state_file: Optional[str] = None,
        auto_rotate: bool = True
    ):
        """
        Initialize the API key manager.
        
        Args:
            provider: Name of the provider (e.g., 'serper', 'google')
            keys: List of API keys (will also check environment variables if not provided)
            state_file: Path to state file for persistent storage
            auto_rotate: Whether to automatically rotate keys
        """
        self.provider = provider
        self.auto_rotate = auto_rotate
        self.state_file = state_file or self._get_default_state_file()
        
        # Initialize key status tracking
        self.keys: Dict[str, ApiKeyStatus] = {}
        self.active_keys: Set[str] = set()
        self.current_key: Optional[str] = None
        
        # Load keys
        self._load_keys(keys)
        
        # Load state if available
        self._load_state()
        
        # Set initial current key
        self._select_current_key()
    
    def _get_default_state_file(self) -> str:
        """Get default state file path based on provider"""
        state_dir = os.environ.get("API_KEY_STATE_DIR", "./.api_key_state")
        os.makedirs(state_dir, exist_ok=True)
        return f"{state_dir}/{self.provider}_key_state.json"
    
    def _load_keys(self, keys: Optional[List[str]]) -> None:
        """
        Load API keys from the provided list and environment variables.
        
        Args:
            keys: Optional explicit list of keys
        """
        loaded_keys = set()
        
        # Load from provided keys
        if keys:
            for key in keys:
                if key and key.strip():
                    loaded_keys.add(key.strip())
        
        # Load from environment variables
        self._load_keys_from_env(loaded_keys)
        
        # Initialize key status for all keys
        for key in loaded_keys:
            self.keys[key] = ApiKeyStatus(key=key, provider=self.provider)
            self.active_keys.add(key)
        
        logger.info(f"Loaded {len(self.keys)} API keys for {self.provider}")
        
        if not self.keys:
            logger.warning(f"No API keys found for {self.provider}")
    
    def _load_keys_from_env(self, loaded_keys: Set[str]) -> None:
        """
        Load API keys from environment variables.
        
        Args:
            loaded_keys: Set to add discovered keys to
        """
        # Check provider-specific single key env var
        env_var_name = f"{self.provider.upper()}_API_KEY"
        if key := os.environ.get(env_var_name):
            loaded_keys.add(key.strip())
        
        # Check general API key env var with provider prefix
        env_var_name = f"API_KEY_{self.provider.upper()}"
        if key := os.environ.get(env_var_name):
            loaded_keys.add(key.strip())
        
        # Check for numbered keys for rotation
        for i in range(1, 10):  # Check up to 9 numbered keys
            env_var_name = f"{self.provider.upper()}_API_KEY_{i}"
            if key := os.environ.get(env_var_name):
                loaded_keys.add(key.strip())
    
    def _load_state(self) -> None:
        """Load key state from state file if available"""
        if not self.state_file or not Path(self.state_file).exists():
            return
        
        try:
            with FileLock(f"{self.state_file}.lock"):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Process each key
                for key_data in state_data.get('keys', []):
                    key = key_data.get('key')
                    if key in self.keys:
                        # Update existing key status
                        self.keys[key] = ApiKeyStatus.from_dict(key_data)
                        
                        # Update active keys set
                        if self.keys[key].is_active:
                            self.active_keys.add(key)
                        else:
                            self.active_keys.discard(key)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading API key state: {str(e)}")
    
    def _save_state(self) -> None:
        """Save current key state to state file"""
        if not self.state_file:
            return
        
        try:
            state_dir = os.path.dirname(self.state_file)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
                
            with FileLock(f"{self.state_file}.lock"):
                state_data = {
                    'provider': self.provider,
                    'updated_at': datetime.now().isoformat(),
                    'keys': [key_status.to_dict() for key_status in self.keys.values()]
                }
                
                with open(self.state_file, 'w') as f:
                    json.dump(state_data, f, indent=2)
        except OSError as e:
            logger.error(f"Error saving API key state: {str(e)}")
    
    def _select_current_key(self) -> Optional[str]:
        """
        Select the best key to use based on health and usage.
        
        Returns:
            Selected API key or None if no keys available
        """
        if not self.active_keys:
            self.current_key = None
            return None
        
        # Get candidate keys (not rate limited)
        candidates = [k for k in self.active_keys if not self.keys[k].is_rate_limited()]
        
        if not candidates:
            # If all keys are rate limited, find the one that will reset soonest
            min_reset_time = float('inf')
            soonest_key = None
            
            for key in self.active_keys:
                reset_time = self.keys[key].time_to_reset()
                if reset_time is not None and reset_time < min_reset_time:
                    min_reset_time = reset_time
                    soonest_key = key
            
            self.current_key = soonest_key
            return soonest_key
        
        # Select key with fewest daily calls
        candidates.sort(key=lambda k: self.keys[k].daily_calls)
        self.current_key = candidates[0]
        return self.current_key
    
    def get_key(self) -> str:
        """
        Get the current API key to use.
        
        Returns:
            Current API key
            
        Raises:
            ApiKeyError: If no valid keys are available
        """
        if not self.keys:
            raise ApiKeyError(f"No API keys configured for {self.provider}")
        
        if not self.active_keys:
            raise ApiKeyError(f"All API keys for {self.provider} are invalid or deactivated")
        
        # Check if current key needs rotation
        if self.auto_rotate and self.current_key:
            status = self.keys[self.current_key]
            
            if status.is_rate_limited() or not status.is_active:
                self._select_current_key()
        
        # If no current key, select one
        if not self.current_key:
            self._select_current_key()
        
        # If still no key, all are rate limited
        if not self.current_key:
            min_wait = min(
                self.keys[k].time_to_reset() or 60 
                for k in self.active_keys
            )
            raise ApiKeyError(
                f"All API keys for {self.provider} are rate limited. "
                f"Try again in {min_wait} seconds."
            )
        
        # Update usage statistics
        self.keys[self.current_key].update_usage()
        self._save_state()
        
        return self.current_key
    
    def report_success(self, key: Optional[str] = None, calls_remaining: Optional[int] = None) -> None:
        """
        Report successful API call for a key.
        
        Args:
            key: The key that was used (defaults to current key)
            calls_remaining: Optional rate limit information
        """
        key = key or self.current_key
        if not key or key not in self.keys:
            return
        
        status = self.keys[key]
        status.reset_errors()
        
        if calls_remaining is not None:
            status.calls_remaining = calls_remaining
        
        self._save_state()
    
    def report_error(
        self,
        key: Optional[str] = None,
        is_rate_limit: bool = False,
        reset_time: Optional[float] = None,
        is_authentication_error: bool = False
    ) -> None:
        """
        Report an error with an API key.
        
        Args:
            key: The key that had an error (defaults to current key)
            is_rate_limit: Whether this was a rate limit error
            reset_time: When the rate limit resets (if known)
            is_authentication_error: Whether this was an authentication error
        """
        key = key or self.current_key
        if not key or key not in self.keys:
            return
        
        status = self.keys[key]
        
        # Handle authentication errors (invalid key)
        if is_authentication_error:
            status.is_valid = False
            status.is_active = False
            self.active_keys.discard(key)
            logger.warning(f"API key {status.get_masked_key()} for {self.provider} is invalid")
        
        # Handle rate limit errors
        elif is_rate_limit:
            status.mark_error(is_rate_limit=True, reset_time=reset_time)
            logger.info(
                f"API key {status.get_masked_key()} for {self.provider} hit rate limit. "
                f"Will reset in {status.time_to_reset()} seconds."
            )
        
        # Handle other errors
        else:
            status.mark_error()
            
            # Deactivate key if too many errors
            if not status.is_active:
                self.active_keys.discard(key)
                logger.warning(
                    f"API key {status.get_masked_key()} for {self.provider} deactivated "
                    f"due to {status.error_count} consecutive errors"
                )
        
        # Select a new key if current key had an error
        if key == self.current_key and (is_rate_limit or not status.is_active):
            self.current_key = None
            self._select_current_key()
        
        self._save_state()
    
    def reactivate_key(self, key: str) -> bool:
        """
        Attempt to reactivate a deactivated key.
        
        Args:
            key: The key to reactivate
            
        Returns:
            True if key was reactivated, False otherwise
        """
        if key not in self.keys:
            return False
        
        status = self.keys[key]
        
        # Only reactivate if key is valid but inactive
        if status.is_valid and not status.is_active:
            status.is_active = True
            status.error_count = 0
            self.active_keys.add(key)
            
            logger.info(f"Reactivated API key {status.get_masked_key()} for {self.provider}")
            self._save_state()
            return