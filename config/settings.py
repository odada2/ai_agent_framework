"""
Configuration System

This module provides functionality for managing application configuration.
"""

import json
import logging
import os
import yaml
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Settings:
    """
    Manages configuration settings for the AI agent framework.
    
    This class handles loading and accessing configuration from various sources
    with a clear precedence order: environment variables > config file > defaults.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        env_prefix: str = "AI_AGENT",
        defaults_path: str = "config/default_config.yaml"
    ):
        """
        Initialize the settings manager.
        
        Args:
            config_path: Optional path to a configuration file
            env_prefix: Prefix for environment variables to include
            defaults_path: Path to the default configuration file
        """
        self.env_prefix = env_prefix
        self.config_path = config_path
        self.defaults_path = defaults_path
        
        # Configuration storage with precedence:
        # 1. Environment variables
        # 2. Config file
        # 3. Default config
        self._env_config = {}
        self._file_config = {}
        self._default_config = {}
        
        # Load configurations in reverse order of precedence
        self._load_default_config()
        if config_path:
            self._load_config_file()
        self._load_env_variables()
    
    def _load_default_config(self) -> None:
        """
        Load default configuration from the default config file.
        """
        try:
            if os.path.exists(self.defaults_path):
                with open(self.defaults_path, 'r') as f:
                    self._default_config = yaml.safe_load(f) or {}
                    logger.debug(f"Loaded default configuration from {self.defaults_path}")
        except Exception as e:
            logger.warning(f"Error loading default config: {str(e)}")
            self._default_config = {}
    
    def _load_config_file(self) -> None:
        """
        Load configuration from the specified config file.
        """
        if not self.config_path:
            return
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    # Determine file type based on extension
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        self._file_config = yaml.safe_load(f) or {}
                    elif self.config_path.endswith('.json'):
                        self._file_config = json.load(f) or {}
                    else:
                        logger.warning(f"Unsupported config file format: {self.config_path}")
                        return
                    
                    logger.debug(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.warning(f"Error loading config file: {str(e)}")
            self._file_config = {}
    
    def _load_env_variables(self) -> None:
        """
        Load configuration from environment variables matching the prefix.
        
        Environment variables should follow the pattern:
        PREFIX_SECTION_KEY=value
        
        This will become:
        {
            "section": {
                "key": "value"
            }
        }
        """
        prefix = f"{self.env_prefix}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and split by underscore
                parts = key[len(prefix):].lower().split('_')
                
                if len(parts) < 2:
                    # Need at least section and key
                    continue
                
                # Last part is the key, everything before is the nested path
                config_key = parts[-1]
                section_path = parts[:-1]
                
                # Convert string values to appropriate types
                if value.lower() == 'true':
                    typed_value = True
                elif value.lower() == 'false':
                    typed_value = False
                elif value.isdigit():
                    typed_value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    typed_value = float(value)
                else:
                    typed_value = value
                
                # Navigate to the right section
                current = self._env_config
                for section in section_path:
                    if section not in current:
                        current[section] = {}
                    current = current[section]
                
                # Set the key value
                current[config_key] = typed_value
        
        logger.debug(f"Loaded {len(self._env_config)} configuration sections from environment variables")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration key (e.g., 'llm.claude.api_key')
            default: Default value if the key doesn't exist in any config source
            
        Returns:
            The configuration value or the default
        """
        parts = key_path.lower().split('.')
        
        # Check environment config first
        value = self._get_nested(self._env_config, parts)
        if value is not None:
            return value
        
        # Then check file config
        value = self._get_nested(self._file_config, parts)
        if value is not None:
            return value
        
        # Finally check default config
        value = self._get_nested(self._default_config, parts)
        if value is not None:
            return value
        
        # Return the provided default if key not found
        return default
    
    def _get_nested(self, config: Dict[str, Any], parts: list) -> Any:
        """
        Get a nested value from a dictionary using a list of keys.
        
        Args:
            config: The configuration dictionary
            parts: List of keys to navigate the nested dictionary
            
        Returns:
            The value if found, None otherwise
        """
        current = config
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current
    
    def set(self, key_path: str, value: Any, save: bool = False) -> None:
        """
        Set a configuration value.
        
        Args:
            key_path: Dot-separated path to the configuration key
            value: Value to set
            save: Whether to save changes to the config file
        """
        parts = key_path.lower().split('.')
        
        # Set in the file config
        current = self._file_config
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Last part is the key
                current[part] = value
            else:
                # Navigate to the right section
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Save to file if requested
        if save and self.config_path:
            self.save_config()
    
    def save_config(self, filepath: Optional[str] = None) -> None:
        """
        Save the current file configuration to disk.
        
        Args:
            filepath: Optional alternative path to save to
        """
        save_path = filepath or self.config_path
        if not save_path:
            logger.error("No config file path specified for saving")
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save based on file extension
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(self._file_config, f, default_flow_style=False)
                elif save_path.endswith('.json'):
                    json.dump(self._file_config, f, indent=2)
                else:
                    logger.error(f"Unsupported config file format: {save_path}")
                    return
            
            logger.debug(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the complete configuration as a dictionary.
        
        This merges all configuration sources according to precedence.
        
        Returns:
            Dictionary containing all configuration values
        """
        # Start with default config
        result = {}
        self._deep_merge(result, self._default_config)
        
        # Apply file config
        self._deep_merge(result, self._file_config)
        
        # Apply env config
        self._deep_merge(result, self._env_config)
        
        return result
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge source dict into target dict.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # If both are dicts, recursively merge them
                self._deep_merge(target[key], value)
            else:
                # Otherwise, source overwrites target
                target[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to configuration.
        
        Args:
            key: The configuration key
            
        Returns:
            The configuration value
            
        Raises:
            KeyError: If the key doesn't exist
        """
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: The configuration key
            
        Returns:
            True if the key exists, False otherwise
        """
        return self.get(key) is not None