# ai_agent_framework/config/settings.py

"""
Configuration System

This module provides functionality for managing application configuration.
Loads .env files automatically if python-dotenv is installed.
"""

import json
import logging
import os
import yaml
from typing import Any, Dict, Optional

# --- Attempt to import and use dotenv ---
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
# -----------------------------------------

logger = logging.getLogger(__name__)


class Settings:
    """
    Manages configuration settings for the AI agent framework.

    Handles loading and accessing configuration from various sources
    with precedence: environment variables > config file > defaults.
    Automatically loads variables from a .env file if python-dotenv is installed.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        env_prefix: str = "AI_AGENT",
        defaults_path: Optional[str] = None, # Made optional
        load_env_file: bool = True # Added flag to control .env loading
    ):
        """
        Initialize the settings manager.

        Args:
            config_path: Optional path to a configuration file.
            env_prefix: Prefix for environment variables to include.
            defaults_path: Optional path to the default configuration file. If None, tries common paths.
            load_env_file: If True, attempts to load a .env file using python-dotenv.
        """
        self.env_prefix = env_prefix
        self.config_path = config_path

        # --- Determine default config path ---
        if defaults_path is None:
             # Look in common locations relative to this file or CWD
             script_dir = os.path.dirname(os.path.abspath(__file__))
             potential_defaults = [
                  os.path.join(script_dir, "default_config.yaml"), # Next to settings.py
                  os.path.join(os.path.dirname(script_dir), "config", "default_config.yaml"), # ../config/
                  "config/default_config.yaml", # Relative to CWD
                  "default_config.yaml" # Relative to CWD
             ]
             self.defaults_path = next((p for p in potential_defaults if os.path.exists(p)), None)
        else:
             self.defaults_path = defaults_path
        # -------------------------------------


        # --- Load .env file FIRST if requested and possible ---
        if load_env_file:
            if DOTENV_AVAILABLE:
                # find_dotenv() searches current dir and parents
                # load_dotenv() loads it into os.environ
                from dotenv import find_dotenv
                env_path = find_dotenv(usecwd=True) # Start search from CWD
                if env_path:
                     loaded = load_dotenv(dotenv_path=env_path, override=False) # override=False means existing env vars take precedence
                     if loaded:
                          logger.info(f"Loaded environment variables from: {env_path}")
                     else:
                          logger.debug("No .env file loaded (or it was empty).")
                else:
                     logger.debug("No .env file found to load.")
            else:
                logger.warning("python-dotenv package not found. Cannot load .env file.")
        # --------------------------------------------------------

        # Configuration storage (remains the same)
        self._env_config = {}
        self._file_config = {}
        self._default_config = {}

        # Load configurations (default, file, env) - env vars read AFTER .env load
        self._load_default_config()
        if self.config_path: # Check resolved path
            self._load_config_file()
        self._load_env_variables() # Now os.environ includes .env vars

    def _load_default_config(self) -> None:
        """Load default configuration from the default config file."""
        if not self.defaults_path or not os.path.exists(self.defaults_path):
             logger.warning("Default configuration file not found or specified.")
             self._default_config = {}
             return
        try:
            with open(self.defaults_path, 'r', encoding='utf-8') as f: # Specify encoding
                self._default_config = yaml.safe_load(f) or {}
                logger.debug(f"Loaded default configuration from {self.defaults_path}")
        except Exception as e:
            logger.warning(f"Error loading default config '{self.defaults_path}': {e}")
            self._default_config = {}

    def _load_config_file(self) -> None:
        """Load configuration from the specified config file."""
        if not self.config_path or not os.path.exists(self.config_path):
            logger.debug(f"Config file not specified or not found: {self.config_path}")
            self._file_config = {}
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f: # Specify encoding
                # Determine file type based on extension
                if self.config_path.endswith(('.yaml', '.yml')):
                    self._file_config = yaml.safe_load(f) or {}
                elif self.config_path.endswith('.json'):
                    self._file_config = json.load(f) or {}
                else:
                    logger.warning(f"Unsupported config file format: {self.config_path}. Only .yaml, .yml, .json supported.")
                    self._file_config = {}
                    return # Don't proceed if format is wrong

                logger.debug(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.warning(f"Error loading config file '{self.config_path}': {e}")
            self._file_config = {}

    def _load_env_variables(self) -> None:
        """
        Load configuration from environment variables matching the prefix.
        Environment variables take precedence over .env file variables if override=False was used in load_dotenv.
        """
        loaded_count = 0
        prefix = f"{self.env_prefix}_" if self.env_prefix else "" # Handle empty prefix case

        for key, value in os.environ.items():
            if not prefix or key.startswith(prefix): # Check prefix if it exists
                # Remove prefix and split by double underscore (common practice) or single
                # Try double underscore first for structure like AI_AGENT__LLM__MODEL=...
                if '__' in key:
                    parts = key[len(prefix):].lower().split('__')
                else: # Fallback to single underscore
                     parts = key[len(prefix):].lower().split('_')

                if not parts or len(parts) < 1: # Need at least one part (the key itself)
                    continue

                config_key = parts[-1]
                section_path = parts[:-1]

                # Convert string values to appropriate types (basic conversion)
                if value.lower() == 'true':
                    typed_value: Any = True
                elif value.lower() == 'false':
                    typed_value = False
                elif value.isdigit():
                    typed_value = int(value)
                # Slightly more robust float check
                elif value.replace('.', '', 1).replace('-', '', 1).isdigit() and value.count('.') <= 1:
                    try:
                        typed_value = float(value)
                    except ValueError: # Handle cases like '-' or '.'
                        typed_value = value
                elif value.startswith('[') and value.endswith(']'): # Basic list check (e.g., from JSON)
                     try: typed_value = json.loads(value)
                     except json.JSONDecodeError: typed_value = value # Keep as string if not valid JSON list
                elif value.startswith('{') and value.endswith('}'): # Basic dict check
                     try: typed_value = json.loads(value)
                     except json.JSONDecodeError: typed_value = value
                else:
                    typed_value = value

                # Navigate to the right section
                current = self._env_config
                try:
                    for section in section_path:
                         # Create nested dict if needed
                         current = current.setdefault(section, {})
                         if not isinstance(current, dict): # Check if a non-dict value exists at this level
                             logger.warning(f"Environment variable '{key}' conflicts with existing non-dictionary structure at '{'.'.join(section_path)}'. Skipping.")
                             current = None # Mark as invalid path
                             break
                except TypeError: # Handles case where setdefault fails (current is not dict) - should be caught above
                      logger.warning(f"Environment variable '{key}' path conflicts. Skipping.")
                      current = None


                # Set the key value if path was valid
                if current is not None and isinstance(current, dict):
                    current[config_key] = typed_value
                    loaded_count += 1
                elif current is None:
                    # Logged warning above
                    pass


        if loaded_count > 0:
            logger.debug(f"Loaded {loaded_count} configuration values from environment variables (prefix: '{prefix}').")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its key path (e.g., 'section.subsection.key').
        Precedence: Environment > Config File > Defaults.

        Args:
            key_path: Dot-separated path to the configuration key.
            default: Default value if the key doesn't exist.

        Returns:
            The configuration value or the default.
        """
        parts = key_path.lower().split('.')

        # Check environment config first
        value = self._get_nested(self._env_config, parts)
        if value is not None:
            # logger.debug(f"Setting '{key_path}' found in environment variables.")
            return value

        # Then check file config
        value = self._get_nested(self._file_config, parts)
        if value is not None:
            # logger.debug(f"Setting '{key_path}' found in config file.")
            return value

        # Finally check default config
        value = self._get_nested(self._default_config, parts)
        if value is not None:
            # logger.debug(f"Setting '{key_path}' found in default config.")
            return value

        # logger.debug(f"Setting '{key_path}' not found, returning default: {default}")
        return default

    def _get_nested(self, config: Dict[str, Any], parts: list) -> Any:
        """Helper to get a nested value from a dictionary."""
        current = config
        for part in parts:
            if not isinstance(current, dict):
                 # Trying to access a key on a non-dictionary
                 return None
            # Use .get() for safer access
            current = current.get(part)
            if current is None:
                # Key not found at this level
                return None
        return current

    # set and save_config remain largely the same, ensure UTF-8 writing
    def set(self, key_path: str, value: Any, save: bool = False) -> None:
        """Set a configuration value (primarily affects file config for saving)."""
        parts = key_path.lower().split('.')
        current = self._file_config
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current[part] = value
            else:
                current = current.setdefault(part, {})
                if not isinstance(current, dict):
                     logger.error(f"Cannot set nested key '{key_path}': Path conflicts with non-dictionary value.")
                     return # Cannot proceed

        if save:
            self.save_config()

    def save_config(self, filepath: Optional[str] = None) -> None:
        """Save the current file configuration (_file_config) to disk."""
        save_path = filepath or self.config_path
        if not save_path:
            logger.error("No config file path specified for saving.")
            return

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f: # Use UTF-8
                if save_path.endswith(('.yaml', '.yml')):
                    # Ensure PyYAML is installed if saving YAML
                    try:
                        yaml.dump(self._file_config, f, default_flow_style=False, allow_unicode=True)
                    except NameError:
                        logger.error("PyYAML not installed, cannot save YAML. `pip install pyyaml`")
                        # Optionally fallback to JSON
                        # json.dump(self._file_config, f, indent=2)
                elif save_path.endswith('.json'):
                    json.dump(self._file_config, f, indent=2)
                else:
                    logger.error(f"Unsupported config file format for saving: {save_path}. Use .yaml or .json.")
                    return
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to '{save_path}': {e}")

    def as_dict(self) -> Dict[str, Any]:
        """Get the merged configuration as a dictionary (Env > File > Default)."""
        result = {}
        self._deep_merge(result, self._default_config)
        self._deep_merge(result, self._file_config)
        self._deep_merge(result, self._env_config) # Env vars override last
        return result

    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Recursively merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            # Handle merging lists if needed, otherwise source overwrites
            # elif key in target and isinstance(target[key], list) and isinstance(value, list):
            #     target[key].extend(value) # Or replace: target[key] = value
            else:
                target[key] = value

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found.")
        return value

    def __contains__(self, key: str) -> bool:
        """Check if key exists (e.g., `if 'section.key' in settings:`)."""
        # Check across all merged configs implicitly via get()
        return self.get(key) is not None