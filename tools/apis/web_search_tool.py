"""
Web Search Tool

This module provides a simplified import for the WebSearchTool class.

The actual implementation resides in the web_search package to support modular
organization and better maintainability. This file serves as a convenience wrapper
to maintain backward compatibility and simplify imports in agent configurations.

Usage:
    from ai_agent_framework.tools.apis.web_search_tool import WebSearchTool
    
    # Create a web search tool
    search_tool = WebSearchTool()
    
    # Register with agent
    agent.register_tool(search_tool)

For advanced configurations and customization, you can import directly from the package:
    from ai_agent_framework.tools.apis.web_search.web_search import WebSearchTool
    from ai_agent_framework.tools.apis.web_search.providers.google import GoogleSearchProvider
    
    provider = GoogleSearchProvider(api_keys=["your_key"])
    search_tool = WebSearchTool(providers={"google": provider})
"""

import logging
from typing import Dict, List, Optional, Any
from importlib import import_module

logger = logging.getLogger(__name__)

try:
    # Import the real implementation
    from .web_search.web_search import WebSearchTool
    
    # Re-export for backward compatibility
    __all__ = ['WebSearchTool']
    
except ImportError as e:
    # Handle missing dependencies gracefully
    logger.warning(f"Could not import WebSearchTool: {str(e)}")
    logger.warning("Make sure all required dependencies are installed.")
    logger.warning("You can install them with: pip install 'ai_agent_framework[web]'")
    
    # Define a placeholder class for type checking
    class WebSearchTool:
        """
        Placeholder for WebSearchTool when dependencies are missing.
        
        This class raises ImportError when instantiated to provide a clear error message.
        """
        
        def __init__(self, *args, **kwargs):
            """Raise ImportError when instantiated."""
            raise ImportError(
                "WebSearchTool dependencies are not installed. "
                "Install them with: pip install 'ai_agent_framework[web]'"
            )