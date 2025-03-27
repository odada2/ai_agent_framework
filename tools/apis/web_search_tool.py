"""
Web Search Tool

This module provides a simplified import for the WebSearchTool class.
The actual implementation has been moved to the web_search package.
"""

import os
import logging
from typing import Dict, List, Optional, Any

from ...core.tools.base import BaseTool
from .web_search import WebSearchTool, SearchResult

logger = logging.getLogger(__name__)


# Re-export the WebSearchTool for backward compatibility
__all__ = ['WebSearchTool']