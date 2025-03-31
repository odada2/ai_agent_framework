# ai_agent_framework/core/tools/registry.py

"""
Tool Registry

This module provides a registry for managing and accessing tools asynchronously.
"""

import asyncio
import logging
import json # <-- IMPORT ADDED HERE
from typing import Any, Dict, List, Optional, Union

# Assuming BaseTool can be imported correctly
try:
    from .base import BaseTool
except ImportError:
    # Fallback for different execution contexts
    from core.tools.base import BaseTool


logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing and accessing tools.

    Provides asynchronous methods for executing tools, handling both sync and async tool implementations.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {} # category_name -> List[tool_name]

    def register_tool(self, tool: BaseTool, categories: Optional[List[str]] = None) -> None:
        """
        Register a tool with the registry. (Synchronous)

        Args:
            tool: The tool instance to register.
            categories: Optional list of categories for organization.

        Raises:
            ValueError: If a tool with the same name is already registered or tool is invalid.
            TypeError: If the provided tool is not an instance of BaseTool.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Object provided is not an instance of BaseTool: {type(tool)}")
        if not tool.name or not isinstance(tool.name, str):
             raise ValueError("Tool must have a valid string name.")

        if tool.name in self._tools:
            # Allow overriding/updating if needed? For now, raise error.
            raise ValueError(f"A tool with name '{tool.name}' is already registered")

        self._tools[tool.name] = tool

        # Register categories if provided
        if categories:
            for category in categories:
                if not isinstance(category, str): continue # Skip non-string categories
                cat_list = self._categories.setdefault(category, [])
                if tool.name not in cat_list:
                    cat_list.append(tool.name)

        logger.debug(f"Registered tool: {tool.name} with categories: {categories or 'None'}")

    def unregister_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the registry. (Synchronous)
        """
        if tool_name not in self._tools:
            # Optionally return False or log warning instead of raising error
            logger.warning(f"Attempted to unregister non-existent tool: {tool_name}")
            return # Or raise ValueError(f"Tool '{tool_name}' is not registered")

        # Remove from categories
        categories_to_remove_from = []
        for category, tools_in_cat in self._categories.items():
            if tool_name in tools_in_cat:
                tools_in_cat.remove(tool_name)
                if not tools_in_cat: # Check if category is now empty
                     categories_to_remove_from.append(category)

        # Remove empty categories
        for category in categories_to_remove_from:
             if category in self._categories: # Check existence before deleting
                  del self._categories[category]

        # Remove from tools map
        if tool_name in self._tools:
             del self._tools[tool_name]
             logger.debug(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> BaseTool:
        """
        Get a tool by name. (Synchronous)
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        return tool

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered. (Synchronous)"""
        return tool_name in self._tools

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool asynchronously by name with the provided parameters.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Parameters to pass to the tool's execute method.

        Returns:
            The result of the tool execution, or an error dictionary.

        Raises:
            ValueError: If the tool is not registered.
        """
        try:
            tool = self.get_tool(tool_name) # get_tool is sync and raises ValueError if not found
        except ValueError as e:
            logger.error(f"Cannot execute tool: {e}")
            return {"error": str(e)} # Return error dict for consistency with BaseTool.execute

        logger.debug(f"Executing tool asynchronously: {tool_name}")
        # Await the tool's async execute method
        # BaseTool.execute now catches exceptions and returns an error dict
        return await tool.execute(**kwargs)

    # Methods below remain synchronous as they only read internal state

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a specific category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_tool_names(self) -> List[str]:
        """Get the names of all registered tools."""
        return list(self._tools.keys())

    def get_categories(self) -> List[str]:
        """Get all registered categories."""
        return list(self._categories.keys())

    def get_tool_definitions(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get definitions for specified tools (or all tools) in a format suitable for LLMs.

        Args:
            tool_names: Optional list of specific tool names to get definitions for.
                        If None, returns definitions for all registered tools.

        Returns:
            List of tool definition dictionaries.
        """
        target_tools = []
        names_to_get = tool_names if tool_names is not None else self.get_tool_names()

        for name in names_to_get:
             try:
                  target_tools.append(self.get_tool(name))
             except ValueError:
                  logger.warning(f"Requested tool definition for unknown tool: {name}")

        return [tool.get_definition() for tool in target_tools]


    def get_tool_descriptions(self) -> str:
        """
        Get a formatted string describing all tools for inclusion in prompts.
        """
        if not self._tools:
            return "No tools available."

        descriptions = []
        for tool_name, tool in sorted(self._tools.items()): # Sort for consistent order
            # Start with name and description
            desc_parts = [f"- {tool_name}: {tool.description}"]

            # Add parameter information
            params = tool.parameters # Access the property
            properties = params.get("properties")
            if properties: # Check if properties exist
                desc_parts.append("  Parameters:")
                required = params.get("required", [])
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_str = "required" if param_name in required else "optional"
                    default_val = param_info.get("default")
                    # Use json.dumps for default value formatting
                    default_str = f" (default: {json.dumps(default_val)})" if default_val is not None else ""
                    enum_str = f" (options: {', '.join(map(str, param_info['enum']))})" if "enum" in param_info else ""

                    desc_parts.append(f"    - {param_name} ({param_type}, {req_str}{default_str}{enum_str}): {param_desc}")

            descriptions.append("\n".join(desc_parts))

        return "\n\n".join(descriptions)

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return self.has_tool(tool_name)