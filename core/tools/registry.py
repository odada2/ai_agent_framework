"""
Tool Registry

This module provides a registry for managing and accessing tools.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing and accessing tools.
    
    This class provides functionality to register, access, and execute tools.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register_tool(self, tool: BaseTool, categories: Optional[List[str]] = None) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: The tool to register
            categories: Optional list of categories to organize the tool
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"A tool with name '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        
        # Register categories if provided
        if categories:
            for category in categories:
                if category not in self._categories:
                    self._categories[category] = []
                self._categories[category].append(tool.name)
        
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the registry.
        
        Args:
            tool_name: Name of the tool to remove
            
        Raises:
            ValueError: If the tool is not registered
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        # Remove from categories
        for category, tools in self._categories.items():
            if tool_name in tools:
                self._categories[category].remove(tool_name)
        
        # Remove from tools
        del self._tools[tool_name]
        logger.debug(f"Unregistered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> BaseTool:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            The requested tool
            
        Raises:
            ValueError: If the tool is not registered
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        return self._tools[tool_name]
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool is registered, False otherwise
        """
        return tool_name in self._tools
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name with the provided parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If the tool is not registered
        """
        tool = self.get_tool(tool_name)
        logger.debug(f"Executing tool: {tool_name}")
        return tool.execute(**kwargs)
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        return list(self._tools.values())
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of tools in the category
        """
        if category not in self._categories:
            return []
        
        return [self._tools[name] for name in self._categories[category]]
    
    def get_tool_names(self) -> List[str]:
        """
        Get the names of all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_categories(self) -> List[str]:
        """
        Get all registered categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get definitions for all tools in a format suitable for LLMs.
        
        Returns:
            List of tool definitions
        """
        return [tool.get_definition() for tool in self._tools.values()]
    
    def get_tool_descriptions(self) -> str:
        """
        Get a formatted string describing all tools for inclusion in prompts.
        
        Returns:
            Formatted string describing all tools
        """
        if not self._tools:
            return "No tools available."
        
        descriptions = []
        for tool_name, tool in self._tools.items():
            desc = f"- {tool_name}: {tool.description}"
            
            # Add parameter information
            params = tool.parameters.get("properties", {})
            if params:
                desc += "\n  Parameters:"
                for param_name, param_info in params.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    required = "required" if param_name in tool.parameters.get("required", []) else "optional"
                    desc += f"\n    - {param_name} ({param_type}, {required}): {param_desc}"
            
            descriptions.append(desc)
        
        return "\n\n".join(descriptions)
    
    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return self.has_tool(tool_name)