"""
Tool Documentation Utilities

This module provides utilities for generating and formatting tool documentation.
"""

import json
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


class ToolDocumenter:
    """
    Utility class for generating and formatting tool documentation.
    """
    
    @staticmethod
    def generate_tool_guide(
        registry: ToolRegistry,
        format_type: str = "markdown",
        include_examples: bool = True
    ) -> str:
        """
        Generate formatted documentation for all tools in a registry.
        
        Args:
            registry: The tool registry to document
            format_type: The format to generate (markdown, text)
            include_examples: Whether to include examples
            
        Returns:
            Formatted tool documentation
        """
        if format_type.lower() == "markdown":
            return ToolDocumenter._generate_markdown_docs(registry, include_examples)
        else:
            return ToolDocumenter._generate_text_docs(registry, include_examples)
    
    @staticmethod
    def _generate_markdown_docs(registry: ToolRegistry, include_examples: bool) -> str:
        """
        Generate markdown documentation for tools.
        
        Args:
            registry: The tool registry to document
            include_examples: Whether to include examples
            
        Returns:
            Markdown formatted documentation
        """
        docs = ["# Available Tools\n"]
        
        # First list all tools as a table of contents
        docs.append("## Tool List\n")
        for tool_name in registry.get_tool_names():
            docs.append(f"- [{tool_name}](#{tool_name.lower().replace(' ', '-')})")
        docs.append("\n")
        
        # Then document each tool in detail
        docs.append("## Tool Details\n")
        for tool_name in registry.get_tool_names():
            tool = registry.get_tool(tool_name)
            
            docs.append(f"### {tool_name}\n")
            docs.append(f"{tool.description}\n")
            
            # Parameters section
            if tool.parameters and "properties" in tool.parameters:
                docs.append("#### Parameters\n")
                required = tool.parameters.get("required", [])
                
                for param_name, param_info in tool.parameters["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_str = "**Required**" if param_name in required else "Optional"
                    
                    docs.append(f"- `{param_name}` ({param_type}, {req_str}): {param_desc}")
                
                docs.append("\n")
            
            # Examples section
            if include_examples and tool.examples:
                docs.append("#### Examples\n")
                
                for i, example in enumerate(tool.examples, 1):
                    docs.append(f"**Example {i}:**\n")
                    
                    if "description" in example:
                        docs.append(f"{example['description']}\n")
                    
                    if "parameters" in example:
                        docs.append("```json\n")
                        docs.append(json.dumps(example["parameters"], indent=2))
                        docs.append("\n```\n")
                    
                    if "result" in example:
                        docs.append("Result:\n")
                        docs.append("```\n")
                        if isinstance(example["result"], (dict, list)):
                            docs.append(json.dumps(example["result"], indent=2))
                        else:
                            docs.append(str(example["result"]))
                        docs.append("\n```\n")
            
            docs.append("\n")
        
        return "\n".join(docs)
    
    @staticmethod
    def _generate_text_docs(registry: ToolRegistry, include_examples: bool) -> str:
        """
        Generate plain text documentation for tools.
        
        Args:
            registry: The tool registry to document
            include_examples: Whether to include examples
            
        Returns:
            Plain text formatted documentation
        """
        docs = ["AVAILABLE TOOLS", "==============="]
        
        for tool_name in registry.get_tool_names():
            tool = registry.get_tool(tool_name)
            
            docs.append(f"\n{tool_name}")
            docs.append("-" * len(tool_name))
            docs.append(f"{tool.description}\n")
            
            # Parameters section
            if tool.parameters and "properties" in tool.parameters:
                docs.append("Parameters:")
                required = tool.parameters.get("required", [])
                
                for param_name, param_info in tool.parameters["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_str = "REQUIRED" if param_name in required else "optional"
                    
                    docs.append(f"  {param_name} ({param_type}, {req_str}): {param_desc}")
                
                docs.append("")
            
            # Examples section
            if include_examples and tool.examples:
                docs.append("Examples:")
                
                for i, example in enumerate(tool.examples, 1):
                    docs.append(f"  Example {i}:")
                    
                    if "description" in example:
                        docs.append(f"  {example['description']}")
                    
                    if "parameters" in example:
                        docs.append("  Parameters:")
                        params_str = json.dumps(example["parameters"], indent=2)
                        docs.append("    " + params_str.replace("\n", "\n    "))
                    
                    if "result" in example:
                        docs.append("  Result:")
                        if isinstance(example["result"], (dict, list)):
                            result_str = json.dumps(example["result"], indent=2)
                            docs.append("    " + result_str.replace("\n", "\n    "))
                        else:
                            docs.append(f"    {example['result']}")
                    
                    docs.append("")
        
        return "\n".join(docs)