"""
Tool Call Parser

This module provides functionality for parsing and handling tool calls from LLM responses.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ToolCallParser:
    """
    Parser for extracting and validating tool calls from LLM responses.
    """
    
    # Regex patterns for different tool call formats
    # The specific patterns depend on the LLM output format
    _CLAUDE_FUNCTION_PATTERN = r'<function_call>\s*(\{.*?\})\s*</function_call>'
    _MARKDOWN_CODE_PATTERN = r'```(?:json)?\s*(\{.*?\})\s*```'
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the tool call parser.
        
        Args:
            strict_mode: If True, will raise exceptions for malformed tool calls
        """
        self.strict_mode = strict_mode
    
    def parse_tool_calls(self, response: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and parse tool calls from an LLM response.
        
        Args:
            response: LLM response as a string or a structured response object
            
        Returns:
            List of parsed tool calls
            
        Raises:
            ValueError: If strict_mode is True and parsing fails
        """
        # Handle structured API responses (e.g., from Claude API)
        if isinstance(response, dict):
            return self._parse_structured_response(response)
        
        # Handle text responses
        return self._parse_text_response(response)
    
    def _parse_structured_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse tool calls from a structured API response.
        
        Args:
            response: Structured response from an LLM API
            
        Returns:
            List of parsed tool calls
        """
        tool_calls = []
        
        # Handle Claude-style tool calls
        if "tool_calls" in response:
            return response["tool_calls"]
        
        # Handle content blocks (Claude API)
        if "content" in response and isinstance(response["content"], list):
            for block in response["content"]:
                if block.get("type") == "tool_use":
                    tool_calls.append({
                        "name": block.get("name", ""),
                        "parameters": block.get("input", {})
                    })
        
        return tool_calls
    
    def _parse_text_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from a text response.
        
        Args:
            text: LLM response as a string
            
        Returns:
            List of parsed tool calls
            
        Raises:
            ValueError: If strict_mode is True and parsing fails
        """
        tool_calls = []
        
        # Try parsing Claude-style function calls
        claude_calls = self._extract_claude_function_calls(text)
        if claude_calls:
            tool_calls.extend(claude_calls)
        
        # Try parsing from markdown code blocks
        if not tool_calls:
            markdown_calls = self._extract_markdown_tool_calls(text)
            if markdown_calls:
                tool_calls.extend(markdown_calls)
        
        return tool_calls
    
    def _extract_claude_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Claude-style function_call tags.
        
        Args:
            text: Text to parse
            
        Returns:
            List of parsed tool calls
        """
        tool_calls = []
        
        matches = re.finditer(self._CLAUDE_FUNCTION_PATTERN, text, re.DOTALL)
        for match in matches:
            try:
                json_str = match.group(1)
                call_data = json.loads(json_str)
                
                if "name" in call_data and "parameters" in call_data:
                    tool_calls.append(call_data)
                else:
                    # Try to adapt the format
                    adapted_call = {
                        "name": call_data.get("name") or call_data.get("function") or "",
                        "parameters": call_data.get("parameters") or call_data.get("arguments") or {}
                    }
                    if adapted_call["name"]:
                        tool_calls.append(adapted_call)
                    elif self.strict_mode:
                        logger.warning(f"Malformed tool call, missing name: {json_str}")
            except json.JSONDecodeError as e:
                if self.strict_mode:
                    raise ValueError(f"Failed to parse tool call JSON: {str(e)}")
                logger.warning(f"Failed to parse tool call JSON: {str(e)}")
        
        return tool_calls

    def _extract_markdown_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from markdown code blocks.
        
        Args:
            text: Text to parse
            
        Returns:
            List of parsed tool calls
        """
        tool_calls = []
        
        matches = re.finditer(self._MARKDOWN_CODE_PATTERN, text, re.DOTALL)
        for match in matches:
            try:
                json_str = match.group(1)
                call_data = json.loads(json_str)
                
                # Check if this looks like a tool call
                if isinstance(call_data, dict) and ("name" in call_data or "function" in call_data):
                    adapted_call = {
                        "name": call_data.get("name") or call_data.get("function") or "",
                        "parameters": call_data.get("parameters") or call_data.get("arguments") or {}
                    }
                    if adapted_call["name"]:
                        tool_calls.append(adapted_call)
            except json.JSONDecodeError:
                # Not a valid JSON object, might be other code
                pass
        
        return tool_calls