# ai_agent_framework/core/tools/parser.py

"""
Tool Call Parser

This module provides functionality for parsing and handling tool calls from LLM responses.
Handles JSON directly, within <function_call> tags, or within markdown code blocks.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ToolCallParser:
    """
    Parser for extracting and validating tool calls from LLM responses.
    Handles direct JSON, Claude <function_call> tags, and markdown ```json blocks.
    """

    # Regex patterns - adjusted markdown pattern slightly
    _CLAUDE_FUNCTION_PATTERN = r'<function_call>\s*(\{.*?\})\s*</function_call>'
    # Matches ```json {..} ``` or ``` {..} ```
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
            List of parsed tool calls (containing 'name' and 'parameters' keys)

        Raises:
            ValueError: If strict_mode is True and parsing fails (optional)
        """
        # Handle structured API responses (e.g., from Claude API with native tool use)
        if isinstance(response, dict):
            parsed = self._parse_structured_response(response)
            if parsed: return parsed # Return if structured parsing worked

        # Handle text responses otherwise, or if structured parsing yielded nothing
        if isinstance(response, dict):
            # If it was a dict but structured parsing failed, get the text content
            response_text = response.get("content", "")
            if not isinstance(response_text, str):
                 response_text = str(response_text) # Convert potential non-string content
        elif isinstance(response, str):
            response_text = response
        else:
            # Cannot parse non-dict/non-string input
             logger.warning(f"Cannot parse tool calls from type: {type(response)}")
             return []

        return self._parse_text_response(response_text)

    def _parse_structured_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse tool calls from a structured API response (e.g., Claude's tool_use).
        """
        tool_calls = []
        # Handle content blocks (Claude API)
        if "content" in response and isinstance(response["content"], list):
            for block in response["content"]:
                if block.get("type") == "tool_use":
                    # Ensure parameters is a dict, defaults to empty if missing/wrong type
                    params = block.get("input", {})
                    if not isinstance(params, dict):
                        logger.warning(f"Tool call parameters for tool '{block.get('name')}' is not a dictionary: {params}. Using empty dict.")
                        params = {}

                    tool_calls.append({
                        "name": block.get("name", ""),
                        "parameters": params
                    })
        # Handle older direct tool_calls if present (less common now)
        elif "tool_calls" in response and isinstance(response["tool_calls"], list):
             # Basic pass-through, assuming structure is correct [{name:.., parameters:..}]
             # Add validation if needed
             return response["tool_calls"]

        return tool_calls

    def _parse_text_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from a text response, trying various formats.
        """
        text = text.strip()
        if not text:
            return []

        # 1. Try parsing as direct JSON object/list
        try:
            # Check if the entire string is likely a JSON object/list
            if (text.startswith('{') and text.endswith('}')) or \
               (text.startswith('[') and text.endswith(']')):
                data = json.loads(text)
                # If it's a single call object, wrap in list
                if isinstance(data, dict) and ("name" in data or "tool" in data):
                     parsed = self._validate_and_adapt_call(data)
                     if parsed: return [parsed]
                # If it's a list of calls
                elif isinstance(data, list):
                     parsed_list = [self._validate_and_adapt_call(item) for item in data if isinstance(item, dict)]
                     valid_list = [p for p in parsed_list if p]
                     if valid_list: return valid_list
        except json.JSONDecodeError:
            pass # Not direct JSON, proceed to other methods

        # 2. Try parsing Claude-style function calls
        claude_calls = self._extract_claude_function_calls(text)
        if claude_calls:
            return claude_calls # Return immediately if found

        # 3. Try parsing from markdown code blocks (more robustly)
        markdown_calls = self._extract_markdown_tool_calls(text)
        if markdown_calls:
            return markdown_calls # Return immediately if found

        # 4. If no specific format found, return empty list
        return []

    def _validate_and_adapt_call(self, call_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
         """Validates if dict looks like a tool call and adapts keys."""
         name = call_data.get("name") or call_data.get("tool")
         parameters = call_data.get("parameters", call_data.get("input")) # Allow 'input' or 'parameters'

         # Basic validation
         if not name or not isinstance(name, str):
              logger.debug(f"Skipping invalid tool call data (missing/invalid name): {call_data}")
              return None
         # Ensure parameters is a dictionary, default to empty if missing or wrong type
         if parameters is None:
              parameters = {}
         elif not isinstance(parameters, dict):
              logger.warning(f"Tool call parameters for tool '{name}' is not a dictionary: {parameters}. Using empty dict.")
              parameters = {}

         return {"name": name, "parameters": parameters}


    def _extract_claude_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from Claude-style <function_call> tags."""
        tool_calls = []
        matches = re.finditer(self._CLAUDE_FUNCTION_PATTERN, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                json_str = match.group(1)
                call_data = json.loads(json_str)
                parsed = self._validate_and_adapt_call(call_data)
                if parsed:
                    tool_calls.append(parsed)
                elif self.strict_mode:
                     logger.warning(f"Malformed <function_call> content: {json_str}")

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse <function_call> JSON: {e}. Content: '{match.group(1)}'"
                if self.strict_mode: raise ValueError(error_msg)
                logger.warning(error_msg)
        return tool_calls

    def _extract_markdown_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from markdown ```json code blocks."""
        tool_calls = []
        # Find all markdown blocks (json or generic) containing potential JSON objects
        matches = re.finditer(self._MARKDOWN_CODE_PATTERN, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            json_str = match.group(1).strip() # Get the content inside ```
            try:
                # Check if it's a plausible JSON object before full parse
                if json_str.startswith('{') and json_str.endswith('}'):
                    call_data = json.loads(json_str)
                    parsed = self._validate_and_adapt_call(call_data)
                    if parsed:
                        tool_calls.append(parsed)
                    # else: # It was valid JSON but not a valid tool call structure
                    #    logger.debug(f"Valid JSON in markdown block, but not a tool call: {json_str}")
            except json.JSONDecodeError:
                # Not a valid JSON object, might be other code or malformed
                logger.debug(f"Could not decode JSON from markdown block: {json_str[:100]}...")
                # If strict, should we error here? Maybe not, could be unrelated code block.
        return tool_calls