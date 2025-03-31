# ai_agent_framework/tools/file_system/list_dir.py

"""
File System List Directory Tool

This module provides a tool for listing files and directories within allowed paths.
"""

import os
import logging
from typing import Dict, Optional, List, Any

# Framework components
from ai_agent_framework.core.tools.base import BaseTool
from ai_agent_framework.core.exceptions import ToolError # Assuming ToolError exists

logger = logging.getLogger(__name__)

class ListDirectoryTool(BaseTool):
    """
    Tool for listing the contents (files and subdirectories) of a specified directory.
    Restricted by allowed_directories for security.
    """

    def __init__(
        self,
        name: str = "list_dir",
        description: str = "Lists files and subdirectories within a specified directory path relative to the allowed base paths. Defaults to the current allowed directory if no path is provided.",
        allowed_directories: Optional[List[str]] = None,
    ):
        """
        Initialize the list directory tool.

        Args:
            name: Name of the tool.
            description: Description of what the tool does.
            allowed_directories: List of absolute base paths from which listing is permitted.
                                 If None or empty, listing is disabled for safety.
        """
        if not allowed_directories:
            logger.warning("ListDirectoryTool initialized without allowed_directories. Listing will be disabled.")
            # Ensure it's a list, even if empty, for consistent checks later
            self.allowed_directories = []
        else:
            # Resolve and store allowed directories securely
            self.allowed_directories = [os.path.realpath(os.path.abspath(d)) for d in allowed_directories]
            if not self.allowed_directories:
                 logger.warning("ListDirectoryTool allowed_directories resolved to an empty list. Listing will be disabled.")

        # Define parameters schema
        parameters_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional path relative to the allowed directories to list contents from. Defaults to the base allowed directory if empty.",
                    "default": "." # Default to current relative path within allowed dir
                }
            },
            "required": [] # Path is optional
        }

        # Define examples
        examples = [
            {
                "description": "List contents of the current allowed directory.",
                "parameters": {},
                "result_summary": "['file1.txt', 'subdir1', 'image.jpg']"
            },
            {
                "description": "List contents of a subdirectory named 'data' within the allowed path.",
                "parameters": {
                    "path": "data"
                },
                "result_summary": "['report.csv', 'log.txt']"
            }
        ]

        super().__init__(
            name=name,
            description=description,
            parameters=parameters_schema,
            examples=examples,
            required_permissions=["file_read"] # Reuse file_read permission? Or define file_list?
        )

    def _is_path_allowed(self, target_path: str) -> bool:
        """Check if the target path is within or is one of the allowed directories."""
        if not self.allowed_directories:
            return False # Deny if no directories are allowed

        # Resolve the target path to its absolute, real path
        try:
            real_target_path = os.path.realpath(os.path.abspath(target_path))
        except OSError as e:
             logger.warning(f"Could not resolve path '{target_path}': {e}")
             return False # Cannot resolve path safely

        # Check if the resolved path is within any of the allowed directory paths
        for allowed_dir in self.allowed_directories:
            # Ensure allowed_dir itself is valid
            if not os.path.isdir(allowed_dir):
                 logger.warning(f"Configured allowed directory '{allowed_dir}' is not a valid directory.")
                 continue

            # Check if the target path IS the allowed directory or is INSIDE it
            # Use commonpath to handle traversal attempts robustly
            try:
                 common = os.path.commonpath([real_target_path, allowed_dir])
                 if common == allowed_dir:
                      return True
            except ValueError: # Can happen on Windows with different drives
                 # Fallback check for simple cases or different drives
                 if real_target_path == allowed_dir or real_target_path.startswith(allowed_dir + os.sep):
                      return True

        logger.warning(f"Path '{real_target_path}' is outside allowed directories: {self.allowed_directories}")
        return False

    # _run is synchronous, BaseTool.execute handles running it in a thread
    def _run(self, path: str = ".") -> Dict[str, Any]:
        """
        List the contents of the specified directory path.

        Args:
            path: Relative path within the allowed directories. Defaults to '.'.

        Returns:
            Dictionary containing the list of contents or an error.
        """
        if not self.allowed_directories:
            raise ToolError("Directory listing is disabled (no allowed_directories configured).")

        # Combine allowed base with relative path safely ONLY if relative path is safe
        # We check the combined path against ALL allowed bases
        # Assume path is relative to one of the allowed directories. Iterate to find base.
        target_abs_path = None
        found_allowed_base = False
        for base_dir in self.allowed_directories:
            potential_path = os.path.join(base_dir, path)
            # Crucially, check if this potential path is allowed BEFORE using it
            if self._is_path_allowed(potential_path):
                target_abs_path = os.path.abspath(potential_path)
                found_allowed_base = True
                break # Found a valid base

        if not found_allowed_base or target_abs_path is None:
             # This condition might be redundant if _is_path_allowed covers it, but double-check
             logger.error(f"Access denied: Relative path '{path}' could not be resolved within any allowed directory.")
             raise ToolError(f"Access denied: Path '{path}' is not permitted.")

        # --- Security Check (Redundant if _is_path_allowed is robust, but good practice) ---
        if not self._is_path_allowed(target_abs_path):
            # This should ideally not be reached if the loop logic above is correct
            logger.error(f"Security Check Failed: Resolved path '{target_abs_path}' is outside allowed directories.")
            raise ToolError(f"Access denied: Path '{path}' resolves outside permitted areas.")

        try:
            # Check if the resolved path actually exists and is a directory
            if not os.path.exists(target_abs_path):
                 raise ToolError(f"Path not found: '{path}' (resolved to '{target_abs_path}')")
            if not os.path.isdir(target_abs_path):
                 raise ToolError(f"Path is not a directory: '{path}' (resolved to '{target_abs_path}')")

            # List directory contents
            contents = os.listdir(target_abs_path)

            logger.info(f"Successfully listed contents for '{target_abs_path}'. Found {len(contents)} items.")
            return {
                "status": "success",
                "path": path, # Return the relative path requested
                "absolute_path": target_abs_path,
                "contents": contents # List of filenames/subdir names
            }
        except OSError as e:
            logger.error(f"OS error listing directory '{target_abs_path}': {e}", exc_info=True)
            raise ToolError(f"Failed to list directory '{path}': OS Error ({e})")
        except Exception as e:
            logger.exception(f"Unexpected error listing directory '{target_abs_path}': {e}")
            raise ToolError(f"An unexpected error occurred while listing directory '{path}': {e}")