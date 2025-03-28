# ai_agent_framework/tools/file_system/write.py

"""
File System Write Tool

This module provides a tool for writing content to files within allowed directories.
"""

import os
import logging
from typing import Dict, Optional, List, Literal

# Framework components
from ai_agent_framework.core.tools.base import BaseTool
from ai_agent_framework.core.exceptions import ToolError # Use specific error if available

logger = logging.getLogger(__name__)

class FileWriteTool(BaseTool):
    """
    Tool for writing or appending content to files on the file system.

    Security Note: This tool is restricted by `allowed_directories`. Ensure
    this list is configured securely to prevent unintended file access/modification.
    """

    def __init__(
        self,
        name: str = "write_file",
        description: str = "Write or append text content to a specified file path.",
        allowed_directories: Optional[List[str]] = None,
        default_encoding: str = "utf-8",
        max_file_size: Optional[int] = None # Optional: limit write size
    ):
        """
        Initialize the file write tool.

        Args:
            name: Name of the tool.
            description: Description of what the tool does.
            allowed_directories: List of absolute paths where writing is permitted.
                                 If None or empty, writing is disabled for safety.
            default_encoding: Default encoding to use when writing.
            max_file_size: Optional maximum size in bytes for the resulting file.
        """
        if not allowed_directories:
            logger.warning("FileWriteTool initialized without allowed_directories. Writing will be disabled.")
            self.allowed_directories = [] # Ensure it's a list
        else:
             # Ensure directories are absolute and resolved for security checks
            self.allowed_directories = [os.path.realpath(os.path.abspath(d)) for d in allowed_directories]

        self.default_encoding = default_encoding
        self.max_file_size = max_file_size

        # Define parameters schema
        parameters_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write to the file."
                },
                "mode": {
                    "type": "string",
                    "description": "Write mode: 'w' to overwrite, 'a' to append.",
                    "enum": ["w", "a"],
                    "default": "w"
                },
                "encoding": {
                    "type": "string",
                    "description": f"Encoding to use (default: {self.default_encoding}).",
                    "default": self.default_encoding
                }
            },
            "required": ["path", "content"]
        }

        # Define examples
        examples = [
            {
                "description": "Overwrite a text file",
                "parameters": {
                    "path": "output/report.txt",
                    "content": "This is the final report.",
                    "mode": "w"
                },
                "result_summary": "Successfully wrote content to output/report.txt"
            },
            {
                "description": "Append a line to a log file",
                "parameters": {
                    "path": "logs/agent.log",
                    "content": "\nINFO: Agent completed task.",
                    "mode": "a",
                    "encoding": "utf-8"
                },
                "result_summary": "Successfully appended content to logs/agent.log"
            }
        ]

        super().__init__(
            name=name,
            description=description,
            parameters=parameters_schema,
            examples=examples,
            required_permissions=["file_write"] # Define permission needed
        )

    def _is_path_allowed(self, target_path: str) -> bool:
        """Check if the target path is within the allowed directories."""
        if not self.allowed_directories:
            return False # Deny if no directories are allowed

        # Resolve the target path to its absolute, real path to prevent traversal attacks
        try:
            real_target_path = os.path.realpath(os.path.abspath(target_path))
        except OSError as e:
             logger.warning(f"Could not resolve path '{target_path}': {e}")
             return False # Cannot resolve path safely

        # Check if the resolved path starts with any of the allowed directory paths
        for allowed_dir in self.allowed_directories:
            # Ensure allowed_dir itself is valid
            if not os.path.isdir(allowed_dir):
                 logger.warning(f"Configured allowed directory '{allowed_dir}' is not a valid directory.")
                 continue
            # Use os.path.commonpath (more robust than startswith for paths)
            # Or check if real_target_path is within the directory tree of allowed_dir
            try:
                 # Check if the common path of the target and allowed dir is the allowed dir itself
                 # This prevents paths like /allowed/../other/file
                 if os.path.commonpath([real_target_path, allowed_dir]) == allowed_dir:
                      return True
            except ValueError: # Can happen if paths are on different drives on Windows
                 # Fallback to startswith check for simple cases or different drives
                 if real_target_path.startswith(allowed_dir + os.sep):
                      return True

        return False

    def _run(
        self,
        path: str,
        content: str,
        mode: Literal["w", "a"] = "w",
        encoding: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Write or append content to a specified file.

        Args:
            path: Path to the file.
            content: Text content to write.
            mode: 'w' for overwrite, 'a' for append.
            encoding: File encoding to use.

        Returns:
            Dictionary indicating success or failure.

        Raises:
            ToolError: If writing is disallowed, path is invalid, or I/O error occurs.
        """
        if not self.allowed_directories:
             raise ToolError("File writing is disabled (no allowed_directories configured).")

        target_abs_path = os.path.abspath(path)

        # --- Security Check ---
        if not self._is_path_allowed(target_abs_path):
            logger.error(f"Access denied: Attempted write to '{path}' (resolved: '{target_abs_path}') which is outside allowed directories.")
            raise ToolError(f"Access denied: Writing to path '{path}' is not permitted.")

        resolved_encoding = encoding or self.default_encoding
        write_mode = mode if mode in ["w", "a"] else "w" # Default to overwrite

        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(target_abs_path)
            if parent_dir: # Avoid trying to create '' if path is just filename
                 os.makedirs(parent_dir, exist_ok=True)

            # Optional: Check resulting file size if appending and limit is set
            if write_mode == 'a' and self.max_file_size is not None and os.path.exists(target_abs_path):
                 current_size = os.path.getsize(target_abs_path)
                 content_bytes = content.encode(resolved_encoding)
                 if current_size + len(content_bytes) > self.max_file_size:
                      raise ToolError(f"Cannot append content: Resulting file size would exceed maximum limit of {self.max_file_size} bytes.")
            elif write_mode == 'w' and self.max_file_size is not None:
                 content_bytes = content.encode(resolved_encoding)
                 if len(content_bytes) > self.max_file_size:
                       raise ToolError(f"Cannot write content: Content size exceeds maximum limit of {self.max_file_size} bytes.")


            # Write the file
            with open(target_abs_path, write_mode, encoding=resolved_encoding) as f:
                bytes_written = f.write(content)

            logger.info(f"Successfully wrote {bytes_written} characters to '{target_abs_path}' (mode: {write_mode}, encoding: {resolved_encoding}).")
            return {
                "status": "success",
                "message": f"Successfully { 'wrote' if write_mode == 'w' else 'appended' } content to '{path}'.",
                "path": path,
                "absolute_path": target_abs_path,
                "bytes_written": bytes_written # Note: len(content) might differ due to encoding
            }
        except IOError as e:
            logger.error(f"I/O error writing to file '{target_abs_path}': {e}", exc_info=True)
            raise ToolError(f"Failed to write to file '{path}': I/O Error ({e})")
        except UnicodeEncodeError as e:
            logger.error(f"Encoding error writing to file '{target_abs_path}' with encoding '{resolved_encoding}': {e}", exc_info=True)
            raise ToolError(f"Failed to write to file '{path}': Encoding Error using '{resolved_encoding}' ({e}). Try a different encoding.")
        except OSError as e:
            logger.error(f"OS error related to file '{target_abs_path}': {e}", exc_info=True)
            raise ToolError(f"Failed to write to file '{path}': OS Error ({e})")
        except Exception as e:
            logger.exception(f"Unexpected error writing to file '{target_abs_path}': {e}")
            raise ToolError(f"An unexpected error occurred while writing to '{path}': {e}")