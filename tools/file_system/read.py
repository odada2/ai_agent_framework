"""
File System Read Tool

This module provides a tool for reading files from the file system.
"""

import os
from typing import Dict, Optional

from ...core.tools.base import BaseTool


class FileReadTool(BaseTool):
    """
    Tool for reading files from the file system.
    """
    
    def __init__(
        self,
        name: str = "read_file",
        description: str = "Read the contents of a file.",
        allowed_directories: Optional[list] = None,
        max_file_size: int = 10 * 1024 * 1024  # 10MB
    ):
        """
        Initialize the file read tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            allowed_directories: Optional list of allowed directories (None = all)
            max_file_size: Maximum file size in bytes
        """
        parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "Encoding to use when reading the file (default: utf-8)"
                }
            },
            "required": ["path"]
        }
        
        examples = [
            {
                "description": "Read a text file",
                "parameters": {
                    "path": "data/example.txt"
                },
                "result": "This is the content of the example file."
            },
            {
                "description": "Read a CSV file with specific encoding",
                "parameters": {
                    "path": "data/data.csv",
                    "encoding": "latin-1"
                },
                "result": "id,name,value\n1,Item 1,10.5\n2,Item 2,20.75"
            }
        ]
        
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            examples=examples,
            required_permissions=["file_read"]
        )
        
        self.allowed_directories = allowed_directories
        self.max_file_size = max_file_size
    
    def _run(self, path: str, encoding: str = "utf-8") -> Dict:
        """
        Read the contents of a file.
        
        Args:
            path: Path to the file to read
            encoding: Encoding to use (default: utf-8)
            
        Returns:
            Dictionary containing the file content and metadata
            
        Raises:
            ValueError: If the file isn't allowed, doesn't exist, or exceeds size limit
        """
        # Convert to absolute path for security checks
        abs_path = os.path.abspath(path)
        
        # Check if file is in an allowed directory
        if self.allowed_directories:
            allowed = False
            for directory in self.allowed_directories:
                dir_abs = os.path.abspath(directory)
                if abs_path.startswith(dir_abs):
                    allowed = True
                    break
            
            if not allowed:
                raise ValueError(
                    f"Access denied: File {path} is outside of allowed directories"
                )
        
        # Check if file exists
        if not os.path.exists(abs_path):
            raise ValueError(f"File not found: {path}")
        
        # Check if it's a regular file (not a directory or special file)
        if not os.path.isfile(abs_path):
            raise ValueError(f"Not a regular file: {path}")
        
        # Check file size
        if os.path.getsize(abs_path) > self.max_file_size:
            raise ValueError(
                f"File exceeds maximum size of {self.max_file_size} bytes"
            )
        
        # Read the file
        try:
            with open(abs_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Get file metadata
            stats = os.stat(abs_path)
            
            return {
                "content": content,
                "metadata": {
                    "path": path,
                    "size": stats.st_size,
                    "last_modified": stats.st_mtime,
                    "encoding": encoding
                }
            }
        except UnicodeDecodeError:
            raise ValueError(
                f"Failed to decode file with encoding '{encoding}'. "
                "Try a different encoding."
            )
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")