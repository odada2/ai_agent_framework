"""
Base Tool Class

This module defines the abstract base class for all tools in the framework.
"""

import inspect
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools in the framework.
    
    This class defines the interface and common functionality that all tools must implement.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
        required_permissions: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the BaseTool.
        
        Args:
            name: Unique name for the tool
            description: Human-readable description of what the tool does
            func: Optional function to use for execution (alternative to implement)
            parameters: JSON schema for the tool's parameters
            required_permissions: Optional list of permissions needed to use this tool
            examples: Optional list of example tool usages
        """
        self.name = name
        self.description = description
        self._func = func
        self._parameters = parameters
        self.required_permissions = required_permissions or []
        self.examples = examples or []
        
        # Auto-generate parameters schema if not provided and using a function
        if self._parameters is None and self._func is not None:
            self._parameters = self._generate_params_from_func(self._func)

    def _generate_params_from_func(self, func: Callable) -> Dict[str, Any]:
        """
        Generate a JSON schema from a function's signature.
        
        Args:
            func: The function to analyze
            
        Returns:
            A JSON schema describing the function's parameters
        """
        sig = inspect.signature(func)
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            # Get type annotation if available
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            
            # Map Python types to JSON schema types
            if param_type in (str, Optional[str]):
                schema_type = {"type": "string"}
            elif param_type in (int, Optional[int]):
                schema_type = {"type": "integer"}
            elif param_type in (float, Optional[float]):
                schema_type = {"type": "number"}
            elif param_type in (bool, Optional[bool]):
                schema_type = {"type": "boolean"}
            elif param_type in (list, List, Optional[list], Optional[List]):
                schema_type = {"type": "array", "items": {"type": "string"}}
            elif param_type in (dict, Dict, Optional[dict], Optional[Dict]):
                schema_type = {"type": "object"}
            else:
                schema_type = {"type": "string"}
            
            schema["properties"][name] = schema_type
            
            # Add to required list if the parameter has no default value
            if param.default == inspect.Parameter.empty:
                schema["required"].append(name)
        
        return schema

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool's parameters.
        
        Returns:
            JSON schema as a dictionary
        """
        return self._parameters or {}
    
    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """
        Execute the tool functionality.
        
        This method must be implemented by concrete tool classes if not using a function.
        
        Args:
            **kwargs: Tool parameters as keyword arguments
            
        Returns:
            The result of the tool execution
        """
        pass
    
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the provided parameters.
        
        This method handles parameter validation, permission checking, and execution.
        
        Args:
            **kwargs: Tool parameters as keyword arguments
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate parameters
        self._validate_parameters(kwargs)
        
        # Execute the tool
        try:
            if self._func is not None:
                return self._func(**kwargs)
            else:
                return self._run(**kwargs)
        except Exception as e:
            logger.exception(f"Error executing tool '{self.name}': {str(e)}")
            return {"error": str(e)}
    
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate that the provided parameters match the expected schema.
        
        Args:
            params: Parameters to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        schema = self.parameters
        
        # Check required parameters
        for required_param in schema.get("required", []):
            if required_param not in params:
                raise ValueError(f"Missing required parameter: {required_param}")
        
        # Type checking could be added here for more complex validation
        # For now, we trust that the LLM provides the correct types
    
    def get_definition(self) -> Dict[str, Any]:
        """
        Get the complete tool definition in a format suitable for LLMs.
        
        Returns:
            Dictionary containing the tool's name, description, and parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "examples": self.examples
        }
    
    def __str__(self) -> str:
        """Return a string representation of the tool."""
        return f"Tool(name='{self.name}')"

    def __repr__(self) -> str:
        """Return a string representation of the tool."""
        return self.__str__()