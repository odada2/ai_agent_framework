# ai_agent_framework/core/tools/base.py

"""
Base Tool Class

This module defines the abstract base class for all tools in the framework.
Handles async execution of both sync and async _run methods.
"""

import asyncio # Import asyncio
import inspect
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

# Assuming ToolError is defined in exceptions
# from ..exceptions import ToolError # Use this if ToolError exists

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools in the framework.

    Handles parameter validation and asynchronous execution of the underlying
    tool logic (_run or func), whether it's synchronous or asynchronous.
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
            func: Optional function to use for execution (alternative to implement _run).
                  Can be sync or async.
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

        # Determine if the primary execution logic is async
        self._is_async = asyncio.iscoroutinefunction(self._func) if self._func else asyncio.iscoroutinefunction(self._run)

        # Auto-generate parameters schema if not provided and using a function
        if self._parameters is None and self._func is not None:
            # Note: _generate_params_from_func is synchronous and inspects the signature
            self._parameters = self._generate_params_from_func(self._func)

    def _generate_params_from_func(self, func: Callable) -> Dict[str, Any]:
        """
        Generate a JSON schema from a function's signature. (Remains Synchronous)
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

            # Map Python types to JSON schema types (basic mapping)
            prop_details: Dict[str, Any] = {}
            origin_type = getattr(param_type, '__origin__', None)
            if origin_type is Union: # Handle Optional[T] which is Union[T, NoneType]
                 # Check if NoneType is one of the args, assume optional
                 type_args = getattr(param_type, '__args__', ())
                 is_optional = type(None) in type_args
                 # Get the first non-None type
                 actual_type = next((t for t in type_args if t is not type(None)), Any)
                 origin_type = getattr(actual_type, '__origin__', None) # Check origin of actual type
                 param_type = actual_type # Use actual type for mapping

            if param_type is str: prop_details["type"] = "string"
            elif param_type is int: prop_details["type"] = "integer"
            elif param_type is float: prop_details["type"] = "number"
            elif param_type is bool: prop_details["type"] = "boolean"
            elif origin_type is list or param_type is list:
                 prop_details["type"] = "array"
                 # Try to infer item type if List[T] annotation used
                 item_type = getattr(param_type, '__args__', (Any,))[0]
                 if item_type is str: prop_details["items"] = {"type": "string"}
                 elif item_type is int: prop_details["items"] = {"type": "integer"}
                 elif item_type is float: prop_details["items"] = {"type": "number"}
                 elif item_type is bool: prop_details["items"] = {"type": "boolean"}
                 elif item_type is dict: prop_details["items"] = {"type": "object"}
                 # Default items to allow anything if type is complex/Any
                 else: prop_details["items"] = {}
            elif origin_type is dict or param_type is dict:
                 prop_details["type"] = "object"
                 # Basic assumption: values are strings. Can be enhanced.
                 prop_details["additionalProperties"] = {"type": "string"}
            elif origin_type is Literal: # Handle Literal[...] for enums
                 prop_details["type"] = "string"
                 prop_details["enum"] = list(getattr(param_type, '__args__', ()))
            else: # Default to string for Any or complex types
                 prop_details["type"] = "string"

            # Add description from docstring if possible (more advanced)
            # prop_details["description"] = ...

            schema["properties"][name] = prop_details

            # Add to required list if the parameter has no default value
            if param.default == inspect.Parameter.empty:
                schema["required"].append(name)
            else:
                 # Add default value to schema if simple type
                 if isinstance(param.default, (str, int, float, bool, type(None))):
                      prop_details["default"] = param.default


        return schema


    @property
    def parameters(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return self._parameters or {"type": "object", "properties": {}}

    # Keep _run abstract, implementations will be sync or async
    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """
        Execute the tool functionality. Sync or Async.

        This method must be implemented by concrete tool classes if not using `func`.
        """
        pass

    # Make execute asynchronous
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool asynchronously with the provided parameters.

        Handles parameter validation, permission checking (optional),
        and runs the underlying sync or async logic appropriately.

        Args:
            **kwargs: Tool parameters as keyword arguments

        Returns:
            The result of the tool execution

        Raises:
            ValueError: If required parameters are missing or invalid
            ToolError: If execution fails (or specific subclass)
        """
        # Validate parameters (synchronous)
        try:
             self._validate_parameters(kwargs)
        except ValueError as e:
             logger.error(f"Parameter validation failed for tool '{self.name}': {e}")
             # Return error dict or raise? Raising is cleaner for control flow.
             # raise ToolError(f"Parameter validation failed: {e}") from e
             return {"error": f"Parameter validation failed: {e}"} # Return error dict

        # TODO: Add permission checking logic here if required by the framework

        # Execute the underlying logic (sync or async)
        target_func = self._func if self._func else self._run
        if target_func is None:
             # Should not happen if class structure is correct
             raise NotImplementedError(f"Tool '{self.name}' has neither '_run' method nor 'func' configured.")

        try:
            if self._is_async:
                # If the target function (_run or _func) is async, await it directly
                result = await target_func(**kwargs)
            else:
                # If the target function is synchronous, run it in a thread executor
                # Use asyncio.to_thread for Python 3.9+
                result = await asyncio.to_thread(target_func, **kwargs)

            return result

        except Exception as e:
            logger.exception(f"Error executing tool '{self.name}': {e}")
            # Return error dict or raise ToolError? Returning dict for now.
            # raise ToolError(f"Execution failed for tool '{self.name}': {e}") from e
            return {"error": str(e)}

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate that the provided parameters match the expected schema. (Synchronous)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid
        """
        schema = self.parameters

        # Check required parameters
        required_params = schema.get("required", [])
        missing = [req for req in required_params if req not in params]
        if missing:
            raise ValueError(f"Missing required parameter(s): {', '.join(missing)}")

        # Optional: Add JSON Schema type validation here if needed
        # Example using jsonschema library (would add dependency):
        # try:
        #     import jsonschema
        #     jsonschema.validate(instance=params, schema=schema)
        # except ImportError:
        #     logger.warning("jsonschema library not installed, skipping detailed parameter type validation.")
        # except jsonschema.ValidationError as e:
        #     raise ValueError(f"Parameter validation failed: {e.message}")

        # Basic type checks based on schema (optional, LLMs often get this right)
        properties = schema.get("properties", {})
        for name, value in params.items():
            if name in properties:
                 expected_type = properties[name].get("type")
                 # Basic checks, can be expanded
                 if expected_type == "string" and not isinstance(value, str):
                      logger.warning(f"Parameter '{name}' expected type '{expected_type}', got '{type(value).__name__}'. Attempting to proceed.")
                 elif expected_type == "integer" and not isinstance(value, int):
                      logger.warning(f"Parameter '{name}' expected type '{expected_type}', got '{type(value).__name__}'. Attempting to proceed.")
                 elif expected_type == "number" and not isinstance(value, (int, float)):
                      logger.warning(f"Parameter '{name}' expected type '{expected_type}', got '{type(value).__name__}'. Attempting to proceed.")
                 elif expected_type == "boolean" and not isinstance(value, bool):
                      logger.warning(f"Parameter '{name}' expected type '{expected_type}', got '{type(value).__name__}'. Attempting to proceed.")
                 elif expected_type == "array" and not isinstance(value, list):
                       logger.warning(f"Parameter '{name}' expected type '{expected_type}', got '{type(value).__name__}'. Attempting to proceed.")
                 elif expected_type == "object" and not isinstance(value, dict):
                       logger.warning(f"Parameter '{name}' expected type '{expected_type}', got '{type(value).__name__}'. Attempting to proceed.")


    def get_definition(self) -> Dict[str, Any]:
        """
        Get the complete tool definition in a format suitable for LLMs. (Synchronous)
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            # Examples might need adjustment if they contain complex objects
            "examples": self.examples
        }

    def __str__(self) -> str:
        """Return a string representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Return a string representation of the tool."""
        return self.__str__()