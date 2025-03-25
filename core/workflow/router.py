"""
Router Workflow

This module implements the routing workflow pattern, which classifies an input
and directs it to a specialized followup task.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from ..llm.base import BaseLLM
from ..tools.registry import ToolRegistry
from .base import BaseWorkflow

logger = logging.getLogger(__name__)


class Router(BaseWorkflow):
    """
    Implementation of the routing workflow pattern.
    
    This workflow classifies an input and directs it to specialized handlers.
    It allows for separation of concerns, enabling optimization for different
    input types without affecting performance on other inputs.
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        routes: Dict[str, Dict[str, Any]],
        tools: Optional[ToolRegistry] = None,
        max_steps: int = 5,
        default_route: Optional[str] = None,
        use_llm_for_routing: bool = True,
        route_descriptions: Optional[Dict[str, str]] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the routing workflow.
        
        Args:
            name: Name of the workflow
            llm: LLM instance to use
            routes: Dictionary mapping route names to handler configurations
            tools: Optional tool registry for tool-based routes
            max_steps: Maximum number of steps to execute
            default_route: Optional default route if classification fails
            use_llm_for_routing: Whether to use LLM for routing (vs custom classifier)
            route_descriptions: Optional descriptions for each route
            verbose: Whether to log detailed information
            **kwargs: Additional parameters for specific routes
        """
        super().__init__(name=name, max_steps=max_steps, verbose=verbose)
        
        self.llm = llm
        self.tools = tools
        self.routes = routes
        self.default_route = default_route
        self.use_llm_for_routing = use_llm_for_routing
        self.route_descriptions = route_descriptions or {}
        self.kwargs = kwargs
        
        # Custom classifier if not using LLM
        self.classifier = kwargs.get("classifier")
        
        # Validate routes
        self._validate_routes()
    
    def _validate_routes(self) -> None:
        """
        Validate that the routes are properly configured.
        
        Raises:
            ValueError: If routes are not properly configured
        """
        if not self.routes:
            raise ValueError("At least one route must be defined")
        
        for route_name, route_config in self.routes.items():
            if "handler" not in route_config:
                raise ValueError(f"Route '{route_name}' missing required 'handler' configuration")
            
            handler_type = route_config.get("type", "workflow")
            
            if handler_type not in ["workflow", "llm", "function"]:
                raise ValueError(f"Invalid handler type '{handler_type}' for route '{route_name}'")
            
            if handler_type == "workflow" and not isinstance(route_config["handler"], BaseWorkflow):
                raise ValueError(f"Handler for route '{route_name}' must be a BaseWorkflow instance")
            
            if handler_type == "function" and not callable(route_config["handler"]):
                raise ValueError(f"Handler for route '{route_name}' must be a callable function")
    
    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the routing workflow.
        
        Args:
            input_data: Input to route
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary containing the execution result
        """
        self.reset()
        
        # Prepare input
        if isinstance(input_data, str):
            input_data = {"input": input_data}
        
        original_input = input_data.get("input", str(input_data))
        
        # Initialize result structure
        result = {
            "input": original_input,
            "selected_route": None,
            "routes": list(self.routes.keys()),
            "result": None
        }
        
        try:
            # Step 1: Classify the input to select a route
            self.current_step += 1
            self._log_step("classify", original_input, None)
            
            selected_route = await self._classify_input(original_input, **kwargs)
            result["selected_route"] = selected_route
            
            if not selected_route or selected_route not in self.routes:
                if self.default_route and self.default_route in self.routes:
                    selected_route = self.default_route
                    result["selected_route"] = selected_route
                    logger.info(f"Using default route: {selected_route}")
                else:
                    error_msg = f"No valid route selected and no default route available"
                    self._log_step("classify", original_input, None, error=Exception(error_msg))
                    self._mark_finished(success=False, error=error_msg)
                    result["error"] = error_msg
                    return result
            
            # Step 2: Execute the selected route's handler
            if not self._increment_step():
                return result
            
            route_config = self.routes[selected_route]
            handler_type = route_config.get("type", "workflow")
            handler = route_config["handler"]
            
            # Prepare context for the handler
            handler_context = {
                "input": original_input,
                "route": selected_route,
                **kwargs
            }
            
            # Add any route-specific parameters
            if "params" in route_config:
                handler_context.update(route_config["params"])
            
            # Execute the appropriate handler type
            if handler_type == "workflow":
                # Handler is a workflow
                handler_result = await handler.execute(handler_context)
                result["result"] = handler_result.get("final_result", handler_result)
            
            elif handler_type == "llm":
                # Handler is a direct LLM prompt
                prompt_template = handler
                system_prompt = route_config.get("system_prompt")
                
                # Format the prompt template with the input
                prompt = prompt_template.format(input=original_input)
                
                llm_response = await self.llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=route_config.get("temperature", 0.7),
                    max_tokens=route_config.get("max_tokens")
                )
                
                result["result"] = llm_response.get("content", "")
            
            elif handler_type == "function":
                # Handler is a function
                if asyncio.iscoroutinefunction(handler):
                    handler_result = await handler(handler_context)
                else:
                    handler_result = handler(handler_context)
                
                result["result"] = handler_result
            
            self._log_step("execute_handler", handler_context, result["result"])
            self._mark_finished(success=True)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in routing workflow: {str(e)}")
            self._mark_finished(success=False, error=str(e))
            
            result["error"] = str(e)
            return result
    
    async def _classify_input(self, input_text: str, **kwargs) -> str:
        """
        Classify the input to determine the appropriate route.
        
        Args:
            input_text: The input text to classify
            **kwargs: Additional classification parameters
            
        Returns:
            The selected route name
        """
        # If we have a custom classifier and not using LLM, use that
        if self.classifier and not self.use_llm_for_routing:
            return self.classifier(input_text, **kwargs)
        
        # Otherwise use the LLM for classification
        route_options = []
        for route_name in self.routes.keys():
            description = self.route_descriptions.get(route_name, f"Route: {route_name}")
            route_options.append(f"- {route_name}: {description}")
        
        classification_prompt = (
            f"Please classify the following input into one of the available categories.\n\n"
            f"Input: {input_text}\n\n"
            f"Available categories:\n"
            f"{chr(10).join(route_options)}\n\n"
            f"Respond with ONLY the category name that best matches the input."
        )
        
        classification_response = await self.llm.generate(
            prompt=classification_prompt,
            temperature=0.1,  # Low temperature for more deterministic classification
            max_tokens=20     # We only need a short response
        )
        
        response_text = classification_response.get("content", "").strip()
        
        # Match response with available routes
        for route_name in self.routes.keys():
            if route_name.lower() in response_text.lower():
                return route_name
        
        # If no direct match, try to extract from the response
        if ":" in response_text:
            potential_route = response_text.split(":", 1)[0].strip()
            if potential_route in self.routes:
                return potential_route
        
        # Default to returning the raw response if it's one of our routes
        if response_text in self.routes:
            return response_text
        
        # No valid route found
        logger.warning(f"No valid route match found for classification response: '{response_text}'")
        return self.default_route if self.default_route else ""