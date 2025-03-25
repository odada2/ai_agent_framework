"""
Prompt Chaining Workflow

This module implements the prompt chaining workflow pattern.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from ..llm.base import BaseLLM
from ..tools.registry import ToolRegistry
from .base import BaseWorkflow

logger = logging.getLogger(__name__)


class PromptChain(BaseWorkflow):
    """
    Implementation of the prompt chaining workflow pattern.
    
    This workflow executes a series of LLM calls in sequence,
    with each step using the output of the previous step.
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        steps: List[Dict[str, Any]],
        tools: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        verbose: bool = False
    ):
        """
        Initialize the prompt chaining workflow.
        
        Args:
            name: Name of the workflow
            llm: LLM instance to use
            steps: List of step configurations
            tools: Optional tool registry for tool-based steps
            max_steps: Maximum number of steps to execute
            verbose: Whether to log detailed information
        """
        super().__init__(name=name, max_steps=max_steps, verbose=verbose)
        
        self.llm = llm
        self.tools = tools
        
        # Validate and prepare steps
        self.steps = []
        for i, step_config in enumerate(steps):
            if "name" not in step_config:
                step_config["name"] = f"step_{i+1}"
            
            if "prompt" not in step_config and "prompt_template" not in step_config:
                raise ValueError(f"Step {i+1} must have either 'prompt' or 'prompt_template'")
            
            self.steps.append(step_config)
    
    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the prompt chain with the given input.
        
        Args:
            input_data: Initial input for the chain
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary containing the final result and execution data
        """
        self.reset()
        
        # Prepare initial state
        result = {
            "input": input_data,
            "intermediate_results": [],
            "final_result": None,
        }
        
        current_input = input_data
        
        try:
            # Execute each step in sequence
            for i, step in enumerate(self.steps):
                self.current_step = i + 1
                
                # Skip steps if we've reached max_steps
                if not self._increment_step():
                    break
                
                step_name = step["name"]
                step_result = await self._execute_step(step, current_input)
                
                # Check for gate function if present
                if "gate" in step and callable(step["gate"]):
                    gate_result = step["gate"](step_result)
                    
                    if not gate_result:
                        # Gate failed, end the chain
                        self._log_step(
                            step_name, 
                            current_input, 
                            step_result, 
                            error="Gate check failed"
                        )
                        result["final_result"] = step_result
                        result["intermediate_results"].append({
                            "step": step_name,
                            "input": current_input,
                            "output": step_result,
                            "gate_passed": False
                        })
                        self._mark_finished(success=False, error="Gate check failed")
                        return result
                
                # Log successful step
                self._log_step(step_name, current_input, step_result)
                
                # Store intermediate result
                result["intermediate_results"].append({
                    "step": step_name,
                    "input": current_input,
                    "output": step_result,
                    "gate_passed": True if "gate" in step else None
                })
                
                # Use this step's output as the next step's input
                current_input = step_result
            
            # Set final result to the output of the last step
            result["final_result"] = current_input
            self._mark_finished(success=True)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in prompt chain workflow: {str(e)}")
            self._mark_finished(success=False, error=str(e))
            
            result["error"] = str(e)
            result["final_result"] = current_input
            
            return result
    
    async def _execute_step(self, step: Dict[str, Any], input_data: Any) -> Any:
        """
        Execute a single step in the chain.
        
        Args:
            step: Step configuration
            input_data: Input data for this step
            
        Returns:
            Step execution result
        """
        step_name = step["name"]
        
        # Get prompt to send to LLM
        if "prompt" in step:
            prompt = step["prompt"]
        else:  # Must be prompt_template
            prompt_template = step["prompt_template"]
            prompt = self._format_prompt_template(prompt_template, input_data)
        
        # Get system prompt if provided
        system_prompt = step.get("system_prompt")
        
        # Handle tool-based steps
        if "use_tools" in step and step["use_tools"] and self.tools:
            # Get tool subset if specified
            tool_names = step.get("tools", None)
            tools_to_use = []
            
            if tool_names:
                for tool_name in tool_names:
                    if self.tools.has_tool(tool_name):
                        tool = self.tools.get_tool(tool_name)
                        tools_to_use.append(tool.get_definition())
            else:
                # Use all tools
                tools_to_use = self.tools.get_tool_definitions()
            
            # Generate with tools
            response = await self.llm.generate_with_tools(
                prompt=prompt,
                tools=tools_to_use,
                system_prompt=system_prompt,
                temperature=step.get("temperature"),
                max_tokens=step.get("max_tokens")
            )
            
            # Execute any tool calls if present
            if "tool_calls" in response and response["tool_calls"]:
                # Note: This executes tools sequentially.
                # For parallel execution, you would use asyncio.gather()
                tool_results = []
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["name"]
                    
                    if self.tools.has_tool(tool_name):
                        tool_result = self.tools.execute_tool(
                            tool_name, 
                            **tool_call.get("parameters", {})
                        )
                        tool_results.append({
                            "tool": tool_name,
                            "parameters": tool_call.get("parameters", {}),
                            "result": tool_result
                        })
                
                # If specified, include tool results in step output
                if step.get("return_tool_results", False):
                    return {
                        "content": response.get("content", ""),
                        "tool_results": tool_results
                    }
        else:
            # Regular LLM generation
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=step.get("temperature"),
                max_tokens=step.get("max_tokens")
            )
        
        # Apply output parser if provided
        if "output_parser" in step and callable(step["output_parser"]):
            parser = step["output_parser"]
            return parser(response)
        
        # Otherwise return content or full response based on setting
        if step.get("return_full_response", False):
            return response
        else:
            return response.get("content", "")
    
    def _format_prompt_template(self, template: str, data: Any) -> str:
        """
        Format a prompt template with the provided data.
        
        Args:
            template: The prompt template string with placeholders
            data: Data to use for formatting
            
        Returns:
            Formatted prompt string
        """
        # For simple string data
        if isinstance(data, str):
            return template.replace("{input}", data)
        
        # For dictionary data
        if isinstance(data, dict):
            # Start with {input} replacement for backward compatibility
            result = templ