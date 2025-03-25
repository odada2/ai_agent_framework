"""
Workflow Agent

This module provides an implementation of an agent that uses predefined workflows
to accomplish tasks. The WorkflowAgent follows the workflow pattern described in
the Anthropic document on building effective agents.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.llm.base import BaseLLM
from ..core.memory.conversation import ConversationMemory
from ..core.tools.registry import ToolRegistry
from ..core.tools.parser import ToolCallParser
from ..core.workflow.base import BaseWorkflow
from ..core.workflow.chain import PromptChain
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class WorkflowAgent(BaseAgent):
    """
    An agent implementation that uses predefined workflows to accomplish tasks.
    
    The WorkflowAgent follows a more structured approach than autonomous agents,
    using predefined workflows to handle different types of tasks. This makes the
    agent more predictable and easier to debug, while still providing flexibility
    through workflow composition.
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        workflows: Optional[Dict[str, BaseWorkflow]] = None,
        default_workflow: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the WorkflowAgent.
        
        Args:
            name: A unique name for this agent instance
            llm: The LLM implementation to use for this agent
            tools: Optional registry of tools available to the agent
            memory: Optional conversation memory for maintaining context
            workflows: Optional dictionary of workflows available to the agent
            default_workflow: Optional name of the default workflow to use
            system_prompt: Optional system prompt to guide the agent's behavior
            max_iterations: Maximum number of iterations the agent can perform in a run
            verbose: Whether to log detailed information about the agent's operations
            **kwargs: Additional agent-specific parameters
        """
        super().__init__(
            name=name,
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
            **kwargs
        )
        
        # Initialize workflows dictionary
        self.workflows = workflows or {}
        self.default_workflow = default_workflow
        
        # Set up tool call parser
        self.tool_call_parser = ToolCallParser()
        
        # Add a generic chain workflow if none provided
        if not self.workflows and not self.default_workflow:
            self._setup_default_chain_workflow()
    
    def _setup_default_chain_workflow(self) -> None:
        """
        Set up a default prompt chain workflow if none was provided.
        This gives the agent a basic capability to handle tasks.
        """
        # Define steps for a simple two-step chain:
        # 1. Plan approach to the task
        # 2. Execute the task with tools
        steps = [
            {
                "name": "plan",
                "prompt_template": (
                    "You are tasked with helping the user with the following request:\n\n"
                    "{input}\n\n"
                    "Please analyze this request and create a plan to address it. "
                    "Consider what tools might be helpful and outline the steps you'll take."
                ),
                "system_prompt": "You are a helpful planning assistant that breaks down tasks into clear steps."
            },
            {
                "name": "execute",
                "prompt_template": (
                    "You are tasked with helping the user with the following request:\n\n"
                    "{original_input}\n\n"
                    "You've created the following plan:\n{input}\n\n"
                    "Now, execute this plan to fulfill the user's request. "
                    "Use the available tools when necessary."
                ),
                "use_tools": True,
                "return_tool_results": True,
                "system_prompt": "You are a helpful assistant that executes plans to fulfill user requests."
            }
        ]
        
        # Create a default chain workflow
        default_chain = PromptChain(
            name="default_chain",
            llm=self.llm,
            steps=steps,
            tools=self.tools,
            max_steps=len(steps)
        )
        
        # Add workflow and set as default
        self.workflows["default"] = default_chain
        self.default_workflow = "default"
    
    async def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """
        Run the agent on the given input.
        
        This method processes the input, selects the appropriate workflow,
        executes the workflow, and returns the result.
        
        Args:
            input_data: The input data for the agent to process (string query or structured data)
            **kwargs: Additional runtime parameters
            
        Returns:
            A dictionary containing the agent's response and any additional metadata
        """
        # Reset state for a new run
        self.reset()
        
        # Convert input to dictionary if it's a string
        if isinstance(input_data, str):
            input_data = {"input": input_data, "original_input": input_data}
        elif isinstance(input_data, dict) and "input" in input_data:
            input_data["original_input"] = input_data["input"]
        else:
            input_data = {"input": str(input_data), "original_input": str(input_data)}
        
        # Add input to memory
        user_input = input_data.get("input", "")
        self.memory.add_user_message(user_input)
        
        # Select workflow - either from kwargs, or use default
        workflow_name = kwargs.get("workflow", self.default_workflow)
        if not workflow_name or workflow_name not in self.workflows:
            # Fall back to workflow selection via LLM if no valid workflow specified
            workflow_name = await self._select_workflow(user_input)
        
        # Check if we have a valid workflow
        if workflow_name not in self.workflows:
            error_msg = f"No valid workflow available. Available workflows: {list(self.workflows.keys())}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get the selected workflow
        workflow = self.workflows[workflow_name]
        
        # Execute workflow
        try:
            workflow_result = await workflow.execute(input_data, **kwargs)
            
            # Extract result from the workflow output
            result = {}
            
            if "final_result" in workflow_result:
                final_result = workflow_result["final_result"]
                
                # Extract tool calls if present in a structured response
                tool_calls = []
                if isinstance(final_result, dict):
                    if "tool_results" in final_result:
                        tool_calls = final_result.get("tool_results", [])
                        result["response"] = final_result.get("content", "")
                    else:
                        result["response"] = final_result
                else:
                    # Try to parse tool calls from text response
                    result["response"] = final_result
                    tool_calls = self.tool_call_parser.parse_tool_calls(final_result)
                
                if tool_calls:
                    result["tool_calls"] = tool_calls
            else:
                result["response"] = "The workflow did not produce a final result."
            
            # Add workflow execution summary
            result["workflow"] = {
                "name": workflow_name,
                "steps_executed": workflow.steps_executed,
                "success": workflow.success
            }
            
            # Store assistant response in memory
            self.memory.add_assistant_message(result["response"])
            
            # Mark execution as finished
            self._mark_finished(success=workflow.success, error=workflow.error)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error executing workflow '{workflow_name}': {str(e)}")
            return {
                "error": f"Error executing workflow: {str(e)}",
                "workflow": workflow_name
            }
    
    async def _select_workflow(self, user_input: str) -> str:
        """
        Select the appropriate workflow based on the user input.
        
        If multiple workflows are available, this method uses the LLM to determine
        which workflow is most appropriate for the given input.
        
        Args:
            user_input: The user's input query
            
        Returns:
            The name of the selected workflow
        """
        # If only one workflow is available, use it
        if len(self.workflows) == 1:
            return next(iter(self.workflows.keys()))
        
        # If no workflows are available, return empty string
        if not self.workflows:
            return ""
        
        # Create a prompt to select the appropriate workflow
        workflow_descriptions = []
        for name, workflow in self.workflows.items():
            description = f"- {name}: {getattr(workflow, 'description', 'No description available.')}"
            workflow_descriptions.append(description)
        
        prompt = (
            f"Based on the following user request, select the most appropriate workflow:\n\n"
            f"User request: {user_input}\n\n"
            f"Available workflows:\n"
            f"{chr(10).join(workflow_descriptions)}\n\n"
            f"Please respond with just the name of the most appropriate workflow."
        )
        
        system_prompt = "You are a helpful workflow selection assistant that analyzes user requests and selects the most appropriate workflow."
        
        # Get LLM response
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for more deterministic selection
            )
            
            # Extract workflow name from response
            workflow_name = response.get("content", "").strip().lower()
            
            # Match with available workflow names (case-insensitive)
            workflow_map = {name.lower(): name for name in self.workflows.keys()}
            
            # Check if selected workflow exists
            if workflow_name in workflow_map:
                selected = workflow_map[workflow_name]
                logger.info(f"Selected workflow '{selected}' for input: {user_input[:50]}...")
                return selected
            
            # If no match, use default or first available
            logger.warning(f"Selected workflow '{workflow_name}' not found. Using default.")
            return self.default_workflow or next(iter(self.workflows.keys()))
            
        except Exception as e:
            logger.error(f"Error selecting workflow: {str(e)}")
            # Fall back to default or first available
            return self.default_workflow or next(iter(self.workflows.keys()))
    
    def add_workflow(self, name: str, workflow: BaseWorkflow) -> None:
        """
        Add a workflow to the agent.
        
        Args:
            name: Name of the workflow
            workflow: The workflow instance to add
        """
        self.workflows[name] = workflow
        
        # Set as default if this is the first workflow
        if self.default_workflow is None:
            self.default_workflow = name
        
        if self.verbose:
            logger.info(f"Added workflow '{name}' to agent '{self.name}'")
    
    def remove_workflow(self, name: str) -> None:
        """
        Remove a workflow from the agent.
        
        Args:
            name: Name of the workflow to remove
        """
        if name in self.workflows:
            del self.workflows[name]
            
            # Update default workflow if needed
            if self.default_workflow == name:
                self.default_workflow = next(iter(self.workflows.keys())) if self.workflows else None
            
            if self.verbose:
                logger.info(f"Removed workflow '{name}' from agent '{self.name}'")
    
    async def chat(self, message: str, **kwargs) -> str:
        """
        Simple interface for chat-based interactions.
        
        This method provides a simpler interface for chat-based interactions,
        returning just the text response.
        
        Args:
            message: User message
            **kwargs: Additional parameters for the run method
            
        Returns:
            The agent's text response
        """
        result = await self.run(message, **kwargs)
        return result.get("response", "I'm not sure how to respond to that.")