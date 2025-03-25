"""
Autonomous Agent

This module provides an implementation of an autonomous agent that can plan and
execute tasks using language models and tools. Unlike the workflow agent which follows
predefined paths, the autonomous agent dynamically determines its own process flow.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..core.llm.base import BaseLLM
from ..core.memory.conversation import ConversationMemory
from ..core.tools.registry import ToolRegistry
from ..core.tools.parser import ToolCallParser
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AutonomousAgent(BaseAgent):
    """
    An agent implementation that operates autonomously to accomplish tasks.
    
    The AutonomousAgent is more flexible than the WorkflowAgent, as it dynamically
    determines the actions to take based on the task at hand rather than following
    predefined workflows. This makes it suitable for open-ended problems where
    the required steps are not known in advance.
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        reflection_threshold: int = 3,
        max_planning_depth: int = 2,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the AutonomousAgent.
        
        Args:
            name: A unique name for this agent instance
            llm: The LLM implementation to use for this agent
            tools: Optional registry of tools available to the agent
            memory: Optional conversation memory for maintaining context
            system_prompt: Optional system prompt to guide the agent's behavior
            max_iterations: Maximum number of iterations the agent can perform in a run
            reflection_threshold: Number of iterations after which the agent reflects on its progress
            max_planning_depth: Maximum depth of planning recursion
            verbose: Whether to log detailed information about the agent's operations
            **kwargs: Additional agent-specific parameters
        """
        # Use a default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an autonomous AI assistant that helps users accomplish tasks. "
                "You have access to tools that allow you to interact with external systems. "
                "When presented with a task:\n"
                "1. Think through what's needed and form a plan\n"
                "2. Use available tools when necessary\n"
                "3. Adapt your plan based on new information\n"
                "4. Present clear, concise results to the user\n"
                "Be thorough but efficient, and provide helpful responses."
            )
        
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
        
        # Additional parameters specific to autonomous agents
        self.reflection_threshold = reflection_threshold
        self.max_planning_depth = max_planning_depth
        
        # Set up tool call parser
        self.tool_call_parser = ToolCallParser()
        
        # Execution state
        self.execution_state = {
            "plan": None,
            "current_step": None,
            "completed_steps": [],
            "tool_results": [],
            "reflections": []
        }
    
    async def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """
        Run the agent on the given input.
        
        This method processes the input, plans the approach, executes the plan,
        and returns the result.
        
        Args:
            input_data: The input data for the agent to process (string query or structured data)
            **kwargs: Additional runtime parameters
            
        Returns:
            A dictionary containing the agent's response and any additional metadata
        """
        # Reset for a new run
        self.reset()
        
        # Extract task
        task = input_data if isinstance(input_data, str) else str(input_data)
        
        # Add task to memory
        self.memory.add_user_message(task)
        
        # Initialize state for this run
        self.state = {
            "task": task,
            "start_time": time.time(),
            "last_reflection_iteration": 0
        }
        
        # Main execution loop
        final_response = None
        
        try:
            # Create initial plan
            self.execution_state["plan"] = await self._create_plan(task)
            
            while not self.finished and self._increment_iteration():
                # Check if we need to reflect on progress
                if (self.current_iteration - self.state["last_reflection_iteration"] 
                        >= self.reflection_threshold):
                    await self._reflect_on_progress()
                    self.state["last_reflection_iteration"] = self.current_iteration
                
                # Determine next action
                action_result = await self._determine_next_action()
                
                # If we got a final response, we're done
                if action_result.get("finished", False):
                    final_response = action_result.get("response")
                    self._mark_finished(success=True)
                    break
                
                # Otherwise, execute the determined action
                if "tool_call" in action_result:
                    tool_result = await self._execute_tool(action_result["tool_call"])
                    self.execution_state["tool_results"].append({
                        "tool": action_result["tool_call"]["name"],
                        "input": action_result["tool_call"].get("parameters", {}),
                        "result": tool_result
                    })
                
                # Update execution state
                if "step" in action_result:
                    self.execution_state["current_step"] = action_result["step"]
                    if action_result.get("step_completed", False):
                        self.execution_state["completed_steps"].append(action_result["step"])
                        self.execution_state["current_step"] = None
            
            # If we didn't get a final response, generate one based on the accumulated context
            if not final_response:
                final_response = await self._generate_final_response(task)
            
            # Store in memory
            self.memory.add_assistant_message(final_response)
            
            # Prepare the result object
            result = {
                "response": final_response,
                "iterations": self.current_iteration,
                "tool_calls": self.execution_state["tool_results"],
                "finished": self.finished,
                "success": self.success
            }
            
            return result
            
        except Exception as e:
            logger.exception(f"Error running autonomous agent: {str(e)}")
            error_response = f"I encountered an error while working on your task: {str(e)}"
            self.memory.add_assistant_message(error_response)
            
            return {
                "response": error_response,
                "error": str(e),
                "iterations": self.current_iteration,
                "tool_calls": self.execution_state["tool_results"],
                "finished": True,
                "success": False
            }
    
    async def _create_plan(self, task: str, depth: int = 0) -> Dict[str, Any]:
        """
        Create a plan for accomplishing the task.
        
        Args:
            task: The task to plan for
            depth: Current planning recursion depth
            
        Returns:
            Dictionary containing the plan details
        """
        # Prevent excessive recursion
        if depth >= self.max_planning_depth:
            return {"steps": ["Complete the task directly"], "reasoning": "Plan simplified due to complexity."}
        
        # Generate tool descriptions if tools are available
        tool_descriptions = ""
        if self.tools and len(self.tools) > 0:
            tool_descriptions = f"\n\nAvailable tools:\n{self.tools.get_tool_descriptions()}"
        
        # Create planning prompt
        planning_prompt = (
            f"Task: {task}\n\n"
            f"Create a plan to accomplish this task.{tool_descriptions}\n\n"
            f"Provide your plan in the following format:\n"
            f"Reasoning: <your analysis of the task and approach>\n"
            f"Plan:\n"
            f"1. <first step>\n"
            f"2. <second step>\n"
            f"..."
        )
        
        # Generate plan
        planning_response = await self.llm.generate(
            prompt=planning_prompt,
            system_prompt="You are a strategic planning assistant that breaks down tasks into clear, actionable steps.",
            temperature=0.3  # Lower temperature for more focused planning
        )
        
        # Parse the response to extract the plan
        response_text = planning_response.get("content", "")
        
        # Extract reasoning and steps
        reasoning = ""
        if "Reasoning:" in response_text:
            reasoning_parts = response_text.split("Reasoning:")[1].split("Plan:")
            reasoning = reasoning_parts[0].strip()
        
        # Extract steps, handling different formats
        steps = []
        if "Plan:" in response_text:
            plan_text = response_text.split("Plan:")[1].strip()
            # Handle numbered list
            for line in plan_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove number/bullet and strip
                    step = line
                    if line[0].isdigit() and ". " in line:
                        step = line.split(". ", 1)[1]
                    elif line.startswith("- "):
                        step = line[2:]
                    steps.append(step)
        
        # If no steps were found, extract them differently
        if not steps:
            # Fallback: just use the lines after "Plan:" as steps
            if "Plan:" in response_text:
                plan_text = response_text.split("Plan:")[1].strip()
                steps = [line.strip() for line in plan_text.split("\n") if line.strip()]
            else:
                # Last resort: treat the whole response as a single step
                steps = [response_text.strip()]
        
        return {
            "reasoning": reasoning,
            "steps": steps,
            "created_at": time.time()
        }
    
    async def _determine_next_action(self) -> Dict[str, Any]:
        """
        Determine the next action to take based on the current state.
        
        Returns:
            Dictionary containing the next action details
        """
        # Prepare context with task, plan, and execution history
        context = self._prepare_execution_context()
        
        # Generate action prompt
        action_prompt = (
            f"{context}\n\n"
            f"Determine the next action to take. You can:\n"
            f"1. Use a tool to gather information or perform an action\n"
            f"2. Mark the current step as completed and move to the next step\n"
            f"3. Complete the task and provide a final response to the user\n\n"
            f"Respond in ONE of the following formats:\n\n"
            f"Option 1 - Use a tool:\n"
            f"```json\n"
            f'{{"action": "use_tool", "tool": "<tool_name>", "parameters": {{"param1": "value", "param2": "value"}}}}\n'
            f"```\n\n"
            f"Option 2 - Move to next step:\n"
            f"```json\n"
            f'{{"action": "next_step", "completed": "<current step description>", "reasoning": "<why this step is complete>"}}\n'
            f"```\n\n"
            f"Option 3 - Complete task:\n"
            f"```json\n"
            f'{{"action": "complete", "response": "<final response to user>"}}\n'
            f"```\n"
        )
        
        # Generate action decision
        action_response = await self.llm.generate(
            prompt=action_prompt,
            system_prompt="You are a decision-making assistant that determines the next action to take in a task.",
            temperature=0.2  # Lower temperature for more focused decisions
        )
        
        # Parse the response to extract the action
        response_text = action_response.get("content", "")
        
        # Extract JSON from response if present
        tool_calls = self.tool_call_parser.parse_tool_calls(response_text)
        
        if tool_calls:
            action_data = tool_calls[0]  # Use the first parsed tool call
        else:
            # No structured data found, assume it's a final response
            logger.warning("No structured action found in response, treating as final response")
            return {"finished": True, "response": response_text}
        
        # Process based on action type
        action_type = action_data.get("action", "").lower()
        
        if action_type == "use_tool":
            tool_name = action_data.get("tool", "")
            parameters = action_data.get("parameters", {})
            
            # Check if tool exists
            if not tool_name or not self.tools or not self.tools.has_tool(tool_name):
                logger.warning(f"Tool '{tool_name}' not found or not available")
                return {
                    "step": self.execution_state["current_step"],
                    "error": f"Tool '{tool_name}' not found or not available"
                }
            
            return {
                "step": self.execution_state["current_step"],
                "tool_call": {"name": tool_name, "parameters": parameters}
            }
            
        elif action_type == "next_step":
            completed_step = action_data.get("completed", "")
            reasoning = action_data.get("reasoning", "")
            
            # Mark current step as completed
            return {
                "step": completed_step,
                "step_completed": True,
                "reasoning": reasoning
            }
            
        elif action_type == "complete":
            final_response = action_data.get("response", "")
            
            # Task is complete
            return {
                "finished": True,
                "response": final_response
            }
            
        else:
            # Unknown action type, treat as a continuation
            return {
                "step": self.execution_state["current_step"]
            }
    
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_call: Dictionary containing tool name and parameters
            
        Returns:
            The result of the tool execution
        """
        tool_name = tool_call["name"]
        parameters = tool_call.get("parameters", {})
        
        logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
        
        try:
            result = self.tools.execute_tool(tool_name, **parameters)
            return result
        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}': {str(e)}")
            return {"error": str(e)}
    
    async def _reflect_on_progress(self) -> None:
        """
        Reflect on the progress made so far and adjust the plan if needed.
        """
        context = self._prepare_execution_context()
        
        reflection_prompt = (
            f"{context}\n\n"
            f"Reflect on the progress made so far. Consider:\n"
            f"1. What has been accomplished?\n"
            f"2. What challenges or obstacles have been encountered?\n"
            f"3. Is the current plan still effective or does it need adjustment?\n"
            f"4. What should be the focus for the next steps?\n\n"
            f"Provide your reflection and any adjustments to the plan."
        )
        
        reflection_response = await self.llm.generate(
            prompt=reflection_prompt,
            system_prompt="You are a reflective assistant that evaluates progress and adjusts plans accordingly.",
            temperature=0.3
        )
        
        reflection_text = reflection_response.get("content", "")
        
        # Store the reflection
        self.execution_state["reflections"].append({
            "iteration": self.current_iteration,
            "reflection": reflection_text
        })
        
        # Check if we need to update the plan
        if "plan needs adjustment" in reflection_text.lower() or "adjust the plan" in reflection_text.lower():
            # Extract the remaining task if possible
            remaining_task = self.state["task"]
            if self.execution_state["completed_steps"]:
                remaining_task = f"Original task: {self.state['task']}\n\nProgress so far: {', '.join(self.execution_state['completed_steps'])}\n\nComplete the remaining work."
            
            # Create a new plan for the remaining work
            new_plan = await self._create_plan(remaining_task, depth=1)
            
            # Update the plan, keeping the completed steps
            self.execution_state["plan"] = {
                "reasoning": new_plan["reasoning"],
                "steps": self.execution_state["completed_steps"] + new_plan["steps"],
                "created_at": time.time(),
                "adjusted": True
            }
            
            logger.info(f"Plan adjusted at iteration {self.current_iteration}")
    
    async def _generate_final_response(self, task: str) -> str:
        """
        Generate a final response to the user based on the task and execution state.
        
        Args:
            task: The original task
            
        Returns:
            Final response to the user
        """
        context = self._prepare_execution_context()
        
        final_prompt = (
            f"{context}\n\n"
            f"The task has been completed or the maximum number of iterations has been reached. "
            f"Generate a final response to the user that summarizes what was accomplished, "
            f"what information was gathered, and the final result or recommendation."
        )
        
        final_response = await self.llm.generate(
            prompt=final_prompt,
            system_prompt="You are a helpful assistant that provides clear, concise summaries of completed tasks.",
            temperature=0.5
        )
        
        return final_response.get("content", "I've completed the task to the best of my ability.")
    
    def _prepare_execution_context(self) -> str:
        """
        Prepare the execution context for prompts based on current state.
        
        Returns:
            Formatted context string
        """
        context = [f"Task: {self.state['task']}"]
        
        # Include plan
        if self.execution_state["plan"]:
            plan = self.execution_state["plan"]
            context.append("\nPlan:")
            if plan.get("reasoning"):
                context.append(f"Reasoning: {plan['reasoning']}")
            
            context.append("Steps:")
            for i, step in enumerate(plan["steps"], 1):
                # Mark completed steps
                if step in self.execution_state["completed_steps"]:
                    context.append(f"{i}. [✓] {step}")
                elif step == self.execution_state["current_step"]:
                    context.append(f"{i}. [Current] {step}")
                else:
                    context.append(f"{i}. {step}")
        
        # Include tool execution history
        if self.execution_state["tool_results"]:
            context.append("\nTool Execution History:")
            for i, result in enumerate(self.execution_state["tool_results"], 1):
                context.append(f"{i}. Tool: {result['tool']}")
                context.append(f"   Input: {result['input']}")
                context.append(f"   Result: {result['result']}")
        
        # Include recent reflections (last one only to save space)
        if self.execution_state["reflections"]:
            recent_reflection = self.execution_state["reflections"][-1]
            context.append(f"\nLast Reflection (Iteration {recent_reflection['iteration']}):")
            context.append(recent_reflection["reflection"])
        
        # Include iteration count
        context.append(f"\nCurrent Iteration: {self.current_iteration}/{self.max_iterations}")
        
        return "\n".join(context)
    
    def reset(self) -> None:
        """Reset the agent's state for a new run."""
        super().reset()
        
        self.execution_state = {
            "plan": None,
            "current_step": None,
            "completed_steps": [],
            "tool_results": [],
            "reflections": []
        }
    
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