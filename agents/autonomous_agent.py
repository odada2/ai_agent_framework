# ai_agent_framework/agents/autonomous_agent.py

"""
Autonomous Agent

This module provides an implementation of an autonomous agent that can plan and
execute tasks using language models and tools asynchronously.
"""

import asyncio # Ensure asyncio is imported
import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..core.llm.base import BaseLLM
from ..core.memory.conversation import ConversationMemory
from ..core.tools.registry import ToolRegistry
from ..core.tools.parser import ToolCallParser
from .base_agent import BaseAgent
# Assuming ToolError exists in core.exceptions
# from ..core.exceptions import ToolError

logger = logging.getLogger(__name__)


class AutonomousAgent(BaseAgent):
    """
    An agent implementation that operates autonomously to accomplish tasks.

    Uses asynchronous operations for planning, acting (including tool execution),
    and reflection.
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
        (Docstring remains the same)
        """
        # Use a default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an autonomous AI assistant designed to accomplish tasks by planning, "
                "using available tools, and reflecting on progress. Break down complex tasks, "
                "execute steps methodically, adapt to tool results, and provide a clear final answer."
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

        # Execution state (reset in reset method)
        self.execution_state: Dict[str, Any] = {}
        self.reset() # Initialize execution state structure


    async def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """
        Run the agent asynchronously on the given input.
        (Docstring remains the same)
        """
        # Use await if BaseAgent.reset becomes async
        self.reset()

        task = input_data if isinstance(input_data, str) else input_data.get("input", str(input_data))
        logger.info(f"[{self.name}] Starting run for task: {task[:100]}...")
        self.memory.add_user_message(task)

        # Initialize state for this run
        self.state = {
            "task": task,
            "start_time": time.monotonic(),
            "last_reflection_iteration": 0
        }

        final_response: Optional[str] = None

        try:
            # Create initial plan (ensure _create_plan is async)
            self.execution_state["plan"] = await self._create_plan(task)
            logger.debug(f"[{self.name}] Initial plan: {self.execution_state['plan'].get('steps')}")

            # Main execution loop
            while not self.finished:
                can_continue = self._increment_iteration() # Sync check
                if not can_continue:
                     logger.warning(f"[{self.name}] Max iterations reached.")
                     self._mark_finished(success=False, error="Max iterations reached")
                     break

                logger.info(f"[{self.name}] Starting iteration {self.current_iteration}")

                # --- Reflection Step ---
                if (self.current_iteration > 1 and # Reflect after first action
                    (self.current_iteration - self.state["last_reflection_iteration"]) >= self.reflection_threshold):
                    await self._reflect_on_progress()
                    self.state["last_reflection_iteration"] = self.current_iteration
                    # Re-check if finished state changed during reflection
                    if self.finished: break

                # --- Action Step ---
                # Determine next action (ensure _determine_next_action is async)
                action_result = await self._determine_next_action()

                # Process action result
                if action_result.get("finished", False):
                    final_response = action_result.get("response", "Task concluded.")
                    self._mark_finished(success=True)
                    logger.info(f"[{self.name}] Task determined to be finished.")
                    break
                elif "tool_call" in action_result:
                    # Execute the tool (use async _execute_tool)
                    tool_result_info = await self._execute_tool(action_result["tool_call"])
                    # Record tool usage and result (or error)
                    self.execution_state["tool_results"].append({
                        "iteration": self.current_iteration,
                        "tool": action_result["tool_call"]["name"],
                        "input": action_result["tool_call"].get("parameters", {}),
                        "output": tool_result_info # Contains result or error
                    })
                    # Optional: Add tool result summary to conversation memory for context?
                    # summary = f"Tool {action_result['tool_call']['name']} executed. Result summary: {str(tool_result_info)[:100]}..."
                    # self.memory.add_message(summary, role="system")
                elif action_result.get("step_completed", False):
                     completed_step_desc = action_result.get("step", self.execution_state.get("current_step", "Unknown step"))
                     if completed_step_desc not in self.execution_state["completed_steps"]:
                          self.execution_state["completed_steps"].append(completed_step_desc)
                     logger.info(f"[{self.name}] Marked step as complete: {completed_step_desc}")
                     # Move to next logical step based on plan (or let LLM decide in next iteration)
                     self.execution_state["current_step"] = self._get_next_plan_step()
                elif "error" in action_result:
                     # Handle error in action determination
                     logger.error(f"[{self.name}] Error during action determination: {action_result['error']}")
                     # Decide whether to stop or let agent try to recover
                     # For now, let's stop
                     self._mark_finished(success=False, error=f"Action determination failed: {action_result['error']}")
                     break
                else:
                     # No specific action, implies need for more thought/planning in next iteration
                     logger.debug(f"[{self.name}] No specific action determined, proceeding to next iteration.")


            # --- End of Loop ---

            # Generate final response if loop finished without one
            if final_response is None:
                logger.info(f"[{self.name}] Generating final response after loop completion.")
                final_response = await self._generate_final_response(task)
                if not self.finished: # Mark finished if loop ended due to iterations etc.
                     self._mark_finished(success=True) # Assume success if no errors occurred

            # Store final response in memory
            self.memory.add_assistant_message(final_response)

            # Prepare result object
            run_duration = time.monotonic() - self.state["start_time"]
            result = {
                "response": final_response,
                "iterations": self.current_iteration,
                "duration_seconds": run_duration,
                "tool_calls": self.execution_state["tool_results"],
                "finished": self.finished,
                "success": self.success,
                "error": self.error # From BaseAgent state
            }
            logger.info(f"[{self.name}] Run finished. Success: {self.success}. Duration: {run_duration:.2f}s")
            return result

        except Exception as e:
            logger.exception(f"[{self.name}] Unhandled error during run: {e}")
            error_response = f"I encountered an unexpected error: {e}"
            self._mark_finished(success=False, error=str(e))
            self.memory.add_assistant_message(error_response)
            run_duration = time.monotonic() - self.state.get("start_time", time.monotonic())
            return {
                "response": error_response,
                "error": str(e),
                "iterations": self.current_iteration,
                 "duration_seconds": run_duration,
                "tool_calls": self.execution_state["tool_results"],
                "finished": True,
                "success": False
            }

    # Make _execute_tool asynchronous
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call asynchronously using the ToolRegistry.

        Args:
            tool_call: Dictionary containing tool name and parameters.

        Returns:
            The result from the tool's execute method (which includes errors).
        """
        tool_name = tool_call.get("name")
        parameters = tool_call.get("parameters", {})

        if not tool_name:
            logger.error(f"[{self.name}] Tool call missing 'name'.")
            return {"error": "Tool call missing 'name'."}

        if not self.tools.has_tool(tool_name):
            logger.error(f"[{self.name}] Tool '{tool_name}' not found in registry.")
            return {"error": f"Tool '{tool_name}' not found."}

        logger.info(f"[{self.name}] Executing tool '{tool_name}' with parameters: {parameters}")
        try:
            # Await the asynchronous execute_tool method from the registry
            result = await self.tools.execute_tool(tool_name, **parameters)
            logger.info(f"[{self.name}] Tool '{tool_name}' executed successfully.")
            logger.debug(f"[{self.name}] Tool '{tool_name}' result: {str(result)[:200]}...") # Log truncated result
            # The result itself might contain an 'error' key if the tool's execution failed internally
            return result
        except Exception as e:
            # Catch errors during the awaiting of execute_tool itself
            logger.exception(f"[{self.name}] Unexpected error awaiting tool execution for '{tool_name}': {e}")
            return {"error": f"Failed to execute tool '{tool_name}': {e}"}


    # --- Other methods (_create_plan, _determine_next_action, _reflect_on_progress, etc.) ---
    # Assume these are already async or update them if they perform async operations (like LLM calls)

    async def _create_plan(self, task: str, depth: int = 0) -> Dict[str, Any]:
        """(Async) Create a plan for accomplishing the task."""
        # (Implementation remains largely the same as provided before, ensure LLM call is awaited)
        if depth >= self.max_planning_depth:
            return {"steps": ["Complete the task directly"], "reasoning": "Plan simplified due to complexity."}

        tool_descriptions = self.tools.get_tool_descriptions() if self.tools and len(self.tools) > 0 else "No tools available."

        planning_prompt = f"Task: {task}\n\nAvailable tools:\n{tool_descriptions}\n\nCreate a step-by-step plan. Format:\nReasoning: <analysis>\nPlan:\n1. <step 1>\n2. <step 2>..."
        system_prompt = "You are a strategic planner."

        planning_response = await self.llm.generate(prompt=planning_prompt, system_prompt=system_prompt, temperature=0.3)
        response_text = planning_response.get("content", "")

        reasoning = ""
        if "Reasoning:" in response_text:
            reasoning = response_text.split("Reasoning:", 1)[1].split("Plan:", 1)[0].strip()

        steps = []
        if "Plan:" in response_text:
            plan_text = response_text.split("Plan:", 1)[1].strip()
            steps = [line.split(". ", 1)[1].strip() if '. ' in line else line.lstrip('- ').strip()
                     for line in plan_text.split('\n') if line.strip() and (line.strip().startswith(tuple(f"{i}." for i in range(10))) or line.strip().startswith('-'))]
        if not steps: steps = [response_text] if response_text else ["No plan generated."]


        return {"reasoning": reasoning, "steps": steps, "created_at": time.monotonic()}


    async def _determine_next_action(self) -> Dict[str, Any]:
        """(Async) Determine the next action based on current state."""
        # (Implementation remains largely the same as provided before, ensure LLM call is awaited)
        context = self._prepare_execution_context() # Sync method

        # Define the action choices clearly for the LLM
        action_prompt = (
            f"{context}\n\n"
            f"## Instruction:\n"
            f"Based on the current state, decide the single best next action. Choose ONE option below and respond ONLY with the corresponding JSON structure:\n\n"
            f"1. **Use a Tool:** If you need external information or action.\n"
            f"   ```json\n"
            f'   {{"action": "use_tool", "tool": "<tool_name>", "parameters": {{"arg1": "value1", ...}}}}\n'
            f"   ```\n\n"
            f"2. **Mark Step Complete:** If the *current* step is finished and you are ready for the next one.\n"
            f"   ```json\n"
            f'   {{"action": "next_step", "completed_step": "<description of the step just finished>"}}\n'
            f"   ```\n\n"
            f"3. **Complete Task:** If the overall task is fully accomplished.\n"
            f"   ```json\n"
            f'   {{"action": "complete", "response": "<Final comprehensive response to the user>"}}\n'
            f"   ```\n"
        )
        system_prompt = "You are an autonomous agent deciding the next action. Analyze the context and choose ONE action format. Respond ONLY in JSON."

        try:
            action_response = await self.llm.generate(prompt=action_prompt, system_prompt=system_prompt, temperature=0.1)
            response_text = action_response.get("content", "").strip()

            # Use ToolCallParser to extract JSON robustly
            parsed_calls = self.tool_call_parser.parse_tool_calls(response_text)

            if parsed_calls:
                 action_data = parsed_calls[0].get("parameters", parsed_calls[0]) # Handle potential nesting by parser
                 action_type = action_data.get("action", "").lower()

                 if action_type == "use_tool":
                      tool_name = action_data.get("tool")
                      parameters = action_data.get("parameters", {})
                      if not tool_name or not isinstance(parameters, dict):
                           return {"error": "Invalid 'use_tool' action format."}
                      if not self.tools.has_tool(tool_name):
                           return {"error": f"Action specifies non-existent tool: '{tool_name}'"}
                      return {"action": "use_tool", "tool_call": {"name": tool_name, "parameters": parameters}}

                 elif action_type == "next_step":
                      completed = action_data.get("completed_step")
                      if not completed: return {"error": "Invalid 'next_step' action format."}
                      return {"action": "next_step", "step_completed": True, "step": completed}

                 elif action_type == "complete":
                      final_resp = action_data.get("response")
                      if final_resp is None: return {"error": "Invalid 'complete' action format."}
                      return {"action": "complete", "finished": True, "response": final_resp}
                 else:
                      return {"error": f"Unknown action type received: '{action_type}'"}
            else:
                 # If no JSON is found, maybe LLM responded directly? Treat as final response?
                 logger.warning(f"Could not parse structured action from LLM response: {response_text}")
                 # This fallback might be risky, depends on desired agent behavior
                 # return {"action": "complete", "finished": True, "response": response_text}
                 return {"error": f"Could not parse valid action JSON from LLM response."}

        except Exception as e:
             logger.exception(f"Error determining next action: {e}")
             return {"error": f"Failed to determine next action: {e}"}


    async def _reflect_on_progress(self) -> None:
        """(Async) Reflect on progress and potentially adjust plan."""
        # (Implementation remains largely the same as provided before, ensure LLM calls are awaited)
        context = self._prepare_execution_context()
        logger.info(f"[{self.name}] Reflecting on progress at iteration {self.current_iteration}")

        reflection_prompt = f"{context}\n\nReflect on progress. Are you on track? Any obstacles? Does the plan need adjustment? What's next?"
        system_prompt = "You are a self-reflecting agent evaluating task progress."

        reflection_response = await self.llm.generate(prompt=reflection_prompt, system_prompt=system_prompt, temperature=0.3)
        reflection_text = reflection_response.get("content", "")
        self.execution_state["reflections"].append({"iteration": self.current_iteration, "reflection": reflection_text})
        logger.debug(f"[{self.name}] Reflection: {reflection_text[:200]}...")

        # Basic check for plan adjustment trigger words
        if "adjust plan" in reflection_text.lower() or "new plan" in reflection_text.lower() or "revise plan" in reflection_text.lower():
            logger.info(f"[{self.name}] Reflection suggests plan adjustment.")
            current_progress_summary = self._summarize_progress_for_replan()
            remaining_task_desc = f"Original Task: {self.state['task']}\nProgress Summary: {current_progress_summary}\nObjective: Complete the original task based on this progress."
            try:
                 new_plan = await self._create_plan(remaining_task_desc, depth=1) # Limit recursion depth
                 self.execution_state["plan"] = new_plan # Replace the plan entirely
                 self.execution_state["completed_steps"] = [] # Reset completed steps relative to new plan
                 self.execution_state["current_step"] = self._get_next_plan_step() # Set current step for new plan
                 logger.info(f"[{self.name}] Plan adjusted based on reflection.")
            except Exception as e:
                 logger.error(f"[{self.name}] Failed to create adjusted plan during reflection: {e}")


    def _summarize_progress_for_replan(self) -> str:
         """Creates a concise summary of completed steps and tool results for replanning."""
         summary_parts = []
         if self.execution_state.get("completed_steps"):
              summary_parts.append("Completed Steps: " + ", ".join(self.execution_state["completed_steps"]))
         if self.execution_state.get("tool_results"):
              summary_parts.append("Recent Tool Results Summary:")
              for res in self.execution_state["tool_results"][-2:]: # Limit context
                   outcome = "Success" if "error" not in res.get("output", {}) else "Error"
                   preview = str(res.get("output",{}).get("result", res.get("output",{}).get("error", "")))[:100]
                   summary_parts.append(f"- {res['tool']}: {outcome} ({preview}...)")
         return "\n".join(summary_parts) if summary_parts else "No significant progress yet."


    async def _generate_final_response(self, task: str) -> str:
        """(Async) Generate final response."""
        # (Implementation remains largely the same as provided before, ensure LLM call is awaited)
        context = self._prepare_execution_context()
        final_prompt = f"{context}\n\nTask is finished or max iterations reached. Generate a final, comprehensive response for the user summarizing the outcome."
        system_prompt = "You summarize the results of an autonomous agent's work."

        final_response = await self.llm.generate(prompt=final_prompt, system_prompt=system_prompt, temperature=0.5)
        return final_response.get("content", "Task processing is complete.")


    def _prepare_execution_context(self) -> str:
        """Prepare context string for LLM prompts. (Synchronous)"""
        # (Implementation remains largely the same as provided before)
        context = [f"## Current Task:\n{self.state.get('task', 'N/A')}"]

        plan = self.execution_state.get("plan")
        if plan and plan.get("steps"):
             context.append("\n## Current Plan:")
             if plan.get("reasoning"): context.append(f"Reasoning: {plan['reasoning']}")
             context.append("Steps:")
             current_step_found = False
             for i, step in enumerate(plan["steps"]):
                  prefix = f"{i+1}."
                  status = ""
                  if step in self.execution_state.get("completed_steps", []): status = "[✓ DONE]"
                  elif step == self.execution_state.get("current_step") and not current_step_found:
                       status = "[▶ CURRENT]"
                       current_step_found = True
                  context.append(f"  {prefix} {status} {step}")
             if not current_step_found and self.execution_state.get("current_step"):
                  # If current step isn't in the plan list, mention it
                  context.append(f"  [▶ CURRENT] {self.execution_state['current_step']} (Potentially deviated from plan)")


        tool_results = self.execution_state.get("tool_results", [])
        if tool_results:
             context.append("\n## Recent Tool Execution History (Max 3):")
             for result in tool_results[-3:]: # Show only last few tool calls
                  outcome = result.get("output", {})
                  result_preview = str(outcome.get("result", outcome.get("error", "No result/error provided")))[:150]
                  context.append(f"- **Iteration {result['iteration']}**: Ran `{result['tool']}` with input `{str(result['input'])[:50]}...` -> Outcome: `{result_preview}...`")


        reflections = self.execution_state.get("reflections", [])
        if reflections:
             last_reflection = reflections[-1]
             context.append(f"\n## Last Reflection (Iteration {last_reflection['iteration']}):\n{last_reflection['reflection'][:300]}...") # Truncate reflection

        context.append(f"\n## Current Status:\nIteration {self.current_iteration}/{self.max_iterations}")

        return "\n".join(context)

    def _get_next_plan_step(self) -> Optional[str]:
         """Gets the next step from the plan that hasn't been completed."""
         plan_steps = self.execution_state.get("plan", {}).get("steps", [])
         completed = self.execution_state.get("completed_steps", [])
         for step in plan_steps:
              if step not in completed:
                   return step
         return None # All plan steps completed


    def reset(self) -> None:
        """Reset the agent's state for a new run."""
        # Use await if BaseAgent.reset becomes async
        super().reset() # Resets iteration, finished, base state, memory
        self.execution_state = {
            "plan": None,
            "current_step": None,
            "completed_steps": [],
            "tool_results": [],
            "reflections": []
        }
        if self.verbose:
            logger.info(f"Reset agent '{self.name}' execution state.")

    async def chat(self, message: str, **kwargs) -> str:
        """Simple async interface for chat-based interactions."""
        # Ensure BaseAgent.chat is async if it calls self.run
        result = await self.run(message, **kwargs)
        return result.get("response", "I encountered an issue and couldn't generate a response.")