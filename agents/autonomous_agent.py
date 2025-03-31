# ai_agent_framework/agents/autonomous_agent.py
# Final Version including _create_plan and other fixes

"""
Autonomous Agent

This module provides an implementation of an autonomous agent that can plan and
execute tasks using language models and tools asynchronously.
"""

import asyncio
import logging
import time
import re # Import re for pattern matching in _create_plan parsing
import json # Import json for parsing tool args and action fallback
from typing import Any, Dict, List, Optional, Union

# Use absolute imports assuming execution via `python -m` or installed package
try:
    from ai_agent_framework.core.llm.base import BaseLLM
    from ai_agent_framework.core.memory.conversation import ConversationMemory
    from ai_agent_framework.core.tools.registry import ToolRegistry
    from ai_agent_framework.core.tools.parser import ToolCallParser
    from ai_agent_framework.agents.base_agent import BaseAgent
except ImportError as e:
     # Handle potential import errors if structure changes or run differently
     print(f"Error importing core components in autonomous_agent.py: {e}", file=sys.stderr)
     # Define BaseAgent as object to allow class definition if import fails badly
     BaseAgent = object


logger = logging.getLogger(__name__)


class AutonomousAgent(BaseAgent):
    """
    An agent implementation that operates autonomously to accomplish tasks.

    Uses asynchronous operations for planning, acting (including tool execution),
    and reflection. Includes robust action parsing with fallback analysis.
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
            reflection_threshold: Number of iterations between reflection steps.
            max_planning_depth: Max depth for recursive planning.
            verbose: Whether to log detailed information about the agent's operations
            **kwargs: Additional agent-specific parameters
        """
        # Use a default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an autonomous AI assistant designed to accomplish tasks by planning, "
                "using available tools, and reflecting on progress. Break down complex tasks, "
                "execute steps methodically, adapt to tool results, and provide a clear final answer."
            )

        # Ensure superclass is properly initialized
        # Check if BaseAgent was successfully imported before calling super().__init__
        if BaseAgent is not object:
             super().__init__(
                 name=name, llm=llm, tools=tools, memory=memory, system_prompt=system_prompt,
                 max_iterations=max_iterations, verbose=verbose, **kwargs
             )
        else:
             # Handle case where BaseAgent import failed (minimal init)
             self.name = name; self.llm = llm; self.tools = tools or ToolRegistry()
             self.memory = memory or ConversationMemory(); self.system_prompt = system_prompt
             self.max_iterations = max_iterations; self.verbose = verbose; self.id=str(uuid.uuid4())
             self.current_iteration = 0; self.finished = False; self.success = False; self.error = None; self.state = {}


        # Additional parameters specific to autonomous agents
        self.reflection_threshold = reflection_threshold
        self.max_planning_depth = max_planning_depth

        # Set up tool call parser
        self.tool_call_parser = ToolCallParser()

        # Execution state (reset in reset method)
        self.execution_state: Dict[str, Any] = {}
        self._initialize_execution_state()


    def _initialize_execution_state(self):
        """Initialize the execution state dictionary structure."""
        self.execution_state = {
            "plan": None,
            "current_step": None,
            "completed_steps": [],
            "tool_results": [],
            "reflections": []
        }

    async def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """Run the agent asynchronously on the given input."""
        # Ensure reset method exists and call it
        if hasattr(self, 'reset') and callable(self.reset):
            if asyncio.iscoroutinefunction(self.reset): await self.reset()
            else: self.reset() # Call sync reset if BaseAgent import failed
        else: self._initialize_execution_state() # Manual reset if method missing

        task = input_data if isinstance(input_data, str) else input_data.get("input", str(input_data))
        logger.info(f"[{self.name}] Starting run for task: {task[:100]}...")

        # Add user message to memory if it exists
        if self.memory and hasattr(self.memory, 'add_user_message'):
             self.memory.add_user_message(task)

        self.state = {"task": task, "start_time": time.monotonic(), "last_reflection_iteration": 0}
        final_response: Optional[str] = None

        try:
            # --- Check if _create_plan method exists before calling ---
            if not hasattr(self, '_create_plan') or not callable(self._create_plan):
                 raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '_create_plan'. Check class definition.")
            # ----------------------------------------------------------
            self.execution_state["plan"] = await self._create_plan(task) # <-- This is line 88 approx
            logger.debug(f"[{self.name}] Initial plan: {self.execution_state['plan'].get('steps')}")
            self.execution_state["current_step"] = self._get_next_plan_step()

            while not self.finished:
                if not self._increment_iteration():
                     logger.warning(f"[{self.name}] Max iterations ({self.max_iterations}) reached.")
                     break # State (finished, error) set by _increment_iteration

                logger.info(f"[{self.name}] Starting iteration {self.current_iteration}")

                if (self.current_iteration > 1 and
                    (self.current_iteration - self.state.get("last_reflection_iteration", 0)) >= self.reflection_threshold):
                    await self._reflect_on_progress()
                    if self.finished: break

                action_result = await self._determine_next_action()

                if "error" in action_result:
                     error_message = f"Action determination failed: {action_result['error']}"
                     logger.error(f"[{self.name}] {error_message}")
                     self.finished = True; self.success = False; self.error = error_message
                     break

                action_type = action_result.get("action", "").lower()

                if action_type == "complete":
                    final_response = action_result.get("response", "Task concluded.")
                    self.finished = True; self.success = True; self.error = None
                    logger.info(f"[{self.name}] Task determined to be finished.")
                    break
                elif action_type == "use_tool":
                    tool_name = action_result.get("tool")
                    parameters = action_result.get("parameters", {})
                    if tool_name:
                         tool_result_info = await self._execute_tool({"name": tool_name, "parameters": parameters})
                         self.execution_state["tool_results"].append({
                             "iteration": self.current_iteration, "tool": tool_name,
                             "input": parameters, "output": tool_result_info
                         })
                         if isinstance(tool_result_info, dict) and "error" in tool_result_info:
                              logger.error(f"[{self.name}] Tool execution failed: {tool_result_info['error']}")
                    else:
                         logger.error(f"[{self.name}] 'use_tool' action missing 'tool' name: {action_result}")
                         self.finished = True; self.success = False; self.error = "Invalid 'use_tool' action data."
                         break
                elif action_type == "next_step":
                     completed_step_desc = action_result.get("completed_step", self.execution_state.get("current_step", "Unknown step"))
                     if completed_step_desc not in self.execution_state.get("completed_steps", []):
                          self.execution_state["completed_steps"].append(completed_step_desc)
                     logger.info(f"[{self.name}] Marked step as complete: {completed_step_desc}")
                     self.execution_state["current_step"] = self._get_next_plan_step()
                else:
                     logger.warning(f"[{self.name}] Unexpected action structure: {action_result}. Proceeding.")

            # --- End of Loop ---
            if final_response is None:
                logger.info(f"[{self.name}] Generating final response after loop completion.")
                final_response = await self._generate_final_response(task)
                if not self.finished:
                     self.finished = True
                     self.success = self.error is None

            if self.memory and hasattr(self.memory, 'add_assistant_message'):
                 self.memory.add_assistant_message(final_response)

            run_duration = time.monotonic() - self.state.get("start_time", time.monotonic())
            result = {"response": final_response, "iterations": self.current_iteration, "duration_seconds": run_duration, "tool_calls": self.execution_state["tool_results"], "finished": self.finished, "success": self.success, "error": self.error}
            logger.info(f"[{self.name}] Run finished. Success: {self.success}. Duration: {run_duration:.2f}s")
            return result

        except AttributeError as e: # Catch specific error if _create_plan is missing
            error_message = f"AttributeError in agent run: {e}"
            logger.exception(f"[{self.name}] {error_message}")
            final_response = f"I encountered an internal configuration error: {e}"
            self.finished = True; self.success = False; self.error = error_message
            if self.memory and hasattr(self.memory, 'add_assistant_message') and (not self.memory.messages or self.memory.messages[-1].get("content") != final_response): self.memory.add_assistant_message(final_response)
            run_duration = time.monotonic() - self.state.get("start_time", time.monotonic())
            return {"response": final_response, "error": error_message, "iterations": self.current_iteration, "duration_seconds": run_duration, "tool_calls": self.execution_state.get("tool_results", []), "finished": True, "success": False}
        except Exception as e:
            error_message = f"Unhandled error in agent run: {e}"
            logger.exception(f"[{self.name}] {error_message}")
            final_response = f"I encountered an unexpected error during execution: {e}"
            self.finished = True; self.success = False; self.error = error_message
            if self.memory and hasattr(self.memory, 'add_assistant_message') and (not self.memory.messages or self.memory.messages[-1].get("content") != final_response): self.memory.add_assistant_message(final_response)
            run_duration = time.monotonic() - self.state.get("start_time", time.monotonic())
            return {"response": final_response, "error": error_message, "iterations": self.current_iteration, "duration_seconds": run_duration, "tool_calls": self.execution_state.get("tool_results", []), "finished": True, "success": False}


    # --- Method Definitions (_create_plan, _determine_next_action, etc.) ---

    async def _create_plan(self, task: str, depth: int = 0) -> Dict[str, Any]:
        """(Async) Create a plan for accomplishing the task."""
        # (Keep implementation from previous versions - MUST be defined)
        if not hasattr(self, 'llm') or not self.llm: raise RuntimeError("LLM not initialized for planning.")
        if depth >= self.max_planning_depth:
            logger.warning(f"[{self.name}] Max planning depth ({self.max_planning_depth}) reached.")
            return {"steps": ["Complete the task directly"], "reasoning": "Plan simplified due to complexity.", "created_at": time.monotonic()}

        tool_descriptions = self.tools.get_tool_descriptions() if self.tools and len(self.tools) > 0 else "No tools available."

        planning_prompt = f"Task: {task}\n\nAvailable tools:\n{tool_descriptions}\n\nCreate a step-by-step plan to accomplish the task. Be concise. Format:\nReasoning: <brief analysis>\nPlan:\n1. <step 1>\n2. <step 2>..."
        system_prompt = "You are a strategic planner. Create a concise, actionable plan."

        planning_response = await self.llm.generate(prompt=planning_prompt, system_prompt=system_prompt, temperature=0.3)

        if planning_response.get("error"):
             logger.error(f"[{self.name}] LLM failed during planning: {planning_response['error']}")
             return {"reasoning": f"Planning failed due to LLM error: {planning_response['error']}", "steps": ["Attempt to complete task directly due to planning error."], "created_at": time.monotonic()}

        response_text = planning_response.get("content", "")
        reasoning = ""
        reasoning_match = re.search(r"Reasoning:(.*?)(Plan:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
             reasoning = reasoning_match.group(1).strip()

        steps = []
        plan_match = re.search(r"Plan:(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if plan_match:
            plan_text = plan_match.group(1).strip()
            steps = [line.strip() for line in re.findall(r"^\s*(?:\d+\.|-)\s*(.*)", plan_text, re.MULTILINE)]

        if not steps:
             logger.warning(f"[{self.name}] Could not parse steps from planning response: {response_text}")
             steps = ["Execute the task based on the request."]

        return {"reasoning": reasoning, "steps": steps, "created_at": time.monotonic()}

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        # (Keep implementation from previous versions)
        tool_name = tool_call.get("name")
        parameters = tool_call.get("parameters", {})
        if not tool_name: logger.error(f"[{self.name}] Tool call missing 'name'."); return {"error": "Tool call missing 'name'."}
        if not self.tools.has_tool(tool_name): logger.error(f"[{self.name}] Tool '{tool_name}' not found."); return {"error": f"Tool '{tool_name}' not found."}
        logger.info(f"[{self.name}] Executing tool '{tool_name}' with parameters: {parameters}")
        try:
            result = await self.tools.execute_tool(tool_name, **parameters)
            if isinstance(result, dict) and "error" in result: logger.error(f"[{self.name}] Tool '{tool_name}' failed: {result['error']}")
            else: logger.info(f"[{self.name}] Tool '{tool_name}' executed successfully."); logger.debug(f"[{self.name}] Tool '{tool_name}' result: {str(result)[:200]}...")
            return result
        except Exception as e: logger.exception(f"[{self.name}] Error awaiting tool '{tool_name}': {e}"); return {"error": f"Failed tool '{tool_name}': {e}"}


    # --- New Helper Methods from Snippet ---
    def _parse_and_validate_action(self, response_data: dict) -> dict:
        """Parses and validates the action JSON from LLM response, with fallback."""
        try:
            if not response_data: logger.warning("Received empty response data from LLM."); return {"error": "Empty response from LLM"}
            llm_error = response_data.get("error")
            if llm_error: logger.error(f"LLM generation returned an error: {llm_error}"); return {"error": f"LLM generation failed: {llm_error}"}

            action_json_str = response_data.get("content"); action_json_str = str(action_json_str or "").strip()

            if not action_json_str: logger.warning("LLM response content was empty after stripping."); return self._analyze_response_for_completion(action_json_str) # Analyze empty string

            potential_json = action_json_str
            # Refined markdown extraction
            md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", action_json_str, re.DOTALL | re.IGNORECASE)
            if md_match:
                 potential_json = md_match.group(1).strip()
            # Also handle direct JSON without markdown fences
            elif not (potential_json.startswith('{') and potential_json.endswith('}')):
                 logger.debug(f"Response not direct JSON or in markdown block: {potential_json[:100]}...")
                 # If not looking like JSON, go straight to fallback
                 return self._analyze_response_for_completion(action_json_str)


            if not potential_json: logger.warning("JSON content inside markdown block is empty."); return self._analyze_response_for_completion(action_json_str)

            parsed = json.loads(potential_json)

            if not isinstance(parsed, dict): logger.warning(f"Parsed JSON is not a dict: {parsed}"); return self._analyze_response_for_completion(action_json_str)

            action_type = parsed.get("action", "").strip().lower()

            if not action_type: logger.warning("Parsed JSON missing 'action' key."); return self._analyze_response_for_completion(action_json_str)

            valid_actions = ["use_tool", "next_step", "complete"]
            if action_type not in valid_actions: logger.warning(f"Invalid action type '{action_type}'."); return self._analyze_response_for_completion(action_json_str)

            # Validate specific actions
            validated_action = {"action": action_type}
            if action_type == "use_tool":
                 tool_name = parsed.get("tool"); parameters = parsed.get("parameters", {})
                 if not tool_name or not isinstance(tool_name, str) or not isinstance(parameters, dict): return {"error": "Invalid 'use_tool' format"}
                 validated_action["tool"] = tool_name; validated_action["parameters"] = parameters
            elif action_type == "next_step":
                 completed_step = parsed.get("completed_step")
                 if not completed_step or not isinstance(completed_step, str): return {"error": "Invalid 'next_step' format"}
                 validated_action["completed_step"] = completed_step
            elif action_type == "complete":
                 final_response = parsed.get("response")
                 if final_response is None: return {"error": "Invalid 'complete' format"}
                 validated_action["response"] = final_response

            logger.debug(f"Parsed and validated action: {action_type}")
            return validated_action

        except json.JSONDecodeError: logger.warning(f"Failed decode JSON: {action_json_str[:200]}..."); return self._analyze_response_for_completion(action_json_str)
        except Exception as e: logger.exception(f"Unexpected error parsing/validating action: {e}"); return {"error": f"Internal error parsing action: {e}"}

    def _analyze_response_for_completion(self, raw_response_content: str) -> dict:
        """Fallback analysis if JSON action is missing/invalid."""
        response_lower = str(raw_response_content or "").lower()
        completion_keywords = ["final answer is:", "here is the final summary:", "the task is complete", "i have finished the task", "summary of the findings:", "conclusion:", "the files contain the following:", "here is a summary of the files:"]
        inability_keywords = ["i cannot proceed", "unable to continue", "i lack the ability", "i cannot access"]

        if any(kw in response_lower for kw in completion_keywords) or any(kw in response_lower for kw in inability_keywords):
            logger.info("Fallback analysis determined action 'complete'.")
            final_resp_text = raw_response_content.strip('` ')
            if final_resp_text.lower().startswith("json"): final_resp_text = final_resp_text[len("json"):].strip()
            return {"action": "complete", "response": final_resp_text, "auto_generated": True}

        logger.warning(f"Fallback analysis could not determine action: {raw_response_content[:200]}...")
        return {"error": "Could not determine action type from response content via fallback analysis.", "response_preview": raw_response_content[:200]}

    # --- Method Using the New Helpers ---
    async def _determine_next_action(self) -> Dict[str, Any]:
        """(Async) Determine the next action based on current state using robust parsing."""
        context = self._prepare_execution_context()
        action_prompt = ( # (Keep the same action prompt as before)
            f"{context}\n\n"
            f"## Instruction:\n"
            f"Based on the current state, decide the single best next action. Choose ONE option below and respond ONLY with the corresponding JSON structure:\n\n"
            f"1. **Use a Tool:** ...\n   ```json\n   {{\"action\": \"use_tool\", ...}}\n   ```\n\n"
            f"2. **Mark Step Complete:** ...\n   ```json\n   {{\"action\": \"next_step\", ...}}\n   ```\n\n"
            f"3. **Complete Task:** ...\n   ```json\n   {{\"action\": \"complete\", ...}}\n   ```\n"
        )
        system_prompt = "You are an autonomous agent deciding the next action. Respond ONLY in JSON."

        logger.debug(f"--- Action Prompt Sent to LLM ---:\n{action_prompt}")

        try:
            # Get the raw response dictionary from the LLM
            action_response_dict = await self.llm.generate(
                prompt=action_prompt,
                system_prompt=system_prompt,
                temperature=0.1 # Low temp for deterministic action choice
            )
            logger.debug(f"--- Raw LLM Response Dict Received ---:\n{action_response_dict}")

            # --- Use the new robust parsing method ---
            parsed_action = self._parse_and_validate_action(action_response_dict)
            # -----------------------------------------

            # Return the parsed action (which might contain an error key)
            # The main run loop will handle the error if present
            return parsed_action

        except Exception as e:
             # Catch other unexpected errors during the process
             logger.exception(f"Unexpected error determining next action: {e}")
             return {"error": f"Failed to determine next action due to unexpected error: {e}"}


    async def _reflect_on_progress(self) -> None:
        # (Implementation unchanged)
        context = self._prepare_execution_context()
        logger.info(f"[{self.name}] Reflecting on progress at iteration {self.current_iteration}")
        reflection_prompt = f"{context}\n\nReflect on progress. Are you on track? Any obstacles? Does the plan need adjustment? What's next?"
        system_prompt = "You are a self-reflecting agent evaluating task progress."
        reflection_response = await self.llm.generate(prompt=reflection_prompt, system_prompt=system_prompt, temperature=0.3)
        if reflection_response.get("error"): logger.error(f"[{self.name}] LLM failed reflection: {reflection_response['error']}"); self.execution_state["reflections"].append({"iteration": self.current_iteration, "reflection": f"Error: {reflection_response['error']}"}); return
        reflection_text = reflection_response.get("content", ""); self.execution_state["reflections"].append({"iteration": self.current_iteration, "reflection": reflection_text})
        logger.debug(f"[{self.name}] Reflection: {reflection_text[:200]}...")
        reflection_lower = reflection_text.lower()
        if "adjust plan" in reflection_lower or "new plan" in reflection_lower or "revise plan" in reflection_lower:
            logger.info(f"[{self.name}] Reflection suggests plan adjustment.")
            summary = self._summarize_progress_for_replan(); task = self.state.get('task', ''); desc = f"Original Task: {task}\nProgress: {summary}\nObjective: Complete original task."
            try: new_plan = await self._create_plan(desc, depth=1); self.execution_state["plan"] = new_plan; self.execution_state["completed_steps"] = []; self.execution_state["current_step"] = self._get_next_plan_step(); logger.info(f"[{self.name}] Plan adjusted.")
            except Exception as e: logger.error(f"[{self.name}] Failed create adjusted plan: {e}")

    def _summarize_progress_for_replan(self) -> str:
        # (Implementation unchanged)
         summary_parts = []; completed_steps = self.execution_state.get("completed_steps", []); tool_results = self.execution_state.get("tool_results", [])
         if completed_steps: summary_parts.append("Completed Steps: " + ", ".join(completed_steps))
         if tool_results:
              summary_parts.append("Recent Tool Results:")
              for res in tool_results[-2:]:
                   name = res.get("tool", "?"); output = res.get("output", {}); status = "Success" if isinstance(output, dict) and "error" not in output else "Error"
                   if isinstance(output, dict): preview = str(output.get("result", output.get("error", "N/A")))[:100]
                   else: preview = str(output)[:100]
                   summary_parts.append(f"- {name}: {status} ({preview}...)")
         return "\n".join(summary_parts) or "No progress yet."

    async def _generate_final_response(self, task: str) -> str:
        # (Implementation unchanged)
        context = self._prepare_execution_context()
        prompt = f"{context}\n\nTask finished or max iterations reached. Generate final response."
        system_prompt = "Summarize results of agent's work."
        response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.5)
        if response.get("error"): logger.error(f"LLM failed final response: {response['error']}"); return "Error generating final summary."
        return response.get("content", "Task complete.")

    def _prepare_execution_context(self) -> str:
        # (Implementation largely unchanged, ensure safe access)
        context = [f"## Task:\n{self.state.get('task', 'N/A')}"]
        plan = self.execution_state.get("plan")
        if plan and isinstance(plan.get("steps"), list):
             context.append("\n## Plan:")
             if plan.get("reasoning"): context.append(f"Reasoning: {plan['reasoning']}")
             context.append("Steps:")
             current_step_found = False; current_step = self.execution_state.get("current_step"); completed_set = set(self.execution_state.get("completed_steps", []))
             for i, step in enumerate(plan["steps"]):
                  status = "[✓ DONE]" if step in completed_set else ("[▶ CURRENT]" if not current_step_found and step == current_step else "")
                  if status == "[▶ CURRENT]": current_step_found = True
                  context.append(f"  {i+1}. {status} {step}")
             if not current_step_found and current_step and current_step not in completed_set: context.append(f"  [▶ CURRENT] {current_step}") # Show if not in list
        tool_results = self.execution_state.get("tool_results", [])
        if tool_results:
             context.append("\n## Recent Tool History (Max 3):")
             for res in tool_results[-3:]:
                  name = res.get("tool", "?"); output = res.get("output", {}); status = "Success" if isinstance(output, dict) and "error" not in output else "Error"
                  if isinstance(output, dict): preview = str(output.get("result", output.get("error", "?")))[:150]
                  else: preview = str(output)[:150]
                  inp_prev = str(res.get("input", {}))[:50]
                  context.append(f"- Itr {res.get('iteration','?')}: Ran `{name}` ({inp_prev}...) -> {status}: `{preview}...`")
        reflections = self.execution_state.get("reflections", [])
        if reflections: context.append(f"\n## Last Reflection (Itr {reflections[-1].get('iteration','?')}):\n{str(reflections[-1].get('reflection','?'))[:300]}...")
        context.append(f"\n## Status: Itr {self.current_iteration}/{self.max_iterations}")
        return "\n".join(context)

    def _get_next_plan_step(self) -> Optional[str]:
        # (Implementation unchanged)
         plan_steps = self.execution_state.get("plan", {}).get("steps", [])
         completed = self.execution_state.get("completed_steps", []); completed = completed if isinstance(completed, list) else []
         completed_set = set(completed)
         for step in plan_steps:
              if step not in completed_set: return step
         return None

    async def reset(self) -> None:
        # (Implementation unchanged)
        # Check if super().reset is actually async now
        if hasattr(super(), 'reset') and asyncio.iscoroutinefunction(super().reset):
            await super().reset()
        else:
             if hasattr(super(), 'reset'): super().reset() # Call sync reset if base is sync
        self._initialize_execution_state()
        if self.verbose: logger.info(f"Reset agent '{self.name}' execution state.")

    async def chat(self, message: str, **kwargs) -> str:
        # (Implementation unchanged)
        result = await self.run(message, **kwargs)
        return result.get("response", "I encountered an issue and couldn't generate a response.")