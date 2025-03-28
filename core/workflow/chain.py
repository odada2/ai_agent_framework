# ai_agent_framework/core/workflow/chain.py

"""
Prompt Chaining Workflow Implementation (Async Tool Execution)
"""

import asyncio # Ensure asyncio is imported
import logging
from typing import Any, Callable, Dict, List, Optional, Union

# Framework components
from ai_agent_framework.core.llm.base import BaseLLM
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.core.workflow.base import BaseWorkflow
from ai_agent_framework.core.exceptions import WorkflowError, ToolError # Assuming ToolError exists

logger = logging.getLogger(__name__)


class PromptChain(BaseWorkflow):
    """
    Implements the prompt chaining workflow pattern. Executes steps sequentially,
    handling async LLM calls and async tool execution.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        steps: List[Dict[str, Any]],
        tools: Optional[ToolRegistry] = None,
        max_steps: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the prompt chaining workflow.
        (Docstring remains the same)
        """
        effective_max_steps = max_steps if max_steps is not None else len(steps)
        super().__init__(name=name, max_steps=effective_max_steps, verbose=verbose, **kwargs)

        if not llm: raise ValueError("An LLM instance is required for PromptChain.")
        if not steps: raise ValueError("At least one step must be defined for PromptChain.")

        self.llm = llm
        if tools is not None and not isinstance(tools, ToolRegistry):
             logger.warning("Provided 'tools' is not a ToolRegistry instance. Tool usage might fail.")
        self.tools = tools
        self.steps = self._validate_and_prepare_steps(steps) # Sync validation is okay

    def _validate_and_prepare_steps(self, steps_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validates step configurations and adds defaults. (Synchronous)"""
        # (Implementation remains the same as provided before)
        validated_steps = []
        for i, step_config in enumerate(steps_config):
            if not isinstance(step_config, dict):
                raise ValueError(f"Step configuration at index {i} must be a dictionary.")

            step_name = step_config.get("name")
            if not step_name:
                step_name = f"step_{i+1}"
                step_config["name"] = step_name

            if "prompt" not in step_config and "prompt_template" not in step_config:
                # Allow steps that *only* call a function/tool without an LLM call?
                # For now, require prompt for LLM-based chain steps.
                 if not step_config.get("execute_function"): # Add check for alternative execution
                      raise ValueError(f"Step '{step_name}' must have 'prompt' or 'prompt_template' (or 'execute_function').")

            validated_steps.append(step_config)
        return validated_steps


    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the prompt chain asynchronously.
        (Docstring remains the same)
        """
        # Use await if BaseWorkflow.reset becomes async
        self.reset()

        initial_input = input_data
        current_step_input = input_data
        result_data = {
            "input": initial_input,
            "intermediate_results": [],
            "final_result": None,
            "error": None,
            "execution_summary": {}
        }

        try:
            for i, step_config in enumerate(self.steps):
                step_name = step_config["name"]
                logger.info(f"[{self.name}] Executing step {i+1}/{len(self.steps)}: '{step_name}'")

                if not self._increment_step(): # Sync check
                    result_data["error"] = self.error
                    break

                # Execute the step asynchronously
                step_output = await self._execute_step(step_config, current_step_input, initial_input)

                step_error = step_output.get("error") if isinstance(step_output, dict) else None
                self._log_step(step_name, current_step_input, step_output, error=step_error) # Sync log

                intermediate_step_result = {
                    "step": step_name,
                    "input": f"Output from step {i}" if not self.verbose else current_step_input,
                    "output": step_output,
                    "gate_passed": None
                }
                result_data["intermediate_results"].append(intermediate_step_result)

                if step_error:
                    raise WorkflowError(f"Step '{step_name}' failed: {step_error}")

                # --- Gate Check ---
                gate_func = step_config.get("gate")
                if callable(gate_func):
                    try:
                        gate_passed = await gate_func(step_output) if asyncio.iscoroutinefunction(gate_func) else gate_func(step_output)
                        intermediate_step_result["gate_passed"] = gate_passed
                        if not gate_passed:
                            logger.warning(f"[{self.name}] Gate check failed after step '{step_name}'. Stopping chain.")
                            result_data["final_result"] = step_output # Use output before failed gate
                            self._mark_finished(success=False, error="Gate check failed")
                            result_data["error"] = "Gate check failed"
                            break # Stop workflow execution
                        else:
                             logger.debug(f"[{self.name}] Gate check passed for step '{step_name}'.")
                    except Exception as gate_e:
                         raise WorkflowError(f"Error executing gate function for step '{step_name}': {gate_e}") from gate_e

                # Prepare input for the next step
                current_step_input = step_output

            # ---- End of Loop ----
            if not self.finished: # If loop completed normally
                 result_data["final_result"] = current_step_input
                 self._mark_finished(success=True)

        except Exception as e:
            logger.exception(f"Workflow '{self.name}' failed during execution: {e}")
            self._mark_finished(success=False, error=str(e))
            result_data["error"] = str(e)
            # Assign last successful output if available
            result_data["final_result"] = result_data["intermediate_results"][-1]["output"] if result_data.get("intermediate_results") else None


        result_data["execution_summary"] = self.get_execution_summary() # Sync get
        return result_data

    async def _execute_step(self, step_config: Dict[str, Any], current_input: Any, initial_input: Any) -> Any:
        """
        Execute a single step asynchronously, handling LLM calls and async tool usage.
        """
        step_name = step_config["name"]
        use_tools_flag = step_config.get("use_tools", False) and self.tools is not None

        try:
            # --- Prepare Prompt ---
            # (Prompt formatting logic remains the same - synchronous)
            prompt_template = step_config.get("prompt_template")
            prompt_str = step_config.get("prompt")
            prompt = None
            if prompt_template:
                format_data = {"input": current_input, "original_input": initial_input}
                if isinstance(current_input, dict): format_data.update(current_input)
                prompt = self._format_prompt_template(prompt_template, format_data)
            elif prompt_str:
                prompt = prompt_str
            else: # If no prompt, maybe it's just a function call step? (Not handled yet)
                 raise ValueError(f"Step '{step_name}' needs 'prompt' or 'prompt_template'.")

            system_prompt = step_config.get("system_prompt")
            llm_temp = step_config.get("temperature")
            llm_max_tokens = step_config.get("max_tokens")

            # --- Execute LLM ---
            llm_response: Dict[str, Any]
            if use_tools_flag:
                tool_names = step_config.get("tools") # Optional list of specific tool names
                available_tools_for_step = self.tools.get_tool_definitions(tool_names) if self.tools else []

                if not available_tools_for_step:
                     logger.warning(f"[{self.name}-{step_name}] Step configured to use tools, but no tools available/selected. Generating without tools.")
                     llm_response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=llm_temp, max_tokens=llm_max_tokens)
                else:
                     logger.debug(f"[{self.name}-{step_name}] Generating with tools: {[t['name'] for t in available_tools_for_step]}")
                     llm_response = await self.llm.generate_with_tools(prompt=prompt, tools=available_tools_for_step, system_prompt=system_prompt, temperature=llm_temp, max_tokens=llm_max_tokens)
            else:
                logger.debug(f"[{self.name}-{step_name}] Generating without tools.")
                llm_response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=llm_temp, max_tokens=llm_max_tokens)

            if "error" in llm_response:
                raise WorkflowError(f"LLM generation failed: {llm_response['error']}")

            # --- Execute Tool Calls (if any) ---
            step_output: Any = llm_response.get("content", "") # Start with text content
            tool_results_list: List[Dict] = []
            llm_tool_calls = llm_response.get("tool_calls", [])

            if use_tools_flag and llm_tool_calls:
                logger.info(f"[{self.name}-{step_name}] Executing {len(llm_tool_calls)} tool call(s)...")
                # Execute tools concurrently using asyncio.gather
                tool_coroutines = [self._execute_single_tool_call(tc) for tc in llm_tool_calls]
                tool_results_list = await asyncio.gather(*tool_coroutines, return_exceptions=True)

                # Check for errors during tool execution
                failed_tool_calls = []
                processed_results = []
                for i, res_or_exc in enumerate(tool_results_list):
                     tool_call_info = llm_tool_calls[i] # Map back to original call
                     if isinstance(res_or_exc, Exception):
                          tool_name = tool_call_info.get("name", "unknown")
                          error_str = f"Unhandled exception: {type(res_or_exc).__name__}({res_or_exc})"
                          logger.error(f"Unhandled exception executing tool '{tool_name}': {res_or_exc}", exc_info=self.verbose)
                          failed_tool_calls.append({"tool_name": tool_name, "error": error_str})
                          processed_results.append({"tool_name": tool_name, "parameters": tool_call_info.get("parameters",{}), "error": error_str}) # Add error record
                     elif isinstance(res_or_exc, dict) and "error" in res_or_exc:
                          failed_tool_calls.append(res_or_exc) # Add error dict from _execute_single_tool_call
                          processed_results.append(res_or_exc)
                     elif isinstance(res_or_exc, dict):
                           processed_results.append(res_or_exc) # Successful result
                     else: # Unexpected result type
                           tool_name = tool_call_info.get("name", "unknown")
                           error_str = f"Unexpected result type from tool execution: {type(res_or_exc).__name__}"
                           logger.error(error_str)
                           failed_tool_calls.append({"tool_name": tool_name, "error": error_str})
                           processed_results.append({"tool_name": tool_name, "parameters": tool_call_info.get("parameters",{}), "error": error_str})

                tool_results_list = processed_results # Update list with processed results/errors


                # Handle tool failures - Option: Raise immediately if any tool fails?
                # if failed_tool_calls:
                #      errors = "; ".join([f"{t['tool_name']}: {t['error']}" for t in failed_tool_calls])
                #      raise ToolError(f"One or more tools failed execution: {errors}")

                # --- Prepare Step Output ---
                if step_config.get("return_tool_results", False):
                    # Return both content and structured tool results
                    step_output = {"content": step_output, "tool_results": tool_results_list}
                # else: Just pass the LLM text content (step_output already set)

            # --- Output Parser ---
            parser = step_config.get("output_parser")
            if callable(parser):
                try:
                    if asyncio.iscoroutinefunction(parser):
                         parsed_output = await parser(step_output)
                    else:
                         parsed_output = parser(step_output)
                    logger.debug(f"[{self.name}-{step_name}] Output parsed successfully.")
                    return parsed_output
                except Exception as parse_e:
                    raise WorkflowError(f"Output parser failed for step '{step_name}': {parse_e}") from parse_e

            return step_output # Return final output for this step

        except Exception as e:
            logger.exception(f"Error during execution of step '{step_name}': {e}")
            # Return error dictionary to be handled by the main execute loop
            return {"error": str(e)}


    # Make _execute_single_tool_call asynchronous
    async def _execute_single_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single tool call dictionary asynchronously."""
        tool_name = tool_call.get("name")
        tool_params = tool_call.get("parameters", {})

        if not tool_name: return {"error": "Tool call missing 'name'."}
        if not isinstance(tool_params, dict): return {"tool_name": tool_name, "error": f"Tool parameters must be a dictionary."}
        if not self.tools or not self.tools.has_tool(tool_name): return {"tool_name": tool_name, "error": f"Tool '{tool_name}' not found."}

        try:
             logger.info(f"Executing tool '{tool_name}' with params: {tool_params}")
             # Await the now-async execute_tool method
             result_or_error = await self.tools.execute_tool(tool_name, **tool_params)
             # execute_tool now returns the result directly, or an error dict
             if isinstance(result_or_error, dict) and "error" in result_or_error:
                   logger.error(f"Tool '{tool_name}' execution returned error: {result_or_error['error']}")
                   return {"tool_name": tool_name, "parameters": tool_params, "error": result_or_error['error']}
             else:
                   return {"tool_name": tool_name, "parameters": tool_params, "result": result_or_error}
        except Exception as e:
             # Catch errors during the awaiting of execute_tool itself
             logger.exception(f"Unexpected error awaiting tool '{tool_name}': {e}")
             return {"tool_name": tool_name, "parameters": tool_params, "error": f"Unexpected await error: {e}"}

    def _format_prompt_template(self, template: str, data: Dict[str, Any]) -> str:
        """Formats a prompt template string using the provided data dictionary. (Synchronous)"""
        # (Implementation remains the same as provided before)
        try:
            if isinstance(data.get("input"), (str, int, float, bool)) and "{input}" in template and len(data) <= 2:
                 return template.format(input=data["input"], original_input=data.get("original_input", ""))
            format_ready_data = {k: str(v) for k, v in data.items()}
            if 'input' in data and 'input' not in format_ready_data: format_ready_data['input'] = str(data['input'])
            if 'original_input' in data and 'original_input' not in format_ready_data: format_ready_data['original_input'] = str(data['original_input'])
            return template.format(**format_ready_data)
        except KeyError as e:
            logger.error(f"Missing key '{e}' for prompt template: {template}. Available: {list(data.keys())}")
            raise KeyError(f"Missing key '{e}' for prompt template.") from e
        except Exception as e:
             logger.error(f"Error formatting prompt template: {e}. Template: {template}")
             raise ValueError(f"Failed to format prompt template: {e}")

    # reset method can remain synchronous if BaseWorkflow.reset is sync
    # def reset(self) -> None:
    #     super().reset()