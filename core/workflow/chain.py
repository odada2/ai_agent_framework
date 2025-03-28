# ai_agent_framework/core/workflow/chain.py

"""
Prompt Chaining Workflow Implementation

Executes a sequence of steps, typically involving LLM calls, where the output
of one step becomes the input for the next. Supports integrating tool usage
within steps.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union

# Framework components (using absolute imports)
from ai_agent_framework.core.llm.base import BaseLLM
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.core.workflow.base import BaseWorkflow
# Assuming exceptions are defined
from ai_agent_framework.core.exceptions import WorkflowError, ToolError

logger = logging.getLogger(__name__)


class PromptChain(BaseWorkflow):
    """
    Implements the prompt chaining workflow pattern.

    Executes a series of configured steps sequentially. Each step can involve
    an LLM call (potentially with tools) or custom logic. The output of a
    step generally serves as the input for the subsequent step.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        steps: List[Dict[str, Any]],
        tools: Optional[ToolRegistry] = None,
        max_steps: Optional[int] = None, # Override BaseWorkflow default if needed
        verbose: bool = False,
        **kwargs # Allow passing additional config
    ):
        """
        Initialize the prompt chaining workflow.

        Args:
            name: Name of the workflow.
            llm: LLM instance to use for steps involving LLM calls.
            steps: List of step configurations. Each step is a dict, typically
                   requiring 'name' and ('prompt' or 'prompt_template'). Other
                   optional keys: 'system_prompt', 'use_tools' (bool), 'tools' (list[str]),
                   'return_tool_results' (bool), 'output_parser' (Callable),
                   'gate' (Callable), 'temperature', 'max_tokens'.
            tools: Optional tool registry available to LLM steps.
            max_steps: Maximum number of steps to execute (defaults to len(steps)).
            verbose: Whether to log detailed execution information.
            **kwargs: Additional configuration passed to BaseWorkflow.
        """
        # Default max_steps to the number of defined steps if not provided
        effective_max_steps = max_steps if max_steps is not None else len(steps)
        super().__init__(name=name, max_steps=effective_max_steps, verbose=verbose, **kwargs)

        if not llm:
            raise ValueError("An LLM instance is required for PromptChain.")
        if not steps:
            raise ValueError("At least one step must be defined for PromptChain.")

        self.llm = llm
        # Ensure tools is a ToolRegistry or None
        if tools is not None and not isinstance(tools, ToolRegistry):
             logger.warning("Provided 'tools' is not a ToolRegistry instance. Tool usage might fail.")
        self.tools = tools
        self.steps = self._validate_and_prepare_steps(steps)

    def _validate_and_prepare_steps(self, steps_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validates step configurations and adds defaults."""
        validated_steps = []
        for i, step_config in enumerate(steps_config):
            if not isinstance(step_config, dict):
                raise ValueError(f"Step configuration at index {i} must be a dictionary.")

            step_name = step_config.get("name")
            if not step_name:
                step_name = f"step_{i+1}"
                step_config["name"] = step_name

            if "prompt" not in step_config and "prompt_template" not in step_config:
                raise ValueError(f"Step '{step_name}' must have either 'prompt' or 'prompt_template'.")

            # TODO: Add validation for other keys like 'output_parser', 'gate' types if needed

            validated_steps.append(step_config)
        return validated_steps

    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the prompt chain asynchronously.

        Args:
            input_data: Initial input for the first step of the chain. Can be a string
                        or a dictionary. If a dictionary, keys can be used in
                        the first step's prompt template.
            **kwargs: Additional runtime parameters (currently not used by chain logic itself).

        Returns:
            Dictionary containing the final result, intermediate results,
            and execution metadata. Keys include: 'input', 'intermediate_results',
            'final_result', 'error' (if any), 'execution_summary'.
        """
        await self.reset() # Use await if BaseWorkflow.reset becomes async

        # Prepare initial state
        initial_input = input_data # Keep track of the very first input
        current_step_input = input_data
        result_data = {
            "input": initial_input,
            "intermediate_results": [],
            "final_result": None,
            "error": None,
            "execution_summary": {}
        }

        try:
            # Execute each step in sequence
            for i, step_config in enumerate(self.steps):
                step_name = step_config["name"]
                logger.info(f"Executing step {i+1}/{len(self.steps)}: '{step_name}'")

                # Check iteration limit (using BaseWorkflow's mechanism)
                if not self._increment_step(): # This also logs warning if max steps reached
                    result_data["error"] = self.error # Get error from BaseWorkflow state
                    break # Exit loop if max steps exceeded

                # Execute the single step
                step_output = await self._execute_step(step_config, current_step_input, initial_input)

                # Log step execution (including potential errors from _execute_step)
                step_error = step_output.get("error") if isinstance(step_output, dict) else None
                self._log_step(step_name, current_step_input, step_output, error=step_error)

                # Store intermediate result
                intermediate_step_result = {
                    "step": step_name,
                    # Avoid logging potentially large inputs/outputs unless verbose
                    "input": current_step_input if self.verbose else f"Output from step {i}",
                    "output": step_output,
                    "gate_passed": None # Gate logic check comes next
                }
                result_data["intermediate_results"].append(intermediate_step_result)

                # If step execution itself returned an error, stop the chain
                if step_error:
                    raise WorkflowError(f"Step '{step_name}' failed: {step_error}")


                # Check optional gate function for this step
                gate_func = step_config.get("gate")
                if gate_func and callable(gate_func):
                    try:
                        # Gate function can be sync or async
                        if asyncio.iscoroutinefunction(gate_func):
                            gate_passed = await gate_func(step_output)
                        else:
                            gate_passed = gate_func(step_output)

                        intermediate_step_result["gate_passed"] = gate_passed # Update intermediate result
                        if not gate_passed:
                            logger.warning(f"Gate check failed after step '{step_name}'. Stopping chain.")
                            # Use the output of the step *before* the failed gate as final result
                            result_data["final_result"] = step_output
                            self._mark_finished(success=False, error="Gate check failed")
                            result_data["error"] = "Gate check failed"
                            result_data["execution_summary"] = self.get_execution_summary()
                            return result_data
                        else:
                             logger.debug(f"Gate check passed for step '{step_name}'.")

                    except Exception as gate_e:
                         raise WorkflowError(f"Error executing gate function for step '{step_name}': {gate_e}")


                # Prepare input for the next step
                current_step_input = step_output

            # If loop completed without errors/gate failures
            result_data["final_result"] = current_step_input # Output of last step is final result
            self._mark_finished(success=True)

        except Exception as e:
            logger.exception(f"Workflow '{self.name}' failed during execution: {e}")
            # Mark finished takes care of setting self.error and self.success
            self._mark_finished(success=False, error=str(e))
            result_data["error"] = str(e)
            # Final result is the output of the last *successful* step, or None
            result_data["final_result"] = result_data["intermediate_results"][-1]["output"] if result_data["intermediate_results"] else None

        result_data["execution_summary"] = self.get_execution_summary()
        return result_data

    async def _execute_step(self, step_config: Dict[str, Any], current_input: Any, initial_input: Any) -> Any:
        """
        Execute a single step in the chain, handling LLM calls and tool usage.

        Args:
            step_config: Configuration dictionary for the step.
            current_input: Input data for this step (output from previous step).
            initial_input: The very first input given to the workflow.

        Returns:
            Step execution result, or a dictionary with an 'error' key if failed.
        """
        step_name = step_config["name"]
        use_tools_flag = step_config.get("use_tools", False) and self.tools is not None

        try:
            # 1. Format the prompt
            prompt_template = step_config.get("prompt_template")
            prompt_str = step_config.get("prompt")

            if prompt_template:
                # Prepare data for formatting, including initial input if needed
                format_data = {
                    "input": current_input, # Output from previous step
                    "original_input": initial_input # The very first input to the chain
                }
                # Add previous intermediate results if needed (use with caution for context limits)
                # format_data["intermediate_results"] = result_data.get("intermediate_results", [])

                # If current_input is a dict, merge it for easier templating
                if isinstance(current_input, dict):
                    format_data.update(current_input)

                prompt = self._format_prompt_template(prompt_template, format_data)
            elif prompt_str:
                prompt = prompt_str # Use literal prompt
            else:
                # This should have been caught by validation, but double-check
                raise ValueError(f"Step '{step_name}' has neither 'prompt' nor 'prompt_template'.")

            system_prompt = step_config.get("system_prompt")
            llm_temp = step_config.get("temperature") # Uses LLM default if None
            llm_max_tokens = step_config.get("max_tokens") # Uses LLM default if None

            # 2. Execute LLM call (with or without tools)
            llm_response: Dict[str, Any]

            if use_tools_flag:
                tool_names = step_config.get("tools") # Optional list of specific tool names for this step
                available_tools_for_step = self.tools.get_tool_definitions(tool_names) # Handles None or list

                if not available_tools_for_step:
                     logger.warning(f"Step '{step_name}' configured to use tools, but no tools are available/selected.")
                     # Fallback to generating without tools
                     llm_response = await self.llm.generate(
                         prompt=prompt, system_prompt=system_prompt,
                         temperature=llm_temp, max_tokens=llm_max_tokens
                     )
                else:
                     logger.debug(f"Step '{step_name}' generating with tools: {[t['name'] for t in available_tools_for_step]}")
                     llm_response = await self.llm.generate_with_tools(
                         prompt=prompt, tools=available_tools_for_step, system_prompt=system_prompt,
                         temperature=llm_temp, max_tokens=llm_max_tokens
                     )
            else:
                logger.debug(f"Step '{step_name}' generating without tools.")
                llm_response = await self.llm.generate(
                    prompt=prompt, system_prompt=system_prompt,
                    temperature=llm_temp, max_tokens=llm_max_tokens
                )

            # Check for errors from LLM generation
            if "error" in llm_response:
                raise WorkflowError(f"LLM generation failed: {llm_response['error']}")

            # 3. Process Tool Calls (if any)
            step_output: Any = llm_response.get("content", "") # Start with text content
            tool_results_list: List[Dict] = []

            if use_tools_flag and "tool_calls" in llm_response and llm_response["tool_calls"]:
                logger.info(f"Step '{step_name}' requires tool execution: {len(llm_response['tool_calls'])} call(s).")
                # NOTE: Executes tools sequentially. For parallel, use asyncio.gather with _execute_single_tool_call
                for tool_call in llm_response["tool_calls"]:
                    tool_result_info = self._execute_single_tool_call(tool_call) # Synchronous execution
                    tool_results_list.append(tool_result_info)
                    # Handle tool error - stop chain or provide error info to next step?
                    if "error" in tool_result_info:
                         logger.error(f"Tool execution failed in step '{step_name}': {tool_result_info}")
                         # Option 1: Raise error, stopping the chain
                         # raise ToolError(f"Tool '{tool_result_info.get('tool_name')}' failed: {tool_result_info['error']}")
                         # Option 2: Include error in results (chosen here)
                         pass # Error is already in tool_results_list


                # Decide what to pass to the next step or output parser
                if step_config.get("return_tool_results", False):
                    # Return both content and tool results if requested
                    step_output = {
                        "content": step_output,
                        "tool_results": tool_results_list
                    }
                # else: Just pass the LLM text content (step_output already set)

            # 4. Apply Output Parser
            parser = step_config.get("output_parser")
            if parser and callable(parser):
                try:
                    parsed_output = parser(step_output) # Pass potentially structured output
                    logger.debug(f"Step '{step_name}' output parsed successfully.")
                    return parsed_output
                except Exception as parse_e:
                    raise WorkflowError(f"Output parser failed for step '{step_name}': {parse_e}")

            # 5. Return final output for this step
            return step_output

        except Exception as e:
            logger.exception(f"Error during execution of step '{step_name}': {e}")
            # Return error dictionary to be handled by the main execute loop
            return {"error": str(e)}

    def _execute_single_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single tool call dictionary."""
        tool_name = tool_call.get("name")
        tool_params = tool_call.get("parameters", {})

        if not tool_name:
            return {"error": "Tool call missing 'name'."}
        if not isinstance(tool_params, dict):
             return {"tool_name": tool_name, "error": f"Tool parameters must be a dictionary, got {type(tool_params)}."}

        if not self.tools or not self.tools.has_tool(tool_name):
             return {"tool_name": tool_name, "error": f"Tool '{tool_name}' not found in registry."}

        try:
             logger.info(f"Executing tool '{tool_name}' with params: {tool_params}")
             # Note: Tool execution itself is currently synchronous based on BaseTool/ToolRegistry
             result = self.tools.execute_tool(tool_name, **tool_params)
             return {"tool_name": tool_name, "parameters": tool_params, "result": result}
        except Exception as e:
             logger.exception(f"Error executing tool '{tool_name}': {e}")
             return {"tool_name": tool_name, "parameters": tool_params, "error": str(e)}


    def _format_prompt_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Format a prompt template string using the provided data dictionary.

        Handles simple '{input}' replacement and dictionary key expansion.

        Args:
            template: The prompt template string (e.g., "Analyze {input} for topic {topic}.").
            data: Dictionary containing data to fill the template. Expected keys
                  include 'input', 'original_input', and potentially others if
                  the previous step returned a dictionary.

        Returns:
            Formatted prompt string.

        Raises:
            KeyError: If the template requires a key not present in the data.
        """
        try:
            # Basic check for simple string input replacement if template uses {input}
            # and data['input'] is the primary value
            if isinstance(data.get("input"), (str, int, float, bool)) and "{input}" in template and len(data) <= 2 :
                 # Primarily handles the case where input is just the previous step's string output
                 return template.format(input=data["input"], original_input=data.get("original_input", ""))

            # Otherwise, use dictionary expansion. Make sure all needed keys are strings.
            format_ready_data = {k: str(v) for k, v in data.items()}

            # Include 'input' and 'original_input' explicitly if not already strings
            if 'input' in data and 'input' not in format_ready_data:
                 format_ready_data['input'] = str(data['input'])
            if 'original_input' in data and 'original_input' not in format_ready_data:
                 format_ready_data['original_input'] = str(data['original_input'])


            return template.format(**format_ready_data)
        except KeyError as e:
            logger.error(f"Missing key '{e}' in data for prompt template: {template}. Available keys: {list(data.keys())}")
            raise KeyError(f"Missing key '{e}' for prompt template. Available data: {list(data.keys())}") from e
        except Exception as e:
             logger.error(f"Error formatting prompt template: {e}. Template: {template}, Data keys: {list(data.keys())}")
             raise ValueError(f"Failed to format prompt template: {e}")

    async def reset(self) -> None:
        """Resets the workflow's execution state."""
        # Call BaseWorkflow reset which handles state variables
        super().reset()
        # Add any PromptChain specific state reset if needed in future