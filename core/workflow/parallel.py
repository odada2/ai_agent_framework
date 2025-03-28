# ai_agent_framework/core/workflow/parallel.py

"""
Parallel Workflow Implementation

Executes multiple branches (sub-workflows or tasks) concurrently
and aggregates their results.
"""

import asyncio
import logging
import time
import dataclasses # Use dataclasses for BranchDefinition
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable

# Framework components
from .base import BaseWorkflow
# Import specific exceptions if needed, e.g., from ..exceptions import WorkflowError
from ai_agent_framework.core.exceptions import WorkflowError # Assuming path

logger = logging.getLogger(__name__)

# Define structure for defining a branch
# A branch can be another BaseWorkflow instance or an async function/coroutine
BranchTarget = Union[BaseWorkflow, Callable[..., Awaitable[Any]]]

@dataclasses.dataclass
class BranchDefinition:
    """
    Defines a single branch to be executed within a ParallelWorkflow.

    Attributes:
        name: A unique identifier for this branch within the workflow.
        target: The BaseWorkflow instance or async function (coroutine) to execute.
        input_args: Optional. Defines how the main workflow input maps to this branch's input.
                    - None: Pass the main workflow input directly to the branch.
                    - Dict[str, str]: Maps keys from the main input (if it's a dict) to
                                      the argument names of the branch's target (if it's a function)
                                      or to keys in the input dict for a sub-workflow.
                                      Example: {'main_input_key': 'branch_arg_name'}
                    - Callable[[Any], Any]: A function that takes the main workflow input
                                            and returns the input specifically for this branch.
    """
    name: str
    target: BranchTarget
    input_args: Optional[Union[Dict[str, str], Callable[[Any], Any]]] = None

class ParallelWorkflow(BaseWorkflow):
    """
    Implements the parallel execution workflow pattern using asyncio.gather.

    Runs multiple defined "branches" concurrently and collects their results.
    Each branch can be another workflow or an async function.
    """

    def __init__(
        self,
        name: str,
        branches: List[BranchDefinition],
        description: Optional[str] = "Executes multiple tasks or workflows in parallel.",
        aggregate_results: bool = True, # If True, results are dict[branch_name, result], else list[result]
        require_all_success: bool = True, # If True, workflow fails if any branch fails
        verbose: bool = False,
        **kwargs # Allow passing additional BaseWorkflow config like max_steps (usually 1 for parallel)
    ):
        """
        Initialize the parallel workflow.

        Args:
            name: Name of the workflow.
            branches: A list of BranchDefinition objects defining the parallel tasks.
            description: Optional description.
            aggregate_results: If True, results are returned in a dict keyed by branch name.
                               If False, returns a list of results in the order branches were defined.
            require_all_success: If True, the entire workflow is marked as failed if any
                                 single branch fails. If False, it succeeds as long as
                                 the execution completes, reporting individual branch errors.
            verbose: Enable detailed logging.
            **kwargs: Additional arguments for BaseWorkflow (max_steps defaults to 1).
        """
        # Parallel execution is conceptually one step of concurrency
        super().__init__(name=name, description=description, max_steps=kwargs.pop('max_steps', 1), verbose=verbose, **kwargs)

        if not branches:
            raise ValueError("At least one branch must be defined for ParallelWorkflow.")
        if not isinstance(branches, list) or not all(isinstance(b, BranchDefinition) for b in branches):
             raise TypeError("`branches` must be a list of BranchDefinition objects.")
        # Ensure branch names are unique
        branch_names = [b.name for b in branches]
        if len(set(branch_names)) != len(branches):
            # Find duplicate names for a more helpful error message
            from collections import Counter
            duplicates = [name for name, count in Counter(branch_names).items() if count > 1]
            raise ValueError(f"Branch names must be unique within a ParallelWorkflow. Duplicates found: {duplicates}")

        self.branches = branches
        self.aggregate_results = aggregate_results
        self.require_all_success = require_all_success

    def _prepare_branch_input(self, branch_def: BranchDefinition, main_input: Any) -> Any:
         """Prepares the specific input for a branch based on its definition."""
         mapping = branch_def.input_args
         branch_name = branch_def.name

         if mapping is None:
              if self.verbose: logger.debug(f"Branch '{branch_name}': Passing main input directly.")
              return main_input # Pass main input directly if no mapping specified

         elif isinstance(mapping, dict):
              # Map keys from main_input (must be dict) to branch input dict
              if not isinstance(main_input, dict):
                   raise ValueError(f"Branch '{branch_name}' expects input mapping from a dict, but main input type is {type(main_input).__name__}.")

              branch_input = {}
              missing_keys = []
              for main_key, branch_key in mapping.items():
                   if main_key in main_input:
                        branch_input[branch_key] = main_input[main_key]
                   else:
                        missing_keys.append(main_key)

              if missing_keys:
                    raise ValueError(f"Input mapping failed for branch '{branch_name}': Missing required keys in main input: {missing_keys}")

              if self.verbose: logger.debug(f"Branch '{branch_name}': Prepared input via mapping: {list(branch_input.keys())}")
              return branch_input

         elif callable(mapping):
              # Use the provided function to transform main input
              try:
                   branch_input = mapping(main_input)
                   if self.verbose: logger.debug(f"Branch '{branch_name}': Prepared input via function: type {type(branch_input).__name__}")
                   return branch_input
              except Exception as e:
                   raise ValueError(f"Input mapping function failed for branch '{branch_name}': {e}") from e
         else:
              raise TypeError(f"Invalid input_args type for branch '{branch_name}'. Must be None, dict, or callable.")


    async def _execute_branch(self, branch_def: BranchDefinition, branch_input: Any) -> Dict[str, Any]:
        """Executes a single branch target (workflow or function) safely."""
        target = branch_def.target
        branch_name = branch_def.name
        start_time = time.monotonic()
        logger.info(f"Starting parallel branch: '{branch_name}'")

        # Initialize result payload structure
        result_payload = {
            "branch": branch_name,
            "success": False, # Default to False
            "result": None,
            "error": None,
            "duration": 0.0
        }

        try:
            if isinstance(target, BaseWorkflow):
                # Execute sub-workflow
                workflow_result_dict = await target.execute(branch_input)
                result_payload["success"] = target.success # Use workflow's success status
                # Extract relevant result part if needed, or return full dict
                result_payload["result"] = workflow_result_dict.get("final_result", workflow_result_dict)
                result_payload["error"] = target.error # Capture error from sub-workflow state
                log_suffix = f"(Workflow: {target.name})"
            elif asyncio.iscoroutinefunction(target):
                # Execute async function
                result = await target(branch_input)
                result_payload["success"] = True # Assume success if function doesn't raise error
                result_payload["result"] = result
                log_suffix = "(Function)"
            else:
                # Target type not supported
                raise TypeError(f"Unsupported target type for branch '{branch_name}': {type(target)}. Must be BaseWorkflow or async function.")

            duration = time.monotonic() - start_time
            result_payload["duration"] = duration
            logger.info(f"Branch '{branch_name}' {log_suffix} completed in {duration:.3f}s. Success: {result_payload['success']}")
            return result_payload # Return the structured payload

        except Exception as e:
            # Catch errors during the execution of the branch target itself
            duration = time.monotonic() - start_time
            result_payload["duration"] = duration
            result_payload["error"] = str(e)
            result_payload["success"] = False # Ensure success is False on error
            logger.error(f"Branch '{branch_name}' failed execution after {duration:.3f}s: {e}", exc_info=self.verbose)
            return result_payload # Return the payload containing the error


    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute all defined branches concurrently using asyncio.gather.

        Args:
            input_data: Input data for the workflow. Passed to branches based
                        on BranchDefinition input_args.
            **kwargs: Additional runtime parameters (currently not used directly).

        Returns:
            Dictionary containing aggregated results from all branches
            and execution metadata. Keys include: 'input', 'results' (dict or list),
            'errors' (dict), 'execution_summary'.
        """
        # Use await if BaseWorkflow.reset becomes async
        self.reset()
        start_time = time.monotonic()

        # Prepare coroutines for each branch
        coroutines = []
        branch_definitions_executed = [] # Keep track of branches we actually try to run
        prep_errors: Dict[str, str] = {}

        for branch_def in self.branches:
            try:
                branch_input = self._prepare_branch_input(branch_def, input_data)
                # Wrap _execute_branch call in another coroutine to ensure
                # that exceptions within _execute_branch are caught by gather
                coroutines.append(self._execute_branch(branch_def, branch_input))
                branch_definitions_executed.append(branch_def)
            except (ValueError, TypeError) as prep_e:
                 # Handle errors during input prep immediately
                 logger.error(f"Skipping branch '{branch_def.name}' due to input preparation error: {prep_e}")
                 prep_errors[branch_def.name] = str(prep_e)

        if not coroutines: # No branches were prepared successfully
             error_msg = f"No branches executed. Preparation errors: {prep_errors}" if prep_errors else "No branches defined."
             self._mark_finished(success=False, error=error_msg)
             return {
                  "input": input_data,
                  "results": {} if self.aggregate_results else [],
                  "errors": prep_errors,
                  "execution_summary": self.get_execution_summary()
             }

        # Execute branches concurrently
        logger.info(f"Executing {len(coroutines)} branches in parallel for workflow '{self.name}'...")
        # Increment step counter once for the parallel execution phase
        self._increment_step()

        # Gather results, capturing exceptions from the coroutines themselves (less likely now)
        # and results/errors packaged within the dictionaries returned by _execute_branch
        branch_outputs_or_exceptions = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results and exceptions
        final_results_agg: Dict[str, Any] = {}
        final_results_list: List[Any] = [None] * len(branch_definitions_executed)
        errors_agg: Dict[str, str] = prep_errors.copy() # Include prep errors
        overall_success = not bool(prep_errors) # Start assuming success if prep passed

        for i, res_or_exc in enumerate(branch_outputs_or_exceptions):
            # Map result/exception back to the branch definition based on order
            # This assumes asyncio.gather preserves the order of input coroutines.
            if i >= len(branch_definitions_executed):
                logger.error(f"Mismatch between gathered results ({len(branch_outputs_or_exceptions)}) and executed branches ({len(branch_definitions_executed)}). Skipping extra result.")
                continue
            branch_def = branch_definitions_executed[i]
            branch_name = branch_def.name

            branch_success = False # Default for this branch
            branch_error_str: Optional[str] = None
            branch_result: Any = None

            if isinstance(res_or_exc, Exception):
                # Exception occurred during gather/await itself (e.g., cancellation, or unhandled error in _execute_branch)
                logger.error(f"Unhandled exception during gather for branch '{branch_name}': {res_or_exc}", exc_info=self.verbose)
                branch_error_str = f"Unhandled Exception: {type(res_or_exc).__name__}({res_or_exc})"
            elif isinstance(res_or_exc, dict) and "branch" in res_or_exc:
                # Got the structured dictionary from _execute_branch
                branch_result_payload = res_or_exc
                branch_success = branch_result_payload.get("success", False)
                branch_result = branch_result_payload.get("result")
                branch_error_str = branch_result_payload.get("error") # Will be None if success is True
            else:
                 # Unexpected return type from _execute_branch coroutine
                 logger.error(f"Unexpected result type received from branch '{branch_name}' execution: {type(res_or_exc)}")
                 branch_error_str = f"Unexpected internal result type: {type(res_or_exc).__name__}"

            # Aggregate results and errors
            final_results_agg[branch_name] = branch_result
            final_results_list[i] = branch_result if branch_success else {"error": branch_error_str}

            if branch_error_str:
                 errors_agg[branch_name] = branch_error_str
            # Update overall success based on require_all_success flag
            if not branch_success:
                 overall_success = False # If any branch fails, overall_success becomes False

        # Determine final workflow success status
        final_workflow_success = overall_success if self.require_all_success else (not bool(prep_errors))


        # Log overall execution step
        log_output_summary = {"results_count": len(final_results_agg), "error_count": len(errors_agg)}
        step_error_for_log = Exception(f"Branches failed: {list(errors_agg.keys())}") if errors_agg else None
        # BaseWorkflow._log_step is sync
        self._log_step(f"{self.name}_parallel_gather", input_data, log_output_summary, error=step_error_for_log)


        # Mark workflow as finished
        end_time = time.monotonic()
        final_error_message = "; ".join(f"{k}: {v}" for k, v in errors_agg.items()) if errors_agg else None
        # BaseWorkflow._mark_finished is sync
        self._mark_finished(success=final_workflow_success, error=final_error_message)

        logger.info(f"Parallel workflow '{self.name}' finished in {end_time - start_time:.3f}s. Overall Success: {final_workflow_success}")

        # Prepare final output dictionary
        return {
            "input": input_data,
            "results": final_results_agg if self.aggregate_results else final_results_list,
            "errors": errors_agg, # Dictionary of errors keyed by branch name
            "execution_summary": self.get_execution_summary() # Sync get
        }