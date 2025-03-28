"""
Orchestrator Workflow Pattern

This module implements the orchestrator workflow pattern, which coordinates
multiple specialized workers to accomplish complex tasks.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable

from ..llm.base import BaseLLM
from ..tools.registry import ToolRegistry
from .base import BaseWorkflow

logger = logging.getLogger(__name__)


class OrchestratorWorkflow(BaseWorkflow):
    """
    Implementation of the orchestrator workflow pattern.
    
    This workflow manages multiple worker agents/workflows, delegating subtasks
    to the most appropriate worker and synthesizing their results.
    
    Features:
    - Central coordination of specialized workers
    - Intelligent task decomposition and delegation
    - Result aggregation and synthesis
    - Parallel execution options
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        workers: Dict[str, Union[BaseWorkflow, Callable]],
        tools: Optional[ToolRegistry] = None,
        max_steps: int = 15,
        parallel: bool = True,
        max_parallel_workers: int = 3,
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the orchestrator workflow.
        
        Args:
            name: Name of the workflow
            llm: LLM instance for the orchestrator
            workers: Dictionary of worker workflows/agents to coordinate
            tools: Optional tool registry for the orchestrator's own tools
            max_steps: Maximum number of orchestration steps
            parallel: Whether to allow parallel execution of workers
            max_parallel_workers: Maximum workers to run in parallel
            system_prompt: Optional system prompt for the orchestrator
            verbose: Whether to log detailed information
        """
        super().__init__(
            name=name,
            max_steps=max_steps,
            verbose=verbose
        )
        
        self.llm = llm
        self.workers = workers
        self.tools = tools or ToolRegistry()
        self.parallel = parallel
        self.max_parallel_workers = max_parallel_workers
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Execution state
        self.task_results = {}
        self.pending_tasks = {}
        self.completed_workers = set()
        
        # Validate workers
        self._validate_workers()
    
    def _default_system_prompt(self) -> str:
        """Provide a default system prompt for the orchestrator."""
        return (
            "You are an orchestrator responsible for coordinating multiple specialized agents. "
            "Your job is to break down complex tasks into subtasks, delegate them to appropriate agents, "
            "and synthesize their results into a coherent response. "
            "Think carefully about which agent is best suited for each subtask."
        )
    
    def _validate_workers(self) -> None:
        """Validate that workers are properly configured."""
        for name, worker in self.workers.items():
            if not isinstance(worker, (BaseWorkflow, Callable)):
                raise ValueError(f"Worker '{name}' must be a BaseWorkflow instance or callable")
    
    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the orchestrator workflow.
        
        The orchestrator:
        1. Analyzes the task
        2. Breaks it into subtasks
        3. Assigns subtasks to appropriate workers
        4. Monitors execution
        5. Synthesizes results
        
        Args:
            input_data: Input data for the workflow
            **kwargs: Additional execution parameters
            
        Returns:
            Dict containing the execution results
        """
        self.reset()
        
        # Extract the task
        if isinstance(input_data, str):
            task = input_data
            context = {}
        elif isinstance(input_data, dict):
            task = input_data.get("input", str(input_data))
            context = input_data
        else:
            task = str(input_data)
            context = {}
        
        # Initialize result structure
        result = {
            "input": task,
            "subtasks": [],
            "worker_results": {},
            "final_result": None
        }
        
        try:
            # Step 1: Task Analysis and Decomposition
            subtasks = await self._analyze_and_decompose_task(task, context)
            result["subtasks"] = subtasks
            
            if self.verbose:
                logger.info(f"Decomposed task into {len(subtasks)} subtasks")
            
            # Step 2: Worker Assignment and Execution
            if self.parallel:
                worker_results = await self._execute_workers_parallel(subtasks, context)
            else:
                worker_results = await self._execute_workers_sequential(subtasks, context)
                
            result["worker_results"] = worker_results
            
            # Step 3: Result Synthesis
            final_result = await self._synthesize_results(task, subtasks, worker_results, context)
            result["final_result"] = final_result
            
            self._mark_finished(success=True)
            return result
            
        except Exception as e:
            logger.exception(f"Error in orchestrator workflow: {str(e)}")
            self._mark_finished(success=False, error=str(e))
            
            result["error"] = str(e)
            return result
    
    async def _analyze_and_decompose_task(
        self, 
        task: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze the task and break it down into subtasks.
        
        Args:
            task: The main task to decompose
            context: Additional context
            
        Returns:
            List of subtask specifications
        """
        # Construct analysis prompt
        worker_descriptions = "\n".join([
            f"- {name}: {self._get_worker_description(worker)}"
            for name, worker in self.workers.items()
        ])
        
        analysis_prompt = (
            f"Task: {task}\n\n"
            f"Available workers:\n{worker_descriptions}\n\n"
            f"Break down this task into subtasks that can be delegated to the available workers. "
            f"For each subtask, specify:\n"
            f"1. A clear, specific subtask description\n"
            f"2. Which worker should handle it\n"
            f"3. The priority (1-5, where 1 is highest)\n"
            f"4. Whether this subtask depends on the results of other subtasks\n\n"
            f"Respond in the following JSON format:\n"
            f"```json\n"
            f'{{"subtasks": [\n'
            f'  {{"id": "subtask1", "description": "...", "worker": "...", "priority": 1, "dependencies": []}},'
            f'  {{"id": "subtask2", "description": "...", "worker": "...", "priority": 2, "dependencies": ["subtask1"]}}\n'
            f']}}\n'
            f"```\n\n"
            f"Consider which tasks can be done in parallel and which need to be sequential. "
            f"Ensure that each subtask is assigned to a suitable worker based on their capabilities."
        )
        
        # Get LLM response
        response = await self.llm.generate(
            prompt=analysis_prompt,
            system_prompt=self.system_prompt,
            temperature=0.2  # Low temperature for more deterministic planning
        )
        
        # Extract JSON from response
        import json
        import re
        
        response_text = response.get("content", "")
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data.get("subtasks", [])
            except json.JSONDecodeError:
                logger.error("Failed to parse subtasks JSON")
                
                # Fallback: try to extract without code blocks
                try:
                    # Look for JSON object pattern
                    alt_match = re.search(r"\{[\s\S]*\"subtasks\"[\s\S]*\}", response_text)
                    if alt_match:
                        data = json.loads(alt_match.group(0))
                        return data.get("subtasks", [])
                except Exception:
                    pass
        
        # If parsing fails, create a simple fallback task for each worker
        logger.warning("Using fallback task decomposition")
        subtasks = []
        for i, (name, _) in enumerate(self.workers.items()):
            subtasks.append({
                "id": f"subtask_{i+1}",
                "description": task,
                "worker": name,
                "priority": i+1,
                "dependencies": []
            })
        
        return subtasks
    
    async def _execute_workers_sequential(
        self, 
        subtasks: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute workers sequentially based on priority and dependencies.
        
        Args:
            subtasks: List of subtask specifications
            context: Additional context
            
        Returns:
            Dict of worker results
        """
        results = {}
        completed_tasks = set()
        
        # Sort subtasks by priority and dependencies
        sorted_subtasks = sorted(subtasks, key=lambda x: x.get("priority", 99))
        
        # Process subtasks in order
        for subtask in sorted_subtasks:
            subtask_id = subtask["id"]
            worker_name = subtask["worker"]
            description = subtask["description"]
            dependencies = subtask.get("dependencies", [])
            
            # Skip if worker doesn't exist
            if worker_name not in self.workers:
                logger.warning(f"Worker '{worker_name}' not found, skipping subtask '{subtask_id}'")
                continue
            
            # Check dependencies
            dependency_results = {}
            all_deps_complete = True
            
            for dep_id in dependencies:
                if dep_id not in completed_tasks:
                    all_deps_complete = False
                    logger.warning(f"Dependency '{dep_id}' not complete for subtask '{subtask_id}'")
                    break
                dependency_results[dep_id] = results.get(dep_id, {})
            
            if not all_deps_complete:
                logger.warning(f"Skipping subtask '{subtask_id}' due to incomplete dependencies")
                continue
            
            # Prepare input for worker
            worker_input = {
                "input": description,
                "context": context,
                "dependencies": dependency_results
            }
            
            # Execute worker
            worker = self.workers[worker_name]
            try:
                if self.verbose:
                    logger.info(f"Executing worker '{worker_name}' for subtask '{subtask_id}'")
                
                # Increment step counter
                if not self._increment_step():
                    break
                
                # Execute worker (different handling for workflow vs callable)
                if isinstance(worker, BaseWorkflow):
                    worker_result = await worker.execute(worker_input)
                else:
                    # Assume it's a callable
                    worker_result = await worker(worker_input)
                
                # Store result
                results[subtask_id] = worker_result
                completed_tasks.add(subtask_id)
                self.completed_workers.add(worker_name)
                
                if self.verbose:
                    logger.info(f"Completed worker '{worker_name}' for subtask '{subtask_id}'")
                
            except Exception as e:
                logger.error(f"Error executing worker '{worker_name}': {str(e)}")
                results[subtask_id] = {"error": str(e)}
        
        return results
    
    async def _execute_workers_parallel(
        self, 
        subtasks: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute workers in parallel when possible.
        
        Args:
            subtasks: List of subtask specifications
            context: Additional context
            
        Returns:
            Dict of worker results
        """
        results = {}
        completed_tasks = set()
        dependency_map = {task["id"]: set(task.get("dependencies", [])) for task in subtasks}
        
        # Map subtasks to their workers
        worker_map = {task["id"]: task["worker"] for task in subtasks}
        description_map = {task["id"]: task["description"] for task in subtasks}
        
        # Process subtasks in waves based on dependencies
        while len(completed_tasks) < len(subtasks) and self._increment_step():
            # Find eligible tasks (dependencies satisfied)
            eligible_tasks = []
            
            for task in subtasks:
                task_id = task["id"]
                
                # Skip already completed tasks
                if task_id in completed_tasks:
                    continue
                
                # Check if all dependencies are satisfied
                deps = dependency_map[task_id]
                if all(dep in completed_tasks for dep in deps):
                    eligible_tasks.append(task_id)
            
            if not eligible_tasks:
                # Circular dependency or all tasks completed
                break
            
            # Limit parallel execution
            current_batch = eligible_tasks[:self.max_parallel_workers]
            
            # Prepare tasks
            tasks = []
            for task_id in current_batch:
                worker_name = worker_map[task_id]
                if worker_name not in self.workers:
                    logger.warning(f"Worker '{worker_name}' not found, skipping subtask '{task_id}'")
                    continue
                
                worker = self.workers[worker_name]
                
                # Gather dependency results
                dependency_results = {}
                for dep_id in dependency_map[task_id]:
                    dependency_results[dep_id] = results.get(dep_id, {})
                
                # Prepare input
                worker_input = {
                    "input": description_map[task_id],
                    "context": context,
                    "dependencies": dependency_results,
                    "subtask_id": task_id
                }
                
                # Create task
                if isinstance(worker, BaseWorkflow):
                    task = worker.execute(worker_input)
                else:
                    # Assume it's a callable
                    task = worker(worker_input)
                
                tasks.append((task_id, worker_name, task))
            
            # Execute batch in parallel
            if self.verbose:
                logger.info(f"Executing batch of {len(tasks)} tasks in parallel")
            
            # Create tasks for asyncio.gather
            async_tasks = [task for _, _, task in tasks]
            
            # Wait for all tasks to complete
            task_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Process results
            for i, (task_id, worker_name, _) in enumerate(tasks):
                result = task_results[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Error executing worker '{worker_name}': {str(result)}")
                    results[task_id] = {"error": str(result)}
                else:
                    results[task_id] = result
                    self.completed_workers.add(worker_name)
                
                completed_tasks.add(task_id)
                
                if self.verbose:
                    status = "completed" if not isinstance(result, Exception) else "failed"
                    logger.info(f"Task '{task_id}' {status} by worker '{worker_name}'")
        
        return results
    
    async def _synthesize_results(
        self,
        task: str,
        subtasks: List[Dict[str, Any]],
        worker_results: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Synthesize worker results into a coherent response.
        
        Args:
            task: Original task
            subtasks: List of subtask specifications
            worker_results: Results from workers
            context: Additional context
            
        Returns:
            Synthesized final result
        """
        # Construct synthesis prompt
        subtask_results = []
        
        for subtask in subtasks:
            subtask_id = subtask["id"]
            worker_name = subtask["worker"]
            description = subtask["description"]
            
            # Get result if available
            result = worker_results.get(subtask_id, {})
            result_content = result.get("final_result", result)
            
            if isinstance(result_content, dict) and "error" in result_content:
                result_str = f"ERROR: {result_content['error']}"
            elif isinstance(result_content, (dict, list)):
                import json
                result_str = json.dumps(result_content, indent=2)
            else:
                result_str = str(result_content)
            
            subtask_results.append(
                f"Subtask: {description}\n"
                f"Worker: {worker_name}\n"
                f"Result:\n{result_str}\n"
            )
        
        synthesis_prompt = (
            f"Task: {task}\n\n"
            f"You have coordinated multiple workers to address this task. "
            f"Below are the results from each worker:\n\n"
            f"{'-' * 40}\n"
            f"{'\n' + '-' * 40 + '\n'.join(subtask_results)}\n"
            f"{'-' * 40}\n\n"
            f"Synthesize these results into a coherent, comprehensive response to the original task. "
            f"Be sure to integrate the information from all workers, addressing any contradictions "
            f"and filling in gaps. The final response should be concise but complete."
        )
        
        # Get LLM response
        response = await self.llm.generate(
            prompt=synthesis_prompt,
            system_prompt=self.system_prompt,
            temperature=0.5  # Moderate temperature for creative synthesis
        )
        
        return response.get("content", "Failed to synthesize results")
    
    def _get_worker_description(self, worker: Union[BaseWorkflow, Callable]) -> str:
        """Get a description of a worker for task delegation."""
        if isinstance(worker, BaseWorkflow):
            return getattr(worker, 'description', f"Workflow: {worker.name}")
        else:
            # For callables, use docstring or a default description
            return worker.__doc__ or "Function-based worker"