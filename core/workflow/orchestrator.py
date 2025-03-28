"""
Workflow Orchestrator Module

This module provides the Orchestrator class that manages the execution of workflows,
handling task scheduling, worker allocation, and error recovery mechanisms.

The Orchestrator supports distributed execution across multiple worker nodes and
implements fault tolerance strategies to ensure reliable workflow completion.
"""

import os
import time
import json
import logging
import threading
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Core imports
from core.workflow.task import Task, TaskStatus
from core.workflow.workflow import Workflow
from core.communication.agent_protocol import AgentMessage, AgentProtocol
from core.exceptions import OrchestratorError, WorkerError, SchedulingError
from core.utils.telemetry import TelemetryTracker

# Worker pool management
from core.workflow.worker_pool import WorkerPool, Worker, WorkerStatus

# Configuration management
from config.settings import Settings

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Workflow Orchestrator manages the execution of tasks across a pool of workers.
    
    The Orchestrator is responsible for:
    1. Distributing tasks to appropriate workers
    2. Tracking workflow execution progress
    3. Handling failures and implementing recovery strategies
    4. Optimizing resource utilization
    5. Collecting execution metrics
    
    Attributes:
        name (str): Name identifier for the orchestrator
        worker_pool (WorkerPool): Pool of available worker nodes
        protocol (AgentProtocol): Communication protocol
        settings (Settings): Configuration settings
        active_workflows (Dict[str, Workflow]): Currently running workflows
        telemetry (TelemetryTracker): Performance metrics tracker
    """
    
    def __init__(
        self, 
        name: str,
        settings_path: Optional[str] = None,
        protocol: Optional[AgentProtocol] = None,
    ):
        """
        Initialize a new Orchestrator.
        
        Args:
            name: Unique identifier for this orchestrator
            settings_path: Path to settings file (if None, uses default)
            protocol: Communication protocol for worker messaging
        """
        self.name = name
        self.settings = Settings(settings_path)
        self.protocol = protocol or AgentProtocol()
        self.worker_pool = WorkerPool()
        self.active_workflows: Dict[str, Workflow] = {}
        self.completed_workflows: Dict[str, Workflow] = {}
        self.failed_workflows: Dict[str, Workflow] = {}
        self.telemetry = TelemetryTracker()
        
        # Thread-safety for workflow operations
        self._workflows_lock = threading.RLock()
        
        # Scheduling configuration
        self._load_orchestration_config()
        
        # Start the background monitoring thread
        self._stop_monitoring = threading.Event()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_workflows,
            daemon=True,
            name=f"{self.name}-monitor"
        )
        self._monitoring_thread.start()
        
        logger.info(f"Orchestrator '{name}' initialized")
    
    def _load_orchestration_config(self) -> None:
        """
        Load orchestration configuration from settings.
        
        Initializes scheduling parameters, worker configurations, and fault tolerance settings.
        """
        try:
            # Load orchestration settings
            orch_config = self.settings.get("orchestration", {})
            
            # Configure scheduling parameters
            self.max_concurrent_workflows = orch_config.get("max_concurrent_workflows", 10)
            self.max_retries = orch_config.get("max_retries", 3)
            self.retry_delay = orch_config.get("retry_delay", 5)
            self.monitoring_interval = orch_config.get("monitoring_interval", 10)
            
            # Task prioritization strategy
            priority_strategy = orch_config.get("priority_strategy", "fifo")
            if priority_strategy == "fifo":
                self._prioritize_tasks = self._prioritize_fifo
            elif priority_strategy == "deadline":
                self._prioritize_tasks = self._prioritize_deadline
            else:
                logger.warning(f"Unknown priority strategy '{priority_strategy}', falling back to FIFO")
                self._prioritize_tasks = self._prioritize_fifo
            
            # Worker selection strategy
            worker_strategy = orch_config.get("worker_selection_strategy", "round_robin")
            if worker_strategy == "round_robin":
                self._select_worker = self._select_worker_round_robin
            elif worker_strategy == "least_loaded":
                self._select_worker = self._select_worker_least_loaded
            elif worker_strategy == "capability_match":
                self._select_worker = self._select_worker_capability_match
            else:
                logger.warning(f"Unknown worker selection strategy '{worker_strategy}', falling back to round-robin")
                self._select_worker = self._select_worker_round_robin
            
            # Initialize any additional components from config
            self._initialize_workers_from_config(orch_config.get("workers", []))
            
            logger.info(f"Orchestration configuration loaded: {len(self.worker_pool.workers)} workers configured")
            
        except Exception as e:
            logger.error(f"Error loading orchestration configuration: {str(e)}")
            logger.debug(traceback.format_exc())
            # Use default configurations
            self.max_concurrent_workflows = 10
            self.max_retries = 3
            self.retry_delay = 5
            self.monitoring_interval = 10
            self._prioritize_tasks = self._prioritize_fifo
            self._select_worker = self._select_worker_round_robin
    
    def _initialize_workers_from_config(self, worker_configs: List[Dict[str, Any]]) -> None:
        """
        Initialize workers from configuration.
        
        Args:
            worker_configs: List of worker configuration dictionaries
        """
        for config in worker_configs:
            worker_id = config.get("id")
            if not worker_id:
                logger.warning("Skipping worker with missing ID in configuration")
                continue
                
            try:
                worker = Worker(
                    id=worker_id,
                    endpoint=config.get("endpoint"),
                    capabilities=config.get("capabilities", []),
                    max_concurrent_tasks=config.get("max_concurrent_tasks", 5)
                )
                self.worker_pool.add_worker(worker)
                logger.info(f"Initialized worker '{worker_id}' from configuration")
                
            except Exception as e:
                logger.error(f"Failed to initialize worker '{worker_id}': {str(e)}")
    
    def register_worker(self, worker: Worker) -> None:
        """
        Register a new worker with the orchestrator.
        
        Args:
            worker: Worker instance to register
            
        Raises:
            ValueError: If a worker with the same ID is already registered
        """
        self.worker_pool.add_worker(worker)
        logger.info(f"Worker '{worker.id}' registered with capabilities: {worker.capabilities}")
    
    def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker from the orchestrator.
        
        Args:
            worker_id: ID of the worker to unregister
            
        Raises:
            ValueError: If the worker is not registered
        """
        self.worker_pool.remove_worker(worker_id)
        logger.info(f"Worker '{worker_id}' unregistered")
    
    def submit_workflow(self, workflow: Workflow) -> str:
        """
        Submit a workflow for execution.
        
        Args:
            workflow: The workflow to execute
            
        Returns:
            ID of the submitted workflow
            
        Raises:
            OrchestratorError: If the workflow cannot be submitted
        """
        with self._workflows_lock:
            # Check if we can accept more workflows
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                raise OrchestratorError(
                    f"Maximum concurrent workflows limit reached ({self.max_concurrent_workflows})"
                )
            
            # Check if workflow already exists
            if workflow.id in self.active_workflows:
                raise OrchestratorError(f"Workflow with ID '{workflow.id}' is already active")
            
            # Add workflow to active workflows
            self.active_workflows[workflow.id] = workflow
            
            # Start telemetry tracking
            self.telemetry.start_workflow(workflow.id)
            
            logger.info(f"Workflow '{workflow.id}' submitted with {len(workflow.tasks)} tasks")
            
            # Schedule initial tasks
            self._schedule_ready_tasks(workflow)
            
            return workflow.id
    
    def _schedule_ready_tasks(self, workflow: Workflow) -> None:
        """
        Schedule all ready tasks in a workflow.
        
        Args:
            workflow: The workflow containing tasks to schedule
        """
        # Find all tasks with PENDING status and no dependencies or all dependencies satisfied
        ready_tasks = []
        for task in workflow.tasks:
            if task.status == TaskStatus.PENDING and self._are_dependencies_satisfied(workflow, task):
                ready_tasks.append(task)
        
        if not ready_tasks:
            logger.debug(f"No ready tasks to schedule in workflow '{workflow.id}'")
            return
        
        # Prioritize tasks
        prioritized_tasks = self._prioritize_tasks(ready_tasks)
        
        # Schedule each task
        for task in prioritized_tasks:
            self._schedule_task(workflow, task)
    
    def _are_dependencies_satisfied(self, workflow: Workflow, task: Task) -> bool:
        """
        Check if all dependencies of a task are satisfied.
        
        Args:
            workflow: The workflow containing the task
            task: The task to check dependencies for
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        dependencies = task.metadata.get("dependencies", [])
        if not dependencies:
            return True
            
        # Check each dependency
        for dep_id in dependencies:
            # Find the dependency task
            dep_task = next((t for t in workflow.tasks if t.id == dep_id), None)
            if not dep_task:
                logger.warning(f"Dependency '{dep_id}' not found for task '{task.id}'")
                return False
                
            # Check if dependency is completed
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _prioritize_fifo(self, tasks: List[Task]) -> List[Task]:
        """
        Prioritize tasks using First-In-First-Out strategy.
        
        Args:
            tasks: List of tasks to prioritize
            
        Returns:
            Prioritized list of tasks
        """
        # In FIFO, we don't change the order
        return tasks
    
    def _prioritize_deadline(self, tasks: List[Task]) -> List[Task]:
        """
        Prioritize tasks based on deadline (earliest first).
        
        Args:
            tasks: List of tasks to prioritize
            
        Returns:
            Prioritized list of tasks
        """
        return sorted(
            tasks,
            key=lambda task: task.metadata.get("deadline", float("inf"))
        )
    
    def _schedule_task(self, workflow: Workflow, task: Task) -> None:
        """
        Schedule a task for execution on a worker.
        
        Args:
            workflow: The workflow containing the task
            task: The task to schedule
        """
        # Select the best worker for this task
        worker = self._select_worker(task)
        if not worker:
            logger.warning(f"No suitable worker found for task '{task.id}' in workflow '{workflow.id}'")
            task.status = TaskStatus.FAILED
            task.error = "No suitable worker available"
            self._check_workflow_completion(workflow)
            return
        
        # Update task status
        task.status = TaskStatus.SCHEDULED
        task.assigned_worker = worker.id
        
        # Start telemetry for this task
        self.telemetry.start_task(workflow.id, task.id)
        
        # Create task execution message
        task_message = AgentMessage(
            sender=self.name,
            recipient=worker.id,
            content=task.to_dict(),
            message_type="task_execute"
        )
        
        # Send task to worker
        try:
            logger.info(f"Scheduling task '{task.id}' on worker '{worker.id}'")
            self.protocol.send(task_message)
            
            # Update worker status
            worker.active_tasks += 1
            
        except Exception as e:
            logger.error(f"Failed to schedule task '{task.id}' on worker '{worker.id}': {str(e)}")
            task.status = TaskStatus.FAILED
            task.error = f"Scheduling failed: {str(e)}"
            self._check_workflow_completion(workflow)
    
    def _select_worker_round_robin(self, task: Task) -> Optional[Worker]:
        """
        Select a worker using round-robin scheduling.
        
        Args:
            task: The task to find a worker for
            
        Returns:
            Selected worker or None if no suitable worker is found
        """
        return self.worker_pool.get_next_available_worker()
    
    def _select_worker_least_loaded(self, task: Task) -> Optional[Worker]:
        """
        Select the least loaded worker for a task.
        
        Args:
            task: The task to find a worker for
            
        Returns:
            Selected worker or None if no suitable worker is found
        """
        return self.worker_pool.get_least_loaded_worker()
    
    def _select_worker_capability_match(self, task: Task) -> Optional[Worker]:
        """
        Select a worker based on capability matching.
        
        This selects workers that match all required capabilities for the task,
        prioritizing workers with the fewest extra capabilities (most specialized).
        
        Args:
            task: The task to find a worker for
            
        Returns:
            Selected worker or None if no suitable worker is found
        """
        required_capabilities = task.metadata.get("required_capabilities", [])
        
        # Find workers with all required capabilities
        suitable_workers = []
        for worker in self.worker_pool.get_available_workers():
            if worker.status != WorkerStatus.ONLINE:
                continue
                
            # Check if worker has all required capabilities
            if all(cap in worker.capabilities for cap in required_capabilities):
                # Calculate how specialized this worker is for the task
                # (fewer extra capabilities = more specialized)
                extra_capabilities = len(worker.capabilities) - len(required_capabilities)
                suitable_workers.append((worker, extra_capabilities))
        
        if not suitable_workers:
            return None
            
        # Sort by number of extra capabilities (ascending) and then by load (ascending)
        suitable_workers.sort(key=lambda x: (x[1], x[0].active_tasks))
        return suitable_workers[0][0]
    
    def handle_task_completion(self, worker_id: str, task_id: str, result: Any) -> None:
        """
        Handle the completion of a task.
        
        Args:
            worker_id: ID of the worker that completed the task
            task_id: ID of the completed task
            result: Result data from the task
        """
        # Find the workflow and task
        workflow = None
        task = None
        
        with self._workflows_lock:
            for wf in self.active_workflows.values():
                for t in wf.tasks:
                    if t.id == task_id:
                        workflow = wf
                        task = t
                        break
                if workflow:
                    break
        
        if not workflow or not task:
            logger.warning(f"Received completion for unknown task '{task_id}' from worker '{worker_id}'")
            return
            
        # Update task status
        task.status = TaskStatus.COMPLETED
        task.result = result
        
        # Update worker status
        worker = self.worker_pool.get_worker(worker_id)
        if worker:
            worker.active_tasks -= 1
        
        # Record completion in telemetry
        self.telemetry.end_task(workflow.id, task.id)
        
        logger.info(f"Task '{task_id}' in workflow '{workflow.id}' completed by worker '{worker_id}'")
        
        # Schedule dependent tasks
        self._schedule_ready_tasks(workflow)
        
        # Check if workflow is complete
        self._check_workflow_completion(workflow)
    
    def handle_task_failure(self, worker_id: str, task_id: str, error: str) -> None:
        """
        Handle a task failure.
        
        Args:
            worker_id: ID of the worker that reported the failure
            task_id: ID of the failed task
            error: Error message or description
        """
        # Find the workflow and task
        workflow = None
        task = None
        
        with self._workflows_lock:
            for wf in self.active_workflows.values():
                for t in wf.tasks:
                    if t.id == task_id:
                        workflow = wf
                        task = t
                        break
                if workflow:
                    break
        
        if not workflow or not task:
            logger.warning(f"Received failure for unknown task '{task_id}' from worker '{worker_id}'")
            return
            
        logger.warning(f"Task '{task_id}' in workflow '{workflow.id}' failed on worker '{worker_id}': {error}")
        
        # Update worker status
        worker = self.worker_pool.get_worker(worker_id)
        if worker:
            worker.active_tasks -= 1
        
        # Check if we should retry
        retry_count = task.metadata.get("retry_count", 0)
        
        if retry_count < self.max_retries:
            # Update retry count
            task.metadata["retry_count"] = retry_count + 1
            
            # Reset task status
            task.status = TaskStatus.PENDING
            task.assigned_worker = None
            
            logger.info(f"Retrying task '{task_id}' (attempt {retry_count + 1}/{self.max_retries})")
            
            # Wait before retrying
            time.sleep(self.retry_delay)
            
            # Re-schedule the task
            self._schedule_task(workflow, task)
            
        else:
            # Mark task as failed
            task.status = TaskStatus.FAILED
            task.error = error
            
            # Record failure in telemetry
            self.telemetry.record_task_failure(workflow.id, task.id, error)
            
            # Check if workflow needs to be marked as failed
            self._check_workflow_completion(workflow)
    
    def _check_workflow_completion(self, workflow: Workflow) -> None:
        """
        Check if a workflow is complete or failed and update its status.
        
        A workflow is complete if all tasks are either completed or failed.
        A workflow is failed if any task has failed and affects the critical path.
        
        Args:
            workflow: The workflow to check
        """
        with self._workflows_lock:
            # Check if any tasks are still in progress
            in_progress = False
            has_failed = False
            
            for task in workflow.tasks:
                if task.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED, TaskStatus.RUNNING]:
                    in_progress = True
                elif task.status == TaskStatus.FAILED:
                    has_failed = True
                    # Check if this failure affects the critical path
                    if task.metadata.get("critical", False):
                        # Mark workflow as failed
                        workflow.status = "failed"
                        
                        # Move to failed workflows
                        self.failed_workflows[workflow.id] = workflow
                        if workflow.id in self.active_workflows:
                            del self.active_workflows[workflow.id]
                            
                        logger.error(f"Workflow '{workflow.id}' failed due to critical task '{task.id}'")
                        
                        # End telemetry for this workflow
                        self.telemetry.end_workflow(workflow.id)
                        return
            
            # If no tasks are in progress, the workflow is complete
            if not in_progress:
                if has_failed:
                    # Some non-critical tasks failed
                    workflow.status = "completed_with_failures"
                else:
                    # All tasks completed successfully
                    workflow.status = "completed"
                
                # Move to completed workflows
                self.completed_workflows[workflow.id] = workflow
                if workflow.id in self.active_workflows:
                    del self.active_workflows[workflow.id]
                    
                # End telemetry for this workflow
                self.telemetry.end_workflow(workflow.id)
                
                logger.info(f"Workflow '{workflow.id}' {workflow.status}")
    
    def _monitor_workflows(self) -> None:
        """
        Background thread that monitors workflow progress and handles timeouts.
        """
        while not self._stop_monitoring.is_set():
            try:
                with self._workflows_lock:
                    current_time = time.time()
                    
                    # Check each active workflow
                    for workflow_id, workflow in list(self.active_workflows.items()):
                        # Check for workflow timeout
                        if "timeout" in workflow.metadata:
                            start_time = workflow.metadata.get("start_time", 0)
                            timeout = workflow.metadata.get("timeout")
                            
                            if current_time - start_time > timeout:
                                logger.warning(f"Workflow '{workflow_id}' timed out after {timeout}s")
                                workflow.status = "timeout"
                                
                                # Move to failed workflows
                                self.failed_workflows[workflow_id] = workflow
                                del self.active_workflows[workflow_id]
                                
                                # End telemetry
                                self.telemetry.end_workflow(workflow_id)
                                continue
                        
                        # Check for task heartbeats and timeouts
                        for task in workflow.tasks:
                            if task.status == TaskStatus.RUNNING:
                                # Check if task has a heartbeat timeout
                                last_heartbeat = task.metadata.get("last_heartbeat", 0)
                                heartbeat_timeout = task.metadata.get("heartbeat_timeout", 300)  # 5 minutes default
                                
                                if current_time - last_heartbeat > heartbeat_timeout:
                                    logger.warning(f"Task '{task.id}' in workflow '{workflow_id}' missed heartbeat")
                                    
                                    # Mark task as failed
                                    self.handle_task_failure(
                                        task.assigned_worker or "unknown",
                                        task.id,
                                        "Task missed heartbeat timeout"
                                    )
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in workflow monitoring thread: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(self.monitoring_interval)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow.
        
        Args:
            workflow_id: ID of the workflow to check
            
        Returns:
            Dict containing workflow status information
            
        Raises:
            ValueError: If the workflow doesn't exist
        """
        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
        # Check completed workflows
        elif workflow_id in self.completed_workflows:
            workflow = self.completed_workflows[workflow_id]
        # Check failed workflows
        elif workflow_id in self.failed_workflows:
            workflow = self.failed_workflows[workflow_id]
        else:
            raise ValueError(f"Workflow '{workflow_id}' not found")
            
        return {
            "workflow_id": workflow_id,
            "status": workflow.status,
            "tasks": [
                {
                    "id": task.id,
                    "status": task.status,
                    "assigned_worker": task.assigned_worker,
                    "has_error": task.error is not None
                }
                for task in workflow.tasks
            ],
            "metrics": self.telemetry.get_workflow_metrics(workflow_id)
        }
        
    def cancel_workflow(self, workflow_id: str) -> None:
        """
        Cancel an in-progress workflow.
        
        Args:
            workflow_id: ID of the workflow to cancel
            
        Raises:
            ValueError: If the workflow doesn't exist
        """
        with self._workflows_lock:
            if workflow_id not in self.active_workflows:
                raise ValueError(f"Workflow '{workflow_id}' not found or not active")
                
            workflow = self.active_workflows[workflow_id]
            
            # Cancel any running or scheduled tasks
            for task in workflow.tasks:
                if task.status in [TaskStatus.SCHEDULED, TaskStatus.RUNNING]:
                    # Try to send cancel message to the assigned worker
                    if task.assigned_worker:
                        cancel_message = AgentMessage(
                            sender=self.name,
                            recipient=task.assigned_worker,
                            content={"task_id": task.id},
                            message_type="task_cancel"
                        )
                        
                        try:
                            self.protocol.send(cancel_message)
                        except Exception as e:
                            logger.warning(f"Failed to send cancel message for task '{task.id}': {str(e)}")
                    
                    task.status = TaskStatus.CANCELLED
            
            workflow.status = "cancelled"
            
            # Move to failed workflows
            self.failed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]
            
            # Record cancellation in telemetry
            self.telemetry.record_workflow_cancellation(workflow_id)
            
            logger.info(f"Workflow '{workflow_id}' cancelled")
    
    def shutdown(self) -> None:
        """
        Gracefully shut down the orchestrator.
        
        This cancels all active workflows and stops background threads.
        """
        logger.info(f"Shutting down orchestrator '{self.name}'")
        
        # Stop the monitoring thread
        self._stop_monitoring.set()
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        # Cancel all active workflows
        with self._workflows_lock:
            for workflow_id in list(self.active_workflows.keys()):
                try:
                    self.cancel_workflow(workflow_id)
                except Exception as e:
                    logger.error(f"Error cancelling workflow '{workflow_id}' during shutdown: {str(e)}")
        
        logger.info(f"Orchestrator '{self.name}' shutdown complete")