# ai_agent_framework/core/workflow/orchestrator.py

"""
Asynchronous Workflow Orchestrator Module (Updated)

Manages asynchronous execution of workflows, handling task scheduling,
worker allocation via an async communication protocol, and error recovery,
using the implemented WorkerPool, TelemetryTracker, and Exceptions.
"""

import asyncio
import time
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Core framework components (Absolute Imports)
from ai_agent_framework.core.workflow.task import Task, TaskStatus
from ai_agent_framework.core.workflow.workflow import Workflow
from ai_agent_framework.core.communication.agent_protocol import AgentMessage, AgentProtocol
# Using newly created exceptions
from ai_agent_framework.core.exceptions import (
    OrchestratorError, SchedulingError, CommunicationError, ProtocolError, AgentFrameworkError
)
# Using newly created worker pool
from ai_agent_framework.core.workflow.worker_pool import WorkerPool, Worker, WorkerStatus
# Assuming Settings exists
from ai_agent_framework.config.settings import Settings
# Using newly created telemetry
from ai_agent_framework.core.utils.telemetry import TelemetryTracker

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Asynchronous Workflow Orchestrator manages task execution across workers.
    Uses WorkerPool, TelemetryTracker, AgentProtocol, and custom exceptions.
    """

    def __init__(
        self,
        name: str,
        settings_path: Optional[str] = None,
        protocol: Optional[AgentProtocol] = None,
        worker_pool: Optional[WorkerPool] = None,
        telemetry: Optional[TelemetryTracker] = None,
    ):
        """
        Initialize the asynchronous Orchestrator.

        Args:
            name: Unique identifier for this orchestrator.
            settings_path: Path to settings file.
            protocol: Async communication protocol instance.
            worker_pool: Pool of available worker nodes.
            telemetry: Telemetry tracking instance.
        """
        self.name = name
        self.settings = Settings(config_path=settings_path) # Assuming Settings handles None path
        self.protocol = protocol or AgentProtocol(own_id=f"{name}_proto")
        self.worker_pool = worker_pool or WorkerPool()
        self.telemetry = telemetry or TelemetryTracker()
        self.active_workflows: Dict[str, Workflow] = {}
        self.completed_workflows: Dict[str, Workflow] = {}
        self.failed_workflows: Dict[str, Workflow] = {}

        self._workflows_lock = asyncio.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        self._load_orchestration_config()
        logger.info(f"Orchestrator '{name}' initialized.")

    def _load_orchestration_config(self) -> None:
        """Load orchestration configuration from settings."""
        try:
            orch_config = self.settings.get("orchestration", {})
            self.max_concurrent_workflows = orch_config.get("max_concurrent_workflows", 10)
            self.max_retries = orch_config.get("max_retries", 3)
            self.retry_delay = float(orch_config.get("retry_delay", 5.0)) # Ensure float
            self.monitoring_interval = float(orch_config.get("monitoring_interval", 10.0)) # Ensure float
            self.default_task_timeout = float(orch_config.get("default_task_timeout", 300.0)) # Added default task timeout

            priority_strategy = orch_config.get("priority_strategy", "fifo")
            self._prioritize_tasks = getattr(self, f"_prioritize_{priority_strategy}", self._prioritize_fifo)

            worker_strategy = orch_config.get("worker_selection_strategy", "capability_match")
            self._select_worker = getattr(self, f"_select_worker_{worker_strategy}", self._select_worker_capability_match)

            self._initialize_workers_from_config(orch_config.get("workers", []))
            logger.info(f"Orchestration config loaded for '{self.name}': {len(self.worker_pool.workers)} workers.")
        except Exception as e:
            logger.error(f"Error loading orchestration config for '{self.name}': {e}", exc_info=True)
            # Apply safe defaults
            self.max_concurrent_workflows = 10; self.max_retries = 3; self.retry_delay = 5.0
            self.monitoring_interval = 10.0; self.default_task_timeout = 300.0
            self._prioritize_tasks = self._prioritize_fifo
            self._select_worker = self._select_worker_capability_match

    def _initialize_workers_from_config(self, worker_configs: List[Dict[str, Any]]) -> None:
        """Initialize workers from configuration data using register_worker."""
        # Run synchronously as it modifies the pool directly before async operations start
        # If worker registration becomes async, this needs `asyncio.gather`
        for config in worker_configs:
            worker_id = config.get("id"); endpoint = config.get("endpoint")
            if not worker_id or not endpoint: continue
            try:
                worker = Worker(id=worker_id, endpoint=endpoint, capabilities=config.get("capabilities", []),
                                max_concurrent_tasks=config.get("max_concurrent_tasks", 5), status=WorkerStatus.ONLINE)
                # Use asyncio.run_coroutine_threadsafe or similar if called from non-async context
                # For now, assume this init runs before the event loop starts or use direct sync add
                # Simplified: Directly add if WorkerPool allows sync add or use run_until_complete if needed
                asyncio.get_event_loop().run_until_complete(self.register_worker(worker)) # Or handle differently if loop running
            except Exception as e: logger.error(f"Failed to init worker '{worker_id}' from config: {e}")

    async def register_worker(self, worker: Worker):
        """Register a new worker and its endpoint asynchronously."""
        # WorkerPool methods are now async
        await self.worker_pool.add_worker(worker)
        if worker.endpoint:
            self.protocol.register_endpoint(worker.id, worker.endpoint) # Protocol register is sync
        logger.info(f"Worker '{worker.id}' registered for '{self.name}'.")

    async def unregister_worker(self, worker_id: str):
        """Unregister a worker asynchronously."""
        try:
            await self.worker_pool.remove_worker(worker_id)
            # Optionally remove endpoint from protocol
            logger.info(f"Worker '{worker_id}' unregistered from '{self.name}'.")
        except ValueError as e: # WorkerPool should raise ValueError if not found
             logger.warning(f"Attempted to unregister unknown worker '{worker_id}': {e}")

    async def start_monitoring(self):
        """Start the background monitoring task."""
        if self._monitoring_task is None or self._monitoring_task.done():
             self._stop_event.clear()
             self._monitoring_task = asyncio.create_task(self._monitor_workflows(), name=f"{self.name}-monitor")
             logger.info(f"Workflow monitor started for '{self.name}'.")

    async def stop_monitoring(self):
         """Stop the background monitoring task."""
         if self._monitoring_task and not self._monitoring_task.done():
              self._stop_event.set()
              try: await asyncio.wait_for(self._monitoring_task, timeout=self.monitoring_interval + 2.0)
              except asyncio.TimeoutError: logger.warning(f"Monitor task for '{self.name}' didn't stop gracefully."); self._monitoring_task.cancel()
              except asyncio.CancelledError: pass
              self._monitoring_task = None
              logger.info(f"Workflow monitor stopped for '{self.name}'.")

    async def submit_workflow(self, workflow: Workflow) -> str:
        """Asynchronously submit a workflow for execution."""
        async with self._workflows_lock:
            if len(self.active_workflows) >= self.max_concurrent_workflows: raise OrchestratorError("Max concurrent workflows reached.")
            if workflow.id in self.active_workflows or workflow.id in self.completed_workflows or workflow.id in self.failed_workflows: raise OrchestratorError(f"Workflow ID '{workflow.id}' exists.")
            workflow.status = "submitted"; workflow.metadata["submit_time"] = time.monotonic() # Use monotonic time
            self.active_workflows[workflow.id] = workflow
            self.telemetry.start_workflow(workflow.id)
            logger.info(f"Workflow '{workflow.id}' submitted to '{self.name}'.")
            asyncio.create_task(self._schedule_ready_tasks(workflow.id)) # Schedule by ID
            await self.start_monitoring() # Ensure monitor is running
            return workflow.id

    async def _schedule_ready_tasks(self, workflow_id: str) -> None:
        """Find and schedule ready tasks for a specific workflow ID."""
        async with self._workflows_lock:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow: return # Workflow no longer active

            ready_tasks = [task for task in workflow.tasks if task.status == TaskStatus.PENDING and self._are_dependencies_satisfied(workflow, task)]

        if not ready_tasks:
            async with self._workflows_lock: await self._check_workflow_completion(workflow) # Check completion if no tasks are ready
            return

        prioritized_tasks = self._prioritize_tasks(ready_tasks)
        logger.debug(f"Found {len(prioritized_tasks)} ready tasks for workflow '{workflow_id}'.")
        scheduling_coroutines = [self._schedule_task(workflow_id, task.id) for task in prioritized_tasks] # Schedule by ID
        await asyncio.gather(*scheduling_coroutines)

    def _are_dependencies_satisfied(self, workflow: Workflow, task: Task) -> bool:
        """Check if task dependencies are met (synchronous check)."""
        # Assumes workflow lock is held or workflow state is stable during check
        for dep_id in task.metadata.get("dependencies", []):
            dep_task = workflow.get_task(dep_id)
            if not dep_task:
                task.status = TaskStatus.FAILED; task.error = f"Dependency '{dep_id}' missing."; logger.error(f"Task '{task.id}' failed: {task.error}"); return False
            if dep_task.status != TaskStatus.COMPLETED: return False
        return True

    # --- Prioritization and Worker Selection ---
    def _prioritize_fifo(self, tasks: List[Task]) -> List[Task]: return tasks
    def _prioritize_deadline(self, tasks: List[Task]) -> List[Task]: return sorted(tasks, key=lambda t: t.metadata.get("deadline", float('inf')))

    async def _select_worker(self, task: Task) -> Optional[Worker]:
        """Select worker using configured strategy (now async)."""
        # Ensure worker pool selection methods are async
        required_caps = task.metadata.get("required_capabilities", [])
        if self._select_worker.__name__ == "_select_worker_round_robin":
             return await self.worker_pool.get_next_available_worker(required_caps)
        elif self._select_worker.__name__ == "_select_worker_least_loaded":
             return await self.worker_pool.get_least_loaded_worker(required_caps)
        elif self._select_worker.__name__ == "_select_worker_capability_match":
             return await self.worker_pool.get_best_capability_match_worker(required_caps)
        else: # Fallback
             logger.warning("Unknown worker selection strategy, using capability match.")
             return await self.worker_pool.get_best_capability_match_worker(required_caps)

    async def _schedule_task(self, workflow_id: str, task_id: str) -> None:
        """Asynchronously schedule a single task by ID."""
        async with self._workflows_lock:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow: return
            task = workflow.get_task(task_id)
            if not task or task.status != TaskStatus.PENDING: return

            worker = await self._select_worker(task) # Await worker selection
            if not worker or not worker.endpoint:
                logger.warning(f"No suitable worker for task '{task_id}'. Keeping PENDING.")
                return # Keep pending for next scheduling cycle

            task.status = TaskStatus.SCHEDULED; task.assigned_worker = worker.id
            task.metadata["scheduled_time"] = time.monotonic()
            await self.worker_pool.assign_task_to_worker(worker.id, task.id) # Await pool update

        self.telemetry.start_task(workflow_id, task_id)
        task_content = task.to_dict(); task_content["workflow_id"] = workflow_id # Add workflow context
        task_message = AgentMessage(sender=self.name, recipient=worker.id, content=task_content, message_type="task_execute", message_id=f"task_{task_id}_{task.metadata.get('retry_count', 0)}")

        logger.info(f"Scheduling task '{task.id}' on worker '{worker.id}'.")
        try:
            await self.protocol.send(task_message) # Await async send
        except (CommunicationError, ProtocolError, Exception) as e:
            logger.error(f"Failed dispatching task '{task.id}' to '{worker.id}': {e}", exc_info=True)
            async with self._workflows_lock: await self.worker_pool.release_task_from_worker(worker.id, task.id) # Ensure release on dispatch failure
            await self.handle_task_failure("orchestrator", task.id, f"Dispatch failure: {e}") # Trigger failure handling

    async def handle_task_completion(self, worker_id: str, task_id: str, result: Any) -> None:
        """Async handler for task completion."""
        async with self._workflows_lock:
            workflow, task = self._find_workflow_and_task(task_id)
            if not workflow or not task or workflow.id not in self.active_workflows: return
            if task.status not in [TaskStatus.SCHEDULED, TaskStatus.RUNNING]: return

            logger.info(f"Task '{task_id}' completed by '{worker_id}'.")
            task.status = TaskStatus.COMPLETED; task.result = result; task.error = None
            task.metadata["completed_time"] = time.monotonic()
            self.telemetry.end_task(workflow.id, task.id, success=True)
            await self.worker_pool.release_task_from_worker(worker_id, task.id) # Await release
            needs_scheduling = True # Flag to schedule outside lock

        if needs_scheduling: # Schedule next tasks outside lock
             asyncio.create_task(self._schedule_ready_tasks(workflow.id))

    async def handle_task_failure(self, worker_id: str, task_id: str, error: str) -> None:
        """Async handler for task failure."""
        schedule_retry = False
        workflow_id_for_retry = None
        task_for_retry = None

        async with self._workflows_lock:
            workflow, task = self._find_workflow_and_task(task_id)
            if not workflow or not task or workflow.id not in self.active_workflows: return
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]: return

            logger.warning(f"Task '{task_id}' failed on '{worker_id}': {error}")
            task.metadata["last_error"] = str(error)[:500] # Store limited error
            await self.worker_pool.release_task_from_worker(worker_id, task.id)

            retry_count = task.metadata.get("retry_count", 0)
            if retry_count < self.max_retries:
                task.metadata["retry_count"] = retry_count + 1
                task.status = TaskStatus.PENDING; task.assigned_worker = None
                logger.info(f"Task '{task_id}' will be retried (attempt {retry_count + 1}). Delaying {self.retry_delay}s.")
                schedule_retry = True
                workflow_id_for_retry = workflow.id
                task_for_retry = task # Pass task itself for scheduling
            else:
                logger.error(f"Task '{task_id}' FAILED after {self.max_retries} retries.")
                task.status = TaskStatus.FAILED; task.error = error
                task.metadata["completed_time"] = time.monotonic()
                self.telemetry.record_task_failure(workflow.id, task.id, error)
                await self._check_workflow_completion(workflow) # Check completion inside lock after final failure

        if schedule_retry and workflow_id_for_retry and task_for_retry:
            # Schedule the retry call after a delay, outside the lock
            await asyncio.sleep(self.retry_delay)
            # Re-check if workflow still active before scheduling retry task
            async with self._workflows_lock:
                 workflow = self.active_workflows.get(workflow_id_for_retry)
                 if workflow and task_for_retry.status == TaskStatus.PENDING: # Ensure it wasn't cancelled etc.
                      logger.info(f"Scheduling retry for task '{task_for_retry.id}'")
                      asyncio.create_task(self._schedule_task(workflow.id, task_for_retry.id))


    def _find_workflow_and_task(self, task_id: str) -> Tuple[Optional[Workflow], Optional[Task]]:
         """Helper to find workflow and task by task ID (assumes lock is held)."""
         for workflow in self.active_workflows.values():
              task = workflow.get_task(task_id)
              if task: return workflow, task
         return None, None

    async def _check_workflow_completion(self, workflow: Workflow) -> None:
        """Check if workflow is terminal (assumes lock is held)."""
        if workflow.id not in self.active_workflows: return

        all_terminal = all(t.status in Task.TERMINAL_STATUSES for t in workflow.tasks) # Assuming Task has TERMINAL_STATUSES set

        if all_terminal:
             has_critical_failure = any(t.status == TaskStatus.FAILED and t.metadata.get("critical", False) for t in workflow.tasks)
             has_any_failure = any(t.status == TaskStatus.FAILED for t in workflow.tasks)

             del self.active_workflows[workflow.id]
             workflow.metadata["completed_time"] = time.monotonic()

             if has_critical_failure:
                  workflow.status = "failed"; self.failed_workflows[workflow.id] = workflow
                  logger.error(f"Workflow '{workflow.id}' FAILED (critical task).")
             elif has_any_failure:
                  workflow.status = "completed_with_failures"; self.completed_workflows[workflow.id] = workflow
                  logger.warning(f"Workflow '{workflow.id}' COMPLETED with failures.")
             else:
                  workflow.status = "completed"; self.completed_workflows[workflow.id] = workflow
                  logger.info(f"Workflow '{workflow.id}' COMPLETED successfully.")

             self.telemetry.end_workflow(workflow.id, success=(workflow.status == "completed"))


    async def _monitor_workflows(self) -> None:
        """Async background task to monitor workflows."""
        logger.info(f"Starting workflow monitor for '{self.name}'...")
        while not self._stop_event.is_set():
            try:
                tasks_to_fail = []
                async with self._workflows_lock:
                    current_mono_time = time.monotonic()
                    workflows_to_check = list(self.active_workflows.items())

                for workflow_id, workflow in workflows_to_check:
                     # Check workflow timeout
                     wf_timeout = workflow.metadata.get("timeout")
                     wf_start_time = workflow.metadata.get("submit_time")
                     if wf_timeout and wf_start_time and (current_mono_time - wf_start_time > wf_timeout):
                          logger.warning(f"Workflow '{workflow_id}' timed out. Failing.")
                          async with self._workflows_lock:
                              if workflow_id in self.active_workflows:
                                  del self.active_workflows[workflow_id]; workflow.status = "timeout"
                                  workflow.metadata["completed_time"] = current_mono_time; self.failed_workflows[workflow_id] = workflow
                                  self.telemetry.end_workflow(workflow_id, success=False)
                          continue # Move to next workflow

                     # Check task timeouts
                     async with self._workflows_lock:
                          if workflow_id not in self.active_workflows: continue
                          for task in workflow.tasks:
                               if task.status == TaskStatus.SCHEDULED or task.status == TaskStatus.RUNNING:
                                    task_timeout = task.metadata.get("timeout", self.default_task_timeout)
                                    task_start_time = task.metadata.get("scheduled_time") # Check since scheduling
                                    if task_start_time and (current_mono_time - task_start_time > task_timeout):
                                         logger.warning(f"Task '{task.id}' in '{workflow_id}' timed out.")
                                         # Collect tasks to fail outside the lock iteration
                                         tasks_to_fail.append((task.assigned_worker or "orchestrator", task.id, f"Task timed out after {task_timeout}s"))

                # Handle task failures outside the main loop/lock
                if tasks_to_fail:
                    failure_coroutines = [self.handle_task_failure(w_id, t_id, err) for w_id, t_id, err in tasks_to_fail]
                    await asyncio.gather(*failure_coroutines, return_exceptions=True) # Handle failures concurrently

                # Wait before next check
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError: logger.info(f"Monitor for '{self.name}' cancelled."); break
            except Exception as e: logger.error(f"Error in monitor for '{self.name}': {e}", exc_info=True); await asyncio.sleep(self.monitoring_interval * 2)
        logger.info(f"Monitor for '{self.name}' stopped.")


    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        async with self._workflows_lock:
            wf = self.active_workflows.get(workflow_id) or \
                 self.completed_workflows.get(workflow_id) or \
                 self.failed_workflows.get(workflow_id)
            if not wf: raise ValueError(f"Workflow '{workflow_id}' not found.")

            tasks_status = []
            for task in wf.tasks:
                 ts_info = {"id": task.id, "action": getattr(task, 'action', '?'), "status": getattr(task.status, 'value', task.status),
                            "worker": task.assigned_worker, "error": task.error}
                 if task.status == TaskStatus.COMPLETED: ts_info["result_preview"] = str(task.result)[:100] + "..." if task.result else "None"
                 tasks_status.append(ts_info)

            return {"id": wf.id, "status": wf.status, "tasks": tasks_status, "meta": wf.metadata, "metrics": self.telemetry.get_workflow_metrics(wf.id)}


    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel an in-progress workflow."""
        async with self._workflows_lock:
            if workflow_id not in self.active_workflows: raise ValueError(f"Workflow '{workflow_id}' not active.")
            workflow = self.active_workflows.pop(workflow_id); workflow.status = "cancelling"
            logger.info(f"Cancelling workflow '{workflow_id}'...")
            tasks_to_cancel_msgs = []
            for task in workflow.tasks:
                 if task.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED, TaskStatus.RUNNING]:
                      task.status = TaskStatus.CANCELLED; task.metadata["cancelled_time"] = time.monotonic()
                      if task.assigned_worker:
                           cancel_msg = AgentMessage(sender=self.name, recipient=task.assigned_worker, content={"task_id": task.id}, message_type="task_cancel")
                           tasks_to_cancel_msgs.append(self.protocol.send(cancel_msg)) # Collect send coroutines
                           await self.worker_pool.release_task_from_worker(task.assigned_worker, task.id) # Release immediately

        # Send cancel messages concurrently outside lock
        if tasks_to_cancel_msgs:
            results = await asyncio.gather(*tasks_to_cancel_msgs, return_exceptions=True)
            for res in results: # Log any errors sending cancel messages
                 if isinstance(res, Exception): logger.warning(f"Failed sending cancel message during WF {workflow_id} cancel: {res}")

        async with self._workflows_lock: # Final update under lock
             workflow.status = "cancelled"; workflow.metadata["completed_time"] = time.monotonic()
             self.failed_workflows[workflow_id] = workflow # Store in terminal state dict
             self.telemetry.record_workflow_cancellation(workflow_id)
        logger.info(f"Workflow '{workflow_id}' cancelled.")


    async def shutdown(self) -> None:
        """Gracefully shut down the orchestrator."""
        logger.info(f"Shutting down orchestrator '{self.name}'...")
        await self.stop_monitoring()
        async with self._workflows_lock: active_ids = list(self.active_workflows.keys())
        cancel_coros = [self.cancel_workflow(wf_id) for wf_id in active_ids]
        if cancel_coros: await asyncio.gather(*cancel_coros, return_exceptions=True)
        if hasattr(self.protocol, 'shutdown'): await self.protocol.shutdown()
        logger.info(f"Orchestrator '{self.name}' shutdown complete.")