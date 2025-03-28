# ai_agent_framework/core/utils/telemetry.py

"""
Basic Telemetry Tracker for Workflows and Tasks.

Provides a simple in-memory mechanism to track the start, end, and status
of workflows and their constituent tasks.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TelemetryTracker:
    """
    A basic in-memory tracker for workflow and task execution metrics.

    Stores timing and status information. Can be extended to push metrics
    to external monitoring systems (e.g., Prometheus, Datadog).
    """

    def __init__(self):
        # Structure: { workflow_id: { "start": time, "end": time, "status": str, "tasks": { task_id: {...} } } }
        self.workflows: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tasks": {}})
        logger.info("TelemetryTracker initialized.")

    def start_workflow(self, workflow_id: str):
        """Record the start of a workflow."""
        if workflow_id not in self.workflows:
            self.workflows[workflow_id] = {
                "start_time": time.monotonic(),
                "end_time": None,
                "status": "running",
                "tasks": {}
            }
            logger.debug(f"Telemetry started for workflow: {workflow_id}")
        else:
             # Handle workflow restart or concurrent runs if necessary
             logger.warning(f"Telemetry already started for workflow {workflow_id}. Re-initializing.")
             self.workflows[workflow_id]["start_time"] = time.monotonic()
             self.workflows[workflow_id]["end_time"] = None
             self.workflows[workflow_id]["status"] = "running"
             # Optionally clear previous tasks for this workflow ID if restarting
             # self.workflows[workflow_id]["tasks"] = {}


    def end_workflow(self, workflow_id: str, success: bool = True, status: Optional[str] = None):
        """Record the end of a workflow."""
        if workflow_id in self.workflows:
            wf_data = self.workflows[workflow_id]
            if wf_data.get("end_time") is None: # Record only if not already ended
                 wf_data["end_time"] = time.monotonic()
                 wf_data["status"] = status or ("completed" if success else "failed")
                 wf_data["duration"] = wf_data["end_time"] - wf_data.get("start_time", wf_data["end_time"])
                 logger.debug(f"Telemetry ended for workflow: {workflow_id}, Status: {wf_data['status']}, Duration: {wf_data['duration']:.2f}s")
            else:
                 logger.warning(f"Attempted to end telemetry for already ended workflow: {workflow_id}")
        else:
            logger.warning(f"Attempted to end telemetry for unknown workflow: {workflow_id}")

    def start_task(self, workflow_id: str, task_id: str):
        """Record the start of a task within a workflow."""
        if workflow_id in self.workflows:
            if task_id not in self.workflows[workflow_id]["tasks"]:
                 self.workflows[workflow_id]["tasks"][task_id] = {
                     "start_time": time.monotonic(),
                     "end_time": None,
                     "status": "running",
                     "retries": 0, # Initialize retries
                     "errors": []
                 }
                 logger.debug(f"Telemetry started for task: {task_id} in workflow: {workflow_id}")
            else:
                # Handle task restart/retry
                task_data = self.workflows[workflow_id]["tasks"][task_id]
                task_data["start_time"] = time.monotonic()
                task_data["end_time"] = None
                task_data["status"] = "running"
                # Increment retry count if applicable (might be better handled by orchestrator)
                # task_data["retries"] = task_data.get("retries", 0) + 1
                logger.debug(f"Telemetry restarted for task: {task_id} (Retry scenario?)")

        else:
            logger.warning(f"Attempted to start task telemetry for unknown workflow: {workflow_id}")

    def end_task(self, workflow_id: str, task_id: str, success: bool = True, status: Optional[str] = None):
        """Record the end of a task."""
        if workflow_id in self.workflows and task_id in self.workflows[workflow_id]["tasks"]:
            task_data = self.workflows[workflow_id]["tasks"][task_id]
            if task_data.get("end_time") is None: # Record only if not already ended
                 task_data["end_time"] = time.monotonic()
                 task_data["status"] = status or ("completed" if success else "failed")
                 task_data["duration"] = task_data["end_time"] - task_data.get("start_time", task_data["end_time"])
                 logger.debug(f"Telemetry ended for task: {task_id}, Status: {task_data['status']}, Duration: {task_data['duration']:.2f}s")
            else:
                 logger.warning(f"Attempted to end telemetry for already ended task: {task_id}")
        else:
            logger.warning(f"Attempted to end task telemetry for unknown workflow/task: {workflow_id}/{task_id}")

    def record_task_failure(self, workflow_id: str, task_id: str, error: str):
        """Record a failure event for a task."""
        # Ensure end_task is also called to mark it as terminal
        self.end_task(workflow_id, task_id, success=False, status="failed")
        # Add error details if task exists
        if workflow_id in self.workflows and task_id in self.workflows[workflow_id]["tasks"]:
             task_data = self.workflows[workflow_id]["tasks"][task_id]
             task_data.setdefault("errors", []).append({"time": time.monotonic(), "message": str(error)[:500]}) # Limit error message size
             task_data["retries"] = task_data.get("retries", 0) + 1 # Increment retries on failure recording


    def record_workflow_cancellation(self, workflow_id: str):
        """Record the cancellation of a workflow."""
        self.end_workflow(workflow_id, success=False, status="cancelled")

    def get_workflow_metrics(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metrics for a specific workflow."""
        return self.workflows.get(workflow_id)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
         """Retrieve all collected workflow metrics."""
         return self.workflows.copy() # Return a copy

    def clear_metrics(self, workflow_id: Optional[str] = None):
         """Clear metrics for a specific workflow or all workflows."""
         if workflow_id:
              if workflow_id in self.workflows:
                   del self.workflows[workflow_id]
                   logger.info(f"Cleared telemetry for workflow: {workflow_id}")
         else:
              self.workflows.clear()
              logger.info("Cleared all telemetry data.")