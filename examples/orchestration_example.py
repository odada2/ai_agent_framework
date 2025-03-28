"""
Orchestration Examples (Refactored)

This module provides practical examples of using a simulated orchestration framework
for various use cases and patterns.

Examples include:
1. Basic workflow orchestration
2. Pipeline pattern implementation
3. Dynamic workflow with conditional branching
4. Distributed processing with specialized workers
5. Event-driven workflow with external triggers

Refactoring focuses on fixing minor bugs, improving comments, and clarifying
simulation limitations.
"""

import os
import sys
import time
import json
import random
import logging
import traceback
import argparse
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field

# Add parent directory to path to allow imports
# Note: For larger projects, proper packaging is recommended over sys.path manipulation.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import framework components (assuming they exist in the specified paths)
try:
    from core.workflow.orchestrator import Orchestrator
    from core.workflow.worker_pool import Worker, WorkerStatus
    from core.workflow.task import Task, TaskStatus
    from core.workflow.workflow import Workflow
    from core.communication.agent_protocol import AgentProtocol, AgentMessage
    from core.exceptions import OrchestratorError, WorkerError, SchedulingError, CommunicationError
    from config.settings import Settings
except ImportError as e:
    print(f"Error importing framework components: {e}")
    print("Please ensure the 'core' directory is correctly structured and accessible.")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MAX_WAIT_TIME = 15 # seconds
DEFAULT_CHECK_INTERVAL = 0.5 # seconds

# --- Mock Worker Environment ---

@dataclass
class MockWorkerInfo:
    """Information about a simulated worker."""
    capabilities: List[str]
    endpoint: str
    tasks_processed: int = 0
    last_task: Optional[str] = None
    max_concurrent_tasks: int = 1 # Simplified concurrency limit for simulation

@dataclass
class MockEnvironmentState:
    """Shared state accessible by mock task handlers."""
    # Example states used by different examples
    pipeline_stages: Dict[str, Any] = field(default_factory=dict)
    current_stage: Optional[str] = None
    decision: Optional[str] = None
    path_taken: Optional[str] = None
    tasks_added_dynamically: List[str] = field(default_factory=list)
    processing_result: Optional[Dict[str, Any]] = None
    final_result: Optional[Dict[str, Any]] = None
    processed_counts: Dict[str, int] = field(default_factory=lambda: {"text": 0, "image": 0})
    failed_counts: Dict[str, int] = field(default_factory=lambda: {"text": 0, "image": 0})
    aggregated: bool = False
    aggregated_results: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    processed_event_data: List[Dict[str, Any]] = field(default_factory=list)
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    current_workflow_id: Optional[str] = None


class MockWorkerEnvironment:
    """Helper class to simulate a worker environment for testing purposes."""

    def __init__(self, name: str, failure_rate: float = 0.05):
        """Initialize the mock environment."""
        self.name = name
        self.workers: Dict[str, MockWorkerInfo] = {}
        self.task_handlers: Dict[str, Callable] = {}
        # WARNING: Direct mutation of shared state in handlers is simple for simulation,
        # but can lead to issues in real concurrent systems. Use with caution.
        self.shared_state: MockEnvironmentState = MockEnvironmentState()
        self.failure_rate = failure_rate
        self._orchestrator_ref: Optional[Orchestrator] = None # To report results back

    def register_mock_worker(self, worker_id: str, capabilities: List[str], endpoint: str = None, max_tasks: int = 1):
        """Register a mock worker."""
        resolved_endpoint = endpoint or f"http://localhost:8{len(self.workers) + 1:03d}"
        self.workers[worker_id] = MockWorkerInfo(
            capabilities=capabilities,
            endpoint=resolved_endpoint,
            max_concurrent_tasks=max_tasks
        )
        logger.debug(f"Registered mock worker: {worker_id} with caps {capabilities} at {resolved_endpoint}")

    def register_task_handler(self, task_action: str, handler: Callable):
        """Register a handler for a specific task action."""
        self.task_handlers[task_action] = handler
        logger.debug(f"Registered handler for action: {task_action}")

    def _simulate_task_execution(self, message: AgentMessage):
        """Simulates a worker receiving and executing a task."""
        if self._orchestrator_ref is None:
             logger.error("Orchestrator reference not set in MockWorkerEnvironment!")
             return

        task_id = message.content.get("id")
        task_action = message.content.get("action")
        task_params = message.content.get("parameters", {})
        worker_id = message.recipient

        # Simulate random failures
        if random.random() < self.failure_rate:
            error_msg = f"Simulated random failure on worker {worker_id}"
            logger.warning(f"{error_msg} for task {task_id}")
            self._orchestrator_ref.handle_task_failure(worker_id, task_id, error_msg)
            # Update simulation state for tracking
            if task_action == "process_text": self.shared_state.failed_counts["text"] += 1
            if task_action == "process_image": self.shared_state.failed_counts["image"] += 1
            return

        # Track task execution in simulation
        if worker_id in self.workers:
            self.workers[worker_id].tasks_processed += 1
            self.workers[worker_id].last_task = task_id

        # Simulate processing time
        delay = random.uniform(0.1, 0.6) # Shorter delays for faster examples
        time.sleep(delay)

        # Get task result based on action
        result_data = None
        error_msg = None
        try:
            if task_action in self.task_handlers:
                handler = self.task_handlers[task_action]
                # Pass mutable shared state to handler (use carefully)
                result_data = handler(task_id, task_params, worker_id, self.shared_state)
            else:
                # Default handler if none registered
                logger.warning(f"No specific handler for action '{task_action}'. Using default success.")
                result_data = {
                    "task_id": task_id,
                    "action": task_action,
                    "processed": True,
                    "status": "success_default_handler",
                    "processing_time": delay
                }
        except Exception as e:
             logger.error(f"Error executing handler for task {task_id} ({task_action}): {e}", exc_info=True)
             error_msg = f"Handler execution error: {e}"

        # Send completion or failure back to orchestrator
        if error_msg:
             self._orchestrator_ref.handle_task_failure(worker_id, task_id, error_msg)
        else:
             self._orchestrator_ref.handle_task_completion(worker_id, task_id, result_data)

    def _simulate_status_response(self, message: AgentMessage, protocol: AgentProtocol):
         """Simulates a worker responding to a status request."""
         worker_id = message.recipient
         worker_info = self.workers.get(worker_id)
         if worker_info:
             response = AgentMessage(
                 sender=worker_id,
                 recipient=message.sender,
                 message_type="status_response",
                 content={
                     "status": "online", # Simplified status
                     "active_tasks": 0, # Simplified for mock
                     "processed_count": worker_info.tasks_processed,
                     "capabilities": worker_info.capabilities
                 },
                 correlation_id=message.message_id
             )
             # In a real system, this goes over the network. Here, we inject it back.
             protocol.receive(response.to_dict())
         else:
              logger.warning(f"Received status request for unknown worker: {worker_id}")


    def get_mock_protocol(self, orchestrator: Orchestrator) -> AgentProtocol:
        """Create a mock protocol with simulated message sending."""
        self._orchestrator_ref = orchestrator # Store orchestrator reference
        protocol = AgentProtocol()

        def mock_send(message: AgentMessage):
            """Simulates sending a message, routing based on type."""
            try:
                logger.debug(f"MockSend: To={message.recipient}, Type={message.message_type}, Task={message.content.get('id', message.content.get('task_id', 'N/A'))}")

                if message.message_type == "task_execute":
                    self._simulate_task_execution(message)

                elif message.message_type == "task_cancel":
                    task_id = message.content.get("task_id")
                    worker_id = message.recipient
                    logger.info(f"Simulating task '{task_id}' cancellation request to worker '{worker_id}' (no action taken in mock)")
                    # In a real system, worker would attempt cancellation and report back.

                elif message.message_type == "status_request":
                     self._simulate_status_response(message, protocol)

                else:
                     logger.warning(f"MockSend received unhandled message type: {message.message_type}")

            except Exception as e:
                logger.error(f"Error in mock_send for message to {message.recipient}: {e}", exc_info=True)
                # Attempt to report failure if it was a task execution message
                if message.message_type == "task_execute" and self._orchestrator_ref:
                    self._orchestrator_ref.handle_task_failure(
                        message.recipient,
                        message.content.get("id", "unknown_task"),
                        f"MockSend simulation error: {e}"
                    )

        protocol.send = mock_send
        return protocol

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about worker usage and final shared state."""
        return {
            "workers": {
                worker_id: {
                    "tasks_processed": info.tasks_processed,
                    "capabilities": info.capabilities
                }
                for worker_id, info in self.workers.items()
            },
            "final_shared_state": self.shared_state # Return the entire final state object
        }

# --- Helper Function ---

def wait_for_workflow_completion(orchestrator: Orchestrator, workflow_id: str,
                                 description: str = "Workflow",
                                 max_wait_time: int = DEFAULT_MAX_WAIT_TIME,
                                 check_interval: float = DEFAULT_CHECK_INTERVAL) -> Dict[str, Any]:
    """Helper function to wait for a workflow to complete by polling."""
    logger.info(f"Waiting for {description} ({workflow_id}) to complete...")
    start_time = time.time() # Correctly scoped start time
    last_status = {}

    try:
        while time.time() - start_time < max_wait_time:
            try:
                status = orchestrator.get_workflow_status(workflow_id)
                last_status = status # Store the latest status seen

                # Count tasks by status for logging
                status_counts = {}
                for task_status in status.get("tasks", []):
                    s = task_status.get("status", "unknown")
                    status_counts[s] = status_counts.get(s, 0) + 1

                logger.info(f"{description} status: {status['status']}, Tasks: {status_counts}")

                if status["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value, "completed_with_failures"]: # Use enum value if available
                    logger.info(f"{description} ({workflow_id}) finished with status: {status['status']}")
                    return status

                time.sleep(check_interval)
            except OrchestratorError as e:
                logger.error(f"Error getting workflow status for {workflow_id}: {e}")
                # Decide if this error is terminal for waiting
                if "not found" in str(e).lower():
                     logger.error(f"Workflow {workflow_id} not found. Stopping wait.")
                     return {"status": TaskStatus.FAILED.value, "error": f"Workflow not found: {e}"}
                time.sleep(check_interval * 2) # Wait a bit longer after an error
            except Exception as e:
                 logger.error(f"Unexpected error waiting for workflow {workflow_id}: {e}", exc_info=True)
                 time.sleep(check_interval * 2)


        # If loop finishes, it's a timeout
        logger.warning(f"{description} ({workflow_id}) did not complete within {max_wait_time} seconds.")
        return last_status or {"status": "timeout", "error": f"Timed out after {max_wait_time}s"}

    except KeyboardInterrupt:
        logger.warning(f"Wait for {description} ({workflow_id}) interrupted by user.")
        return last_status or {"status": "interrupted", "error": "Monitoring interrupted"}


# --- Example Definitions ---

def example_1_basic_workflow():
    """Example 1: Basic Workflow Orchestration"""
    logger.info("\n--- Example 1: Basic Workflow Orchestration ---")
    mock_env = MockWorkerEnvironment("basic-workflow", failure_rate=0.0) # No failures for basic

    # Register workers
    mock_env.register_mock_worker("worker-proc", ["data_processing"])
    mock_env.register_mock_worker("worker-exec", ["task_execution"])
    mock_env.register_mock_worker("worker-valid", ["data_validation"])

    # Register task handlers (simple simulations)
    mock_env.register_task_handler("process_data", lambda tid, p, wid, s: {"file": p.get("file"), "records": random.randint(100,500)})
    mock_env.register_task_handler("execute_logic", lambda tid, p, wid, s: {"func": p.get("function"), "result": "ok"})
    mock_env.register_task_handler("validate_data", lambda tid, p, wid, s: {"schema": p.get("schema"), "valid": True})

    orchestrator = Orchestrator(name="basic-orchestrator")
    orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)

    for worker_id, worker_info in mock_env.workers.items():
        orchestrator.register_worker(Worker(id=worker_id, endpoint=worker_info.endpoint, capabilities=worker_info.capabilities))

    try:
        # Define tasks (independent)
        tasks = [
            Task(id="t1", action="process_data", parameters={"file": "data1.csv"}, metadata={"required_capabilities": ["data_processing"]}),
            Task(id="t2", action="process_data", parameters={"file": "data2.csv"}, metadata={"required_capabilities": ["data_processing"]}),
            Task(id="t3", action="execute_logic", parameters={"function": "analyze"}, metadata={"required_capabilities": ["task_execution"]}),
            Task(id="t4", action="validate_data", parameters={"schema": "cust"}, metadata={"required_capabilities": ["data_validation"]})
        ]
        workflow = Workflow(id="wf-basic", tasks=tasks)

        logger.info(f"Submitting workflow {workflow.id} with {len(tasks)} tasks...")
        workflow_id = orchestrator.submit_workflow(workflow)

        final_status = wait_for_workflow_completion(orchestrator, workflow_id, description="Basic Workflow")

        logger.info(f"\nFinal Status ({workflow_id}): {final_status.get('status', 'unknown')}")
        logger.info("Task Details:")
        for task_status in final_status.get("tasks", []):
            logger.info(f"  - {task_status['id']} ({task_status['action']}): {task_status['status']}")
            if task_status.get('result'):
                logger.info(f"      Result: {str(task_status['result'])[:80]}...") # Truncate long results
            if task_status.get('error'):
                logger.error(f"      Error: {task_status['error']}")

        stats = mock_env.get_stats()
        logger.info("\nWorker Statistics:")
        for worker_id, worker_stats in stats["workers"].items():
            logger.info(f"  - {worker_id}: Processed {worker_stats['tasks_processed']} tasks")

    finally:
        orchestrator.shutdown()
        logger.info("Example 1 finished.")


def example_2_pipeline_pattern():
    """Example 2: Pipeline Pattern"""
    logger.info("\n--- Example 2: Pipeline Pattern ---")
    mock_env = MockWorkerEnvironment("pipeline-pattern")

    # Register workers for pipeline stages
    mock_env.register_mock_worker("loader", ["data_loading"])
    mock_env.register_mock_worker("processor", ["data_processing"])
    mock_env.register_mock_worker("analyzer", ["data_analysis"])
    mock_env.register_mock_worker("reporter", ["data_reporting"])

    # Register task handlers that modify shared state
    def load_data(tid, params, worker_id, state: MockEnvironmentState):
        rows = random.randint(800, 1200)
        result = {"source": params.get("source"), "rows": rows}
        state.pipeline_stages["loaded"] = result # Store result for next stage
        state.current_stage = "loaded"
        return result

    def process_data(tid, params, worker_id, state: MockEnvironmentState):
        input_rows = state.pipeline_stages.get("loaded", {}).get("rows", 0)
        valid_rows = max(0, input_rows - random.randint(10, 50))
        result = {"input_rows": input_rows, "valid_rows": valid_rows, "ops": params.get("ops")}
        state.pipeline_stages["processed"] = result
        state.current_stage = "processed"
        return result

    def analyze_data(tid, params, worker_id, state: MockEnvironmentState):
        input_rows = state.pipeline_stages.get("processed", {}).get("valid_rows", 0)
        insights = [f"Insight_{i}" for i in range(random.randint(2, 5))]
        result = {"valid_rows": input_rows, "insights": insights, "metrics": params.get("metrics")}
        state.pipeline_stages["analyzed"] = result
        state.current_stage = "analyzed"
        return result

    def report_data(tid, params, worker_id, state: MockEnvironmentState):
        insights_count = len(state.pipeline_stages.get("analyzed", {}).get("insights", []))
        result = {"insights_count": insights_count, "format": params.get("format")}
        state.pipeline_stages["reported"] = result
        state.current_stage = "reported"
        return result

    mock_env.register_task_handler("load", load_data)
    mock_env.register_task_handler("process", process_data)
    mock_env.register_task_handler("analyze", analyze_data)
    mock_env.register_task_handler("report", report_data)

    orchestrator = Orchestrator(name="pipeline-orchestrator")
    orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)

    for worker_id, worker_info in mock_env.workers.items():
         orchestrator.register_worker(Worker(id=worker_id, endpoint=worker_info.endpoint, capabilities=worker_info.capabilities))

    try:
        # Define tasks with dependencies
        t_load = Task(id="t-load", action="load", parameters={"source": "db"}, metadata={"required_capabilities": ["data_loading"]})
        t_process = Task(id="t-process", action="process", parameters={"ops": ["clean", "norm"]}, metadata={"required_capabilities": ["data_processing"], "dependencies": ["t-load"]})
        t_analyze = Task(id="t-analyze", action="analyze", parameters={"metrics": ["corr"]}, metadata={"required_capabilities": ["data_analysis"], "dependencies": ["t-process"]})
        t_report = Task(id="t-report", action="report", parameters={"format": "pdf"}, metadata={"required_capabilities": ["data_reporting"], "dependencies": ["t-analyze"]})

        workflow = Workflow(id="wf-pipeline", tasks=[t_load, t_process, t_analyze, t_report])

        logger.info(f"Submitting workflow {workflow.id}...")
        workflow_id = orchestrator.submit_workflow(workflow)

        # Custom wait loop to show intermediate state
        pipeline_start_time = time.time()
        max_pipeline_wait = 20
        final_status = {}
        while time.time() - pipeline_start_time < max_pipeline_wait:
            current_stage = mock_env.shared_state.current_stage
            logger.info(f"Pipeline stage: {current_stage or 'Starting'}...")
            try:
                 status = orchestrator.get_workflow_status(workflow_id)
                 final_status = status # Keep track of last known status
                 if status["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value, "completed_with_failures"]:
                      break
            except Exception as e:
                 logger.error(f"Error checking status in pipeline wait: {e}")
            time.sleep(1.0) # Check less frequently
        else:
             logger.warning(f"Pipeline workflow {workflow_id} might have timed out.")

        logger.info(f"\nFinal Status ({workflow_id}): {final_status.get('status', 'unknown')}")
        logger.info("Pipeline Stages Results:")
        for stage, data in mock_env.shared_state.pipeline_stages.items():
             logger.info(f"  - {stage}: {str(data)[:100]}...") # Truncate

        stats = mock_env.get_stats()
        logger.info("\nWorker Statistics:")
        for worker_id, worker_stats in stats["workers"].items():
            logger.info(f"  - {worker_id}: Processed {worker_stats['tasks_processed']} tasks")

    finally:
        orchestrator.shutdown()
        logger.info("Example 2 finished.")


def example_3_dynamic_workflow():
    """Example 3: Dynamic Workflow with Conditional Branching"""
    logger.info("\n--- Example 3: Dynamic Workflow with Conditional Branching ---")
    mock_env = MockWorkerEnvironment("dynamic-workflow")

    mock_env.register_mock_worker("decider", ["decision_making"])
    mock_env.register_mock_worker("proc-a", ["process_type_a"])
    mock_env.register_mock_worker("proc-b", ["process_type_b"])
    mock_env.register_mock_worker("finalizer", ["finalization"])

    # Task handlers
    def decide(tid, params, worker_id, state: MockEnvironmentState):
        decision = random.choice(["path_a", "path_b"])
        state.decision = decision
        logger.info(f"Task {tid}: Made decision -> {decision}")
        return {"decision": decision}

    def process_a(tid, params, worker_id, state: MockEnvironmentState):
        result = {"processed_by": "A", "quality": random.uniform(0.8, 1.0)}
        state.path_taken = "path_a"
        state.processing_result = result
        return result

    def process_b(tid, params, worker_id, state: MockEnvironmentState):
        result = {"processed_by": "B", "efficiency": random.uniform(0.7, 0.95)}
        state.path_taken = "path_b"
        state.processing_result = result
        return result

    def finalize(tid, params, worker_id, state: MockEnvironmentState):
        result = {"finalized": True, "path": state.path_taken, "input_result": state.processing_result}
        state.final_result = result
        return result

    mock_env.register_task_handler("decide", decide)
    mock_env.register_task_handler("proc_a", process_a)
    mock_env.register_task_handler("proc_b", process_b)
    mock_env.register_task_handler("finalize", finalize)

    orchestrator = Orchestrator(name="dynamic-orchestrator")
    orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)

    for worker_id, worker_info in mock_env.workers.items():
         orchestrator.register_worker(Worker(id=worker_id, endpoint=worker_info.endpoint, capabilities=worker_info.capabilities))

    try:
        # Initial task
        t_decide = Task(id="t-decide", action="decide", metadata={"required_capabilities": ["decision_making"]})
        workflow = Workflow(id="wf-dynamic", tasks=[t_decide])

        logger.info(f"Submitting workflow {workflow.id} with initial task {t_decide.id}...")
        workflow_id = orchestrator.submit_workflow(workflow)
        mock_env.shared_state.current_workflow_id = workflow_id # Store for potential use

        # Wait for the initial decision task to complete
        logger.info("Waiting for decision task to complete...")
        decision_status = wait_for_workflow_completion(orchestrator, workflow_id, description="Decision Task", max_wait_time=5)

        # --- Simulation of Dynamic Task Addition ---
        # NOTE: This part simulates adding tasks based on the result.
        # In a real system, this logic might be in the orchestrator reacting to task completion,
        # or an external system calling an orchestrator API like `orchestrator.add_task(...)`.
        # The `wait_for_workflow_completion` called later won't track these added tasks accurately
        # because they weren't part of the initial submission in this simulation.
        # ---
        if decision_status.get("status") == TaskStatus.COMPLETED.value:
            decision_result = next((t['result'] for t in decision_status.get('tasks', []) if t['id'] == 't-decide'), None)
            decision = decision_result.get("decision") if decision_result else None
            mock_env.shared_state.decision = decision # Ensure state is updated if handler didn't

            logger.info(f"Decision task completed. Decision: {decision}")

            next_task_id = None
            if decision == "path_a":
                logger.info("Simulating addition of 'process A' task...")
                task_content = {"id": "t-proc-a", "action": "proc_a", "parameters": {}}
                # Simulate sending task to appropriate worker
                orchestrator.protocol.send(AgentMessage(sender=orchestrator.name, recipient="proc-a", content=task_content, message_type="task_execute"))
                mock_env.shared_state.tasks_added_dynamically.append("t-proc-a")
                next_task_id = "t-proc-a"
            elif decision == "path_b":
                logger.info("Simulating addition of 'process B' task...")
                task_content = {"id": "t-proc-b", "action": "proc_b", "parameters": {}}
                # Simulate sending task to appropriate worker
                orchestrator.protocol.send(AgentMessage(sender=orchestrator.name, recipient="proc-b", content=task_content, message_type="task_execute"))
                mock_env.shared_state.tasks_added_dynamically.append("t-proc-b")
                next_task_id = "t-proc-b"
            else:
                logger.warning("No valid decision made, cannot proceed dynamically.")

            if next_task_id:
                 # Wait for the dynamically added processing task (best effort in simulation)
                 time.sleep(1.5) # Give some time for simulated execution

                 # Add finalizer task
                 logger.info("Simulating addition of 'finalize' task...")
                 task_content = {"id": "t-finalize", "action": "finalize", "parameters": {}}
                 orchestrator.protocol.send(AgentMessage(sender=orchestrator.name, recipient="finalizer", content=task_content, message_type="task_execute"))
                 mock_env.shared_state.tasks_added_dynamically.append("t-finalize")

                 # Final wait (acknowledging limitations)
                 logger.info("Waiting for finalization (status check might be incomplete due to dynamic tasks)...")
                 # This wait only sees the *original* workflow status
                 final_wf_status = wait_for_workflow_completion(orchestrator, workflow_id, description="Dynamic Workflow (Final)", max_wait_time=5)
                 logger.info(f"Orchestrator status for original WF: {final_wf_status.get('status', 'unknown')}")

        else:
             logger.error(f"Decision task failed or timed out. Status: {decision_status.get('status')}")

        # Log final simulated state
        logger.info("\nFinal Simulated State:")
        logger.info(f"  Decision: {mock_env.shared_state.decision}")
        logger.info(f"  Path Taken: {mock_env.shared_state.path_taken}")
        logger.info(f"  Dynamically Added Tasks (Simulated): {mock_env.shared_state.tasks_added_dynamically}")
        logger.info(f"  Processing Result: {str(mock_env.shared_state.processing_result)[:100]}...")
        logger.info(f"  Final Result: {str(mock_env.shared_state.final_result)[:100]}...")

    finally:
        orchestrator.shutdown()
        logger.info("Example 3 finished.")


def example_4_distributed_processing():
    """Example 4: Distributed Processing with Specialized Workers"""
    logger.info("\n--- Example 4: Distributed Processing ---")
    mock_env = MockWorkerEnvironment("distributed-proc", failure_rate=0.1) # Increased failure rate

    # Register specialized workers
    for i in range(3): mock_env.register_mock_worker(f"text-worker-{i}", ["text_processing"])
    for i in range(2): mock_env.register_mock_worker(f"image-worker-{i}", ["image_processing"])
    mock_env.register_mock_worker("aggregator", ["result_aggregation"])

    # Task handlers
    def process_text(tid, params, worker_id, state: MockEnvironmentState):
        result = {"file": params.get("file"), "tokens": random.randint(500, 2000)}
        state.processed_counts["text"] += 1
        state.aggregated_results.append({"type": "text", "id": tid, **result})
        return result

    def process_image(tid, params, worker_id, state: MockEnvironmentState):
        result = {"file": params.get("file"), "objects": random.randint(0, 10)}
        state.processed_counts["image"] += 1
        state.aggregated_results.append({"type": "image", "id": tid, **result})
        return result

    def aggregate(tid, params, worker_id, state: MockEnvironmentState):
        # In a real scenario, this might query results from a DB or shared storage
        # Here, it uses the potentially incomplete shared state from successful handlers
        summary = {
            "total_processed": state.processed_counts["text"] + state.processed_counts["image"],
            "text_files": state.processed_counts["text"],
            "image_files": state.processed_counts["image"],
            "text_failures": state.failed_counts["text"],
            "image_failures": state.failed_counts["image"],
            "avg_tokens": (sum(r['tokens'] for r in state.aggregated_results if r['type'] == 'text') / max(1, state.processed_counts["text"])),
            "avg_objects": (sum(r['objects'] for r in state.aggregated_results if r['type'] == 'image') / max(1, state.processed_counts["image"])),
        }
        state.aggregated = True
        state.final_result = summary # Store final aggregation
        return summary

    mock_env.register_task_handler("proc_text", process_text)
    mock_env.register_task_handler("proc_image", process_image)
    mock_env.register_task_handler("aggregate", aggregate)

    orchestrator = Orchestrator(name="distrib-orchestrator")
    orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)

    for worker_id, worker_info in mock_env.workers.items():
         orchestrator.register_worker(Worker(id=worker_id, endpoint=worker_info.endpoint, capabilities=worker_info.capabilities))

    try:
        tasks = []
        task_ids = []
        # Create many processing tasks
        for i in range(5):
             t_id = f"text-{i}"
             tasks.append(Task(id=t_id, action="proc_text", parameters={"file": f"doc_{i}.txt"}, metadata={"required_capabilities": ["text_processing"]}))
             task_ids.append(t_id)
        for i in range(3):
             t_id = f"img-{i}"
             tasks.append(Task(id=t_id, action="proc_image", parameters={"file": f"img_{i}.jpg"}, metadata={"required_capabilities": ["image_processing"]}))
             task_ids.append(t_id)

        # Aggregation task depends on all processing tasks
        t_agg = Task(id="t-agg", action="aggregate", metadata={"required_capabilities": ["result_aggregation"], "dependencies": task_ids})
        tasks.append(t_agg)

        workflow = Workflow(id="wf-distributed", tasks=tasks)
        logger.info(f"Submitting workflow {workflow.id} with {len(tasks)} tasks...")
        workflow_id = orchestrator.submit_workflow(workflow)

        final_status = wait_for_workflow_completion(orchestrator, workflow_id, description="Distributed Workflow", max_wait_time=25)

        logger.info(f"\nFinal Status ({workflow_id}): {final_status.get('status', 'unknown')}")
        # Log final aggregated state if aggregation completed
        if mock_env.shared_state.aggregated:
             logger.info("\nAggregated Results:")
             logger.info(json.dumps(mock_env.shared_state.final_result, indent=2))
        else:
             logger.warning("Aggregation task may not have completed successfully.")
             logger.info(f"Partial counts: {mock_env.shared_state.processed_counts}, Failures: {mock_env.shared_state.failed_counts}")

        stats = mock_env.get_stats()
        logger.info("\nWorker Statistics:")
        for worker_id, worker_stats in stats["workers"].items():
            logger.info(f"  - {worker_id}: Processed {worker_stats['tasks_processed']} tasks")

    finally:
        orchestrator.shutdown()
        logger.info("Example 4 finished.")


def example_5_event_driven_workflow():
    """Example 5: Event-Driven Workflow with External Triggers"""
    logger.info("\n--- Example 5: Event-Driven Workflow ---")
    mock_env = MockWorkerEnvironment("event-driven")

    mock_env.register_mock_worker("listener", ["event_handling"])
    mock_env.register_mock_worker("processor", ["processing"])
    mock_env.register_mock_worker("notifier", ["notification"])

    # Task handlers
    def handle_event(tid, params, worker_id, state: MockEnvironmentState):
        event_type = params.get("event_type", "unknown")
        event_data = params.get("event_data", {})
        event_record = {"type": event_type, "data": event_data, "task_id": tid, "ts": time.time()}
        state.events.append(event_record)
        logger.info(f"Event handler received: {event_type}")
        return {"event_type_received": event_type} # Return confirmation

    def process_event_data(tid, params, worker_id, state: MockEnvironmentState):
        data_id = params.get("data_id", "unknown")
        result = {"data_id": data_id, "status": "processed", "worker": worker_id}
        state.processed_event_data.append(result)
        return result

    def notify(tid, params, worker_id, state: MockEnvironmentState):
        message = params.get("message", "")
        result = {"message": message, "status": "sent", "channel": params.get("channel")}
        state.notifications.append(result)
        return result

    mock_env.register_task_handler("handle_event", handle_event)
    mock_env.register_task_handler("process", process_event_data)
    mock_env.register_task_handler("notify", notify)

    orchestrator = Orchestrator(name="event-orchestrator")
    orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)

    for worker_id, worker_info in mock_env.workers.items():
         orchestrator.register_worker(Worker(id=worker_id, endpoint=worker_info.endpoint, capabilities=worker_info.capabilities))

    try:
        # Create a long-running listener task (or represent it conceptually)
        # In a real system, this might not be a task but the orchestrator listening on a queue/endpoint.
        # For simulation, we submit it but expect external triggers.
        t_listen = Task(id="t-listen", action="handle_event", metadata={"required_capabilities": ["event_handling"]})
        # Set a long duration or make it non-terminating in a real scenario
        workflow = Workflow(id="wf-event-listener", tasks=[t_listen])

        logger.info(f"Submitting workflow {workflow.id} with listener task {t_listen.id}...")
        workflow_id = orchestrator.submit_workflow(workflow)
        mock_env.shared_state.current_workflow_id = workflow_id

        logger.info("Workflow submitted. Simulating external events...")
        time.sleep(0.5) # Allow listener task to notionally start

        # --- Simulation of External Events & Dynamic Tasks ---
        # NOTE: Similar to Example 3, this simulates events triggering new tasks outside
        # the initially submitted workflow. Orchestrator status checks will be limited.
        # ---

        # Simulate Event 1: New Data -> Process Data
        logger.info("Simulating external 'NEW_DATA' event...")
        event1_data = {"data_id": "abc", "value": 123}
        # Conceptually, this event arrives and triggers logic (here, simulated directly)
        # Trigger the handler (as if orchestrator received event and dispatched)
        orchestrator.protocol.send(AgentMessage(sender="external_source", recipient="listener", content={"id": "event-1", "action": "handle_event", "parameters": {"event_type": "NEW_DATA", "event_data": event1_data}}, message_type="task_execute"))
        time.sleep(0.5) # Allow event handler simulation

        # Simulate reaction: Add processing task
        logger.info("Simulating dynamic task 'process' based on NEW_DATA event...")
        process_content = {"id": "proc-abc", "action": "process", "parameters": {"data_id": "abc"}}
        orchestrator.protocol.send(AgentMessage(sender=orchestrator.name, recipient="processor", content=process_content, message_type="task_execute"))
        mock_env.shared_state.tasks_added_dynamically.append("proc-abc")
        time.sleep(1.0) # Allow processing simulation

        # Simulate Event 2: Alert -> Notify
        logger.info("Simulating external 'ALERT' event...")
        event2_data = {"level": "WARN", "msg": "Threshold exceeded"}
        # Trigger the handler
        orchestrator.protocol.send(AgentMessage(sender="monitoring_system", recipient="listener", content={"id": "event-2", "action": "handle_event", "parameters": {"event_type": "ALERT", "event_data": event2_data}}, message_type="task_execute"))
        time.sleep(0.5) # Allow event handler simulation

        # Simulate reaction: Add notification task
        logger.info("Simulating dynamic task 'notify' based on ALERT event...")
        notify_content = {"id": "notify-warn", "action": "notify", "parameters": {"channel": "slack", "message": "WARN: Threshold exceeded"}}
        orchestrator.protocol.send(AgentMessage(sender=orchestrator.name, recipient="notifier", content=notify_content, message_type="task_execute"))
        mock_env.shared_state.tasks_added_dynamically.append("notify-warn")
        time.sleep(1.0) # Allow notification simulation

        logger.info("\nEvent simulation finished.")

        # Log final simulated state
        logger.info("\nFinal Simulated State:")
        logger.info(f"  Events Handled: {len(mock_env.shared_state.events)}")
        for i, ev in enumerate(mock_env.shared_state.events): logger.info(f"    - Event {i+1}: Type={ev['type']}, Data={str(ev['data'])[:50]}...")
        logger.info(f"  Data Processed: {len(mock_env.shared_state.processed_event_data)}")
        for i, pd in enumerate(mock_env.shared_state.processed_event_data): logger.info(f"    - Processed {i+1}: ID={pd['data_id']}, Worker={pd['worker']}")
        logger.info(f"  Notifications Sent: {len(mock_env.shared_state.notifications)}")
        for i, nt in enumerate(mock_env.shared_state.notifications): logger.info(f"    - Notification {i+1}: Channel={nt['channel']}, Msg={nt['message'][:50]}...")

        # Check status of the original listener workflow (might still be running/pending)
        listener_status = orchestrator.get_workflow_status(workflow_id)
        logger.info(f"\nStatus of original listener workflow ({workflow_id}): {listener_status.get('status', 'unknown')}")


    finally:
        orchestrator.shutdown()
        logger.info("Example 5 finished.")


# --- Main Runner ---

def run_all_examples():
    """Runs all defined examples sequentially."""
    example_funcs = [
        example_1_basic_workflow,
        example_2_pipeline_pattern,
        example_3_dynamic_workflow,
        example_4_distributed_processing,
        example_5_event_driven_workflow
    ]
    for i, func in enumerate(example_funcs):
        try:
            func()
        except Exception as e:
            logger.error(f"Error running example {i+1} ({func.__name__}): {e}", exc_info=True)
        if i < len(example_funcs) - 1:
             logger.info("--- Delay between examples ---")
             time.sleep(1.5) # Brief pause between examples

    logger.info("\n--- All examples finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run orchestration examples")
    parser.add_argument("--example", type=int, choices=range(1, 6), help="Run a specific example (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all examples (default if none specified)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(args.log_level.upper())

    # Determine which examples to run
    run_all = args.all or not args.example # Run all if --all or if no specific example chosen

    example_map = {
        1: example_1_basic_workflow,
        2: example_2_pipeline_pattern,
        3: example_3_dynamic_workflow,
        4: example_4_distributed_processing,
        5: example_5_event_driven_workflow
    }

    try:
        if run_all:
            run_all_examples()
        elif args.example in example_map:
            example_map[args.example]()
        else:
             # Should not happen due to argparse choices, but good practice
             logger.error(f"Invalid example number: {args.example}")

    except KeyboardInterrupt:
        logger.info("Orchestration examples interrupted by user.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during example execution: {e}", exc_info=True)