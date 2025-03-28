"""
Unit Tests for Orchestrator

This module contains unit tests for the orchestration components,
including the Orchestrator class, worker management, and task scheduling.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from core.workflow.orchestrator import Orchestrator
from core.workflow.worker_pool import WorkerPool, Worker, WorkerStatus
from core.workflow.task import Task, TaskStatus
from core.workflow.workflow import Workflow
from core.communication.agent_protocol import AgentMessage, AgentProtocol
from core.exceptions import OrchestratorError, WorkerError, SchedulingError

class TestWorkerPool(unittest.TestCase):
    """Tests for the WorkerPool class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.worker_pool = WorkerPool()
        
        # Create some test workers
        self.worker1 = Worker(id="worker1", endpoint="http://worker1:8000", capabilities=["task_execution"])
        self.worker2 = Worker(id="worker2", endpoint="http://worker2:8000", capabilities=["data_processing", "task_execution"])
        self.worker3 = Worker(id="worker3", endpoint="http://worker3:8000", capabilities=["api_integration"])
    
    def test_add_worker(self):
        """Test adding workers to the pool."""
        self.worker_pool.add_worker(self.worker1)
        self.assertEqual(len(self.worker_pool.workers), 1)
        self.assertEqual(self.worker_pool.workers["worker1"], self.worker1)
        
        # Adding the same worker again should raise ValueError
        with self.assertRaises(ValueError):
            self.worker_pool.add_worker(self.worker1)
    
    def test_remove_worker(self):
        """Test removing workers from the pool."""
        self.worker_pool.add_worker(self.worker1)
        self.worker_pool.add_worker(self.worker2)
        
        self.worker_pool.remove_worker("worker1")
        self.assertEqual(len(self.worker_pool.workers), 1)
        self.assertNotIn("worker1", self.worker_pool.workers)
        
        # Removing a non-existent worker should raise ValueError
        with self.assertRaises(ValueError):
            self.worker_pool.remove_worker("nonexistent")
    
    def test_get_worker(self):
        """Test retrieving a worker by ID."""
        self.worker_pool.add_worker(self.worker1)
        
        worker = self.worker_pool.get_worker("worker1")
        self.assertEqual(worker, self.worker1)
        
        # Getting a non-existent worker should return None
        self.assertIsNone(self.worker_pool.get_worker("nonexistent"))
    
    def test_get_workers_by_capability(self):
        """Test filtering workers by capability."""
        self.worker_pool.add_worker(self.worker1)
        self.worker_pool.add_worker(self.worker2)
        self.worker_pool.add_worker(self.worker3)
        
        # Workers with "task_execution" capability
        task_execution_workers = self.worker_pool.get_workers_by_capability("task_execution")
        self.assertEqual(len(task_execution_workers), 2)
        self.assertIn(self.worker1, task_execution_workers)
        self.assertIn(self.worker2, task_execution_workers)
        
        # Workers with "api_integration" capability
        api_workers = self.worker_pool.get_workers_by_capability("api_integration")
        self.assertEqual(len(api_workers), 1)
        self.assertIn(self.worker3, api_workers)
        
        # No workers with "nonexistent" capability
        nonexistent_workers = self.worker_pool.get_workers_by_capability("nonexistent")
        self.assertEqual(len(nonexistent_workers), 0)
    
    def test_get_available_workers(self):
        """Test getting available workers."""
        self.worker_pool.add_worker(self.worker1)
        self.worker_pool.add_worker(self.worker2)
        self.worker_pool.add_worker(self.worker3)
        
        # All workers are available by default
        available_workers = self.worker_pool.get_available_workers()
        self.assertEqual(len(available_workers), 3)
        
        # Mark worker1 as offline
        self.worker1.status = WorkerStatus.OFFLINE
        available_workers = self.worker_pool.get_available_workers()
        self.assertEqual(len(available_workers), 2)
        self.assertNotIn(self.worker1, available_workers)
        
        # Mark worker2 as busy (but still available)
        self.worker2.active_tasks = self.worker2.max_concurrent_tasks
        available_workers = self.worker_pool.get_available_workers()
        self.assertEqual(len(available_workers), 1)
        self.assertNotIn(self.worker1, available_workers)
        self.assertNotIn(self.worker2, available_workers)
    
    def test_get_least_loaded_worker(self):
        """Test getting the least loaded worker."""
        self.worker_pool.add_worker(self.worker1)
        self.worker_pool.add_worker(self.worker2)
        self.worker_pool.add_worker(self.worker3)
        
        # Initially, all workers have 0 active tasks
        least_loaded = self.worker_pool.get_least_loaded_worker()
        self.assertIn(least_loaded, [self.worker1, self.worker2, self.worker3])
        
        # Add some tasks to worker1
        self.worker1.active_tasks = 2
        least_loaded = self.worker_pool.get_least_loaded_worker()
        self.assertIn(least_loaded, [self.worker2, self.worker3])
        
        # Add more tasks to all workers
        self.worker1.active_tasks = 2
        self.worker2.active_tasks = 1
        self.worker3.active_tasks = 3
        least_loaded = self.worker_pool.get_least_loaded_worker()
        self.assertEqual(least_loaded, self.worker2)
        
        # Mark worker2 as offline
        self.worker2.status = WorkerStatus.OFFLINE
        least_loaded = self.worker_pool.get_least_loaded_worker()
        self.assertEqual(least_loaded, self.worker1)
        
        # Mark all workers as busy
        self.worker1.active_tasks = self.worker1.max_concurrent_tasks
        self.worker3.active_tasks = self.worker3.max_concurrent_tasks
        least_loaded = self.worker_pool.get_least_loaded_worker()
        self.assertIsNone(least_loaded)


class TestOrchestrator(unittest.TestCase):
    """Tests for the Orchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the AgentProtocol
        self.protocol_mock = MagicMock(spec=AgentProtocol)
        
        # Create an orchestrator with the mock protocol
        self.orchestrator = Orchestrator(name="test-orchestrator", protocol=self.protocol_mock)
        
        # Add some test workers
        self.worker1 = Worker(id="worker1", endpoint="http://worker1:8000", capabilities=["task_execution"])
        self.worker2 = Worker(id="worker2", endpoint="http://worker2:8000", capabilities=["data_processing", "task_execution"])
        self.orchestrator.register_worker(self.worker1)
        self.orchestrator.register_worker(self.worker2)
        
        # Create some test tasks
        self.task1 = Task(id="task1", action="test_action", parameters={"param1": "value1"})
        self.task2 = Task(id="task2", action="test_action", parameters={"param2": "value2"})
        self.task3 = Task(id="task3", action="test_action", parameters={"param3": "value3"})
        
        # Add required capabilities
        self.task1.metadata["required_capabilities"] = ["task_execution"]
        self.task2.metadata["required_capabilities"] = ["data_processing"]
        self.task3.metadata["required_capabilities"] = ["api_integration"]
        
        # Create dependencies
        self.task3.metadata["dependencies"] = ["task1", "task2"]
    
    def test_register_worker(self):
        """Test registering a worker with the orchestrator."""
        worker3 = Worker(id="worker3", endpoint="http://worker3:8000", capabilities=["api_integration"])
        self.orchestrator.register_worker(worker3)
        
        self.assertIn("worker3", self.orchestrator.worker_pool.workers)
        
        # Registering a worker with the same ID should raise ValueError
        with self.assertRaises(ValueError):
            self.orchestrator.register_worker(worker3)
    
    def test_unregister_worker(self):
        """Test unregistering a worker from the orchestrator."""
        self.orchestrator.unregister_worker("worker1")
        
        self.assertNotIn("worker1", self.orchestrator.worker_pool.workers)
        
        # Unregistering a non-existent worker should raise ValueError
        with self.assertRaises(ValueError):
            self.orchestrator.unregister_worker("nonexistent")
    
    def test_submit_workflow(self):
        """Test submitting a workflow for execution."""
        # Create a workflow with two independent tasks
        workflow = Workflow(id="workflow1", tasks=[self.task1, self.task2])
        
        # Submit the workflow
        workflow_id = self.orchestrator.submit_workflow(workflow)
        
        self.assertEqual(workflow_id, "workflow1")
        self.assertIn("workflow1", self.orchestrator.active_workflows)
        
        # The protocol should be called to send messages for both tasks
        self.assertEqual(self.protocol_mock.send.call_count, 2)
        
        # Submitting a workflow with the same ID should raise OrchestratorError
        with self.assertRaises(OrchestratorError):
            self.orchestrator.submit_workflow(workflow)
    
    def test_workflow_dependencies(self):
        """Test workflow with task dependencies."""
        # Create a workflow with dependencies
        workflow = Workflow(id="workflow2", tasks=[self.task1, self.task2, self.task3])
        
        # Submit the workflow
        workflow_id = self.orchestrator.submit_workflow(workflow)
        
        # Only tasks 1 and 2 should be scheduled initially (task 3 has dependencies)
        self.assertEqual(self.protocol_mock.send.call_count, 2)
        
        # Complete task 1
        self.orchestrator.handle_task_completion("worker1", "task1", {"result": "success"})
        
        # Still waiting for task 2, so task 3 should not be scheduled yet
        self.assertEqual(self.protocol_mock.send.call_count, 2)
        
        # Complete task 2
        self.orchestrator.handle_task_completion("worker2", "task2", {"result": "success"})
        
        # Now task 3 should be scheduled
        self.assertEqual(self.protocol_mock.send.call_count, 3)
    
    def test_handle_task_failure(self):
        """Test handling task failure and retry logic."""
        # Create a workflow with a single task
        workflow = Workflow(id="workflow3", tasks=[self.task1])
        
        # Submit the workflow
        workflow_id = self.orchestrator.submit_workflow(workflow)
        
        # First attempt
        self.assertEqual(self.protocol_mock.send.call_count, 1)
        
        # Simulate task failure
        self.orchestrator.handle_task_failure("worker1", "task1", "Test error")
        
        # Should retry
        self.assertEqual(self.protocol_mock.send.call_count, 2)
        
        # Fail again
        self.orchestrator.handle_task_failure("worker1", "task1", "Test error")
        
        # Should retry again
        self.assertEqual(self.protocol_mock.send.call_count, 3)
        
        # Fail for the third time (exceeding max retries)
        self.orchestrator.handle_task_failure("worker1", "task1", "Test error")
        
        # Should not retry anymore
        self.assertEqual(self.protocol_mock.send.call_count, 3)
        
        # Workflow should be marked as failed
        self.assertEqual(workflow.status, "failed")
        self.assertIn("workflow3", self.orchestrator.failed_workflows)
        self.assertNotIn("workflow3", self.orchestrator.active_workflows)
    
    def test_cancel_workflow(self):
        """Test cancelling a workflow."""
        # Create a workflow with two tasks
        workflow = Workflow(id="workflow4", tasks=[self.task1, self.task2])
        
        # Submit the workflow
        workflow_id = self.orchestrator.submit_workflow(workflow)
        
        # Update task status to running
        self.task1.status = TaskStatus.RUNNING
        self.task1.assigned_worker = "worker1"
        
        # Cancel the workflow
        self.orchestrator.cancel_workflow(workflow_id)
        
        # Should send a cancel message for the running task
        self.protocol_mock.send.assert_called_with(unittest.mock.ANY)
        
        # Workflow should be marked as cancelled
        self.assertEqual(workflow.status, "cancelled")
        self.assertIn("workflow4", self.orchestrator.failed_workflows)
        self.assertNotIn("workflow4", self.orchestrator.active_workflows)
        
        # Cancelling a non-existent workflow should raise ValueError
        with self.assertRaises(ValueError):
            self.orchestrator.cancel_workflow("nonexistent")
    
    def test_get_workflow_status(self):
        """Test getting workflow status."""
        # Create a workflow with two tasks
        workflow = Workflow(id="workflow5", tasks=[self.task1, self.task2])
        
        # Submit the workflow
        workflow_id = self.orchestrator.submit_workflow(workflow)
        
        # Get workflow status
        status = self.orchestrator.get_workflow_status(workflow_id)
        
        self.assertEqual(status["workflow_id"], workflow_id)
        self.assertEqual(status["status"], workflow.status)
        self.assertEqual(len(status["tasks"]), 2)
        
        # Getting status for a non-existent workflow should raise ValueError
        with self.assertRaises(ValueError):
            self.orchestrator.get_workflow_status("nonexistent")
    
    def test_worker_selection(self):
        """Test worker selection strategies."""
        # Test round-robin selection
        self.orchestrator._select_worker = self.orchestrator._select_worker_round_robin
        
        worker = self.orchestrator._select_worker(self.task1)
        # Either worker1 or worker2 could be selected
        self.assertIn(worker.id, ["worker1", "worker2"])
        
        # Test capability-based selection
        self.orchestrator._select_worker = self.orchestrator._select_worker_capability_match
        
        # Task1 requires "task_execution" - both workers have it
        worker = self.orchestrator._select_worker(self.task1)
        self.assertIn(worker.id, ["worker1", "worker2"])
        
        # Task2 requires "data_processing" - only worker2 has it
        worker = self.orchestrator._select_worker(self.task2)
        self.assertEqual(worker.id, "worker2")
        
        # Task3 requires "api_integration" - no worker has it
        worker = self.orchestrator._select_worker(self.task3)
        self.assertIsNone(worker)
    
    def tearDown(self):
        """Clean up after tests."""
        self.orchestrator.shutdown()


if __name__ == "__main__":
    unittest.main()