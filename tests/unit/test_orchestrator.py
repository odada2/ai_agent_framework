# ai_agent_framework/tests/unit/test_orchestrator.py

"""
Unit Tests for the Asynchronous Orchestrator.
Uses pytest and pytest-asyncio for testing async functionality.
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch, call

# Assuming components exist at these paths relative to 'ai_agent_framework' package root
from ai_agent_framework.core.workflow.orchestrator import Orchestrator
from ai_agent_framework.core.workflow.worker_pool import WorkerPool, Worker, WorkerStatus
from ai_agent_framework.core.workflow.task import Task, TaskStatus
from ai_agent_framework.core.workflow.workflow import Workflow
from ai_agent_framework.core.communication.agent_protocol import AgentProtocol, AgentMessage
from ai_agent_framework.core.exceptions import OrchestratorError, SchedulingError, CommunicationError # Import exceptions
# Assuming Settings and TelemetryTracker can be mocked or instantiated simply
from ai_agent_framework.config.settings import Settings
from ai_agent_framework.core.utils.telemetry import TelemetryTracker

# --- Test Fixtures ---

@pytest.fixture
def mock_settings():
    """Fixture for mock Settings."""
    settings = MagicMock(spec=Settings)
    settings.get.side_effect = lambda key, default=None: {
        "orchestration.max_concurrent_workflows": 10, "orchestration.max_retries": 3,
        "orchestration.retry_delay": 0.1, "orchestration.monitoring_interval": 1,
        "orchestration.priority_strategy": "fifo", "orchestration.worker_selection_strategy": "capability_match",
        "orchestration.workers": [], "orchestration.default_task_timeout": 30.0,
    }.get(key, default)
    return settings

@pytest.fixture
def mock_protocol():
    """Fixture for mock AgentProtocol with async send."""
    protocol = MagicMock(spec=AgentProtocol)
    protocol.send = AsyncMock()
    protocol.register_endpoint = MagicMock()
    protocol.shutdown = AsyncMock() # Mock shutdown
    return protocol

@pytest.fixture
def mock_worker_pool():
     """Fixture for mock WorkerPool with async methods."""
     pool = MagicMock(spec=WorkerPool)
     pool.workers = {}
     pool.add_worker = AsyncMock()
     pool.remove_worker = AsyncMock()
     pool.get_worker = AsyncMock(side_effect=lambda id: pool.workers.get(id)) # Simulate get
     pool.assign_task_to_worker = AsyncMock()
     pool.release_task_from_worker = AsyncMock()
     pool.get_available_workers = AsyncMock(return_value=[]) # Default to none available
     pool.get_next_available_worker = AsyncMock(return_value=None)
     pool.get_least_loaded_worker = AsyncMock(return_value=None)
     pool.get_best_capability_match_worker = AsyncMock(return_value=None)

     # Simulate add_worker modifying get_available_workers (basic)
     async def _add_worker_side_effect(worker):
         pool.workers[worker.id] = worker
         pool.get_available_workers.return_value = [w for w in pool.workers.values() if w.is_available()]
         # Simulate capability matching more closely based on the real implementation
         async def _get_best_match(caps=[]):
              available = await pool.get_available_workers(caps)
              if not available: return None
              candidates = []
              req_caps = set(caps or [])
              for w in available:
                   w_caps = set(w.capabilities)
                   extra = len(w_caps - req_caps)
                   candidates.append((extra, w.current_load, w))
              candidates.sort(key=lambda x: (x[0], x[1]))
              return candidates[0][2] if candidates else None
         pool.get_best_capability_match_worker.side_effect = _get_best_match
         # Simulate other selection methods if needed for specific tests

     pool.add_worker.side_effect = _add_worker_side_effect

     return pool

@pytest.fixture
def mock_telemetry():
     """Fixture for mock TelemetryTracker."""
     telemetry = MagicMock(spec=TelemetryTracker)
     telemetry.start_workflow = MagicMock()
     telemetry.end_workflow = MagicMock()
     telemetry.start_task = MagicMock()
     telemetry.end_task = MagicMock()
     telemetry.record_task_failure = MagicMock()
     telemetry.record_workflow_cancellation = MagicMock()
     telemetry.get_workflow_metrics = MagicMock(return_value={})
     return telemetry


@pytest.fixture
# Patch the classes within the orchestrator module where they are imported/used
@patch('ai_agent_framework.core.workflow.orchestrator.Settings')
@patch('ai_agent_framework.core.workflow.orchestrator.TelemetryTracker')
@patch('ai_agent_framework.core.workflow.orchestrator.WorkerPool')
@patch('ai_agent_framework.core.workflow.orchestrator.AgentProtocol')
async def orchestrator(MockAgentProtocol, MockWorkerPool, MockTelemetryTracker, MockSettings,
                       mock_settings, mock_protocol, mock_worker_pool, mock_telemetry):
    """Fixture to create an Orchestrator instance with all dependencies mocked."""
    MockSettings.return_value = mock_settings
    MockTelemetryTracker.return_value = mock_telemetry
    MockWorkerPool.return_value = mock_worker_pool
    MockAgentProtocol.return_value = mock_protocol

    orch = Orchestrator(name="test-orchestrator")
    # Override instances just in case patching didn't fully take inside init
    orch.settings = mock_settings
    orch.telemetry = mock_telemetry
    orch.worker_pool = mock_worker_pool
    orch.protocol = mock_protocol

    yield orch
    await orch.shutdown() # Ensure shutdown is awaited

# --- Test Cases (Verified Async) ---
# (Keep test logic largely the same as previous version, ensuring `await` is used for orchestrator methods)

@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator, mock_settings):
    assert orchestrator.name == "test-orchestrator"; assert orchestrator.max_retries == 3

@pytest.mark.asyncio
async def test_register_worker(orchestrator, mock_worker_pool, mock_protocol):
    worker = Worker(id="w1", endpoint="http://w1", capabilities=["test"], status=WorkerStatus.ONLINE)
    await orchestrator.register_worker(worker) # Use await
    mock_worker_pool.add_worker.assert_awaited_once_with(worker)
    mock_protocol.register_endpoint.assert_called_with("w1", "http://w1") # Sync method

@pytest.mark.asyncio
async def test_submit_workflow_success(orchestrator, mock_protocol, mock_worker_pool):
    worker = Worker(id="w1", endpoint="http://w1", capabilities=["exec"], status=WorkerStatus.ONLINE)
    await orchestrator.register_worker(worker)
    # Configure mock pool to return the worker for selection
    mock_worker_pool.get_best_capability_match_worker.return_value = worker

    task = Task(id="t1", action="do", meta={"required_capabilities": ["exec"]})
    workflow = Workflow(id="wf1", tasks=[task])
    wf_id = await orchestrator.submit_workflow(workflow)
    assert wf_id == "wf1"; assert "wf1" in orchestrator.active_workflows
    await asyncio.sleep(0.01) # Allow scheduling task to run
    mock_protocol.send.assert_awaited_once() # Check send was awaited
    sent_msg = mock_protocol.send.await_args[0][0]
    assert sent_msg.recipient == "w1"; assert sent_msg.content["id"] == "t1"
    assert task.status == TaskStatus.SCHEDULED

@pytest.mark.asyncio
async def test_handle_task_completion(orchestrator, mock_protocol, mock_worker_pool):
    w1=Worker(id="w1",endpoint="http://w1",caps=["c1"],status=WorkerStatus.ONLINE)
    w2=Worker(id="w2",endpoint="http://w2",caps=["c2"],status=WorkerStatus.ONLINE)
    await orchestrator.register_worker(w1); await orchestrator.register_worker(w2)
    mock_worker_pool.get_best_capability_match_worker.side_effect = lambda caps=[]: w1 if "c1" in caps else (w2 if "c2" in caps else None)

    t1 = Task(id="t1", action="a1", meta={"required_capabilities":["c1"]})
    t2 = Task(id="t2", action="a2", meta={"required_capabilities":["c2"], "dependencies":["t1"]})
    wf = Workflow(id="wf_comp", tasks=[t1, t2])
    await orchestrator.submit_workflow(wf); await asyncio.sleep(0.01)
    mock_protocol.send.assert_awaited_once() # t1 scheduled
    t1.status = TaskStatus.RUNNING; t1.assigned_worker = "w1" # Manually set status for test

    await orchestrator.handle_task_completion("w1", "t1", {"out": "res1"})
    assert t1.status == TaskStatus.COMPLETED
    mock_worker_pool.release_task_from_worker.assert_awaited_with("w1", "t1") # Check release

    await asyncio.sleep(0.01) # Allow t2 scheduling
    assert mock_protocol.send.await_count == 2
    sent_msg_t2 = mock_protocol.send.await_args_list[1][0][0]
    assert sent_msg_t2.recipient == "w2"; assert sent_msg_t2.content["id"] == "t2"
    assert t2.status == TaskStatus.SCHEDULED

@pytest.mark.asyncio
async def test_handle_task_failure_and_retry(orchestrator, mock_protocol, mock_worker_pool):
    worker = Worker(id="w1", endpoint="http://w1", caps=["exec"], status=WorkerStatus.ONLINE)
    await orchestrator.register_worker(worker)
    mock_worker_pool.get_best_capability_match_worker.return_value = worker
    orchestrator.max_retries = 1; orchestrator.retry_delay = 0.05

    task = Task(id="t1", action="a1", meta={"required_capabilities":["exec"]})
    workflow = Workflow(id="wf_fail", tasks=[task])
    await orchestrator.submit_workflow(workflow); await asyncio.sleep(0.01)
    task.status = TaskStatus.RUNNING; task.assigned_worker = "w1"

    await orchestrator.handle_task_failure("w1", "t1", "Fail 1")
    assert task.status == TaskStatus.PENDING; assert task.metadata.get("retry_count") == 1
    mock_worker_pool.release_task_from_worker.assert_awaited_with("w1", "t1")

    await asyncio.sleep(orchestrator.retry_delay + 0.05) # Wait for retry schedule
    assert mock_protocol.send.await_count == 2
    assert task.status == TaskStatus.SCHEDULED # Rescheduled

    task.status = TaskStatus.RUNNING # Assume retry started
    await orchestrator.handle_task_failure("w1", "t1", "Fail 2")
    assert task.status == TaskStatus.FAILED; assert task.error == "Fail 2"
    async with orchestrator._workflows_lock: assert workflow.id in orchestrator.failed_workflows


# Add other tests (cancel, completion states etc.) similarly using await and async mocks