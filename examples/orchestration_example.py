# ai_agent_framework/examples/orchestration_example.py

"""
Orchestration Examples (Async Verified)

Demonstrates using the asynchronous Orchestrator with a simulated worker environment.
Imports are absolute, uses asyncio primitives.
"""

import asyncio
import time
import json
import random
import logging
import traceback
import argparse
from typing import Dict, List, Any, Optional, Union, Callable, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field

# Framework components (using absolute imports)
from ai_agent_framework.core.workflow.orchestrator import Orchestrator
# Assuming WorkerPool, Worker, WorkerStatus are correctly located
from ai_agent_framework.core.workflow.worker_pool import WorkerPool, Worker, WorkerStatus
from ai_agent_framework.core.workflow.task import Task, TaskStatus
from ai_agent_framework.core.workflow.workflow import Workflow
from ai_agent_framework.core.communication.agent_protocol import AgentProtocol, AgentMessage
# Assuming exceptions are correctly located
from ai_agent_framework.core.exceptions import OrchestratorError, CommunicationError, ProtocolError
# Assuming Settings is correctly located
# from ai_agent_framework.config.settings import Settings # Not strictly needed for this example if Orchestrator loads its own

# Configure logging (basic setup for example)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MAX_WAIT_TIME = 25.0 # seconds
DEFAULT_CHECK_INTERVAL = 0.5 # seconds

# --- Mock Worker Environment (Async Refactor - Verified) ---
# (Keep MockWorkerInfo, MockEnvironmentState dataclasses as previously defined)
@dataclass
class MockWorkerInfo:
    capabilities: List[str]; endpoint: str; tasks_processed: int = 0; last_task: Optional[str] = None; max_concurrent_tasks: int = 1
@dataclass
class MockEnvironmentState:
    pipeline_stages: Dict = field(default_factory=dict); current_stage: Optional[str] = None; decision: Optional[str] = None; path_taken: Optional[str] = None; tasks_added_dynamically: List = field(default_factory=list); processing_result: Optional[Dict] = None; final_result: Optional[Dict] = None; processed_counts: Dict = field(default_factory=lambda: {"text": 0, "image": 0}); failed_counts: Dict = field(default_factory=lambda: {"text": 0, "image": 0}); aggregated: bool = False; aggregated_results: List = field(default_factory=list); events: List = field(default_factory=list); processed_event_data: List = field(default_factory=list); notifications: List = field(default_factory=list); current_workflow_id: Optional[str] = None

class MockWorkerEnvironment:
    """Simulates a worker environment for testing the async Orchestrator."""
    def __init__(self, name: str, failure_rate: float = 0.05):
        self.name = name; self.workers: Dict[str, MockWorkerInfo] = {}; self.task_handlers: Dict[str, Callable] = {}
        self.shared_state: MockEnvironmentState = MockEnvironmentState(); self.failure_rate = failure_rate
        self._orchestrator_ref: Optional[Orchestrator] = None; self._loop = None

    def register_mock_worker(self, worker_id: str, capabilities: List[str], endpoint: str = None, max_tasks: int = 1):
        resolved_endpoint = endpoint or f"http://mock-worker-{worker_id}:{8000 + len(self.workers)}"
        self.workers[worker_id] = MockWorkerInfo(capabilities, resolved_endpoint, max_concurrent_tasks=max_tasks)
        logger.debug(f"[{self.name}] Registered mock worker: {worker_id} caps {capabilities} at {resolved_endpoint}")

    def register_task_handler(self, task_action: str, handler: Callable):
        self.task_handlers[task_action] = handler
        logger.debug(f"[{self.name}] Registered handler for action: {task_action}")

    async def _ensure_loop(self):
         """Ensure event loop is available."""
         if not self._loop:
              self._loop = asyncio.get_running_loop()

    async def _simulate_task_execution(self, message: AgentMessage):
        """Asynchronously simulates worker executing a task."""
        await self._ensure_loop()
        if self._orchestrator_ref is None: logger.error(f"[{self.name}] Orchestrator ref missing!"); return

        task_content = message.content; task_id = task_content.get("id"); task_action = task_content.get("action")
        task_params = task_content.get("parameters", {}); worker_id = message.recipient

        if random.random() < self.failure_rate:
            error_msg = f"Simulated random failure on worker {worker_id}"; logger.warning(f"[{self.name}] {error_msg} for task {task_id}")
            asyncio.create_task(self._orchestrator_ref.handle_task_failure(worker_id, task_id, error_msg))
            if task_action == "proc_text": self.shared_state.failed_counts["text"] += 1 # Use correct action names
            if task_action == "proc_image": self.shared_state.failed_counts["image"] += 1
            return

        if worker_id in self.workers: self.workers[worker_id].tasks_processed += 1; self.workers[worker_id].last_task = task_id
        delay = random.uniform(0.05, 0.2); await asyncio.sleep(delay) # Shorter delay

        result_data = None; error_msg = None
        try:
            if task_action in self.task_handlers:
                handler = self.task_handlers[task_action]
                if asyncio.iscoroutinefunction(handler): result_data = await handler(task_id, task_params, worker_id, self.shared_state)
                else: result_data = handler(task_id, task_params, worker_id, self.shared_state) # Assuming sync handlers are fast
            else:
                logger.warning(f"[{self.name}] No handler for action '{task_action}'. Default success.")
                result_data = {"status": "success_default_handler", "processing_time": delay}
        except Exception as e: logger.error(f"[{self.name}] Handler error task {task_id}: {e}", exc_info=True); error_msg = f"Handler error: {e}"

        if error_msg: asyncio.create_task(self._orchestrator_ref.handle_task_failure(worker_id, task_id, error_msg))
        else: asyncio.create_task(self._orchestrator_ref.handle_task_completion(worker_id, task_id, result_data))

    async def _simulate_status_response(self, message: AgentMessage, protocol: AgentProtocol):
         worker_id = message.recipient; worker_info = self.workers.get(worker_id)
         response_content = {"status": "unknown"}
         if worker_info: response_content = {"status": "online", "active_tasks": 0, "processed_count": worker_info.tasks_processed, "capabilities": worker_info.capabilities}
         else: logger.warning(f"[{self.name}] Status req for unknown worker: {worker_id}")
         response_msg = AgentMessage(sender=worker_id, recipient=message.sender, content=response_content, message_type="status_response", correlation_id=message.message_id)
         protocol.process_received_data(response_msg.to_dict()) # Simulate receive

    def get_mock_protocol(self, orchestrator: Orchestrator) -> AgentProtocol:
        self._orchestrator_ref = orchestrator
        protocol = AgentProtocol(own_id=f"{self.name}_mock_proto")

        async def mock_send(message: AgentMessage, request_timeout: float = 5.0):
            logger.debug(f"[{self.name}] MockSend: To={message.recipient}, Type={message.message_type}, Task={message.content.get('id', '?')}")
            if message.recipient in self.workers:
                if message.message_type == "task_execute": asyncio.create_task(self._simulate_task_execution(message))
                elif message.message_type == "task_cancel": logger.info(f"[{self.name}] Simulating task cancel for {message.content.get('task_id')}")
                elif message.message_type == "status_request": asyncio.create_task(self._simulate_status_response(message, protocol))
                else: logger.warning(f"[{self.name}] MockSend unhandled type for worker: {message.message_type}")
            else:
                logger.error(f"[{self.name}] MockSend: Recipient '{message.recipient}' not found.")
                if message.message_type == "task_execute":
                     # Ensure failure handler is called async
                     asyncio.create_task(self._orchestrator_ref.handle_task_failure("orchestrator", message.content.get("id", "unknown"), f"Recipient '{message.recipient}' not found."))

        protocol.send = mock_send # Override protocol's send
        return protocol

    def get_stats(self) -> Dict[str, Any]:
        return {"workers": {w_id: {"tasks_processed": info.tasks_processed, "caps": info.capabilities} for w_id, info in self.workers.items()},
                "final_shared_state": self.shared_state}


# --- Async Wait Helper (Verified) ---
async def wait_for_workflow_completion(orchestrator: Orchestrator, workflow_id: str,
                                       description: str = "Workflow",
                                       max_wait_time: float = DEFAULT_MAX_WAIT_TIME,
                                       check_interval: float = DEFAULT_CHECK_INTERVAL) -> Dict[str, Any]:
    """Async helper to wait for a workflow to complete by polling."""
    logger.info(f"Waiting for {description} ('{workflow_id}') to complete...")
    start_time = time.monotonic()
    last_status = {}
    last_log_time = 0
    terminal_statuses = ["completed", "failed", "cancelled", "timeout", "completed_with_failures"]

    try:
        while time.monotonic() - start_time < max_wait_time:
            try:
                status = await orchestrator.get_workflow_status(workflow_id)
                last_status = status
                current_time = time.monotonic()
                wf_status = status.get("status")

                # Log status periodically
                if current_time - last_log_time > 5.0:
                     status_counts = {}
                     for task_status_info in status.get("tasks", []):
                          s = task_status_info.get("status", "unknown")
                          status_counts[s] = status_counts.get(s, 0) + 1
                     logger.info(f"[{description} {workflow_id}] Status: {wf_status}, Tasks: {status_counts}")
                     last_log_time = current_time

                if wf_status in terminal_statuses:
                    logger.info(f"{description} ('{workflow_id}') finished with status: {wf_status}")
                    return status

                await asyncio.sleep(check_interval)

            except ValueError: # ID not found yet
                 logger.warning(f"{description} ('{workflow_id}') not found yet, continuing wait...")
                 await asyncio.sleep(check_interval * 2)
            except Exception as e:
                 logger.error(f"Error polling status for {workflow_id}: {e}", exc_info=True)
                 await asyncio.sleep(check_interval * 2)

        logger.warning(f"{description} ('{workflow_id}') did not complete within {max_wait_time:.1f} seconds.")
        return last_status or {"workflow_id": workflow_id, "status": "timeout", "error": f"Timed out after {max_wait_time:.1f}s"}

    except asyncio.CancelledError:
        logger.warning(f"Wait for {description} ('{workflow_id}') cancelled.")
        return last_status or {"workflow_id": workflow_id, "status": "cancelled", "error": "Monitoring cancelled"}


# --- Async Example Definitions (Verified Imports/Async Calls) ---
# (Keep example functions mostly as previously defined, ensuring they use await correctly)
async def example_1_basic_workflow():
    logger.info("\n--- Example 1: Basic Workflow Orchestration (Async) ---")
    mock_env = MockWorkerEnvironment("basic-workflow-async", 0.0)
    mock_env.register_mock_worker("w-proc", ["data_processing"]); mock_env.register_mock_worker("w-exec", ["task_execution"]); mock_env.register_mock_worker("w-valid", ["data_validation"])
    mock_env.register_task_handler("process_data", lambda tid, p, wid, s: {"f": p.get("file"), "r": random.randint(100,500)})
    mock_env.register_task_handler("execute_logic", lambda tid, p, wid, s: {"fn": p.get("function"), "res": "ok"})
    mock_env.register_task_handler("validate_data", lambda tid, p, wid, s: {"s": p.get("schema"), "v": True})
    orchestrator = Orchestrator(name="basic-orch-async"); orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)
    for wid,winfo in mock_env.workers.items(): orchestrator.register_worker(Worker(id=wid, endpoint=winfo.endpoint, capabilities=winfo.capabilities))
    try:
        tasks = [Task(id=f"t{i+1}", action=a, parameters=p, metadata={"required_capabilities": rc}) for i,(a,p,rc) in enumerate([("process_data", {"file": "d1.csv"}, ["data_processing"]), ("process_data", {"file": "d2.csv"}, ["data_processing"]), ("execute_logic", {"function": "analyze"}, ["task_execution"]), ("validate_data", {"schema": "cust"}, ["data_validation"])])]
        workflow = Workflow(id="wf-basic-async", tasks=tasks)
        logger.info(f"Submitting {workflow.id}..."); wf_id = await orchestrator.submit_workflow(workflow)
        final_status = await wait_for_workflow_completion(orchestrator, wf_id, "Basic Async WF")
        logger.info(f"\nFinal Status ({wf_id}): {final_status.get('status', '?')}")
        for ts in final_status.get("tasks",[]): logger.info(f"  - {ts.get('id','?')} ({ts.get('action','?')}) @{ts.get('assigned_worker','?')}: {ts.get('status','?')}")
    finally: await orchestrator.shutdown(); logger.info("Example 1 finished.")

async def example_2_pipeline_pattern():
     logger.info("\n--- Example 2: Pipeline Pattern (Async) ---")
     mock_env = MockWorkerEnvironment("pipeline-pattern-async")
     mock_env.register_mock_worker("loader", ["data_loading"]); mock_env.register_mock_worker("processor", ["data_processing"]); mock_env.register_mock_worker("analyzer", ["data_analysis"]); mock_env.register_mock_worker("reporter", ["data_reporting"])
     def load(tid, p, wid, s: MockEnvironmentState): res={"s": p.get("source"), "r": random.randint(800,1200)}; s.pipeline_stages["loaded"]=res; s.current_stage="loaded"; return res
     def proc(tid, p, wid, s: MockEnvironmentState): ir=s.pipeline_stages.get("loaded",{}).get("r",0); vr=max(0,ir-random.randint(10,50)); res={"ir":ir,"vr":vr,"ops": p.get("ops")}; s.pipeline_stages["processed"]=res; s.current_stage="processed"; return res
     def analyze(tid, p, wid, s: MockEnvironmentState): vr=s.pipeline_stages.get("processed",{}).get("vr",0); ins=[f"I_{i}" for i in range(random.randint(2,5))]; res={"vr":vr,"ins":ins,"m": p.get("metrics")}; s.pipeline_stages["analyzed"]=res; s.current_stage="analyzed"; return res
     def report(tid, p, wid, s: MockEnvironmentState): ic=len(s.pipeline_stages.get("analyzed",{}).get("ins",[])); res={"ic":ic,"f": p.get("format")}; s.pipeline_stages["reported"]=res; s.current_stage="reported"; return res
     mock_env.register_task_handler("load", load); mock_env.register_task_handler("process", proc); mock_env.register_task_handler("analyze", analyze); mock_env.register_task_handler("report", report)
     orchestrator = Orchestrator(name="pipeline-orch-async"); orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)
     for wid,winfo in mock_env.workers.items(): orchestrator.register_worker(Worker(id=wid, endpoint=winfo.endpoint, capabilities=winfo.capabilities))
     try:
          t_load = Task(id="tL", action="load", parameters={"source": "db"}, metadata={"required_capabilities": ["data_loading"]})
          t_proc = Task(id="tP", action="process", parameters={"ops": ["clean"]}, metadata={"required_capabilities": ["data_processing"], "dependencies": ["tL"]})
          t_analyze = Task(id="tA", action="analyze", parameters={"metrics": ["corr"]}, metadata={"required_capabilities": ["data_analysis"], "dependencies": ["tP"]})
          t_report = Task(id="tR", action="report", parameters={"format": "pdf"}, metadata={"required_capabilities": ["data_reporting"], "dependencies": ["tA"]})
          workflow = Workflow(id="wf-pipe-async", tasks=[t_load, t_proc, t_analyze, t_report])
          logger.info(f"Submitting {workflow.id}..."); wf_id = await orchestrator.submit_workflow(workflow)
          final_status = await wait_for_workflow_completion(orchestrator, wf_id, "Pipeline Async WF")
          logger.info(f"\nFinal Status ({wf_id}): {final_status.get('status', '?')}")
          logger.info("Pipeline Stages:"); [logger.info(f"  - {st}: {str(dat)[:100]}...") for st,dat in mock_env.shared_state.pipeline_stages.items()]
     finally: await orchestrator.shutdown(); logger.info("Example 2 finished.")

# Keep other example functions (3, 4, 5) and main execution block as previously defined,
# ensuring they use `await` for orchestrator methods and `asyncio.sleep`.
async def example_3_dynamic_workflow():
     # ... (previous async implementation for example 3) ...
     logger.info("\n--- Example 3: Dynamic Workflow (Async Simulation) ---")
     mock_env = MockWorkerEnvironment("dynamic-workflow-async")
     mock_env.register_mock_worker("decider", ["decision_making"]); mock_env.register_mock_worker("proc-a", ["process_type_a"]); mock_env.register_mock_worker("proc-b", ["process_type_b"]); mock_env.register_mock_worker("finalizer", ["finalization"])
     def decide(tid, p, wid, s: MockEnvironmentState): decision = random.choice(["path_a", "path_b"]); s.decision = decision; logger.info(f"Task {tid}: Decided -> {decision}"); return {"decision": decision}
     def process_a(tid, p, wid, s: MockEnvironmentState): res = {"p_by": "A", "q": random.uniform(0.8, 1.0)}; s.path_taken = "path_a"; s.processing_result = res; return res
     def process_b(tid, p, wid, s: MockEnvironmentState): res = {"p_by": "B", "e": random.uniform(0.7, 0.95)}; s.path_taken = "path_b"; s.processing_result = res; return res
     def finalize(tid, p, wid, s: MockEnvironmentState): res = {"fin": True, "path": s.path_taken, "in_res": s.processing_result}; s.final_result = res; return res
     mock_env.register_task_handler("decide", decide); mock_env.register_task_handler("proc_a", process_a); mock_env.register_task_handler("proc_b", process_b); mock_env.register_task_handler("finalize", finalize)
     orchestrator = Orchestrator(name="dynamic-orch-async"); orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)
     for wid,winfo in mock_env.workers.items(): orchestrator.register_worker(Worker(id=wid, endpoint=winfo.endpoint, capabilities=winfo.capabilities))
     try:
          t_decide = Task(id="t-decide", action="decide", metadata={"required_capabilities": ["decision_making"]})
          workflow = Workflow(id="wf-dynamic-async", tasks=[t_decide])
          logger.info(f"Submitting {workflow.id}..."); wf_id = await orchestrator.submit_workflow(workflow); mock_env.shared_state.current_workflow_id = wf_id
          logger.info("Waiting for decision..."); decision_status = await wait_for_workflow_completion(orchestrator, wf_id, "Decision Task", max_wait_time=10)
          next_task_id = None; task_to_add = None
          if decision_status.get("status") == "completed":
               decision_task_info = next((t for t in decision_status.get('tasks', []) if t['id'] == 't-decide'), None)
               decision = decision_task_info.get('result_preview', {}).get('decision') if decision_task_info and decision_task_info.get('result_preview') else None; mock_env.shared_state.decision = decision
               logger.info(f"Decision: {decision}")
               if decision == "path_a": logger.info("Simulating add 'proc_a'..."); next_task_id="t-proc-a"; task_to_add=AgentMessage(sender=orchestrator.name, recipient="proc-a", content={"id":next_task_id, "action":"proc_a"}, message_type="task_execute")
               elif decision == "path_b": logger.info("Simulating add 'proc_b'..."); next_task_id="t-proc-b"; task_to_add=AgentMessage(sender=orchestrator.name, recipient="proc-b", content={"id":next_task_id, "action":"proc_b"}, message_type="task_execute")
               if task_to_add:
                    await orchestrator.protocol.send(task_to_add); mock_env.shared_state.tasks_added_dynamically.append(task_to_add.content["id"]); await asyncio.sleep(0.5)
                    logger.info("Simulating add 'finalize'..."); finalize_id="t-finalize"; finalize_msg=AgentMessage(sender=orchestrator.name, recipient="finalizer", content={"id":finalize_id, "action":"finalize"}, message_type="task_execute")
                    await orchestrator.protocol.send(finalize_msg); mock_env.shared_state.tasks_added_dynamically.append(finalize_id); await asyncio.sleep(0.5)
          else: logger.error(f"Decision task failed: {decision_status.get('status')}")
          logger.info("\nFinal Simulated State (Dynamic):"); logger.info(f" Decision: {mock_env.shared_state.decision}, Path: {mock_env.shared_state.path_taken}"); logger.info(f" Added Tasks: {mock_env.shared_state.tasks_added_dynamically}"); logger.info(f" Final Result: {str(mock_env.shared_state.final_result)[:100]}...")
     finally: await orchestrator.shutdown(); logger.info("Example 3 finished.")

async def example_4_distributed_processing():
     # ... (previous async implementation for example 4) ...
     logger.info("\n--- Example 4: Distributed Processing (Async) ---")
     mock_env = MockWorkerEnvironment("distrib-proc-async", 0.1)
     for i in range(3): mock_env.register_mock_worker(f"txt-w{i}", ["text_processing"])
     for i in range(2): mock_env.register_mock_worker(f"img-w{i}", ["image_processing"])
     mock_env.register_mock_worker("agg", ["result_aggregation"])
     def p_txt(tid, p, wid, s: MockEnvironmentState): res={"f": p.get("file"), "tok": random.randint(500,2000)}; s.processed_counts["text"]+=1; s.aggregated_results.append({"type":"text", "id":tid, **res}); return res
     def p_img(tid, p, wid, s: MockEnvironmentState): res={"f": p.get("file"), "obj": random.randint(0,10)}; s.processed_counts["image"]+=1; s.aggregated_results.append({"type":"image", "id":tid, **res}); return res
     def agg(tid, p, wid, s: MockEnvironmentState): nt=s.processed_counts["text"]; ni=s.processed_counts["image"]; nf_t=s.failed_counts["text"]; nf_i=s.failed_counts["image"]; summary={"tot":nt+ni,"txt":nt,"img":ni,"fail_t":nf_t,"fail_i":nf_i, "avg_tok":(sum(r['tok'] for r in s.aggregated_results if r['type']=='text')/max(1,nt)) if nt>0 else 0, "avg_obj":(sum(r['obj'] for r in s.aggregated_results if r['type']=='image')/max(1,ni)) if ni>0 else 0}; s.aggregated=True; s.final_result=summary; return summary
     mock_env.register_task_handler("proc_text", p_txt); mock_env.register_task_handler("proc_image", p_img); mock_env.register_task_handler("aggregate", agg)
     orchestrator = Orchestrator(name="distrib-orch-async"); orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)
     for wid,winfo in mock_env.workers.items(): orchestrator.register_worker(Worker(id=wid, endpoint=winfo.endpoint, capabilities=winfo.capabilities))
     try:
          tasks = []; task_ids = []
          for i in range(5): tid=f"txt-{i}"; tasks.append(Task(id=tid, action="proc_text", p={"file":f"d{i}.txt"}, meta={"required_capabilities":["text_processing"]})); task_ids.append(tid)
          for i in range(3): tid=f"img-{i}"; tasks.append(Task(id=tid, action="proc_image", p={"file":f"i{i}.jpg"}, meta={"required_capabilities":["image_processing"]})); task_ids.append(tid)
          t_agg = Task(id="t-agg", action="aggregate", meta={"required_capabilities":["result_aggregation"], "dependencies": task_ids})
          tasks.append(t_agg); workflow = Workflow(id="wf-distrib-async", tasks=tasks)
          logger.info(f"Submitting {workflow.id}..."); wf_id = await orchestrator.submit_workflow(workflow)
          final_status = await wait_for_workflow_completion(orchestrator, wf_id, "Distrib Async WF", max_wait_time=30)
          logger.info(f"\nFinal Status ({wf_id}): {final_status.get('status', '?')}")
          if mock_env.shared_state.aggregated: logger.info(f"Aggregated:\n{json.dumps(mock_env.shared_state.final_result, indent=2)}")
          else: logger.warning("Aggregation may be incomplete.")
     finally: await orchestrator.shutdown(); logger.info("Example 4 finished.")

async def example_5_event_driven_workflow():
     # ... (previous async implementation for example 5) ...
     logger.info("\n--- Example 5: Event-Driven Workflow (Async Simulation) ---")
     mock_env = MockWorkerEnvironment("event-driven-async")
     mock_env.register_mock_worker("listener", ["event_handling"]); mock_env.register_mock_worker("processor", ["processing"]); mock_env.register_mock_worker("notifier", ["notification"])
     def handle(tid, p, wid, s: MockEnvironmentState): et=p.get("event_type","?"); ed=p.get("event_data",{}); s.events.append({"type":et,"data":ed,"ts":time.time()}); logger.info(f"Event handler got: {et}"); return {"rec":et}
     def proc_ev(tid, p, wid, s: MockEnvironmentState): di=p.get("data_id","?"); res={"di":di,"st":"processed","w":wid}; s.processed_event_data.append(res); return res
     def notify(tid, p, wid, s: MockEnvironmentState): msg=p.get("message",""); res={"msg":msg,"st":"sent","chan":p.get("channel")}; s.notifications.append(res); return res
     mock_env.register_task_handler("handle_event", handle); mock_env.register_task_handler("process", proc_ev); mock_env.register_task_handler("notify", notify)
     orchestrator = Orchestrator(name="event-orch-async"); orchestrator.protocol = mock_env.get_mock_protocol(orchestrator)
     for wid,winfo in mock_env.workers.items(): orchestrator.register_worker(Worker(id=wid, endpoint=winfo.endpoint, capabilities=winfo.capabilities))
     try:
          t_listen=Task(id="t-listen-base", action="handle_event", p={"event_type":"START"}, meta={"required_capabilities":["event_handling"]})
          workflow = Workflow(id="wf-event-listener-async", tasks=[t_listen]); logger.info(f"Submitting {workflow.id}..."); wf_id=await orchestrator.submit_workflow(workflow)
          mock_env.shared_state.current_workflow_id = wf_id; await asyncio.sleep(0.1)
          logger.info("Simulating external events...")
          async def trigger(eid, et, ed, sa=None, sr=None, sp=None):
               logger.info(f"Simulating '{et}' event...")
               ev_msg = AgentMessage(sender="ext", recipient="listener", content={"id":eid, "action":"handle_event", "parameters":{"event_type":et, "event_data":ed}}, message_type="task_execute")
               await orchestrator.protocol.send(ev_msg); await asyncio.sleep(0.1)
               if sa and sr:
                    logger.info(f"Simulating dynamic task '{sa}'...")
                    st_id=f"{eid}-{sa}"; st_msg=AgentMessage(sender=orchestrator.name, recipient=sr, content={"id":st_id,"action":sa,"parameters":sp or {}}, message_type="task_execute")
                    await orchestrator.protocol.send(st_msg); mock_env.shared_state.tasks_added_dynamically.append(st_id); await asyncio.sleep(0.2)
          await trigger("ev1", "NEW_DATA", {"data_id":"abc"}, sa="process", sr="processor", sp={"data_id":"abc"})
          await trigger("ev2", "ALERT", {"level":"WARN"}, sa="notify", sr="notifier", sp={"channel":"log", "message":"WARN event occurred"})
          logger.info("\nEvent simulation finished.")
          logger.info("\nFinal Simulated State (Event):"); logger.info(f" Events: {len(mock_env.shared_state.events)}"); logger.info(f" Processed: {len(mock_env.shared_state.processed_event_data)}"); logger.info(f" Notified: {len(mock_env.shared_state.notifications)}")
     finally: await orchestrator.shutdown(); logger.info("Example 5 finished.")

# --- Async Main Runner (Verified) ---
async def main():
    parser = argparse.ArgumentParser(description="Run Async Orchestration Examples")
    parser.add_argument("--example", type=int, choices=range(1, 6), help="Run specific")
    parser.add_argument("--all", action="store_true", help="Run all")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level.upper())
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    run_all = args.all or not args.example
    example_map = {1: example_1_basic_workflow, 2: example_2_pipeline_pattern, 3: example_3_dynamic_workflow, 4: example_4_distributed_processing, 5: example_5_event_driven_workflow}

    try:
        if run_all:
            for i, func in enumerate(example_map.values()):
                 await func()
                 if i < len(example_map) - 1: logger.info("--- Delay 1s ---"); await asyncio.sleep(1.0)
            logger.info("\n--- All async examples finished ---")
        elif args.example in example_map:
            await example_map[args.example](); logger.info(f"\n--- Async example {args.example} finished ---")
    except asyncio.CancelledError: logger.info("Examples cancelled.")
    except Exception as e: logger.critical(f"Unexpected error running examples: {e}", exc_info=True)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("Interrupted by user.")