# ai_agent_framework/agents/supervisor_agent.py

"""
Supervisor Agent (Async Refactor)

Coordinates multiple specialized agents asynchronously using AgentProtocol
and handles async operations correctly. Includes a basic placeholder
for the missing AgentCommunicator.
"""

import asyncio
import json
import logging
import re
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Set, Tuple, Any, Callable, Awaitable

# Framework components (Absolute Imports)
from ai_agent_framework.core.llm.base import BaseLLM
from ai_agent_framework.core.memory.conversation import ConversationMemory
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.core.tools.parser import ToolCallParser
# Use the refactored AgentProtocol and AgentMessage
from ai_agent_framework.core.communication.agent_protocol import AgentMessage, AgentProtocol, CommunicationError, ProtocolError
# Assuming BaseAgent is defined
from ai_agent_framework.agents.base_agent import BaseAgent
# Assuming Settings exists
from ai_agent_framework.config.settings import Settings
# Import specific exceptions
from ai_agent_framework.core.exceptions import AgentFrameworkError, AgentError

logger = logging.getLogger(__name__)

# --- MessageType Enum (Define if not provided elsewhere) ---
class MessageType(Enum):
    QUERY = auto(); RESPONSE = auto(); UPDATE = auto(); RESULT = auto()
    ERROR = auto(); CONFIRMATION = auto(); INSTRUCTION = auto()
    DELEGATE_TASK = auto(); TASK_STATUS = auto(); HEARTBEAT = auto()
    # Add other necessary types


# --- Placeholder AgentCommunicator (Async) ---
# Needs actual implementation based on requirements
class AgentCommunicator:
    """Async Placeholder for AgentCommunicator using AgentProtocol."""
    def __init__(self, agent_id: str, protocol: AgentProtocol):
        self.agent_id = agent_id
        self.protocol = protocol
        self._receive_queue = asyncio.Queue() # Internal queue for received messages
        self._listener_task: Optional[asyncio.Task] = None
        logger.info(f"[{self.agent_id}] Initialized placeholder AgentCommunicator.")
        # Register handler with protocol to put messages into queue
        self.protocol.register_handler(MessageType.QUERY.name, self._queue_message)
        self.protocol.register_handler(MessageType.RESPONSE.name, self._queue_message)
        self.protocol.register_handler(MessageType.UPDATE.name, self._queue_message)
        self.protocol.register_handler(MessageType.RESULT.name, self._queue_message)
        self.protocol.register_handler(MessageType.ERROR.name, self._queue_message)
        self.protocol.register_handler(MessageType.CONFIRMATION.name, self._queue_message)
        self.protocol.register_handler(MessageType.INSTRUCTION.name, self._queue_message)
        self.protocol.register_handler(MessageType.DELEGATE_TASK.name, self._queue_message)
        # Add more handlers as needed

    async def _queue_message(self, message: AgentMessage):
         """Generic handler to put incoming messages onto the internal queue."""
         await self._receive_queue.put(message)

    async def send(self, content: Any, message_type: MessageType, receiver: str, reference_id: Optional[str] = None, metadata: Optional[Dict] = None):
        content_dict = content if isinstance(content, dict) else {"data": content}
        msg = AgentMessage(sender=self.agent_id, recipient=receiver, content=content_dict,
                           message_type=message_type.name, correlation_id=reference_id, metadata=metadata or {})
        await self.protocol.send(msg)

    async def receive(self, timeout: float = 1.0) -> Optional[AgentMessage]:
         """Receive a message from the internal queue."""
         try:
              return await asyncio.wait_for(self._receive_queue.get(), timeout=timeout)
         except asyncio.TimeoutError:
              return None
         except Exception as e:
              logger.error(f"[{self.agent_id}] Error receiving message: {e}")
              return None

    async def broadcast(self, content: Any, message_type: MessageType, metadata: Optional[Dict] = None):
        # Ensure content is dict
        content_dict = content if isinstance(content, dict) else {"data": content}
        # Get recipients from protocol (requires protocol to expose endpoints)
        recipients = list(self.protocol.endpoints.keys()) if hasattr(self.protocol, 'endpoints') else []
        tasks = []
        for agent_id in recipients:
            if agent_id != self.agent_id:
                 msg = AgentMessage(sender=self.agent_id, recipient=agent_id, content=content_dict,
                                    message_type=message_type.name, metadata=metadata or {})
                 tasks.append(self.protocol.send(msg)) # Collect send coroutines
        if tasks:
             results = await asyncio.gather(*tasks, return_exceptions=True)
             for i, res in enumerate(results):
                  if isinstance(res, Exception): logger.error(f"[{self.agent_id}] Error broadcasting to recipient {i}: {res}")

    async def delegate_task(self, receiver: str, task: str, context: Optional[Dict], wait_for_result: bool, timeout: Optional[float]) -> Dict:
        task_content = {"task": task, "context": context or {}}
        msg_type = MessageType.DELEGATE_TASK.name
        msg_meta = {"wait_for_result": wait_for_result}
        msg = AgentMessage(sender=self.agent_id, recipient=receiver, content=task_content, message_type=msg_type, metadata=msg_meta)

        if wait_for_result:
            effective_timeout = timeout or 60.0
            try:
                response_msg = await self.protocol.send_and_receive(msg, response_timeout=effective_timeout)
                if response_msg:
                     # Check if the response indicates success or failure based on its content/type
                     if response_msg.message_type == MessageType.RESULT.name:
                           return {"status": "completed", "result": response_msg.content, "task_id": msg.message_id}
                     elif response_msg.message_type == MessageType.ERROR.name:
                           return {"status": "failed", "error": response_msg.content.get("error", "Agent reported error"), "task_id": msg.message_id}
                     else: # Unexpected response type
                           return {"status": "failed", "error": f"Unexpected response type: {response_msg.message_type}", "task_id": msg.message_id}
                else: # Should not happen if send_and_receive doesn't timeout
                      return {"status": "failed", "error": "No response object received", "task_id": msg.message_id}
            except asyncio.TimeoutError:
                return {"status": "timeout", "error": f"Timeout waiting for task result ({effective_timeout}s)", "task_id": msg.message_id}
            except (CommunicationError, ProtocolError) as e:
                return {"status": "failed", "error": f"Communication error: {e}", "task_id": msg.message_id}
            except Exception as e:
                 logger.exception(f"Unexpected error in delegate_task (wait): {e}")
                 return {"status": "failed", "error": f"Unexpected error: {e}", "task_id": msg.message_id}
        else:
            await self.protocol.send(msg)
            return {"status": "delegated", "task_id": msg.message_id}

    async def register(self): logger.info(f"[{self.agent_id}] Communicator registered (placeholder).") # Placeholder
    async def unregister(self): logger.info(f"[{self.agent_id}] Communicator unregistered (placeholder).") # Placeholder
    async def shutdown(self): await self.protocol.shutdown()
# --- End Placeholder ---


class SupervisorAgent(BaseAgent):
    """Async supervisor agent coordinating specialized agents via AgentProtocol."""

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        specialized_agents: Dict[str, BaseAgent],
        protocol: Optional[AgentProtocol] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        task_timeout: Optional[float] = None,
        **kwargs # Allow BaseAgent kwargs
    ):
        """Initialize the async SupervisorAgent."""
        system_prompt = system_prompt or (
            "You are a Supervisor Agent coordinating specialized agents. "
            "Break down tasks, delegate to the best agent, synthesize results."
        )
        super().__init__(name=name, llm=llm, tools=tools, memory=memory, system_prompt=system_prompt,
                         max_iterations=max_iterations, verbose=verbose, **kwargs)

        if not specialized_agents: raise ValueError("Supervisor requires specialized agents.")
        self.specialized_agents = specialized_agents
        # Note: Parallel execution depends on how delegation/orchestration is implemented below
        # self.parallel_execution = parallel_execution
        # self.max_parallel_agents = max_parallel_agents

        self.task_timeout = task_timeout
        try: # Load relevant settings if Settings class is available
            settings = Settings()
            self.task_timeout = task_timeout or settings.get("api.agents.task_timeout", 60.0) # Example setting path
        except Exception: logger.warning(f"[{self.name}] Could not load settings, using default task timeout: {self.task_timeout or 60.0}s")
        self.task_timeout = self.task_timeout or 60.0 # Ensure a default value

        # Setup Communication (using async protocol and placeholder communicator)
        self.protocol = protocol or AgentProtocol(own_id=f"{self.name}_proto")
        self.communicator = AgentCommunicator(agent_id=self.id, protocol=self.protocol)
        self._register_agent_endpoints()

        self.agent_health = {aid: {"status": "ready", "last_active": time.monotonic()} for aid in specialized_agents}
        self.active_tasks: Dict[str, Dict] = {} # task_id -> info
        self.completed_tasks: Dict[str, Dict] = {} # task_id -> info
        self.tool_call_parser = ToolCallParser()
        self._message_listener_task: Optional[asyncio.Task] = None

    def _register_agent_endpoints(self):
         """Registers endpoints of specialized agents if available."""
         for agent_id, agent in self.specialized_agents.items():
              endpoint = getattr(agent, 'endpoint', agent.config.get('endpoint')) # Check instance then config
              if endpoint: self.protocol.register_endpoint(agent_id, endpoint)
              else: logger.warning(f"[{self.name}] No endpoint for agent '{agent_id}'.")

    async def start_listening(self):
        """Start background message listener."""
        if self._message_listener_task is None or self._message_listener_task.done():
            self._message_listener_task = asyncio.create_task(self._listen_loop(), name=f"{self.name}-listener")
            logger.info(f"[{self.name}] Started message listener.")

    async def stop_listening(self):
         """Stop background message listener."""
         if self._message_listener_task and not self._message_listener_task.done():
              self._message_listener_task.cancel(); await asyncio.sleep(0) # Allow cancellation
              try: await self._message_listener_task
              except asyncio.CancelledError: pass
              self._message_listener_task = None; logger.info(f"[{self.name}] Stopped message listener.")

    async def _listen_loop(self):
         """Background loop to process incoming messages via communicator."""
         while True:
              try:
                   # Use communicator's receive with a timeout
                   message = await self.communicator.receive(timeout=5.0)
                   if message:
                        await self._process_message(message)
                   else:
                        # No message, yield control
                        await asyncio.sleep(0.1)
              except asyncio.CancelledError: logger.info(f"[{self.name}] Listener loop cancelled."); break
              except Exception as e: logger.error(f"[{self.name}] Error in listener loop: {e}", exc_info=True); await asyncio.sleep(1) # Avoid busy loop on error


    async def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """Asynchronously run the supervisor agent."""
        await self.reset(); await self.start_listening()
        task = input_data if isinstance(input_data, str) else input_data.get("input", str(input_data))
        context = input_data if isinstance(input_data, dict) else {"input": input_data}
        context.setdefault("conversation_id", getattr(self.memory, 'conversation_id', self.id)) # Pass unique ID

        self.memory.add_user_message(task)
        run_state = {"task": task, "delegated_tasks": []} # Local state for this run
        start_time = time.monotonic()

        try:
             # Simplified: Always use direct delegation (async)
             result = await self._run_with_direct_delegation(task, context, run_state, **kwargs)
             final_response = result.get("response", "Task processing completed.")
             self.memory.add_assistant_message(final_response)
             return {"response": final_response, "agent_delegations": result.get("delegations", []),
                     "iterations": self.current_iteration, "duration_seconds": time.monotonic() - start_time,
                     "finished": True, "success": result.get("success", True)}
        except Exception as e:
             logger.exception(f"[{self.name}] Error during run: {e}")
             error_response = f"Supervisor error: {e}"; self.memory.add_assistant_message(error_response)
             return {"response": error_response, "error": str(e), "iterations": self.current_iteration,
                     "duration_seconds": time.monotonic() - start_time, "finished": True, "success": False}


    async def _run_with_direct_delegation(self, task: str, context: Dict, run_state: Dict, **kwargs) -> Dict:
        """Execute task via async direct delegation loop."""
        delegations = run_state["delegated_tasks"]
        current_task_description = task

        for _ in range(self.max_iterations):
             if not self._increment_iteration(): logger.warning(f"[{self.name}] Max iterations reached."); break

             agent_id, subtask = await self._analyze_and_delegate_task(current_task_description, delegations)
             if not agent_id: logger.warning(f"[{self.name}] No suitable agent found. Attempting direct response."); break # Stop if no agent found

             delegation_context = {**context, "history": [d.get("result") for d in delegations if d.get("result")]} # Pass results history
             try:
                  task_result_info = await self.delegate_task_to_agent(agent_id, subtask, delegation_context, True, self.task_timeout)
             except Exception as e: task_result_info = {"status": "failed", "error": f"Delegation exception: {e}"}

             delegation_record = {"agent": agent_id, "task": subtask, "status": task_result_info.get("status"),
                                  "result": task_result_info.get("result", {}).get("response") if task_result_info.get("status") == "completed" else None,
                                  "error": task_result_info.get("error")}
             delegations.append(delegation_record)

             if task_result_info.get("status") != "completed":
                  error_msg = task_result_info.get('error', 'Task failed/timed out.'); logger.error(f"[{self.name}] Agent '{agent_id}' failed: {error_msg}")
                  final_response = f"Sub-task failed by {agent_id}: {error_msg}"
                  return {"response": final_response, "success": False, "delegations": delegations}
             else:
                  # Simple: Assume first success completes task. More complex logic needed for multi-step.
                  result_content = task_result_info.get("result", {}).get("response", "Task completed by agent.")
                  final_response = result_content + (f"\n\n(Handled by: {agent_id})" if self.verbose else "")
                  return {"response": final_response, "success": True, "delegations": delegations}

        # If loop finished or broke early without success
        llm_resp = await self.llm.generate(prompt=f"Task failed after attempts. Summarize the situation based on these steps: {json.dumps(delegations)}", system_prompt=self.system_prompt)
        return {"response": llm_resp.get("content", "Unable to complete the task after multiple steps."), "success": False, "delegations": delegations}


    async def _analyze_and_delegate_task(self, task_description: str, history: List[Dict]) -> Tuple[Optional[str], str]:
         """Analyze task, considering history, and select agent using LLM (async)."""
         if not self._increment_iteration(): return None, task_description # Check iteration limit
         agent_descs = "\n".join([f"- {id}: {getattr(a,'description',id)}" for id, a in self.specialized_agents.items()])
         history_summary = "\n".join([f" - {d['agent']}: {d['task'][:50]}... -> Status: {d['status']}" for d in history]) if history else "None"

         prompt = (f"Previous Steps Summary:\n{history_summary}\n\nRemaining Task: {task_description}\n\n"
                   f"Available agents:\n{agent_descs}\n\nAnalyze the remaining task and history. "
                   f"Select the single best agent ID and formulate the specific next subtask instruction for it.\n\n"
                   f"Respond ONLY in JSON: {{\"agent_id\": \"AGENT_ID\", \"subtask\": \"SUBTASK_INSTRUCTION\"}}")
         system_prompt = "You route tasks to specialized agents. Output ONLY the JSON requested."

         try:
              response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.1)
              content = response.get("content", "").strip()
              json_match = re.search(r'\{.*\}', content, re.DOTALL); data = json.loads(json_match.group(0)) if json_match else None
              agent_id, subtask = data.get("agent_id"), data.get("subtask")
              if agent_id in self.specialized_agents and subtask:
                   logger.info(f"[{self.name}] LLM selected '{agent_id}' for subtask: {subtask[:50]}...")
                   return agent_id, subtask
              else: logger.warning(f"[{self.name}] LLM selected invalid agent/subtask: {data}")
         except Exception as e: logger.exception(f"[{self.name}] Error during LLM task analysis: {e}")
         return None, task_description


    async def delegate_task_to_agent(self, agent_id: str, task: str, context: Optional[Dict], wait_for_result: bool, timeout: Optional[float]) -> Dict:
        """Delegate task via communicator (async)."""
        if agent_id not in self.specialized_agents: raise ValueError(f"Agent {agent_id} not found")
        effective_timeout = timeout if timeout is not None else self.task_timeout
        logger.info(f"[{self.name}] Delegating to {agent_id} (Timeout:{effective_timeout}s): {task[:50]}...")
        try:
            result_info = await self.communicator.delegate_task(receiver=agent_id, task=task, context=context, wait_for_result=wait_for_result, timeout=effective_timeout)
            task_id = result_info.get("task_id", f"delegate_{uuid.uuid4()}") # Get or make an ID
            status = result_info.get("status", "unknown")
            task_record = {"agent_id": agent_id, "task": task, "status": status, **result_info}
            if status == "completed" or status == "failed" or status == "timeout": self.completed_tasks[task_id] = task_record
            elif status == "delegated": self.active_tasks[task_id] = task_record
            if status != "completed": logger.warning(f"Task {task_id} for {agent_id} finished with status {status}: {result_info.get('error')}")
            return result_info
        except Exception as e: logger.exception(f"[{self.name}] Error delegating task via communicator: {e}"); return {"status": "failed", "error": f"Supervisor delegation error: {e}"}


    async def _process_message(self, message: AgentMessage) -> None:
        """Process incoming messages via communicator's queue (async)."""
        sender = message.sender; msg_type = message.message_type; content = message.content
        logger.info(f"[{self.name}] Processing msg type {msg_type} from {sender}")
        # Update health on any message
        if sender in self.agent_health: self.agent_health[sender]["last_active"] = time.monotonic()

        task_id = message.correlation_id or message.metadata.get("task_id")
        if task_id and task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            if msg_type == MessageType.RESULT.name:
                 logger.info(f"Received result for task {task_id} from {sender}")
                 task_info["result"] = content; task_info["status"] = "completed"; task_info["completed_at"] = time.monotonic()
                 self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
            elif msg_type == MessageType.ERROR.name:
                 error_detail = content.get('error', 'Unknown error reported')
                 logger.error(f"Received error for task {task_id} from {sender}: {error_detail}")
                 task_info["error"] = error_detail; task_info["status"] = "failed"; task_info["completed_at"] = time.monotonic()
                 self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
            elif msg_type == MessageType.UPDATE.name:
                 logger.info(f"Status update for task {task_id} from {sender}: {content.get('status')}")
                 task_info["status"] = content.get('status', task_info["status"])
                 task_info["last_update"] = time.monotonic()
            # Handle other message types if needed (e.g., QUERY requires response)
        else:
             logger.warning(f"[{self.name}] Received message type {msg_type} from {sender} for unknown/inactive task ID: {task_id}")
             # Handle general queries/messages not tied to a task if necessary


    async def reset(self) -> None:
        """Reset supervisor state asynchronously."""
        logger.info(f"[{self.name}] Resetting supervisor state...")
        # Use await if BaseAgent.reset becomes async
        super().reset() # Assuming BaseAgent.reset is sync
        self.active_tasks = {}; self.completed_tasks = {}
        for agent_id in self.agent_health: self.agent_health[agent_id] = {"status": "ready", "last_active": time.monotonic()}
        # Optionally stop/clear communicator state if needed
        # await self.communicator.reset()

    async def shutdown(self) -> None:
        """Shutdown supervisor and communicator asynchronously."""
        logger.info(f"[{self.name}] Shutting down...")
        await self.stop_listening()
        await self.communicator.shutdown() # Shutdown protocol via communicator
        logger.info(f"[{self.name}] Shutdown complete.")