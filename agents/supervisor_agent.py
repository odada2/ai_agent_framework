# ai_agent_framework/agents/supervisor_agent.py

"""
Supervisor Agent (Async Refactor)

Coordinates multiple specialized agents asynchronously using AgentProtocol
and handles async operations correctly. Uses a concrete AgentCommunicator.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Dict, List, Optional, Union, Set, Tuple, Any, Callable, Awaitable

# Framework components (Absolute Imports)
from ai_agent_framework.core.llm.base import BaseLLM
from ai_agent_framework.core.memory.conversation import ConversationMemory
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.core.tools.parser import ToolCallParser
# Use the concrete AgentCommunicator and AgentProtocol
from ai_agent_framework.core.communication.agent_communicator import AgentCommunicator
from ai_agent_framework.core.communication.agent_protocol import AgentMessage, AgentProtocol, CommunicationError, ProtocolError
from ai_agent_framework.agents.base_agent import BaseAgent
# Assuming Settings exists and might be used for configuration
from ai_agent_framework.config.settings import Settings
# Import specific exceptions
from ai_agent_framework.core.exceptions import AgentFrameworkError, AgentError

logger = logging.getLogger(__name__)

# --- Agent Definition ---

class SupervisorAgent(BaseAgent):
    """Async supervisor agent coordinating specialized agents via AgentProtocol."""

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        # Takes agent definitions (IDs and metadata/descriptions) instead of instances
        specialized_agents: Dict[str, Dict[str, Any]],
        protocol: Optional[AgentProtocol] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        task_timeout: Optional[float] = None,
        **kwargs # Allow BaseAgent kwargs
    ):
        """
        Initialize the async SupervisorAgent.

        Args:
            name: Unique name for the supervisor.
            llm: The language model instance.
            specialized_agents: Dictionary mapping agent IDs to their metadata
                                (e.g., {"analyzer_agent": {"description": "Analyzes data", "endpoint": "http..."}}).
            protocol: Optional shared AgentProtocol instance. If None, creates a new one.
            tools: Optional tools for the supervisor itself.
            memory: Optional conversation memory.
            system_prompt: Optional system prompt for the supervisor's LLM.
            max_iterations: Max coordination iterations per run.
            verbose: Enable detailed logging.
            task_timeout: Default timeout for delegated tasks.
            **kwargs: Additional base agent parameters.
        """
        system_prompt = system_prompt or (
            "You are a Supervisor Agent coordinating specialized agents. "
            "Your goal is to accomplish the user's main task by breaking it down "
            "and delegating subtasks to the most suitable specialized agent based on their description. "
            "Analyze the history of previous steps and results to decide the next best step. "
            "If an agent fails, decide whether to retry, delegate to another agent, or report failure."
            "Synthesize the final results from specialized agents into a coherent response for the user."
        )
        # BaseAgent init creates self.id
        super().__init__(name=name, llm=llm, tools=tools, memory=memory, system_prompt=system_prompt,
                         max_iterations=max_iterations, verbose=verbose, **kwargs)

        if not specialized_agents: raise ValueError("Supervisor requires specialized agent definitions.")
        # Store agent definitions (metadata, endpoint) rather than instances
        self.specialized_agents_definitions = specialized_agents

        self.task_timeout = task_timeout
        try: # Load relevant settings if Settings class is available
            settings = Settings()
            self.task_timeout = task_timeout or settings.get("supervisor.task_timeout", 60.0)
        except Exception: logger.warning(f"[{self.name}] Could not load settings, using default task timeout: {self.task_timeout or 60.0}s")
        self.task_timeout = self.task_timeout or 60.0 # Ensure a default value

        # Setup Communication using the concrete AgentCommunicator
        # If protocol is not provided, create a default one for the supervisor
        # It's often better if the protocol is shared and passed in.
        self.protocol = protocol or AgentProtocol(own_id=f"{self.name}_proto_{self.id}")
        # Use the concrete AgentCommunicator, passing self.id and the protocol instance
        self.communicator = AgentCommunicator(agent_id=self.id, protocol=self.protocol, default_timeout=self.task_timeout)

        # Register endpoints for specialized agents based on the provided definitions
        self._register_agent_endpoints()

        # Initialize agent health tracking based on the provided definitions
        self.agent_health = {aid: {"status": "ready", "last_active": time.monotonic()} for aid in specialized_agents}

        # Task tracking dictionaries remain the same
        self.active_tasks: Dict[str, Dict] = {} # task_id -> info from delegate_task
        self.completed_tasks: Dict[str, Dict] = {} # task_id -> info from delegate_task
        self.tool_call_parser = ToolCallParser()
        self._message_listener_task: Optional[asyncio.Task] = None # Listener task handle

    def _register_agent_endpoints(self):
         """Registers endpoints of specialized agents if available in definitions."""
         for agent_id, agent_def in self.specialized_agents_definitions.items():
              endpoint = agent_def.get('endpoint')
              if endpoint:
                   self.protocol.register_endpoint(agent_id, endpoint)
                   logger.info(f"[{self.name}] Registered endpoint for agent '{agent_id}': {endpoint}")
              else: logger.warning(f"[{self.name}] No endpoint found in definition for agent '{agent_id}'.")

    async def start_listening(self):
        """Start background message listener."""
        if self._message_listener_task is None or self._message_listener_task.done():
            self._message_listener_task = asyncio.create_task(self._listen_loop(), name=f"{self.name}-listener")
            logger.info(f"[{self.name}] Started message listener.")

    async def stop_listening(self):
         """Stop background message listener."""
         if self._message_listener_task and not self._message_listener_task.done():
              self._message_listener_task.cancel()
              try:
                  await asyncio.wait_for(self._message_listener_task, timeout=1.0)
              except (asyncio.CancelledError, asyncio.TimeoutError):
                   pass # Expected exceptions on cancellation/timeout
              self._message_listener_task = None
              logger.info(f"[{self.name}] Stopped message listener.")

    async def _listen_loop(self):
         """Background loop to process incoming messages via communicator."""
         logger.info(f"[{self.agent_id}] Starting listener loop...") # Use self.agent_id from communicator
         while True:
              try:
                   message = await self.communicator.receive(timeout=5.0) # Use communicator.receive
                   if message:
                        # Pass message to the processing logic
                        await self._process_message(message)
                   else:
                        await asyncio.sleep(0.1) # Yield if no message
              except asyncio.CancelledError:
                   logger.info(f"[{self.agent_id}] Listener loop cancelled.")
                   break
              except Exception as e:
                   logger.error(f"[{self.agent_id}] Error in listener loop: {e}", exc_info=True)
                   await asyncio.sleep(1) # Prevent busy-loop on error


    async def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """Asynchronously run the supervisor agent."""
        await self.reset() # Resets state, clears memory
        await self.start_listening() # Start listening for async responses

        task = input_data if isinstance(input_data, str) else input_data.get("input", str(input_data))
        # Initial context can include more than just the input if provided
        context = input_data if isinstance(input_data, dict) else {"input": task}
        context.setdefault("conversation_id", self.memory.id if hasattr(self.memory, 'id') else self.id) # Pass unique ID

        self.memory.add_user_message(task)
        run_state = {"task": task, "delegated_tasks_history": []} # Local state for this run
        start_time = time.monotonic()

        try:
             # Run the main delegation loop
             result = await self._run_delegation_loop(task, context, run_state, **kwargs)

             # Final synthesis step (using LLM if needed)
             final_response = await self._synthesize_final_response(task, result.get("delegations", []))
             self.memory.add_assistant_message(final_response)

             return {
                 "response": final_response,
                 "agent_delegations": result.get("delegations", []),
                 "iterations": self.current_iteration,
                 "duration_seconds": time.monotonic() - start_time,
                 "finished": True,
                 "success": result.get("success", True) # Success determined by loop
                 }

        except Exception as e:
             logger.exception(f"[{self.name}] Error during run: {e}")
             error_response = f"Supervisor encountered an error: {e}"
             self.memory.add_assistant_message(error_response)
             return {
                 "response": error_response,
                 "error": str(e),
                 "iterations": self.current_iteration,
                 "duration_seconds": time.monotonic() - start_time,
                 "finished": True,
                 "success": False
                 }
        finally:
            await self.stop_listening() # Stop listener when run is complete


    async def _run_delegation_loop(self, task: str, context: Dict, run_state: Dict, **kwargs) -> Dict:
        """Core loop for analyzing task, delegating, and handling results."""
        delegations = run_state["delegated_tasks_history"]
        current_task_description = task
        last_result_summary = "Initial task."

        for _ in range(self.max_iterations):
            if not self._increment_iteration():
                logger.warning(f"[{self.name}] Max iterations ({self.max_iterations}) reached for task: {task[:50]}...")
                return {"response": "Max iterations reached.", "success": False, "delegations": delegations}

            # Analyze current state and decide next step
            agent_id, subtask = await self._analyze_and_delegate_task(current_task_description, delegations, last_result_summary)

            if not agent_id or not subtask:
                logger.warning(f"[{self.name}] Could not determine next step or suitable agent. Ending.")
                # Decide if this is success or failure based on history
                success = len(delegations) > 0 and delegations[-1].get("status") == "completed"
                return {"response": last_result_summary, "success": success, "delegations": delegations}

            # Delegate the subtask
            delegation_context = {
                "original_task": task,
                "conversation_history": self.memory.get_conversation_history(format_type='list'), # Provide history
                **(context or {}) # Merge initial context
            }
            try:
                task_result_info = await self.delegate_task_to_agent(
                    agent_id,
                    subtask,
                    delegation_context,
                    wait_for_result=True, # Supervisor typically waits
                    timeout=self.task_timeout
                )
            except Exception as e:
                 logger.exception(f"[{self.name}] Exception during task delegation to {agent_id}")
                 task_result_info = {"status": "failed", "error": f"Delegation exception: {e}"}

            # Record the delegation attempt and its outcome
            delegation_record = {
                 "iteration": self.current_iteration,
                 "agent": agent_id,
                 "subtask": subtask,
                 "status": task_result_info.get("status"),
                 "result_preview": None, # Populate below
                 "error": task_result_info.get("error")
            }

            # Process result
            if task_result_info.get("status") == "completed":
                result_payload = task_result_info.get("result", {})
                # Assuming the agent's response is in result_payload['response'] or similar
                result_content = result_payload if isinstance(result_payload, str) else result_payload.get("response", str(result_payload))
                delegation_record["result_preview"] = str(result_content)[:200] + "..." # Store preview
                last_result_summary = f"Agent {agent_id} completed subtask '{subtask[:50]}...'. Result: {str(result_content)[:100]}..."

                # --- Check for completion ---
                # Simple check: Ask LLM if main task is done based on last result
                is_complete = await self._check_if_task_completed(task, delegations, last_result_summary)
                if is_complete:
                     logger.info(f"[{self.name}] Main task determined to be complete after delegation to {agent_id}.")
                     delegations.append(delegation_record) # Add final successful step
                     return {"success": True, "delegations": delegations}

            else:
                error_msg = task_result_info.get('error', 'Task failed/timed out.')
                delegation_record["error"] = error_msg
                logger.error(f"[{self.name}] Agent '{agent_id}' failed subtask '{subtask[:50]}...': {error_msg}")
                last_result_summary = f"Agent {agent_id} FAILED subtask '{subtask[:50]}...'. Error: {error_msg}"
                # --- Handle Failure ---
                # Option 1: Stop immediately
                # delegations.append(delegation_record)
                # return {"response": f"Task failed due to error from {agent_id}: {error_msg}", "success": False, "delegations": delegations}
                # Option 2: Allow LLM to decide next step (retry, different agent, give up) - implemented by continuing loop

            delegations.append(delegation_record)
            # Update task description for next analysis step (optional, could just use history)
            # current_task_description = f"Given the previous steps and result/error ({last_result_summary}), continue working on the main task: {task}"

        # Loop finished without meeting completion criteria
        logger.warning(f"[{self.name}] Reached end of loop without explicit task completion for: {task[:50]}...")
        return {"success": False, "delegations": delegations}


    async def _analyze_and_delegate_task(self, task_description: str, history: List[Dict], last_result_summary: str) -> Tuple[Optional[str], Optional[str]]:
         """Analyze task, considering history and last result, select agent, create subtask."""
         if not self._increment_iteration(): return None, None # Check iteration limit

         # Generate descriptions from agent definitions
         agent_descs_list = []
         for agent_id, agent_def in self.specialized_agents_definitions.items():
              description = agent_def.get("description", f"Agent {agent_id}")
              agent_descs_list.append(f"- {agent_id}: {description}")
         agent_descs = "\n".join(agent_descs_list)

         # Create history summary string
         history_str_list = []
         for i, d in enumerate(history):
              status = d['status']
              preview = f"Result preview: {d['result_preview']}" if d.get('result_preview') else f"Error: {d.get('error', 'Unknown')}"
              history_str_list.append(f" Step {d.get('iteration', i+1)}: Delegated '{d['subtask'][:50]}...' to {d['agent']} -> Status: {status}. {preview}")
         history_summary = "\n".join(history_str_list) if history_str_list else "No previous steps."

         prompt = (
             f"## Main Task:\n{task}\n\n"
             f"## Previous Steps History:\n{history_summary}\n\n"
             # f"## Last Result/Error Summary:\n{last_result_summary}\n\n" # Included in history
             f"## Available Specialized Agents:\n{agent_descs}\n\n"
             f"## Instruction:\n"
             f"Analyze the main task and the history. Decide the *single next best action* to progress towards completing the main task. "
             f"Choose the most suitable agent ID from the available list and formulate a *specific, actionable instruction* (subtask) for that agent. "
             f"If the previous step failed, consider if retrying (possibly with modifications) or delegating to a different agent is better. "
             f"If you determine the main task is now fully completed based on the history, respond with action 'complete'.\n\n"
             f"Respond ONLY in JSON format. Choose ONE of the following actions:\n"
             f"1. Delegate a subtask: {{\"action\": \"delegate\", \"agent_id\": \"<AGENT_ID>\", \"subtask\": \"<Specific instruction for the agent>\"}}\n"
             f"2. Complete the main task: {{\"action\": \"complete\"}}\n"
         )

         system_prompt = "You are a supervisor AI coordinating specialized agents. Analyze the task progress and decide the next delegation step or if the task is complete. Respond ONLY with the requested JSON structure."

         try:
              response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.1)
              content = response.get("content", "").strip()
              logger.debug(f"[{self.name}] LLM decision response: {content}")

              # Robust JSON extraction
              json_match = re.search(r'\{.*\}', content, re.DOTALL)
              if not json_match:
                  logger.error(f"[{self.name}] LLM did not provide valid JSON for decision. Response: {content}")
                  return None, None # Cannot proceed

              data = json.loads(json_match.group(0))

              action = data.get("action")

              if action == "delegate":
                  agent_id, subtask = data.get("agent_id"), data.get("subtask")
                  # Validate selected agent ID
                  if agent_id in self.specialized_agents_definitions and subtask:
                      logger.info(f"[{self.name}] LLM decided to delegate to '{agent_id}' subtask: {subtask[:50]}...")
                      return agent_id, subtask
                  else:
                      logger.warning(f"[{self.name}] LLM selected invalid agent ('{agent_id}') or subtask ('{subtask}').")
                      return None, None # Invalid delegation target
              elif action == "complete":
                   logger.info(f"[{self.name}] LLM determined the main task is complete.")
                   # Signal completion by returning (None, None) or a specific marker
                   return "COMPLETE", "COMPLETE" # Use special marker
              else:
                   logger.error(f"[{self.name}] LLM provided unknown action '{action}' in decision JSON.")
                   return None, None

         except json.JSONDecodeError:
              logger.error(f"[{self.name}] Failed to decode LLM decision JSON. Response: {content}")
              return None, None
         except Exception as e:
              logger.exception(f"[{self.name}] Error during LLM task analysis/delegation decision: {e}")
              return None, None

    async def _check_if_task_completed(self, main_task: str, history: List[Dict], last_result_summary: str) -> bool:
         """Uses LLM to determine if the main task is complete based on history."""
         if not history: return False # Cannot be complete without any steps

         history_str_list = []
         for i, d in enumerate(history):
              status = d['status']
              preview = f"Result preview: {d['result_preview']}" if d.get('result_preview') else f"Error: {d.get('error', 'Unknown')}"
              history_str_list.append(f" Step {d.get('iteration', i+1)}: Delegated '{d['subtask'][:50]}...' to {d['agent']} -> Status: {status}. {preview}")
         history_summary = "\n".join(history_str_list)

         prompt = (
              f"## Main Task:\n{main_task}\n\n"
              f"## Execution History:\n{history_summary}\n\n"
              # f"## Last Result Summary:\n{last_result_summary}\n\n" # Included in history
              f"## Question:\nBased *only* on the provided execution history, is the main task now fully completed? "
              f"Consider if the results address all aspects of the main task.\n\n"
              f"Respond ONLY with 'Yes' or 'No'."
         )
         system_prompt = "You are an evaluator determining if a task is complete based on its execution history. Respond ONLY 'Yes' or 'No'."

         try:
              response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.0, max_tokens=5)
              decision = response.get("content", "No").strip().lower()
              logger.debug(f"[{self.name}] LLM completion check decision: '{decision}'")
              return decision == 'yes'
         except Exception as e:
              logger.exception(f"[{self.name}] Error during LLM completion check: {e}")
              return False # Assume not complete if check fails


    async def _synthesize_final_response(self, main_task: str, history: List[Dict]) -> str:
         """Synthesize the final response for the user using LLM."""
         logger.info(f"[{self.name}] Synthesizing final response for task: {main_task[:50]}...")
         if not history:
              return "I was unable to complete the task as no steps were successfully executed."

         history_str_list = []
         final_result_data = "No specific final result captured."
         for i, d in enumerate(history):
              status = d['status']
              preview = f"Result: {d['result_preview']}" if d.get('result_preview') else f"Outcome: {d.get('error', 'No result/error logged')}"
              history_str_list.append(f" Step {d.get('iteration', i+1)}: Agent '{d['agent']}' performed '{d['subtask'][:50]}...' -> Status: {status}. {preview}")
              # Capture the last successful result as potential final data
              if status == "completed" and d.get("result_preview"):
                   final_result_data = f"The final step by {d['agent']} resulted in: {d['result_preview']}"


         history_summary = "\n".join(history_str_list)

         prompt = (
              f"## Original Task:\n{main_task}\n\n"
              f"## Summary of Actions Taken:\n{history_summary}\n\n"
              # f"## Final Result Data:\n{final_result_data}\n\n" # Included in history summary
              f"## Instruction:\nBased on the original task and the actions taken (including any errors), "
              f"synthesize a final, comprehensive response for the user. "
              f"Clearly state the outcome and provide the relevant information gathered or generated. "
              f"If the task failed, explain what happened."
         )

         try:
              response = await self.llm.generate(prompt=prompt, system_prompt=self.system_prompt, temperature=0.5)
              final_text = response.get("content", "Processing complete. Please review the steps taken.")
              logger.info(f"[{self.name}] Final response synthesized.")
              return final_text
         except Exception as e:
              logger.exception(f"[{self.name}] Error during final response synthesis: {e}")
              return f"An error occurred while summarizing the results. Please review the execution history:\n{history_summary}"


    # --- Methods below remain largely the same, ensure they use self.communicator ---

    async def delegate_task_to_agent(self, agent_id: str, task: str, context: Optional[Dict], wait_for_result: bool, timeout: Optional[float]) -> Dict:
        """Delegate task via communicator (async)."""
        # Check agent definition exists
        if agent_id not in self.specialized_agents_definitions:
             logger.error(f"[{self.name}] Attempted to delegate to unknown agent: {agent_id}")
             raise ValueError(f"Agent '{agent_id}' not found in supervisor's definitions.")

        effective_timeout = timeout if timeout is not None else self.task_timeout
        logger.info(f"[{self.name}] Delegating to {agent_id} (Timeout: {effective_timeout}s): {task[:50]}...")
        try:
            # Use the communicator to handle delegation
            result_info = await self.communicator.delegate_task(
                recipient_id=agent_id,
                task_description=task,
                context=context,
                wait_for_result=wait_for_result,
                timeout=effective_timeout
            )

            task_id = result_info.get("task_id", f"delegate_{uuid.uuid4()}") # Get or make an ID
            status = result_info.get("status", "unknown")

            # Update internal task tracking
            task_record = {
                "agent_id": agent_id,
                "task": task,
                "status": status,
                "submitted_at": time.monotonic(),
                **result_info # Include result/error from communicator's response
            }
            if status == "completed" or status == "failed" or status == "timeout":
                task_record["completed_at"] = time.monotonic()
                self.completed_tasks[task_id] = task_record
                if task_id in self.active_tasks: del self.active_tasks[task_id] # Clean up active
            elif status == "delegated":
                self.active_tasks[task_id] = task_record

            if status != "completed":
                 logger.warning(f"[{self.name}] Task {task_id} delegated to {agent_id} finished with status '{status}': {result_info.get('error', 'N/A')}")

            return result_info # Return the dictionary provided by the communicator

        except Exception as e:
             logger.exception(f"[{self.name}] Error delegating task via communicator to {agent_id}: {e}")
             # Return a standard error structure
             return {"status": "failed", "error": f"Supervisor delegation error: {e}"}


    async def _process_message(self, message: AgentMessage) -> None:
        """Process incoming messages via communicator's queue (async)."""
        sender = message.sender
        msg_type = message.message_type
        content = message.content
        task_id = message.correlation_id or message.metadata.get("task_id") # Prefer correlation_id for responses

        logger.info(f"[{self.name}] Processing msg type '{msg_type}' from {sender} (Task ID: {task_id})")

        # Update health on any message from a known agent
        if sender in self.agent_health:
            self.agent_health[sender]["last_active"] = time.monotonic()
            self.agent_health[sender]["status"] = "ready" # Assume ready on message

        # --- Handle messages related to active tasks ---
        if task_id and task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            task_info["last_update"] = time.monotonic() # Update timestamp

            if msg_type == "RESULT": # Check string name
                 logger.info(f"Received result for task {task_id} from {sender}")
                 task_info["result"] = content.get("result") # Assuming result is nested
                 task_info["status"] = "completed"
                 task_info["completed_at"] = time.monotonic()
                 self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                 # TODO: Potentially trigger next step based on this completion? Requires more state.
            elif msg_type == "ERROR": # Check string name
                 error_detail = content.get('error', 'Unknown error reported')
                 logger.error(f"Received error for task {task_id} from {sender}: {error_detail}")
                 task_info["error"] = error_detail
                 task_info["status"] = "failed"
                 task_info["completed_at"] = time.monotonic()
                 self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                 # TODO: Trigger error handling / retry logic? Requires more state.
            elif msg_type == "UPDATE": # Check string name
                 new_status = content.get('status', task_info["status"])
                 logger.info(f"Status update for task {task_id} from {sender}: '{new_status}'")
                 task_info["status"] = new_status
                 # Store update details if needed
                 task_info.setdefault("updates", []).append({"time": time.monotonic(), "status": new_status, "detail": content.get("detail")})
            else:
                 logger.warning(f"[{self.name}] Received unhandled message type '{msg_type}' for active task {task_id} from {sender}")

        # --- Handle general messages not tied to an active task ---
        elif msg_type == "QUERY":
             # Example: Respond to a query directed at the supervisor
             logger.info(f"[{self.name}] Received general query from {sender}: {content.get('query')}")
             # TODO: Implement supervisor's own query handling logic (e.g., using self.llm)
             response_content = {"response": "Supervisor received query, but handling is not fully implemented."}
             await self.communicator.send(sender, "RESPONSE", response_content, correlation_id=message.message_id)
        elif msg_type == "HEARTBEAT":
              logger.debug(f"[{self.name}] Received heartbeat from {sender}")
              # Health already updated above
        else:
             logger.warning(f"[{self.name}] Received message type '{msg_type}' from {sender}. No specific handler or active task match found (Task ID: {task_id}).")


    async def reset(self) -> None:
        """Reset supervisor state asynchronously."""
        logger.info(f"[{self.name}] Resetting supervisor state...")
        # Use await if BaseAgent.reset becomes async
        super().reset() # Resets iterations, memory, etc.
        self.active_tasks = {}
        self.completed_tasks = {}
        # Reset health status
        for agent_id in self.agent_health:
            self.agent_health[agent_id] = {"status": "ready", "last_active": time.monotonic()}
        # Clear communicator's receive queue
        while not self.communicator._receive_queue.empty():
            try:
                self.communicator._receive_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info(f"[{self.name}] Supervisor reset complete.")


    # --- Shutdown method ---
    # (Already updated in the __init__ modification section above)