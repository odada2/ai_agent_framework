"""
Supervisor Agent

This module provides an implementation of a Supervisor Agent that coordinates
multiple specialized agents to accomplish complex tasks.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, Set

from ..core.llm.base import BaseLLM
from ..core.memory.conversation import ConversationMemory
from ..core.tools.registry import ToolRegistry
from ..core.tools.parser import ToolCallParser
from ..core.communication.agent_protocol import (
    AgentCommunicator, Message, MessageType, global_message_queue
)
from ..core.workflow.orchestrator import OrchestratorWorkflow
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """
    A supervisor agent that coordinates multiple specialized agents.
    
    The SupervisorAgent is responsible for:
    - Breaking down complex tasks into subtasks
    - Delegating subtasks to appropriate specialized agents
    - Coordinating communication between agents
    - Synthesizing results into coherent responses
    - Monitoring agent performance and health
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        specialized_agents: Dict[str, BaseAgent],
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        parallel_execution: bool = True,
        max_parallel_agents: int = 3,
        **kwargs
    ):
        """
        Initialize the SupervisorAgent.
        
        Args:
            name: A unique name for this agent instance
            llm: The LLM implementation to use for this agent
            specialized_agents: Dictionary of specialized agents to coordinate
            tools: Optional registry of tools available to the supervisor
            memory: Optional conversation memory for maintaining context
            system_prompt: Optional system prompt to guide the agent's behavior
            max_iterations: Maximum number of iterations the agent can perform
            verbose: Whether to log detailed information about the agent's operations
            parallel_execution: Whether to allow parallel execution of subtasks
            max_parallel_agents: Maximum number of agents to run in parallel
            **kwargs: Additional agent-specific parameters
        """
        # Create default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are a Supervisor Agent responsible for coordinating multiple specialized "
                "agents to solve complex tasks. Your job is to break down tasks into subtasks, "
                "delegate them to appropriate agents, and synthesize their results into "
                "coherent responses. Always delegate tasks to the most qualified agent "
                "and monitor their progress carefully."
            )
        
        super().__init__(
            name=name,
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
            **kwargs
        )
        
        # Set up specialized agents
        self.specialized_agents = specialized_agents
        self.parallel_execution = parallel_execution
        self.max_parallel_agents = max_parallel_agents
        
        # Set up communication
        self.communicator = AgentCommunicator(agent_id=self.id)
        self.agent_health = {agent_id: {"status": "ready", "last_active": time.time()} 
                            for agent_id in specialized_agents.keys()}
        
        # Set up orchestrator (for internal use)
        self._setup_orchestrator()
        
        # Track delegated tasks
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # Set up tool call parser
        self.tool_call_parser = ToolCallParser()
    
    def _setup_orchestrator(self) -> None:
        """Set up an internal orchestrator workflow for task coordination."""
        # Convert agents to worker callables for the orchestrator
        workers = {}
        
        for agent_id, agent in self.specialized_agents.items():
            # Create worker function that delegates to the agent
            async def worker_func(input_data, agent=agent):
                task = input_data.get("input", "")
                context = input_data.get("context", {})
                
                # Execute the agent with the given task
                result = await agent.run(
                    input_data={"input": task, "context": context}
                )
                
                return result
            
            # Set docstring to help with task delegation
            worker_func.__doc__ = f"Agent: {agent.name} - {getattr(agent, 'description', '')}"
            
            # Add to workers dictionary
            workers[agent_id] = worker_func
        
        # Create orchestrator workflow
        self.orchestrator = OrchestratorWorkflow(
            name=f"{self.name}_orchestrator",
            llm=self.llm,
            workers=workers,
            tools=self.tools,
            max_steps=self.max_iterations,
            parallel=self.parallel_execution,
            max_parallel_workers=self.max_parallel_agents,
            system_prompt=self.system_prompt,
            verbose=self.verbose
        )
    
    async def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """
        Run the supervisor agent on the given input.
        
        This method processes the input, plans the approach, delegates tasks to
        specialized agents, and returns the result.
        
        Args:
            input_data: The input data for the agent to process (string query or structured data)
            **kwargs: Additional runtime parameters
            
        Returns:
            A dictionary containing the agent's response and any additional metadata
        """
        # Reset for a new run
        self.reset()
        
        # Extract task
        task = input_data if isinstance(input_data, str) else input_data.get("input", str(input_data))
        context = input_data if isinstance(input_data, dict) else {"input": input_data}
        
        # Add task to memory
        self.memory.add_user_message(task)
        
        # Initialize state for this run
        self.state = {
            "task": task,
            "start_time": time.time(),
            "delegated_tasks": [],
            "active_agents": set()
        }
        
        try:
            # Determine execution strategy based on task complexity
            if kwargs.get("use_orchestrator", True) and self._is_complex_task(task):
                # Use orchestrator for complex tasks
                if self.verbose:
                    logger.info(f"Using orchestrator for complex task: {task[:50]}...")
                
                result = await self._run_with_orchestrator(task, context, **kwargs)
            else:
                # Use direct delegation for simpler tasks
                if self.verbose:
                    logger.info(f"Using direct delegation for task: {task[:50]}...")
                
                result = await self._run_with_direct_delegation(task, context, **kwargs)
            
            # Add response to memory
            self.memory.add_assistant_message(result.get("response", ""))
            
            return result
        
        except Exception as e:
            logger.exception(f"Error running supervisor agent: {str(e)}")
            error_response = f"I encountered an error while coordinating my team: {str(e)}"
            self.memory.add_assistant_message(error_response)
            
            return {
                "response": error_response,
                "error": str(e),
                "iterations": self.current_iteration,
                "finished": True,
                "success": False
            }
    
    async def _run_with_orchestrator(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a complex task using the orchestrator workflow.
        
        Args:
            task: Task description
            context: Task context
            **kwargs: Additional parameters
            
        Returns:
            Execution result
        """
        # Execute the orchestrator
        orchestrator_result = await self.orchestrator.execute(
            input_data={"input": task, "context": context}
        )
        
        # Extract and format the final result
        final_result = orchestrator_result.get("final_result", "")
        
        # Include which agents contributed
        agent_contributions = []
        for subtask in orchestrator_result.get("subtasks", []):
            agent_id = subtask.get("worker")
            description = subtask.get("description")
            agent_contributions.append(f"- {agent_id}: {description}")
        
        agent_summary = ""
        if agent_contributions:
            agent_summary = "\n\nTask delegation summary:\n" + "\n".join(agent_contributions)
        
        # Build the complete response
        response = final_result
        
        # Only append agent summary in verbose mode or if explicitly requested
        if self.verbose or kwargs.get("show_delegation_summary", False):
            response += agent_summary
        
        return {
            "response": response,
            "orchestrator_result": orchestrator_result,
            "iterations": self.current_iteration,
            "finished": True,
            "success": True
        }
    
    async def _run_with_direct_delegation(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a task using direct delegation to specialized agents.
        
        Args:
            task: Task description
            context: Task context
            **kwargs: Additional parameters
            
        Returns:
            Execution result
        """
        # Analyze task to determine appropriate agent
        agent_id, subtask = await self._analyze_and_delegate_task(task)
        
        if not agent_id or agent_id not in self.specialized_agents:
            # If no suitable agent found, handle with LLM directly
            logger.warning(f"No suitable agent found for task: {task[:50]}...")
            
            response = await self.llm.generate(
                prompt=f"I need to answer this without specialized help: {task}",
                system_prompt=self.system_prompt
            )
            
            return {
                "response": response.get("content", "I'm unable to find a suitable agent for this task."),
                "iterations": self.current_iteration,
                "finished": True,
                "success": True
            }
        
        # Get the selected agent
        agent = self.specialized_agents[agent_id]
        
        # Execute the agent
        if self.verbose:
            logger.info(f"Delegating task to agent {agent_id}: {subtask[:50]}...")
        
        agent_result = await agent.run(
            input_data={"input": subtask, "context": context}
        )
        
        # Get agent's response
        agent_response = agent_result.get("response", "")
        
        # Format as coming from the supervisor
        response = (
            f"{agent_response}\n\n"
            f"(This response was provided with assistance from my {agent_id} agent.)"
        ).strip()
        
        return {
            "response": response,
            "agent_id": agent_id,
            "agent_result": agent_result,
            "iterations": self.current_iteration,
            "finished": True,
            "success": True
        }
    
    async def _analyze_and_delegate_task(self, task: str) -> Tuple[Optional[str], str]:
        """
        Analyze a task and determine the most suitable agent.
        
        Args:
            task: Task description
            
        Returns:
            Tuple of (agent_id, reformulated_task) or (None, original_task)
        """
        # Increment iteration counter
        if not self._increment_iteration():
            return None, task
        
        # Construct agent descriptions for the prompt
        agent_descriptions = []
        for agent_id, agent in self.specialized_agents.items():
            agent_desc = getattr(agent, "description", f"Agent {agent_id}")
            agent_descriptions.append(f"- {agent_id}: {agent_desc}")
        
        agent_details = "\n".join(agent_descriptions)
        
        analysis_prompt = (
            f"Task: {task}\n\n"
            f"Available specialized agents:\n{agent_details}\n\n"
            f"Analyze this task and determine which agent would be most appropriate "
            f"to handle it. Then reformulate the task as a clear instruction for that agent.\n\n"
            f"Respond in the following format:\n"
            f"Agent: [agent_id]\n"
            f"Task: [reformulated_task]\n"
            f"Reasoning: [brief explanation]"
        )
        
        # Get LLM response
        response = await self.llm.generate(
            prompt=analysis_prompt,
            system_prompt="You are a task analysis assistant that helps determine which specialized agent should handle a task."
        )
        
        # Parse the response to extract agent and reformulated task
        response_text = response.get("content", "")
        
        agent_match = re.search(r"Agent:\s*(\w+)", response_text)
        task_match = re.search(r"Task:\s*(.*?)(?:\n\s*Reasoning:|$)", response_text, re.DOTALL)
        
        if agent_match and task_match:
            agent_id = agent_match.group(1).strip()
            reformulated_task = task_match.group(1).strip()
            
            # Verify agent exists
            if agent_id in self.specialized_agents:
                return agent_id, reformulated_task
        
        # Fallback: Return None and original task
        logger.warning("Failed to parse agent delegation from LLM response")
        return None, task
    
    def _is_complex_task(self, task: str) -> bool:
        """
        Determine if a task is complex enough to warrant full orchestration.
        
        Args:
            task: Task description
            
        Returns:
            True if the task appears complex
        """
        # Simple heuristics for now - could be replaced with more sophisticated analysis
        complexity_indicators = [
            "multiple", "several", "complex", "comprehensive", 
            "analyze", "compare", "investigate", "research",
            "and", "&", "+", "then", "followed by"
        ]
        
        # Count words and check for complexity indicators
        word_count = len(task.split())
        has_indicators = any(indicator in task.lower() for indicator in complexity_indicators)
        
        return word_count > 20 or has_indicators
    
    async def listen_for_messages(self, timeout: Optional[float] = None) -> None:
        """
        Listen for and process incoming agent messages.
        
        Args:
            timeout: Maximum time to listen, or None for indefinite
        """
        start_time = time.time()
        
        while True:
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                break
            
            # Wait for a message with a short timeout
            message = await self.communicator.receive(timeout=1.0)
            
            if message is None:
                # No message received, check timeout and continue
                continue
            
            # Process the message based on type
            await self._process_message(message)
    
    async def _process_message(self, message: Message) -> None:
        """
        Process an incoming message from another agent.
        
        Args:
            message: The message to process
        """
        sender = message.sender
        
        # Update agent health status
        if sender in self.agent_health:
            self.agent_health[sender]["status"] = "active"
            self.agent_health[sender]["last_active"] = time.time()
        
        # Process based on message type
        if message.message_type == MessageType.QUERY:
            # Agent is asking a question
            await self._handle_query(message)
            
        elif message.message_type == MessageType.UPDATE:
            # Agent is providing a status update
            await self._handle_update(message)
            
        elif message.message_type == MessageType.RESULT:
            # Agent has completed a task
            await self._handle_result(message)
            
        elif message.message_type == MessageType.ERROR:
            # Agent encountered an error
            await self._handle_error(message)
    
    async def _handle_query(self, message: Message) -> None:
        """
        Handle a query message from an agent.
        
        Args:
            message: The query message
        """
        query = message.content
        sender = message.sender
        
        if self.verbose:
            logger.info(f"Received query from {sender}: {query[:50]}...")
        
        # Generate response using LLM
        response_text = await self.llm.generate(
            prompt=f"Agent {sender} has asked: {query}\n\nProvide a helpful response."
        )
        
        # Send response
        await self.communicator.send(
            content=response_text.get("content", "I'm not sure how to answer that."),
            message_type=MessageType.RESPONSE,
            receiver=sender,
            reference_id=message.id
        )
    
    async def _handle_update(self, message: Message) -> None:
        """
        Handle a status update message from an agent.
        
        Args:
            message: The update message
        """
        update = message.content
        sender = message.sender
        
        if self.verbose:
            logger.info(f"Received update from {sender}: {update[:50]}...")
        
        # Check if this is for a specific task
        task_id = message.metadata.get("task_id")
        if task_id and task_id in self.active_tasks:
            # Update task status
            self.active_tasks[task_id]["status"] = update
            self.active_tasks[task_id]["last_update"] = time.time()
        
        # Send acknowledgment
        await self.communicator.send(
            content="Update received",
            message_type=MessageType.CONFIRMATION,
            receiver=sender,
            reference_id=message.id
        )
    
    async def _handle_result(self, message: Message) -> None:
        """
        Handle a task result message from an agent.
        
        Args:
            message: The result message
        """
        result = message.content
        sender = message.sender
        
        if self.verbose:
            logger.info(f"Received result from {sender}: {result[:50]}...")
        
        # Check if this is for a specific task
        task_id = message.metadata.get("task_id")
        if task_id and task_id in self.active_tasks:
            # Move task from active to completed
            task_data = self.active_tasks.pop(task_id)
            task_data["result"] = result
            task_data["completed_at"] = time.time()
            self.completed_tasks[task_id] = task_data
        
        # Send acknowledgment
        await self.communicator.send(
            content="Result received",
            message_type=MessageType.CONFIRMATION,
            receiver=sender,
            reference_id=message.id
        )
    
    async def _handle_error(self, message: Message) -> None:
        """
        Handle an error message from an agent.
        
        Args:
            message: The error message
        """
        error = message.content
        sender = message.sender
        
        logger.error(f"Received error from {sender}: {error}")
        
        # Check if this is for a specific task
        task_id = message.metadata.get("task_id")
        if task_id and task_id in self.active_tasks:
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = error
            self.active_tasks[task_id]["last_update"] = time.time()
        
        # Update agent health status
        if sender in self.agent_health:
            self.agent_health[sender]["status"] = "error"
            self.agent_health[sender]["last_error"] = error
            self.agent_health[sender]["last_error_time"] = time.time()
        
        # Send acknowledgment with guidance if needed
        await self.communicator.send(
            content="Error received. Please try to recover or notify if you need assistance.",
            message_type=MessageType.CONFIRMATION,
            receiver=sender,
            reference_id=message.id
        )
    
    async def delegate_task_to_agent(
        self,
        agent_id: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        wait_for_result: bool = True,
        timeout: Optional[float] = 60.0
    ) -> Dict[str, Any]:
        """
        Delegate a task to a specific agent.
        
        Args:
            agent_id: ID of the agent to delegate to
            task: Task description
            context: Additional context for the task
            wait_for_result: Whether to wait for task completion
            timeout: Maximum time to wait for result
            
        Returns:
            Task result or status
        """
        # Verify agent exists
        if agent_id not in self.specialized_agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        # Create a task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Record in active tasks
        self.active_tasks[task_id] = {
            "agent_id": agent_id,
            "task": task,
            "context": context or {},
            "status": "delegated",
            "created_at": time.time(),
            "last_update": time.time()
        }
        
        if self.verbose:
            logger.info(f"Delegating task {task_id} to agent {agent_id}: {task[:50]}...")
        
        # Update active agents set
        self.state["active_agents"].add(agent_id)
        
        # Delegate through communication protocol if agent has communicator
        agent = self.specialized_agents[agent_id]
        if hasattr(agent, 'communicator'):
            # Delegate using agent communication
            message_result = await self.communicator.delegate_task(
                receiver=agent_id,
                task=task,
                context=context,
                wait_for_result=wait_for_result,
                timeout=timeout
            )
            
            if wait_for_result:
                # Task should now be in completed_tasks
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
                else:
                    # Something went wrong
                    return {
                        "status": "failed",
                        "error": "Task completion not properly recorded",
                        "message_result": message_result
                    }
            else:
                return {"status": "delegated", "task_id": task_id}
        else:
            # Direct execution for agents without communication support
            try:
                # Execute agent
                result = await agent.run(
                    input_data={"input": task, "context": context or {}}
                )
                
                # Update task status
                if wait_for_result:
                    # Move to completed tasks
                    self.active_tasks[task_id]["status"] = "completed"
                    self.active_tasks[task_id]["result"] = result
                    self.active_tasks[task_id]["completed_at"] = time.time()
                    self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                    
                    return self.completed_tasks[task_id]
                else:
                    # Just return delegation status
                    return {"status": "delegated", "task_id": task_id}
                    
            except Exception as e:
                # Update task status with error
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
                self.active_tasks[task_id]["last_update"] = time.time()
                
                logger.error(f"Error executing agent {agent_id}: {str(e)}")
                raise
    
    async def check_agent_health(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Check the health status of agents.
        
        Args:
            agent_id: Specific agent to check or None for all agents
            
        Returns:
            Dictionary with health status information
        """
        if agent_id is not None:
            # Check specific agent
            if agent_id not in self.agent_health:
                return {"agent_id": agent_id, "status": "unknown"}
            
            return {
                "agent_id": agent_id,
                **self.agent_health[agent_id]
            }
        else:
            # Check all agents
            health_data = {}
            for agent_id, health in self.agent_health.items():
                # Calculate time since last activity
                time_since_active = time.time() - health["last_active"]
                
                # Determine status based on time since last activity
                status = health["status"]
                if status == "active" and time_since_active > 300:  # 5 minutes
                    status = "idle"
                elif time_since_active > 3600:  # 1 hour
                    status = "inactive"
                
                health_data[agent_id] = {
                    **health,
                    "status": status,
                    "time_since_active": time_since_active
                }
            
            return {
                "agents": health_data,
                "active_count": sum(1 for h in health_data.values() if h["status"] == "active"),
                "idle_count": sum(1 for h in health_data.values() if h["status"] == "idle"),
                "error_count": sum(1 for h in health_data.values() if h["status"] == "error"),
                "total_count": len(health_data)
            }
    
    async def revive_agent(self, agent_id: str) -> bool:
        """
        Attempt to revive an agent that's in an error state.
        
        Args:
            agent_id: ID of the agent to revive
            
        Returns:
            True if revival was successful
        """
        if agent_id not in self.specialized_agents:
            logger.warning(f"Cannot revive unknown agent: {agent_id}")
            return False
        
        if agent_id not in self.agent_health or self.agent_health[agent_id]["status"] != "error":
            logger.info(f"Agent {agent_id} doesn't need revival")
            return True
        
        try:
            # Get the agent
            agent = self.specialized_agents[agent_id]
            
            # Reset the agent
            if hasattr(agent, 'reset') and callable(agent.reset):
                agent.reset()
            
            # Update health status
            self.agent_health[agent_id]["status"] = "ready"
            self.agent_health[agent_id]["last_active"] = time.time()
            
            # Send revival message if agent has communication
            if hasattr(agent, 'communicator'):
                await self.communicator.send(
                    content="You are being reactivated after an error. Please acknowledge.",
                    message_type=MessageType.INSTRUCTION,
                    receiver=agent_id
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to revive agent {agent_id}: {str(e)}")
            return False
    
    async def reset(self) -> None:
        """Reset the supervisor agent's state."""
        super().reset()
        
        # Clear task tracking
        self.active_tasks = {}
        
        # Reset orchestrator
        if hasattr(self.orchestrator, 'reset') and callable(self.orchestrator.reset):
            self.orchestrator.reset()
        
        # Reset agent health monitoring
        for agent_id in self.agent_health:
            self.agent_health[agent_id]["status"] = "ready"
            self.agent_health[agent_id]["last_active"] = time.time()
    
    async def shutdown(self) -> None:
        """Shut down the supervisor agent and all managed agents."""
        # Broadcast shutdown message
        await self.communicator.broadcast(
            content="The supervisor is shutting down. Please complete current tasks and prepare for shutdown.",
            message_type=MessageType.INSTRUCTION
        )
        
        # Short wait for agents to process the message
        await asyncio.sleep(1)
        
        # Unregister from message queue
        await self.communicator.unregister()
        
        # Shutdown specialized agents that support it
        for agent_id, agent in self.specialized_agents.items():
            if hasattr(agent, 'shutdown') and callable(agent.shutdown):
                try:
                    await agent.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down agent {agent_id}: {str(e)}")
        
        logger.info(f"Supervisor agent {self.name} shut down successfully")