"""
Agent Communication Protocol

This module provides a communication protocol for agents to exchange
messages and data, enabling sophisticated agent-to-agent interactions.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    QUERY = "query"               # Request for information
    RESPONSE = "response"         # Response to a query
    INSTRUCTION = "instruction"   # Instruction to perform a task
    UPDATE = "update"             # Status update
    RESULT = "result"             # Task result
    ERROR = "error"               # Error message
    FEEDBACK = "feedback"         # Feedback on performance
    DELEGATION = "delegation"     # Task delegation
    CONFIRMATION = "confirmation" # Confirmation of receipt
    BROADCAST = "broadcast"       # Message to all agents


class Message:
    """
    A message exchanged between agents in the communication protocol.
    
    Messages have a standardized format that includes metadata, content,
    and optional attachments. This enables agents to properly interpret
    and respond to messages from other agents.
    """
    
    def __init__(
        self,
        content: str,
        message_type: Union[MessageType, str],
        sender: str,
        receiver: Optional[str] = None,
        reference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize a message.
        
        Args:
            content: Main content of the message
            message_type: Type of message
            sender: ID of sending agent
            receiver: ID of receiving agent (None for broadcasts)
            reference_id: Optional reference to another message
            metadata: Additional message metadata
            attachments: Optional data attachments
        """
        self.id = str(uuid.uuid4())
        
        if isinstance(message_type, str):
            try:
                self.message_type = MessageType(message_type)
            except ValueError:
                logger.warning(f"Unknown message type: {message_type}, using QUERY")
                self.message_type = MessageType.QUERY
        else:
            self.message_type = message_type
            
        self.content = content
        self.sender = sender
        self.receiver = receiver
        self.reference_id = reference_id
        self.metadata = metadata or {}
        self.attachments = attachments or []
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "message_type": self.message_type.value,
            "content": self.content,
            "sender": self.sender,
            "receiver": self.receiver,
            "reference_id": self.reference_id,
            "metadata": self.metadata,
            "attachments": self.attachments,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary representation."""
        message = cls(
            content=data["content"],
            message_type=data["message_type"],
            sender=data["sender"],
            receiver=data.get("receiver"),
            reference_id=data.get("reference_id"),
            metadata=data.get("metadata", {}),
            attachments=data.get("attachments", [])
        )
        
        # Override auto-generated id and timestamp with saved values
        message.id = data["id"]
        message.timestamp = data["timestamp"]
        
        return message
    
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.receiver is None or self.message_type == MessageType.BROADCAST
    
    def is_response_to(self, reference_id: str) -> bool:
        """Check if this message is a response to another message."""
        return self.reference_id == reference_id
    
    def create_response(
        self,
        content: str,
        message_type: MessageType = MessageType.RESPONSE,
        metadata: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> 'Message':
        """
        Create a response to this message.
        
        Args:
            content: Response content
            message_type: Type of response message
            metadata: Additional metadata
            attachments: Optional data attachments
            
        Returns:
            Response message
        """
        return Message(
            content=content,
            message_type=message_type,
            sender=self.receiver or "system",
            receiver=self.sender,
            reference_id=self.id,
            metadata=metadata,
            attachments=attachments
        )


class MessageQueue:
    """
    Queue for agent messages with asynchronous access.
    
    The message queue allows agents to send and receive messages
    asynchronously, enabling non-blocking communication.
    """
    
    def __init__(self):
        """Initialize the message queue."""
        self.queues: Dict[str, asyncio.Queue] = {}
        self.broadcast_subscribers: List[str] = []
        self.message_history: List[Message] = []
        self.max_history_size = 1000
    
    async def register(self, agent_id: str, store_history: bool = True) -> None:
        """
        Register an agent to the communication system.
        
        Args:
            agent_id: Unique ID of the agent
            store_history: Whether to store this agent's messages in history
        """
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.Queue()
            
        if store_history and agent_id not in self.broadcast_subscribers:
            self.broadcast_subscribers.append(agent_id)
    
    async def unregister(self, agent_id: str) -> None:
        """
        Unregister an agent from the communication system.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.queues:
            del self.queues[agent_id]
            
        if agent_id in self.broadcast_subscribers:
            self.broadcast_subscribers.remove(agent_id)
    
    async def send(self, message: Message) -> bool:
        """
        Send a message to the specified receiver.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was queued, False if receiver not found
        """
        # Store in history
        self._add_to_history(message)
        
        # Check if broadcast
        if message.is_broadcast():
            # Send to all subscribers
            for receiver_id in self.broadcast_subscribers:
                if receiver_id != message.sender:  # Don't send to self
                    if receiver_id in self.queues:
                        await self.queues[receiver_id].put(message)
            return True
            
        # Direct message
        if message.receiver in self.queues:
            await self.queues[message.receiver].put(message)
            return True
            
        logger.warning(f"Receiver {message.receiver} not found")
        return False
    
    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message for the specified agent.
        
        Args:
            agent_id: ID of the receiving agent
            timeout: Optional timeout in seconds
            
        Returns:
            Message if available, None if timeout
        """
        if agent_id not in self.queues:
            await self.register(agent_id)
            
        try:
            if timeout is not None:
                return await asyncio.wait_for(self.queues[agent_id].get(), timeout)
            else:
                return await self.queues[agent_id].get()
        except asyncio.TimeoutError:
            return None
    
    def _add_to_history(self, message: Message) -> None:
        """Add a message to the history, managing size limits."""
        self.message_history.append(message)
        
        # Trim history if it gets too large
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    def get_history(
        self,
        agent_id: Optional[str] = None,
        message_type: Optional[Union[MessageType, str]] = None,
        limit: int = 100
    ) -> List[Message]:
        """
        Get message history, optionally filtered.
        
        Args:
            agent_id: Filter by agent ID (sender or receiver)
            message_type: Filter by message type
            limit: Maximum number of messages to return
            
        Returns:
            List of messages matching criteria
        """
        # Convert string message type to enum if needed
        if isinstance(message_type, str):
            try:
                message_type = MessageType(message_type)
            except ValueError:
                message_type = None
        
        # Filter messages
        filtered = self.message_history
        
        if agent_id:
            filtered = [
                m for m in filtered
                if m.sender == agent_id or m.receiver == agent_id or m.is_broadcast()
            ]
            
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
            
        # Return most recent messages first, limited to requested count
        return list(reversed(filtered))[-limit:]


# Global message queue instance for module-level access
global_message_queue = MessageQueue()


async def send_message(
    content: str,
    message_type: Union[MessageType, str],
    sender: str,
    receiver: Optional[str] = None,
    reference_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    queue: Optional[MessageQueue] = None
) -> Message:
    """
    Convenience function to send a message.
    
    Args:
        content: Message content
        message_type: Type of message
        sender: ID of sending agent
        receiver: ID of receiving agent (None for broadcasts)
        reference_id: Optional reference to another message
        metadata: Additional message metadata
        attachments: Optional data attachments
        queue: Message queue to use (uses global queue if None)
        
    Returns:
        The sent message
    """
    # Create message
    message = Message(
        content=content,
        message_type=message_type,
        sender=sender,
        receiver=receiver,
        reference_id=reference_id,
        metadata=metadata,
        attachments=attachments
    )
    
    # Send message
    queue = queue or global_message_queue
    await queue.send(message)
    
    return message


class AgentCommunicator:
    """
    Agent interface for the communication protocol.
    
    This class provides a high-level interface for agents to send and
    receive messages, abstracting away the details of the message queue.
    """
    
    def __init__(
        self,
        agent_id: str,
        queue: Optional[MessageQueue] = None,
        auto_register: bool = True
    ):
        """
        Initialize the agent communicator.
        
        Args:
            agent_id: ID of the agent
            queue: Message queue to use (uses global queue if None)
            auto_register: Whether to automatically register with the queue
        """
        self.agent_id = agent_id
        self.queue = queue or global_message_queue
        
        # Auto-register if requested
        if auto_register:
            asyncio.create_task(self.register())
    
    async def register(self) -> None:
        """Register this agent with the message queue."""
        await self.queue.register(self.agent_id)
    
    async def unregister(self) -> None:
        """Unregister this agent from the message queue."""
        await self.queue.unregister(self.agent_id)
    
    async def send(
        self,
        content: str,
        message_type: Union[MessageType, str],
        receiver: Optional[str] = None,
        reference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Message:
        """
        Send a message to another agent.
        
        Args:
            content: Message content
            message_type: Type of message
            receiver: ID of receiving agent (None for broadcasts)
            reference_id: Optional reference to another message
            metadata: Additional message metadata
            attachments: Optional data attachments
            
        Returns:
            The sent message
        """
        message = Message(
            content=content,
            message_type=message_type,
            sender=self.agent_id,
            receiver=receiver,
            reference_id=reference_id,
            metadata=metadata,
            attachments=attachments
        )
        
        await self.queue.send(message)
        return message
    
    async def broadcast(
        self,
        content: str,
        message_type: Union[MessageType, str] = MessageType.BROADCAST,
        metadata: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Message:
        """
        Broadcast a message to all agents.
        
        Args:
            content: Message content
            message_type: Type of message
            metadata: Additional message metadata
            attachments: Optional data attachments
            
        Returns:
            The sent message
        """
        return await self.send(
            content=content,
            message_type=message_type,
            receiver=None,  # None indicates broadcast
            metadata=metadata,
            attachments=attachments
        )
    
    async def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message intended for this agent.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Message if available, None if timeout
        """
        return await self.queue.receive(self.agent_id, timeout)
    
    async def receive_filtered(
        self,
        sender: Optional[str] = None,
        message_type: Optional[Union[MessageType, str]] = None,
        timeout: Optional[float] = None,
        predicate: Optional[Callable[[Message], bool]] = None
    ) -> Optional[Message]:
        """
        Receive a message matching specific criteria.
        
        Args:
            sender: Filter by sender
            message_type: Filter by message type
            timeout: Maximum time to wait
            predicate: Optional custom filter function
            
        Returns:
            Matching message or None if timeout
        """
        # Convert string message type to enum if needed
        if isinstance(message_type, str):
            try:
                message_type = MessageType(message_type)
            except ValueError:
                message_type = None
        
        start_time = asyncio.get_event_loop().time()
        remaining_time = timeout
        
        while True:
            # Check timeout
            if timeout is not None:
                if remaining_time <= 0:
                    return None
                
            # Receive message
            message = await self.queue.receive(self.agent_id, remaining_time)
            
            if message is None:
                # Timeout occurred
                return None
            
            # Apply filters
            match = True
            
            if sender and message.sender != sender:
                match = False
                
            if message_type and message.message_type != message_type:
                match = False
                
            if predicate and not predicate(message):
                match = False
            
            if match:
                return message
            
            # Update remaining time if timeout is set
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining_time = timeout - elapsed
    
    async def wait_for_response(
        self,
        message_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Wait for a response to a specific message.
        
        Args:
            message_id: ID of the message to get a response for
            timeout: Maximum time to wait
            
        Returns:
            Response message or None if timeout
        """
        return await self.receive_filtered(
            message_type=MessageType.RESPONSE,
            timeout=timeout,
            predicate=lambda m: m.is_response_to(message_id)
        )
    
    async def query_agent(
        self,
        receiver: str,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[float] = 60.0
    ) -> Optional[Message]:
        """
        Send a query and wait for a response.
        
        Args:
            receiver: Agent to query
            query: Query content
            metadata: Additional metadata
            attachments: Optional attachments
            timeout: Maximum time to wait for response
            
        Returns:
            Response message or None if timeout
        """
        # Send query
        query_msg = await self.send(
            content=query,
            message_type=MessageType.QUERY,
            receiver=receiver,
            metadata=metadata,
            attachments=attachments
        )
        
        # Wait for response
        return await self.wait_for_response(query_msg.id, timeout)
    
    async def instruct_agent(
        self,
        receiver: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        wait_for_confirmation: bool = True,
        timeout: Optional[float] = 5.0
    ) -> Optional[Message]:
        """
        Send an instruction to another agent.
        
        Args:
            receiver: Agent to instruct
            instruction: Instruction content
            metadata: Additional metadata
            attachments: Optional attachments
            wait_for_confirmation: Whether to wait for confirmation
            timeout: Maximum time to wait for confirmation
            
        Returns:
            Confirmation message if requested, otherwise the sent message
        """
        # Send instruction
        instr_msg = await self.send(
            content=instruction,
            message_type=MessageType.INSTRUCTION,
            receiver=receiver,
            metadata=metadata,
            attachments=attachments
        )
        
        # Wait for confirmation if requested
        if wait_for_confirmation:
            return await self.wait_for_response(instr_msg.id, timeout)
        
        return instr_msg
    
    async def delegate_task(
        self,
        receiver: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        wait_for_result: bool = False,
        timeout: Optional[float] = None
    ) -> Union[Message, Dict[str, Any]]:
        """
        Delegate a task to another agent.
        
        Args:
            receiver: Agent to delegate to
            task: Task description
            context: Task context information
            priority: Task priority (1-5, 1 is highest)
            wait_for_result: Whether to wait for task result
            timeout: Maximum time to wait for result
            
        Returns:
            Task result if wait_for_result is True, otherwise delegation message
        """
        # Prepare metadata
        metadata = {
            "task_type": "delegation",
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }
        
        # Prepare attachment with context data
        attachments = [{"type": "context", "data": context}] if context else None
        
        # Send delegation
        deleg_msg = await self.send(
            content=task,
            message_type=MessageType.DELEGATION,
            receiver=receiver,
            metadata=metadata,
            attachments=attachments
        )
        
        # Wait for result if requested
        if wait_for_result:
            result_msg = await self.receive_filtered(
                sender=receiver,
                message_type=MessageType.RESULT,
                timeout=timeout,
                predicate=lambda m: m.is_response_to(deleg_msg.id)
            )
            
            if result_msg:
                # Extract result data if available
                if result_msg.attachments and len(result_msg.attachments) > 0:
                    for attachment in result_msg.attachments:
                        if attachment.get("type") == "result_data":
                            return attachment.get("data", {})
                
                # Fallback to message content
                return result_msg
            
            return None
        
        return deleg_msg
    
    def get_history(
        self,
        with_agent: Optional[str] = None,
        message_type: Optional[Union[MessageType, str]] = None,
        limit: int = 100
    ) -> List[Message]:
        """
        Get message history for this agent.
        
        Args:
            with_agent: Filter by other agent in conversation
            message_type: Filter by message type
            limit: Maximum messages to return
            
        Returns:
            List of matching messages
        """
        # Start with basic filter for this agent
        filtered = self.queue.get_history(agent_id=self.agent_id, message_type=message_type, limit=limit)
        
        # Further filter by conversation partner if requested
        if with_agent:
            filtered = [
                m for m in filtered
                if (m.sender == with_agent and m.receiver == self.agent_id) or
                   (m.sender == self.agent_id and m.receiver == with_agent)
            ]
            
        return filtered[:limit]
    