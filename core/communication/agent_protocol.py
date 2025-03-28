"""
Agent Protocol Module

This module provides the communication protocol for agents to interact with each other
and with the orchestration system. It handles message passing, serialization,
deserialization, and error recovery.

The protocol supports both synchronous and asynchronous communication patterns.
"""

import json
import logging
import uuid
import time
import queue
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict

# Networking
import requests
from urllib.parse import urljoin

# Error handling
from core.exceptions import CommunicationError, ProtocolError, DeserializationError

logger = logging.getLogger(__name__)

@dataclass
class AgentMessage:
    """
    Represents a message exchanged between agents or between agents and the orchestrator.
    
    Attributes:
        sender: Identifier of the sending agent
        recipient: Identifier of the receiving agent
        content: Message payload (must be JSON serializable)
        message_type: Type of message (e.g., "task_execute", "task_complete")
        message_id: Unique identifier for the message
        correlation_id: ID for correlating related messages
        timestamp: Time when the message was created
    """
    sender: str
    recipient: str
    content: Dict[str, Any]
    message_type: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary for serialization.
        
        Returns:
            Dict representation of the message
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """
        Create a message from a dictionary.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            AgentMessage instance
            
        Raises:
            DeserializationError: If the message data is invalid
        """
        required_fields = ['sender', 'recipient', 'content', 'message_type']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                raise DeserializationError(f"Missing required field '{field}' in message data")
        
        # Ensure content is a dictionary
        if not isinstance(data['content'], dict):
            # Try to parse content if it's a string
            if isinstance(data['content'], str):
                try:
                    data['content'] = json.loads(data['content'])
                except json.JSONDecodeError:
                    # If not JSON, wrap in a dictionary
                    data['content'] = {'data': data['content']}
            else:
                # If not a string or dict, wrap in a dictionary
                data['content'] = {'data': data['content']}
        
        try:
            return cls(
                sender=data['sender'],
                recipient=data['recipient'],
                content=data['content'],
                message_type=data['message_type'],
                message_id=data.get('message_id', str(uuid.uuid4())),
                correlation_id=data.get('correlation_id'),
                timestamp=data.get('timestamp', time.time())
            )
        except Exception as e:
            raise DeserializationError(f"Failed to create message from data: {str(e)}")
    
    def validate(self) -> bool:
        """
        Validate the message for correctness.
        
        Returns:
            True if the message is valid, False otherwise
        """
        # Check required fields
        if not self.sender or not self.recipient or not self.message_type:
            return False
        
        # Check content type
        if not isinstance(self.content, dict):
            return False
        
        # Additional validation could be added here
        
        return True

class AgentProtocol:
    """
    Protocol implementation for agent communication.
    
    This class handles:
    1. Serialization and deserialization of messages
    2. Message delivery and receipt
    3. Error recovery and retries
    4. Message queuing for asynchronous communication
    
    Attributes:
        endpoints: Mapping of agent IDs to their API endpoints
        message_handlers: Mapping of message types to handler functions
        inbox: Queue of received messages
        response_queues: Mapping of message IDs to their response queues
    """
    
    def __init__(self, endpoints: Optional[Dict[str, str]] = None):
        """
        Initialize a new AgentProtocol.
        
        Args:
            endpoints: Mapping of agent IDs to their API endpoints
        """
        self.endpoints = endpoints or {}
        self.message_handlers: Dict[str, Callable] = {}
        self.inbox = queue.Queue()
        self.response_queues: Dict[str, queue.Queue] = {}
        
        # Thread for processing incoming messages
        self._stop_processing = threading.Event()
        self._processing_thread = threading.Thread(
            target=self._process_inbox,
            daemon=True,
            name="agent-protocol-processor"
        )
        self._processing_thread.start()
    
    def register_endpoint(self, agent_id: str, endpoint: str) -> None:
        """
        Register an endpoint for an agent.
        
        Args:
            agent_id: Identifier of the agent
            endpoint: API endpoint URL for the agent
        """
        self.endpoints[agent_id] = endpoint
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Function to call when a message of this type is received
        """
        self.message_handlers[message_type] = handler
    
    def send(self, message: AgentMessage, retries: int = 3, timeout: float = 10.0) -> None:
        """
        Send a message to the recipient.
        
        Args:
            message: Message to send
            retries: Number of retry attempts
            timeout: Timeout for request in seconds
            
        Raises:
            CommunicationError: If the message cannot be delivered
        """
        if not message.validate():
            raise ProtocolError("Invalid message")
        
        recipient = message.recipient
        
        # Local delivery
        if recipient == "local":
            self.inbox.put(message)
            return
        
        # Check if we have an endpoint for the recipient
        if recipient not in self.endpoints:
            raise CommunicationError(f"No endpoint registered for agent '{recipient}'")
        
        endpoint = self.endpoints[recipient]
        
        # Serialize the message
        try:
            message_data = json.dumps(message.to_dict())
        except Exception as e:
            raise ProtocolError(f"Failed to serialize message: {str(e)}")
        
        # Send the message with retries
        attempt = 0
        last_error = None
        
        while attempt <= retries:
            try:
                response = requests.post(
                    urljoin(endpoint, "/message"),
                    data=message_data,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    return
                else:
                    error_msg = f"Failed to deliver message: HTTP {response.status_code}"
                    logger.warning(error_msg)
                    last_error = CommunicationError(error_msg)
                    
            except (requests.RequestException, ConnectionError) as e:
                logger.warning(f"Error delivering message to {recipient}: {str(e)}")
                last_error = CommunicationError(f"Communication error: {str(e)}")
                
            # Retry with backoff
            attempt += 1
            if attempt <= retries:
                time.sleep(min(2 ** attempt, 30))  # Exponential backoff, max 30s
        
        # All retries failed
        raise last_error if last_error else CommunicationError("Failed to deliver message after retries")
    
    def send_and_receive(
        self, 
        message: AgentMessage, 
        timeout: float = 30.0,
        retries: int = 3
    ) -> Optional[AgentMessage]:
        """
        Send a message and wait for a response.
        
        Args:
            message: Message to send
            timeout: Timeout for waiting for response in seconds
            retries: Number of retry attempts for sending
            
        Returns:
            Response message or None if no response is received
            
        Raises:
            CommunicationError: If the message cannot be delivered
            TimeoutError: If no response is received within the timeout
        """
        # Create a response queue for this message
        response_queue = queue.Queue()
        self.response_queues[message.message_id] = response_queue
        
        try:
            # Send the message
            self.send(message, retries=retries)
            
            # Wait for response
            try:
                response = response_queue.get(timeout=timeout)
                return response
            except queue.Empty:
                raise TimeoutError(f"No response received within {timeout}s")
                
        finally:
            # Clean up
            if message.message_id in self.response_queues:
                del self.response_queues[message.message_id]
    
    def receive(self, message_data: Dict[str, Any]) -> None:
        """
        Process a received message.
        
        Args:
            message_data: Dictionary containing message data
            
        Raises:
            DeserializationError: If the message data is invalid
        """
        try:
            # Deserialize the message
            message = AgentMessage.from_dict(message_data)
            
            # Put the message in the inbox
            self.inbox.put(message)
            
        except Exception as e:
            logger.error(f"Error processing received message: {str(e)}")
            raise DeserializationError(f"Failed to process message: {str(e)}")
    
    def _process_inbox(self) -> None:
        """
        Background thread for processing incoming messages.
        """
        while not self._stop_processing.is_set():
            try:
                # Get a message from the inbox
                try:
                    message = self.inbox.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if this is a response to a pending request
                if message.correlation_id and message.correlation_id in self.response_queues:
                    self.response_queues[message.correlation_id].put(message)
                    self.inbox.task_done()
                    continue
                
                # Handle based on message type
                if message.message_type in self.message_handlers:
                    try:
                        self.message_handlers[message.message_type](message)
                    except Exception as e:
                        logger.error(f"Error in message handler for '{message.message_type}': {str(e)}")
                else:
                    logger.warning(f"No handler registered for message type '{message.message_type}'")
                
                self.inbox.task_done()
                
            except Exception as e:
                logger.error(f"Error in message processing thread: {str(e)}")
    
    def shutdown(self) -> None:
        """
        Gracefully shut down the protocol.
        """
        self._stop_processing.set()
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)