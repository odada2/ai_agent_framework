# ai_agent_framework/core/communication/agent_protocol.py

"""
Agent Protocol Module (Async Refactor)

Handles the asynchronous communication protocol for agents, including message
serialization, deserialization, delivery via HTTP, and basic error handling.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from urllib.parse import urljoin

import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type

# Assume these exceptions are defined in core.exceptions
# from ..core.exceptions import CommunicationError, ProtocolError, DeserializationError
# Placeholder exceptions if core.exceptions is missing:
class CommunicationError(Exception): pass
class ProtocolError(Exception): pass
class DeserializationError(Exception): pass


logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0 # Base delay for exponential backoff

@dataclass
class AgentMessage:
    """
    Represents a message exchanged between agents or components. (Unchanged from original)

    Attributes:
        sender: Identifier of the sending agent/component.
        recipient: Identifier of the receiving agent/component.
        content: Message payload (must be JSON serializable dictionary).
        message_type: Type of message (e.g., "task_execute", "task_complete").
        message_id: Unique identifier for the message.
        correlation_id: ID for correlating related messages (e.g., request/response).
        timestamp: Time when the message was created.
        metadata: Optional dictionary for additional metadata.
    """
    sender: str
    recipient: str
    content: Dict[str, Any]
    message_type: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create a message from a dictionary."""
        required_fields = ['sender', 'recipient', 'content', 'message_type']
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            raise DeserializationError(f"Missing required field(s): {', '.join(missing)}")

        # Ensure content is a dictionary, attempting deserialization if it's a JSON string
        content = data['content']
        if isinstance(content, str):
            try:
                content = json.loads(content)
                if not isinstance(content, dict):
                     raise DeserializationError("Message content (deserialized from string) must be a dictionary.")
            except json.JSONDecodeError as e:
                raise DeserializationError(f"Failed to decode content string as JSON: {e}")
        elif not isinstance(content, dict):
             raise DeserializationError("Message content must be a dictionary or a valid JSON string representing one.")

        try:
            return cls(
                sender=data['sender'],
                recipient=data['recipient'],
                content=content,
                message_type=data['message_type'],
                message_id=data.get('message_id', str(uuid.uuid4())),
                correlation_id=data.get('correlation_id'),
                timestamp=data.get('timestamp', time.time()),
                metadata=data.get('metadata', {}) # Ensure metadata is always a dict
            )
        except TypeError as e:
            raise DeserializationError(f"Failed to create message from data due to type error: {e}")
        except Exception as e:
            # Catch other potential errors during instantiation
            raise DeserializationError(f"Unexpected error creating message from data: {e}")

    def validate(self) -> bool:
        """Validate the message structure and basic types."""
        if not all([self.sender, self.recipient, self.message_type, self.message_id]):
            return False
        if not isinstance(self.content, dict):
            return False
        if not isinstance(self.metadata, dict):
             return False
        # Basic type checks
        if not isinstance(self.sender, str) or not isinstance(self.recipient, str): return False
        if not isinstance(self.message_type, str): return False
        if self.correlation_id is not None and not isinstance(self.correlation_id, str): return False
        if not isinstance(self.timestamp, (int, float)): return False

        return True


class AgentProtocol:
    """
    Asynchronous protocol implementation for agent communication via HTTP.

    Handles message serialization, delivery with retries, and basic routing
    of incoming messages to registered handlers or response queues.
    """

    DEFAULT_MESSAGE_ENDPOINT_PATH = "/message" # Define as constant

    def __init__(self, endpoints: Optional[Dict[str, str]] = None, own_id: Optional[str] = "protocol_instance"):
        """
        Initialize the asynchronous AgentProtocol.

        Args:
            endpoints: Mapping of agent IDs to their base HTTP API endpoints.
            own_id: Identifier for this protocol instance (used for logging).
        """
        self.endpoints = endpoints or {}
        self._message_handlers: Dict[str, Callable[[AgentMessage], Awaitable[None]]] = {}
        self._response_waiters: Dict[str, asyncio.Future] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._processing_tasks: Set[asyncio.Task] = set()
        self.own_id = own_id

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                # Consider configuring timeouts, connector limits etc. here
                self._session = aiohttp.ClientSession(
                     timeout=aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT)
                )
                logger.info(f"[{self.own_id}] Created new aiohttp session.")
            return self._session

    def register_endpoint(self, agent_id: str, endpoint: str) -> None:
        """Register or update the endpoint URL for an agent."""
        self.endpoints[agent_id] = endpoint
        logger.info(f"[{self.own_id}] Registered endpoint for '{agent_id}': {endpoint}")

    def register_handler(self, message_type: str, handler: Callable[[AgentMessage], Awaitable[None]]) -> None:
        """Register an async handler coroutine for a specific message type."""
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(f"Handler for '{message_type}' must be an async function (coroutine).")
        self._message_handlers[message_type] = handler
        logger.info(f"[{self.own_id}] Registered handler for message type '{message_type}'.")

    @retry(
        stop=stop_after_attempt(DEFAULT_RETRIES + 1),
        wait=wait_exponential(multiplier=DEFAULT_RETRY_DELAY, min=1, max=10), # Exponential backoff
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, CommunicationError)),
        before_sleep=lambda retry_state: logger.warning(f"Retrying send to {retry_state.args[0].recipient} after error: {retry_state.outcome.exception()}. Attempt {retry_state.attempt_number}...")
    )
    async def send(self, message: AgentMessage, request_timeout: float = DEFAULT_REQUEST_TIMEOUT) -> None:
        """
        Asynchronously send a message to the recipient via HTTP POST.

        Args:
            message: The AgentMessage to send.
            request_timeout: Specific timeout for this request.

        Raises:
            ProtocolError: If the message is invalid or serialization fails.
            CommunicationError: If the recipient endpoint is unknown or delivery fails after retries.
            aiohttp.ClientError: For underlying HTTP client issues.
            asyncio.TimeoutError: If the request times out.
        """
        if not message.validate():
            raise ProtocolError(f"[{self.own_id}] Attempted to send invalid message: {message.message_id}")

        recipient = message.recipient
        if recipient not in self.endpoints:
            raise CommunicationError(f"[{self.own_id}] No endpoint registered for agent '{recipient}'.")

        endpoint = self.endpoints[recipient]
        target_url = urljoin(endpoint, self.DEFAULT_MESSAGE_ENDPOINT_PATH)

        try:
            message_json = json.dumps(message.to_dict())
        except TypeError as e:
            raise ProtocolError(f"[{self.own_id}] Failed to serialize message {message.message_id}: {e}")

        session = await self._get_session()
        logger.debug(f"[{self.own_id}] Sending message {message.message_id} to {recipient} at {target_url}")

        try:
            async with session.post(
                target_url,
                data=message_json,
                headers={"Content-Type": "application/json"},
                timeout=request_timeout # Use specific timeout for this request
            ) as response:
                if response.status >= 200 and response.status < 300:
                    logger.debug(f"[{self.own_id}] Message {message.message_id} delivered successfully to {recipient} (Status: {response.status}).")
                    return # Success
                else:
                    # Raise CommunicationError for non-2xx responses to potentially trigger retry
                    error_text = await response.text()
                    raise CommunicationError(f"[{self.own_id}] Failed to deliver message {message.message_id} to {recipient}. Status: {response.status}, Response: {error_text[:200]}")

        except aiohttp.ClientError as e:
             logger.error(f"[{self.own_id}] HTTP Client error sending message {message.message_id} to {recipient}: {e}")
             raise # Allow tenacity to handle retry
        except asyncio.TimeoutError:
             logger.error(f"[{self.own_id}] Timeout sending message {message.message_id} to {recipient} after {request_timeout}s.")
             raise # Allow tenacity to handle retry
        except Exception as e:
             # Catch unexpected errors during send
             logger.exception(f"[{self.own_id}] Unexpected error sending message {message.message_id}: {e}")
             # Wrap in CommunicationError so retry logic might catch it if configured
             raise CommunicationError(f"Unexpected error during send: {e}")


    async def send_and_receive(
        self,
        message: AgentMessage,
        response_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        send_request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    ) -> AgentMessage:
        """
        Asynchronously send a message and wait for a correlated response.

        Args:
            message: The message to send.
            response_timeout: How long to wait for the response message.
            send_request_timeout: Timeout for the initial send request.

        Returns:
            The response AgentMessage.

        Raises:
            asyncio.TimeoutError: If no response is received within the timeout.
            CommunicationError, ProtocolError: If sending fails.
        """
        if not message.message_id:
             raise ProtocolError("Message must have an ID for send_and_receive.")

        future: asyncio.Future[AgentMessage] = asyncio.get_running_loop().create_future()
        self._response_waiters[message.message_id] = future

        try:
            await self.send(message, request_timeout=send_request_timeout)
            logger.debug(f"[{self.own_id}] Message {message.message_id} sent, awaiting response...")
            # Wait for the future to be set by the receive method
            response_message = await asyncio.wait_for(future, timeout=response_timeout)
            return response_message
        except asyncio.TimeoutError:
            logger.warning(f"[{self.own_id}] Timeout waiting for response to message {message.message_id}.")
            raise
        except Exception as e:
            logger.error(f"[{self.own_id}] Error during send_and_receive for message {message.message_id}: {e}")
            raise # Propagate send errors
        finally:
            # Clean up waiter
            self._response_waiters.pop(message.message_id, None)


    def process_received_data(self, message_data: Dict[str, Any]) -> None:
        """
        Entry point for processing received message data (e.g., from an HTTP endpoint).
        Deserializes and routes the message for background processing.

        Args:
            message_data: Raw dictionary data received.
        """
        try:
            message = AgentMessage.from_dict(message_data)
            logger.debug(f"[{self.own_id}] Received message {message.message_id} from {message.sender} for {message.recipient}")

            # Create a task to handle the message processing asynchronously
            task = asyncio.create_task(self._handle_received_message(message))
            self._processing_tasks.add(task)
            task.add_done_callback(self._processing_tasks.discard) # Auto-remove on completion

        except DeserializationError as e:
            logger.error(f"[{self.own_id}] Failed to deserialize received message data: {e}. Data: {message_data}")
            # Optionally, send an error response back if possible/appropriate
        except Exception as e:
            logger.exception(f"[{self.own_id}] Unexpected error receiving/scheduling message processing: {e}")


    async def _handle_received_message(self, message: AgentMessage) -> None:
        """Handles a deserialized message in the background."""
        # Check if this is a response to a waiting request
        if message.correlation_id and message.correlation_id in self._response_waiters:
            future = self._response_waiters.get(message.correlation_id)
            if future and not future.done():
                future.set_result(message)
                # Don't process further via handlers if it was a direct response
                return

        # Handle via registered message handlers
        handler = self._message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.exception(f"[{self.own_id}] Error in handler for message type '{message.message_type}' (ID: {message.message_id}): {e}")
        else:
            logger.warning(f"[{self.own_id}] No handler registered for message type '{message.message_type}' (ID: {message.message_id})")

    async def shutdown(self) -> None:
        """Gracefully shut down the protocol client and background tasks."""
        logger.info(f"[{self.own_id}] Shutting down AgentProtocol...")

        # Cancel any ongoing processing tasks
        if self._processing_tasks:
            logger.info(f"[{self.own_id}] Cancelling {len(self._processing_tasks)} background message processing tasks...")
            for task in list(self._processing_tasks):
                task.cancel()
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
            logger.info(f"[{self.own_id}] Background tasks cancelled.")

        # Close the aiohttp session
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
                logger.info(f"[{self.own_id}] Closed aiohttp session.")
        logger.info(f"[{self.own_id}] AgentProtocol shutdown complete.")
        