# ai_agent_framework/core/communication/agent_communicator.py

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Set

# Framework components
from .agent_protocol import AgentMessage, AgentProtocol, CommunicationError, ProtocolError, DeserializationError

logger = logging.getLogger(__name__)

class AgentCommunicator:
    """
    Handles asynchronous communication for an agent using AgentProtocol.

    Acts as an interface between the agent's logic and the underlying
    communication protocol, managing message sending, receiving (via an
    internal queue), and task delegation with response handling.
    """

    def __init__(self, agent_id: str, protocol: AgentProtocol, default_timeout: float = 60.0):
        """
        Initialize the AgentCommunicator.

        Args:
            agent_id: The unique ID of the agent this communicator serves.
            protocol: The configured AgentProtocol instance to use for communication.
            default_timeout: Default timeout for operations like send_and_receive.
        """
        self.agent_id = agent_id
        if not isinstance(protocol, AgentProtocol):
            raise TypeError("protocol must be an instance of AgentProtocol")
        self.protocol = protocol
        self.default_timeout = default_timeout
        self._receive_queue = asyncio.Queue()
        self._message_handlers: Dict[str, Callable[[AgentMessage], Awaitable[None]]] = {}

        # Register a generic handler with the protocol to feed the queue
        # Register for all expected message types the agent might receive.
        # '*' could be a wildcard if AgentProtocol supports it, otherwise list explicitly.
        expected_message_types = [
            "QUERY", "RESPONSE", "UPDATE", "RESULT", "ERROR",
            "CONFIRMATION", "INSTRUCTION", "DELEGATE_TASK", "TASK_STATUS",
            # Add any other types the Supervisor might need to handle directly
        ]
        for msg_type in expected_message_types:
             # Use a default handler that queues the message
             self.protocol.register_handler(msg_type, self._queue_message)

        logger.info(f"[{self.agent_id}] AgentCommunicator initialized.")

    async def _queue_message(self, message: AgentMessage):
        """Default handler: Puts received messages onto the internal queue."""
        # Optional: Basic filtering (e.g., ignore messages not for this agent_id if protocol doesn't handle it)
        if message.recipient != self.agent_id:
             logger.warning(f"[{self.agent_id}] Ignored message intended for {message.recipient}")
             return
        await self._receive_queue.put(message)
        logger.debug(f"[{self.agent_id}] Queued message {message.message_id} from {message.sender}")

    async def send(
        self,
        recipient_id: str,
        message_type: str,
        content: Dict[str, Any],
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Sends a message to another agent.

        Args:
            recipient_id: The ID of the receiving agent.
            message_type: The type of the message (string).
            content: The message payload dictionary.
            correlation_id: Optional ID for correlating messages.
            metadata: Optional additional metadata.

        Returns:
            The message_id of the sent message.

        Raises:
            CommunicationError: If sending fails after retries.
            ProtocolError: If the message is invalid or serialization fails.
        """
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient_id,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        try:
            await self.protocol.send(message)
            logger.info(f"[{self.agent_id}] Sent {message_type} message {message.message_id} to {recipient_id}")
            return message.message_id
        except (CommunicationError, ProtocolError) as e:
            logger.error(f"[{self.agent_id}] Failed to send {message_type} message {message.message_id} to {recipient_id}: {e}")
            raise # Re-raise the specific error
        except Exception as e:
             # Catch unexpected errors
             logger.exception(f"[{self.agent_id}] Unexpected error sending message {message.message_id} to {recipient_id}: {e}")
             raise CommunicationError(f"Unexpected send error: {e}")


    async def send_and_receive(
        self,
        recipient_id: str,
        message_type: str,
        content: Dict[str, Any],
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Sends a message and waits for a correlated response.

        Args:
            recipient_id: The ID of the receiving agent.
            message_type: The type of the message (string).
            content: The message payload dictionary.
            timeout: Specific timeout for waiting for the response. Uses default if None.
            metadata: Optional additional metadata for the request message.

        Returns:
            The received response AgentMessage.

        Raises:
            asyncio.TimeoutError: If the response is not received within the timeout.
            CommunicationError: If sending fails.
            ProtocolError: If the message is invalid or serialization fails.
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient_id,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
            # correlation_id will be the message_id by default for send_and_receive
        )
        try:
            logger.info(f"[{self.agent_id}] Sending {message_type} message {message.message_id} to {recipient_id} and awaiting response (timeout={effective_timeout}s)")
            # AgentProtocol's send_and_receive handles correlation and waiting
            response_message = await self.protocol.send_and_receive(
                message,
                response_timeout=effective_timeout
            )
            logger.info(f"[{self.agent_id}] Received response {response_message.message_id} for request {message.message_id} from {response_message.sender}")
            return response_message
        except asyncio.TimeoutError:
            logger.error(f"[{self.agent_id}] Timeout waiting for response to message {message.message_id} from {recipient_id}")
            raise
        except (CommunicationError, ProtocolError) as e:
            logger.error(f"[{self.agent_id}] Failed send_and_receive for message {message.message_id} to {recipient_id}: {e}")
            raise
        except Exception as e:
             logger.exception(f"[{self.agent_id}] Unexpected error during send_and_receive for message {message.message_id}: {e}")
             raise CommunicationError(f"Unexpected send_and_receive error: {e}")

    async def receive(self, timeout: float = 1.0) -> Optional[AgentMessage]:
        """
        Receives the next message from the internal queue.

        Args:
            timeout: How long to wait for a message.

        Returns:
            The received AgentMessage or None if timeout occurs.
        """
        try:
            message = await asyncio.wait_for(self._receive_queue.get(), timeout=timeout)
            self._receive_queue.task_done() # Mark task as done for queue management
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error retrieving message from queue: {e}")
            return None

    async def broadcast(
        self,
        message_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Sends a message to all known agents except itself.

        Args:
            message_type: The type of the message (string).
            content: The message payload dictionary.
            metadata: Optional additional metadata.
        """
        # Get recipients from protocol's known endpoints
        if not hasattr(self.protocol, 'endpoints') or not self.protocol.endpoints:
             logger.warning(f"[{self.agent_id}] Cannot broadcast: No endpoints registered in protocol.")
             return

        recipients = list(self.protocol.endpoints.keys())
        tasks = []
        logger.info(f"[{self.agent_id}] Broadcasting {message_type} message to {len(recipients)-1} recipients.")

        for recipient_id in recipients:
            if recipient_id != self.agent_id: # Don't send to self
                message = AgentMessage(
                    sender=self.agent_id,
                    recipient=recipient_id,
                    message_type=message_type,
                    content=content,
                    metadata=metadata or {}
                )
                # Use protocol.send directly for broadcast (fire and forget)
                tasks.append(self.protocol.send(message))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Map results back to recipients for logging
            valid_recipients = [r for r in recipients if r != self.agent_id]
            for i, res in enumerate(results):
                if i < len(valid_recipients):
                    recipient_id = valid_recipients[i]
                    if isinstance(res, Exception):
                        logger.error(f"[{self.agent_id}] Error broadcasting {message_type} message to {recipient_id}: {res}")

    async def delegate_task(
        self,
        recipient_id: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        wait_for_result: bool = True,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Delegates a task to another agent and optionally waits for the result.

        Args:
            recipient_id: The ID of the agent to delegate the task to.
            task_description: A description of the task.
            context: Supporting context or data for the task.
            wait_for_result: If True, waits for a RESULT or ERROR message.
            timeout: Timeout for waiting for the result (if wait_for_result is True).

        Returns:
            A dictionary containing the status ('delegated', 'completed', 'failed', 'timeout')
            and potentially the result or error message. Includes 'task_id' used for delegation.
        """
        message_type = "DELEGATE_TASK" # Use string type consistent with AgentProtocol
        content = {
            "task": task_description,
            "context": context or {},
        }
        metadata = {"wait_for_result": wait_for_result} # Metadata for receiver if needed

        effective_timeout = timeout if timeout is not None else self.default_timeout
        task_id = None # Will be set from the sent message

        try:
            if wait_for_result:
                # Use send_and_receive to handle correlation and waiting
                request_message = AgentMessage( # Manually create to get ID first
                     sender=self.agent_id, recipient=recipient_id,
                     message_type=message_type, content=content, metadata=metadata
                )
                task_id = request_message.message_id # Store the ID used for the request

                response_message = await self.protocol.send_and_receive(
                    request_message,
                    response_timeout=effective_timeout
                )

                # Interpret the response
                # Standardize expected response structure slightly
                if response_message.message_type == "RESULT":
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "result": response_message.content # Assume content IS the result payload
                     }
                elif response_message.message_type == "ERROR":
                     return {
                        "task_id": task_id,
                        "status": "failed",
                        "error": response_message.content.get("error", "Agent reported an error.")
                     }
                else:
                     # Unexpected response type
                     logger.warning(f"[{self.agent_id}] Received unexpected response type '{response_message.message_type}' for delegated task {task_id}")
                     return {
                         "task_id": task_id,
                         "status": "failed",
                         "error": f"Unexpected response type '{response_message.message_type}' received: {str(response_message.content)[:100]}..."
                     }
            else:
                # Fire and forget
                task_id = await self.send(
                    recipient_id=recipient_id,
                    message_type=message_type,
                    content=content,
                    metadata=metadata
                )
                return {"task_id": task_id, "status": "delegated"}

        except asyncio.TimeoutError:
            logger.error(f"[{self.agent_id}] Timeout ({effective_timeout}s) waiting for task result from {recipient_id} for task: {task_description[:50]}...")
            return {"task_id": task_id, "status": "timeout", "error": f"Timeout waiting for task result from {recipient_id}"}
        except (CommunicationError, ProtocolError) as e:
            logger.error(f"[{self.agent_id}] Communication error delegating task to {recipient_id}: {e}")
            return {"task_id": task_id, "status": "failed", "error": f"Communication error: {e}"}
        except Exception as e:
            logger.exception(f"[{self.agent_id}] Unexpected error during task delegation to {recipient_id}: {e}")
            return {"task_id": task_id, "status": "failed", "error": f"Unexpected delegation error: {e}"}

    async def shutdown(self):
        """Perform cleanup, like shutting down the protocol."""
        logger.info(f"[{self.agent_id}] Shutting down AgentCommunicator...")
        # The protocol might be shared, so the agent itself calls protocol.shutdown typically.
        # If the communicator owns the protocol, shut it down here.
        # await self.protocol.shutdown() # Uncomment if communicator owns protocol instance
        logger.info(f"[{self.agent_id}] AgentCommunicator shutdown complete.")