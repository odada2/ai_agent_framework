# ai_agent_framework/core/communication/__init__.py

"""
Communication Package

Handles inter-agent communication protocols and messaging.
"""

from .agent_protocol import AgentMessage, AgentProtocol, CommunicationError, ProtocolError, DeserializationError
# Import the new concrete communicator
from .agent_communicator import AgentCommunicator

__all__ = [
    "AgentMessage",
    "AgentProtocol",
    "AgentCommunicator", # Add communicator to exports
    "CommunicationError",
    "ProtocolError",
    "DeserializationError",
]