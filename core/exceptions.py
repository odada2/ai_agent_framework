# ai_agent_framework/core/exceptions.py

"""
Custom Exception Classes for the AI Agent Framework
"""

class AgentFrameworkError(Exception):
    """Base class for framework-specific exceptions."""
    pass

# --- Communication Errors ---

class CommunicationError(AgentFrameworkError):
    """Error during agent communication (e.g., network issues)."""
    pass

class ProtocolError(CommunicationError):
    """Error related to the communication protocol itself (e.g., invalid message format)."""
    pass

class DeserializationError(ProtocolError):
    """Error deserializing a message."""
    pass

# --- Workflow and Orchestration Errors ---

class OrchestratorError(AgentFrameworkError):
    """Error originating from the Orchestrator."""
    pass

class SchedulingError(OrchestratorError):
    """Error during task scheduling."""
    pass

class WorkflowError(AgentFrameworkError):
    """General error related to workflow execution."""
    pass

# --- Task Errors ---

class TaskError(AgentFrameworkError):
     """Base class for task-related errors."""
     pass

class TaskTimeoutError(TaskError):
     """Error when a task exceeds its execution time limit."""
     pass

# --- Agent Errors ---

class AgentError(AgentFrameworkError):
     """General error related to agent operation."""
     pass

# --- Tool Errors ---

class ToolError(AgentFrameworkError):
     """Error related to tool execution."""
     pass

# --- Memory/Vector Store Errors ---

class VectorStoreError(AgentFrameworkError):
     """Error related to vector store operations."""
     pass

class KnowledgeBaseError(AgentFrameworkError):
     """Error related to knowledge base operations."""
     pass

# --- Configuration Errors ---
class ConfigurationError(AgentFrameworkError):
     """Error related to loading or accessing configuration."""
     pass

# --- Evaluation/Optimization Errors ---
class EvaluationError(WorkflowError):
    """Error during evaluation."""
    pass

class OptimizationError(WorkflowError):
    """Error during optimization."""
    pass

# --- Worker Errors ---
class WorkerError(AgentError):
    """Error originating from or related to a Worker."""
    pass