"""
Base Agent Class

This module provides the BaseAgent class, which serves as the foundation for all agent types
in the AI Agent Framework.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..core.llm.base import BaseLLM
from ..core.memory.conversation import ConversationMemory
from ..core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the framework.
    
    This class defines the interface and common functionality that all agents must implement,
    regardless of their specific type or purpose.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the BaseAgent.
        
        Args:
            name: A unique name for this agent instance
            llm: The LLM implementation to use for this agent
            tools: Optional registry of tools available to the agent
            memory: Optional conversation memory for maintaining context
            system_prompt: Optional system prompt to guide the agent's behavior
            max_iterations: Maximum number of iterations the agent can perform in a run
            verbose: Whether to log detailed information about the agent's operations
            **kwargs: Additional agent-specific parameters
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.config = kwargs
        
        # Iteration tracking
        self.current_iteration = 0
        self.finished = False
        
        # State tracking
        self.state: Dict[str, Any] = {}
        
        if self.verbose:
            logger.info(f"Initialized {self.__class__.__name__} '{self.name}' with ID {self.id}")

    @abstractmethod
    def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        """
        Run the agent on the given input.
        
        This method must be implemented by all concrete agent classes.
        
        Args:
            input_data: The input data for the agent to process (string query or structured data)
            **kwargs: Additional runtime parameters
            
        Returns:
            A dictionary containing the agent's response and any additional metadata
        """
        pass
    
    def _increment_iteration(self) -> bool:
        """
        Increment the iteration counter and check if the maximum has been reached.
        
        Returns:
            True if the agent can continue, False if max iterations reached
        """
        self.current_iteration += 1
        if self.current_iteration >= self.max_iterations:
            if self.verbose:
                logger.warning(f"Agent '{self.name}' reached maximum iterations ({self.max_iterations})")
            return False
        return True
    
    def _prepare_context(self, user_input: str) -> str:
        """
        Prepare the full context for the LLM, including system prompt, memory, and tools.
        
        Args:
            user_input: The current user input to process
            
        Returns:
            The full context string to send to the LLM
        """
        # Start with system prompt if available
        context_parts = []
        if self.system_prompt:
            context_parts.append(self.system_prompt)
        
        # Add tool descriptions if available
        if self.tools and len(self.tools) > 0:
            tool_descriptions = self.tools.get_tool_descriptions()
            context_parts.append(f"You have access to the following tools:\n{tool_descriptions}")
        
        # Add conversation history from memory
        if self.memory:
            conversation_history = self.memory.get_conversation_history()
            if conversation_history:
                context_parts.append(f"Conversation history:\n{conversation_history}")
        
        # Add the current user input
        context_parts.append(f"User input: {user_input}")
        
        return "\n\n".join(context_parts)
    
    def _process_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the raw LLM response to extract tool calls, actions, and the final response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            A processed response with extracted tool calls, actions, and text
        """
        # This implementation will depend on the specific LLM and response format
        # For now, return the raw response
        return response
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call based on the LLM's request.
        
        Args:
            tool_call: A dictionary containing the tool name and parameters
            
        Returns:
            The result of the tool execution
        """
        tool_name = tool_call.get("name")
        tool_params = tool_call.get("parameters", {})
        
        if not tool_name:
            logger.error("Tool call missing tool name")
            return {"error": "Tool call missing tool name"}
        
        if not self.tools.has_tool(tool_name):
            logger.error(f"Tool '{tool_name}' not found in registry")
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            result = self.tools.execute_tool(tool_name, **tool_params)
            return {"tool_name": tool_name, "result": result}
        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}': {str(e)}")
            return {"tool_name": tool_name, "error": str(e)}
    
    def add_tool(self, tool) -> None:
        """
        Add a new tool to the agent's tool registry.
        
        Args:
            tool: The tool instance to add
        """
        self.tools.register_tool(tool)
        if self.verbose:
            logger.info(f"Added tool '{tool.name}' to agent '{self.name}'")
    
    def reset(self) -> None:
        """
        Reset the agent's state, including iteration counter and memory.
        """
        self.current_iteration = 0
        self.finished = False
        self.state = {}
        if self.memory:
            self.memory.clear()
        if self.verbose:
            logger.info(f"Reset agent '{self.name}'")
    
    def save_state(self, path: str) -> None:
        """
        Save the agent's current state to a file.
        
        Args:
            path: Path to save the state to
        """
        # Implementation depends on your serialization approach
        pass
    
    def load_state(self, path: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            path: Path to load the state from
        """
        # Implementation depends on your serialization approach
        pass
    
    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', id='{self.id}')"