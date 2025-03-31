# ai_agent_framework/agents/base_agent.py

"""
Base Agent Class

This module provides the BaseAgent class, which serves as the foundation for all agent types
in the AI Agent Framework.
"""

import logging
import uuid
import asyncio # Added asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# Core framework components (Ensure correct relative imports based on your structure)
# If running scripts using `python -m`, absolute imports like below work.
# If running scripts directly, you might need relative imports (e.g., from ..core.llm.base import BaseLLM)
# Assuming execution via `python -m` or package installation:
from ai_agent_framework.core.llm.base import BaseLLM
from ai_agent_framework.core.memory.conversation import ConversationMemory
from ai_agent_framework.core.tools.registry import ToolRegistry


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
        # --- Added success/error state initialization ---
        self.success = False
        self.error: Optional[str] = None
        # -------------------------------------------

        # State tracking
        self.state: Dict[str, Any] = {}

        if self.verbose:
            logger.info(f"Initialized {self.__class__.__name__} '{self.name}' with ID {self.id}")

    @abstractmethod
    async def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]: # Ensure run is async
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
        # Use >= to allow exactly max_iterations runs before stopping
        if self.current_iteration > self.max_iterations:
            if self.verbose:
                logger.warning(f"Agent '{self.name}' reached maximum iterations ({self.max_iterations})")
            # --- Set state directly here ---
            self.finished = True
            self.success = False
            self.error = "Maximum iterations reached"
            # -------------------------------
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
            # Make sure get_conversation_history is appropriate (sync or async)
            # Assuming it's synchronous based on original code structure
            conversation_history = self.memory.get_conversation_history(format_type="string")
            if conversation_history:
                context_parts.append(f"Conversation history:\n{conversation_history}")

        # Add the current user input
        context_parts.append(f"User input: {user_input}")

        return "\n\n".join(context_parts)

    # --- Removed redundant _execute_tool_call method ---
    # Agents should directly call `await self.tools.execute_tool(...)`

    def add_tool(self, tool) -> None:
        """
        Add a new tool to the agent's tool registry.

        Args:
            tool: The tool instance to add (must be instance of BaseTool)
        """
        # Optional: Add type check for safety
        # from .core.tools.base import BaseTool # Import locally if needed
        # if not isinstance(tool, BaseTool):
        #     raise TypeError("tool must be an instance of BaseTool")
        self.tools.register_tool(tool)
        if self.verbose:
            logger.info(f"Added tool '{tool.name}' to agent '{self.name}'")

    # --- Made reset async ---
    async def reset(self) -> None:
        """
        Reset the agent's state, including iteration counter and memory.
        """
        self.current_iteration = 0
        self.finished = False
        # --- Reset success/error state ---
        self.success = False
        self.error = None
        # -------------------------------
        self.state = {}
        if self.memory:
            # --- Added await ---
            await self.memory.clear()
        if self.verbose:
            logger.info(f"Reset agent '{self.name}'")

    # save_state and load_state remain placeholders
    def save_state(self, path: str) -> None:
        """
        Save the agent's current state to a file. (Placeholder)

        Args:
            path: Path to save the state to
        """
        logger.warning("BaseAgent.save_state is not implemented.")
        pass

    def load_state(self, path: str) -> None:
        """
        Load the agent's state from a file. (Placeholder)

        Args:
            path: Path to load the state from
        """
        logger.warning("BaseAgent.load_state is not implemented.")
        pass

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', id='{self.id}')"

    # Optional: Add an async chat method for convenience if desired
    async def chat(self, message: str, **kwargs) -> str:
        """
        Simple async interface for chat-based interactions.

        Args:
            message: User message
            **kwargs: Additional parameters for the run method

        Returns:
            The agent's text response
        """
        result = await self.run(message, **kwargs)
        # Provide a default message if response key is missing or run failed badly
        return result.get("response", "I encountered an issue and couldn't generate a response.")