# ai_agent_framework/tests/unit/test_base_agent.py

"""
Unit Tests for the BaseAgent class.
"""

import unittest
import uuid
from unittest.mock import MagicMock, patch, create_autospec, PropertyMock

# Import the class to be tested and its dependencies
# Assuming ai_agent_framework is the root package
from ai_agent_framework.agents.base_agent import BaseAgent
from ai_agent_framework.core.llm.base import BaseLLM
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.core.memory.conversation import ConversationMemory

# A concrete implementation of BaseAgent for testing purposes,
# as BaseAgent itself is abstract due to the run method.
class ConcreteTestAgent(BaseAgent):
    """A minimal concrete implementation of BaseAgent for testing."""
    def run(self, input_data: Union[str, Dict], **kwargs) -> Dict[str, Any]:
        # Minimal implementation for the abstract method
        return {"response": f"Processed: {input_data}", "success": True}

class TestBaseAgent(unittest.TestCase):
    """Test suite for the BaseAgent abstract class."""

    def setUp(self):
        """Set up common resources for tests."""
        # Create mocks for dependencies
        self.mock_llm = create_autospec(BaseLLM, instance=True)
        self.mock_tools = create_autospec(ToolRegistry, instance=True)
        self.mock_memory = create_autospec(ConversationMemory, instance=True)

        # Mock return values for methods used in tested functions
        self.mock_tools.get_tool_descriptions.return_value = "- tool1: Does something"
        type(self.mock_tools). __len__.return_value = 1 # Mock __len__ for tool check
        self.mock_memory.get_conversation_history.return_value = "User: Hello\nAssistant: Hi there!"

        # Default parameters for agent creation
        self.agent_name = "test_agent"
        self.system_prompt = "You are a test agent."
        self.max_iterations = 5

        # Instantiate the concrete agent for testing BaseAgent's logic
        self.agent = ConcreteTestAgent(
            name=self.agent_name,
            llm=self.mock_llm,
            tools=self.mock_tools,
            memory=self.mock_memory,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            verbose=False # Keep tests quiet unless debugging
        )

    def test_initialization(self):
        """Test if the agent initializes with correct attributes."""
        self.assertEqual(self.agent.name, self.agent_name)
        self.assertEqual(self.agent.llm, self.mock_llm)
        self.assertEqual(self.agent.tools, self.mock_tools)
        self.assertEqual(self.agent.memory, self.mock_memory)
        self.assertEqual(self.agent.system_prompt, self.system_prompt)
        self.assertEqual(self.agent.max_iterations, self.max_iterations)
        self.assertIsInstance(self.agent.id, str)
        # Check default values
        self.assertEqual(self.agent.current_iteration, 0)
        self.assertFalse(self.agent.finished)
        self.assertEqual(self.agent.state, {})

    def test_initialization_defaults(self):
        """Test initialization with default tools and memory."""
        agent_default = ConcreteTestAgent(name="default_agent", llm=self.mock_llm)
        self.assertIsInstance(agent_default.tools, ToolRegistry)
        self.assertIsInstance(agent_default.memory, ConversationMemory)
        self.assertEqual(len(agent_default.tools), 0) # Check default tool registry is empty

    def test_increment_iteration(self):
        """Test the iteration counter and limit."""
        self.assertEqual(self.agent.current_iteration, 0)
        # Increment within limits
        for i in range(1, self.max_iterations):
            can_continue = self.agent._increment_iteration()
            self.assertTrue(can_continue)
            self.assertEqual(self.agent.current_iteration, i)

        # Increment to the limit
        can_continue = self.agent._increment_iteration()
        self.assertFalse(can_continue) # Max iterations reached
        self.assertEqual(self.agent.current_iteration, self.max_iterations)

        # Try incrementing beyond the limit
        can_continue = self.agent._increment_iteration()
        self.assertFalse(can_continue) # Should remain false
        self.assertEqual(self.agent.current_iteration, self.max_iterations + 1) # Counter still goes up

    def test_prepare_context_all_elements(self):
        """Test context preparation with system prompt, tools, and memory."""
        user_input = "What is the weather?"
        expected_context = (
            f"{self.system_prompt}\n\n"
            f"You have access to the following tools:\n{self.mock_tools.get_tool_descriptions()}\n\n"
            f"Conversation history:\n{self.mock_memory.get_conversation_history()}\n\n"
            f"User input: {user_input}"
        )
        actual_context = self.agent._prepare_context(user_input)
        self.assertEqual(actual_context, expected_context)
        self.mock_tools.get_tool_descriptions.assert_called_once()
        self.mock_memory.get_conversation_history.assert_called_once()

    def test_prepare_context_no_system_prompt(self):
        """Test context preparation without a system prompt."""
        self.agent.system_prompt = None
        user_input = "Tell me a joke."
        expected_context = (
            f"You have access to the following tools:\n{self.mock_tools.get_tool_descriptions()}\n\n"
            f"Conversation history:\n{self.mock_memory.get_conversation_history()}\n\n"
            f"User input: {user_input}"
        )
        actual_context = self.agent._prepare_context(user_input)
        self.assertEqual(actual_context, expected_context)

    def test_prepare_context_no_tools(self):
        """Test context preparation without tools."""
        # Mock tools registry having no tools
        type(self.mock_tools).__len__.return_value = 0
        self.mock_tools.get_tool_descriptions.return_value = "No tools available." # Adjust mock

        self.agent.tools = self.mock_tools # Re-assign potentially modified mock

        user_input = "Summarize this."
        expected_context = (
            f"{self.system_prompt}\n\n"
            # No tool description section expected if len(tools) == 0
            f"Conversation history:\n{self.mock_memory.get_conversation_history()}\n\n"
            f"User input: {user_input}"
        )
        actual_context = self.agent._prepare_context(user_input)
        self.assertEqual(actual_context, expected_context)
        # get_tool_descriptions should not be called if len is 0
        self.mock_tools.get_tool_descriptions.assert_not_called()


    def test_prepare_context_no_memory(self):
        """Test context preparation without memory."""
        self.mock_memory.get_conversation_history.return_value = "" # Mock empty history
        self.agent.memory = self.mock_memory # Re-assign mock

        user_input = "Question?"
        expected_context = (
            f"{self.system_prompt}\n\n"
            f"You have access to the following tools:\n{self.mock_tools.get_tool_descriptions()}\n\n"
            # No conversation history section expected
            f"User input: {user_input}"
        )
        actual_context = self.agent._prepare_context(user_input)
        self.assertEqual(actual_context, expected_context)
        self.mock_memory.get_conversation_history.assert_called_once()


    def test_execute_tool_call_success(self):
        """Test successful tool execution."""
        tool_name = "tool1"
        tool_params = {"param": "value"}
        tool_result = {"data": "result"}
        tool_call = {"name": tool_name, "parameters": tool_params}

        # Mock tool registry behavior
        self.mock_tools.has_tool.return_value = True
        self.mock_tools.execute_tool.return_value = tool_result

        result = self.agent._execute_tool_call(tool_call)

        self.mock_tools.has_tool.assert_called_once_with(tool_name)
        self.mock_tools.execute_tool.assert_called_once_with(tool_name, **tool_params)
        self.assertEqual(result, {"tool_name": tool_name, "result": tool_result})

    def test_execute_tool_call_missing_name(self):
        """Test tool call execution when tool name is missing."""
        tool_call = {"parameters": {"param": "value"}} # Missing 'name'
        result = self.agent._execute_tool_call(tool_call)
        self.assertIn("error", result)
        self.assertIn("missing tool name", result["error"])
        self.mock_tools.has_tool.assert_not_called()
        self.mock_tools.execute_tool.assert_not_called()

    def test_execute_tool_call_tool_not_found(self):
        """Test tool call execution when the tool doesn't exist."""
        tool_name = "unknown_tool"
        tool_call = {"name": tool_name, "parameters": {}}

        self.mock_tools.has_tool.return_value = False # Mock tool not found

        result = self.agent._execute_tool_call(tool_call)

        self.mock_tools.has_tool.assert_called_once_with(tool_name)
        self.mock_tools.execute_tool.assert_not_called()
        self.assertIn("error", result)
        self.assertIn(f"Tool '{tool_name}' not found", result["error"])

    def test_execute_tool_call_execution_error(self):
        """Test tool call execution when the tool raises an exception."""
        tool_name = "error_tool"
        tool_params = {}
        tool_call = {"name": tool_name, "parameters": tool_params}
        error_message = "Something went wrong!"

        self.mock_tools.has_tool.return_value = True
        self.mock_tools.execute_tool.side_effect = Exception(error_message) # Mock tool raising error

        result = self.agent._execute_tool_call(tool_call)

        self.mock_tools.has_tool.assert_called_once_with(tool_name)
        self.mock_tools.execute_tool.assert_called_once_with(tool_name, **tool_params)
        self.assertIn("error", result)
        self.assertEqual(result["error"], error_message)
        self.assertEqual(result["tool_name"], tool_name) # Ensure tool name is still included

    def test_add_tool(self):
        """Test adding a tool to the agent's registry."""
        mock_tool = MagicMock()
        mock_tool.name = "new_tool"

        self.agent.add_tool(mock_tool)

        self.mock_tools.register_tool.assert_called_once_with(mock_tool)

    def test_reset(self):
        """Test resetting the agent's state."""
        # Modify state before reset
        self.agent.current_iteration = 3
        self.agent.finished = True
        self.agent.state = {"key": "value"}

        # Perform reset
        self.agent.reset()

        # Verify state is reset
        self.assertEqual(self.agent.current_iteration, 0)
        self.assertFalse(self.agent.finished)
        self.assertEqual(self.agent.state, {})
        self.mock_memory.clear.assert_called_once() # Verify memory clear was called

    def test_reset_no_memory(self):
        """Test resetting when agent has no memory."""
        agent_no_mem = ConcreteTestAgent(name="no_mem_agent", llm=self.mock_llm, memory=None)
        agent_no_mem.current_iteration = 2

        # Reset should not fail even without memory
        agent_no_mem.reset()
        self.assertEqual(agent_no_mem.current_iteration, 0)


if __name__ == '__main__':
    unittest.main()