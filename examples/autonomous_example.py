"""
Autonomous Agent Example

This script demonstrates how to use the AutonomousAgent class to create and
run an agent that can perform tasks autonomously.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Add parent directory to path to run as standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.agents.autonomous_agent import AutonomousAgent
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.tools.file_system.read import FileReadTool
from ai_agent_framework.tools.apis.web_search import WebSearchTool
from ai_agent_framework.config.settings import Settings
from ai_agent_framework.config.logging_config import setup_logging


async def example_research_task():
    """Example of using the autonomous agent for a research task."""
    
    print("\n=== Example: Research Task ===\n")
    
    # Create LLM
    llm = LLMFactory.create_llm(provider="claude")
    
    # Create tools
    tools = ToolRegistry()
    tools.register_tool(WebSearchTool())
    
    # Create agent
    agent = AutonomousAgent(
        name="research_agent",
        llm=llm,
        tools=tools,
        max_iterations=8,
        reflection_threshold=3,
        verbose=True
    )
    
    # Run the agent
    task = "Research the latest developments in large language model compression techniques"
    print(f"Task: {task}\n")
    
    result = await agent.run(task)
    
    print("\n=== Task Complete ===\n")
    print(f"Response: {result['response']}")
    print(f"\nIterations: {result['iterations']}")
    
    if result.get("tool_calls"):
        print("\nTool Calls:")
        for i, call in enumerate(result["tool_calls"], 1):
            print(f"  {i}. {call['tool']} - Parameters: {call['input']}")


async def example_document_analysis():
    """Example of using the autonomous agent for document analysis."""
    
    print("\n=== Example: Document Analysis ===\n")
    
    # Create LLM
    llm = LLMFactory.create_llm(provider="claude")
    
    # Create tools
    tools = ToolRegistry()
    tools.register_tool(FileReadTool(allowed_directories=["./data", "./docs"]))
    
    # Create agent
    agent = AutonomousAgent(
        name="document_analysis_agent",
        llm=llm,
        tools=tools,
        max_iterations=5,
        verbose=True
    )
    
    # Run the agent
    task = "Analyze the README.md file and suggest improvements"
    print(f"Task: {task}\n")
    
    result = await agent.run(task)
    
    print("\n=== Task Complete ===\n")
    print(f"Response: {result['response']}")
    print(f"\nIterations: {result['iterations']}")
    
    if result.get("tool_calls"):
        print("\nTool Calls:")
        for i, call in enumerate(result["tool_calls"], 1):
            print(f"  {i}. {call['tool']} - Parameters: {call['input']}")


async def interactive_example():
    """Example of using the autonomous agent interactively."""
    
    print("\n=== Interactive Mode ===\n")
    print("Type 'exit' or 'quit' to end the session")
    
    # Create LLM
    llm = LLMFactory.create_llm(provider="claude")
    
    # Create tools
    tools = ToolRegistry()
    tools.register_tool(FileReadTool())
    tools.register_tool(WebSearchTool())
    
    # Create agent
    agent = AutonomousAgent(
        name="interactive_agent",
        llm=llm,
        tools=tools,
        max_iterations=6,
        verbose=True
    )
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending session.")
            break
        
        print("\nAgent is working...")
        result = await agent.run(user_input)
        
        print(f"\nAgent: {result['response']}")
        
        if result.get("tool_calls"):
            print("\nTools Used:")
            for call in result["tool_calls"]:
                print(f"  - {call['tool']}")


async def main():
    """Main function to run the examples."""
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Load settings
    settings = Settings()
    
    # Check environment variables for API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
        print("You may need to set this before running the examples.")
    
    # Choose which example to run
    print("Available examples:")
    print("1. Research Task")
    print("2. Document Analysis")
    print("3. Interactive Mode")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        await example_research_task()
    elif choice == "2":
        await example_document_analysis()
    elif choice == "3":
        await interactive_example()
    else:
        print("Invalid choice. Please run again with a valid option.")


if __name__ == "__main__":
    asyncio.run(main())