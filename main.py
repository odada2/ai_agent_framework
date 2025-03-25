#!/usr/bin/env python3
"""
AI Agent Framework - Main Entry Point

This script serves as the main entry point for the AI Agent Framework, providing
command-line interface to initialize and run different types of agents with various
configurations.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, Optional

# Add the current directory to path to allow running from any location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_agent_framework.config.settings import Settings
from ai_agent_framework.config.logging_config import setup_logging
from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.agents.autonomous_agent import AutonomousAgent


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Agent Framework CLI")
    
    # Basic configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    # Agent selection
    parser.add_argument("--agent-type", type=str, default="workflow",
                        choices=["workflow", "autonomous"],
                        help="Type of agent to run")
    
    # LLM configuration
    parser.add_argument("--llm-provider", type=str, default="claude",
                        help="LLM provider to use")
    parser.add_argument("--model", type=str, 
                        help="Model name to use (defaults to provider's default)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for LLM generation")
    
    # Tool configuration
    parser.add_argument("--enable-filesystem", action="store_true",
                        help="Enable filesystem tools")
    parser.add_argument("--enable-web", action="store_true",
                        help="Enable web tools")
    parser.add_argument("--enable-data-analysis", action="store_true",
                        help="Enable data analysis tools")
    
    # Agent mode
    subparsers = parser.add_subparsers(dest="mode", help="Agent operation mode")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run agent in interactive mode")
    
    # Run a specific task
    task_parser = subparsers.add_parser("task", help="Run agent on a specific task")
    task_parser.add_argument("--task", type=str, required=True, help="Task to run")
    task_parser.add_argument("--input", type=str, help="Input for the task")
    task_parser.add_argument("--input-file", type=str, help="File containing input for the task")
    
    return parser.parse_args()


def setup_tools(args, settings: Settings) -> ToolRegistry:
    """
    Set up and configure tools based on command line arguments and settings.
    
    Args:
        args: Command line arguments
        settings: Configuration settings
        
    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()
    
    # Load filesystem tools if enabled
    if args.enable_filesystem or settings.get("tools.enable_filesystem", False):
        from ai_agent_framework.tools.file_system.read import FileReadTool
        from ai_agent_framework.tools.file_system.write import FileWriteTool
        
        allowed_dirs = settings.get("tools.filesystem.allowed_directories", ["."])
        
        registry.register_tool(
            FileReadTool(allowed_directories=allowed_dirs),
            categories=["filesystem"]
        )
        # Only register write tool if explicitly allowed
        if settings.get("tools.filesystem.allow_write", False):
            registry.register_tool(
                FileWriteTool(allowed_directories=allowed_dirs),
                categories=["filesystem"]
            )
    
    # Load web tools if enabled
    if args.enable_web or settings.get("tools.enable_web_access", False):
        from ai_agent_framework.tools.apis.web_search import WebSearchTool
        from ai_agent_framework.tools.apis.connector import APIConnectorTool
        
        registry.register_tool(WebSearchTool(), categories=["web"])
        
        # Load API connectors from configuration
        api_configs = settings.get("tools.apis.connectors", {})
        for name, config in api_configs.items():
            registry.register_tool(
                APIConnectorTool(name=name, **config),
                categories=["api"]
            )
    
    # Load data analysis tools if enabled
    if args.enable_data_analysis or settings.get("tools.enable_data_analysis", False):
        from ai_agent_framework.tools.data_analysis.analyzer import DataAnalysisTool
        
        registry.register_tool(DataAnalysisTool(), categories=["data"])
    
    return registry


def create_agent(agent_type: str, llm_provider: str, model: Optional[str], 
                 temperature: float, tool_registry: ToolRegistry, 
                 settings: Settings):
    """
    Create an agent based on the specified type and configuration.
    
    Args:
        agent_type: Type of agent to create ('workflow' or 'autonomous')
        llm_provider: LLM provider to use
        model: Model name to use (or None for default)
        temperature: Temperature for LLM generation
        tool_registry: Configured ToolRegistry
        settings: Configuration settings
        
    Returns:
        Configured agent instance
    """
    # Create LLM
    llm = LLMFactory.create_llm(
        provider=llm_provider,
        model_name=model,
        temperature=temperature
    )
    
    # Get agent-specific settings
    max_iterations = settings.get(f"agent.{agent_type}.max_iterations", 10)
    system_prompt = settings.get(f"agent.{agent_type}.system_prompt", None)
    
    # Create the appropriate agent type
    if agent_type == "workflow":
        agent = WorkflowAgent(
            name="workflow_agent",
            llm=llm,
            tools=tool_registry,
            system_prompt=system_prompt,
            max_iterations=max_iterations
        )
    else:  # autonomous
        agent = AutonomousAgent(
            name="autonomous_agent",
            llm=llm,
            tools=tool_registry,
            system_prompt=system_prompt,
            max_iterations=max_iterations
        )
    
    return agent


async def run_interactive_mode(agent):
    """Run agent in interactive mode."""
    print(f"Starting interactive session with {agent.name}")
    print("Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Ending session.")
                break
            
            # Run the agent
            response = await agent.run(user_input)
            
            # Print response
            print(f"\nAgent: {response.get('response', '')}")
            
            # If there were tool calls, show them
            if "tool_calls" in response and response["tool_calls"]:
                print("\nTool Calls:")
                for call in response["tool_calls"]:
                    print(f"  - {call['name']}: {call.get('result', 'No result')}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user. Ending session.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


async def run_task_mode(agent, task: str, input_data: str):
    """Run agent on a specific task."""
    print(f"Running task: {task}")
    print("-" * 50)
    
    try:
        response = await agent.run(input_data)
        
        print("\nResult:")
        print(response.get("response", ""))
        
        # If there were tool calls, show them
        if "tool_calls" in response and response["tool_calls"]:
            print("\nTool Calls:")
            for call in response["tool_calls"]:
                print(f"  - {call['name']}: {call.get('result', 'No result')}")
    
    except Exception as e:
        print(f"Error: {str(e)}")


async def main():
    """Main entry point function."""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    # Load settings
    settings = Settings(config_path=args.config)
    
    # Set up tools
    tool_registry = setup_tools(args, settings)
    
    # Create agent
    agent = create_agent(
        agent_type=args.agent_type,
        llm_provider=args.llm_provider,
        model=args.model,
        temperature=args.temperature,
        tool_registry=tool_registry,
        settings=settings
    )
    
    # Get input data for task mode
    input_data = None
    if args.mode == "task":
        if args.input_file:
            with open(args.input_file, 'r') as f:
                input_data = f.read()
        else:
            input_data = args.input or ""
    
    # Run in the appropriate mode
    if args.mode == "interactive":
        await run_interactive_mode(agent)
    elif args.mode == "task":
        await run_task_mode(agent, args.task, input_data)
    else:
        # Default to interactive if no mode specified
        await run_interactive_mode(agent)


if __name__ == "__main__":
    asyncio.run(main())