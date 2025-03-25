"""
Command Line Interface

This module provides a command-line interface for interacting with the AI Agent Framework.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional, Union

from ..core.llm.factory import LLMFactory
from ..core.tools.registry import ToolRegistry
from ..agents.workflow_agent import WorkflowAgent
from ..agents.autonomous_agent import AutonomousAgent
from ..core.workflow.chain import PromptChain
from ..config.settings import Settings
from ..config.logging_config import setup_logging

logger = logging.getLogger(__name__)


class CLI:
    """Command-line interface for the AI Agent Framework."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.settings = Settings()
        self.agent = None
        self.tools = ToolRegistry()
    
    async def setup(self, args):
        """
        Set up the CLI based on command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Set up logging
        setup_logging(log_level=args.log_level)
        
        # Set up LLM
        llm = LLMFactory.create_llm(
            provider=args.llm_provider,
            model_name=args.model,
            temperature=args.temperature
        )
        
        # Set up tools
        await self._setup_tools(args)
        
        # Create agent
        if args.agent_type == "workflow":
            self.agent = self._create_workflow_agent(llm, args)
        else:  # autonomous
            self.agent = self._create_autonomous_agent(llm, args)
        
        logger.info(f"Set up {args.agent_type} agent with {len(self.tools)} tools")
    
    async def _setup_tools(self, args):
        """
        Set up the tools based on command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Set up filesystem tools if enabled
        if args.enable_filesystem or self.settings.get("tools.enable_filesystem", False):
            from ..tools.file_system.read import FileReadTool
            from ..tools.file_system.write import FileWriteTool
            
            allowed_dirs = self.settings.get("tools.filesystem.allowed_directories", ["."])
            
            self.tools.register_tool(
                FileReadTool(allowed_directories=allowed_dirs),
                categories=["filesystem"]
            )
            
            # Only register write tool if explicitly allowed
            if args.enable_write or self.settings.get("tools.filesystem.allow_write", False):
                self.tools.register_tool(
                    FileWriteTool(allowed_directories=allowed_dirs),
                    categories=["filesystem"]
                )
        
        # Set up web tools if enabled
        if args.enable_web or self.settings.get("tools.enable_web_access", False):
            from ..tools.apis.web_search import WebSearchTool
            
            self.tools.register_tool(WebSearchTool(), categories=["web"])
        
        # Set up data analysis tools if enabled
        if args.enable_data_analysis or self.settings.get("tools.enable_data_analysis", False):
            from ..tools.data_analysis.analyzer import DataAnalysisTool
            
            self.tools.register_tool(DataAnalysisTool(), categories=["data"])
    
    def _create_workflow_agent(self, llm, args):
        """
        Create a workflow agent.
        
        Args:
            llm: The LLM to use
            args: Command-line arguments
            
        Returns:
            Configured WorkflowAgent
        """
        # Set up a simple default workflow
        steps = [
            {
                "name": "process",
                "prompt_template": "{input}",
                "use_tools": True,
                "return_tool_results": True,
            }
        ]
        
        default_workflow = PromptChain(
            name="default_chain",
            llm=llm,
            steps=steps,
            tools=self.tools,
            verbose=args.verbose
        )
        
        # Create workflows dict
        workflows = {"default": default_workflow}
        
        # Create agent
        return WorkflowAgent(
            name="cli_workflow_agent",
            llm=llm,
            tools=self.tools,
            workflows=workflows,
            default_workflow="default",
            system_prompt=args.system_prompt,
            max_iterations=args.max_iterations,
            verbose=args.verbose
        )
    
    def _create_autonomous_agent(self, llm, args):
        """
        Create an autonomous agent.
        
        Args:
            llm: The LLM to use
            args: Command-line arguments
            
        Returns:
            Configured AutonomousAgent
        """
        return AutonomousAgent(
            name="cli_autonomous_agent",
            llm=llm,
            tools=self.tools,
            system_prompt=args.system_prompt,
            max_iterations=args.max_iterations,
            reflection_threshold=args.reflection_threshold,
            verbose=args.verbose
        )
    
    async def run_interactive_mode(self):
        """Run the agent in interactive mode."""
        print(f"Starting interactive session with {self.agent.name}")
        print("Type 'exit' or 'quit' to end the session")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Ending session.")
                    break
                
                # Run the agent
                response = await self.agent.run(user_input)
                
                # Print response
                print(f"\nAgent: {response.get('response', '')}")
                
                # If there were tool calls, show them
                if "tool_calls" in response and response["tool_calls"]:
                    print("\nTool Calls:")
                    for call in response["tool_calls"]:
                        print(f"  - {call['tool']}: {call.get('result', 'No result')}")
            
            except KeyboardInterrupt:
                print("\nInterrupted by user. Ending session.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def run_task_mode(self, task, input_data):
        """
        Run the agent on a specific task.
        
        Args:
            task: Task name
            input_data: Input data for the task
        """
        print(f"Running task: {task}")
        print("-" * 50)
        
        try:
            response = await self.agent.run(input_data)
            
            print("\nResult:")
            print(response.get("response", ""))
            
            # If there were tool calls, show them
            if "tool_calls" in response and response["tool_calls"]:
                print("\nTool Calls:")
                for call in response["tool_calls"]:
                    print(f"  - {call['tool']}: {call.get('result', 'No result')}")
        
        except Exception as e:
            print(f"Error: {str(e)}")


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
    parser.add_argument("--system-prompt", type=str,
                        help="System prompt for the agent")
    
    # Agent parameters
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum number of iterations")
    parser.add_argument("--reflection-threshold", type=int, default=3,
                        help="Number of iterations after which the agent reflects (autonomous only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    # Tool configuration
    parser.add_argument("--enable-filesystem", action="store_true",
                        help="Enable filesystem tools")
    parser.add_argument("--enable-write", action="store_true",
                        help="Enable file writing (requires --enable-filesystem)")
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


async def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Create CLI
    cli = CLI()
    
    # Set up CLI
    await cli.setup(args)
    
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
        await cli.run_interactive_mode()
    elif args.mode == "task":
        await cli.run_task_mode(args.task, input_data)
    else:
        # Default to interactive if no mode specified
        await cli.run_interactive_mode()


def cli_entry_point():
    """Entry point for setuptools console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())