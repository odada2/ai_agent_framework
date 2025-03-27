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
from ..core.memory.knowledge_base import KnowledgeBase
from ..core.memory.embeddings import get_embedder
from ..tools.memory.retrieval_tool import RetrievalTool, ConversationMemoryTool

logger = logging.getLogger(__name__)


class CLI:
    """Command-line interface for the AI Agent Framework."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.settings = Settings()
        self.agent = None
        self.tools = ToolRegistry()
        self.knowledge_base = None
    
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
            
        # Set up knowledge base tools if enabled
        if args.enable_knowledge_base or self.settings.get("tools.enable_knowledge_base", False):
            # Set up embedder
            embedder_type = args.embedder_type or self.settings.get("vector_store.embedding_type", "openai")
            embedder = get_embedder(embedder_type)
            
            # Set up knowledge base
            kb_path = args.knowledge_base_path or self.settings.get("tools.knowledge_base.path", "./data/knowledge_base")
            
            self.knowledge_base = KnowledgeBase(
                name=args.knowledge_base_name or "default",
                embedder=embedder,
                vector_store_type=args.vector_store_type or self.settings.get("vector_store.default_type", "chroma"),
                persist_path=kb_path
            )
            
            # Register retrieval tool
            retrieval_tool = RetrievalTool(
                vector_store=self.knowledge_base.vector_store,
                embedder=embedder,
                default_k=args.retrieval_k or 3
            )
            
            self.tools.register_tool(retrieval_tool, categories=["knowledge"])
            
            # Optionally register conversation memory tool
            if args.enable_memory_search or self.settings.get("tools.enable_conversation_memory", False):
                memory_tool = ConversationMemoryTool(
                    vector_store_type="faiss",
                    embedder=embedder
                )
                
                self.tools.register_tool(memory_tool, categories=["memory"])
    
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
        
        # Add RAG specific system prompt if knowledge base is enabled
        system_prompt = args.system_prompt
        if self.knowledge_base:
            if system_prompt:
                system_prompt += "\n\nYou have access to a knowledge base via the 'retrieve' tool. Use it to find relevant information before responding."
            else:
                system_prompt = "You are a helpful assistant with access to a knowledge base. Use the 'retrieve' tool to find relevant information before responding."
        
        # Create agent
        return WorkflowAgent(
            name="cli_workflow_agent",
            llm=llm,
            tools=self.tools,
            workflows=workflows,
            default_workflow="default",
            system_prompt=system_prompt,
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
        # Add RAG specific system prompt if knowledge base is enabled
        system_prompt = args.system_prompt
        if self.knowledge_base:
            if system_prompt:
                system_prompt += "\n\nYou have access to a knowledge base via the 'retrieve' tool. Use it to find relevant information before responding."
            else:
                system_prompt = "You are a helpful assistant with access to a knowledge base. Use the 'retrieve' tool to find relevant information before responding."
        
        return AutonomousAgent(
            name="cli_autonomous_agent",
            llm=llm,
            tools=self.tools,
            system_prompt=system_prompt,
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
                        print(f"  - {call['tool']}")
            
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
                    print(f"  - {call['tool']}")
        
        except Exception as e:
            print(f"Error: {str(e)}")
    
    async def import_knowledge(self, directory, file_types=None, recursive=True):
        """
        Import knowledge from a directory.
        
        Args:
            directory: Directory path
            file_types: File types to import
            recursive: Whether to search recursively
        """
        if not self.knowledge_base:
            print("Knowledge base not enabled.")
            return
        
        print(f"Importing knowledge from {directory}...")
        
        try:
            results = await self.knowledge_base.add_documents_from_directory(
                directory=directory,
                source_name=os.path.basename(directory),
                file_types=file_types,
                recursive=recursive
            )
            
            print(f"Import complete:")
            print(f"  Added: {results['added']} documents")
            print(f"  Skipped: {results['skipped']} files")
            print(f"  Failed: {results['failed']} files")
            
            if results['failed'] > 0:
                print("\nErrors:")
                for error in results['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error['file']}: {error['error']}")
                
                if len(results['errors']) > 5:
                    print(f"  ...and {len(results['errors']) - 5} more errors")
            
            # Show stats
            stats = await self.knowledge_base.get_stats()
            print(f"\nKnowledge base now contains {stats['documents']} documents")
            
        except Exception as e:
            print(f"Error importing knowledge: {str(e)}")
    
    async def search_knowledge(self, query, k=3, filter=None):
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
        """
        if not self.knowledge_base:
            print("Knowledge base not enabled.")
            return
        
        try:
            results = await self.knowledge_base.query(query, k=k, filter=filter)
            
            print(f"Search results for: {query}")
            print(f"Found {len(results['results'])} matching documents\n")
            
            for i, result in enumerate(results['results'], 1):
                print(f"{i}. Score: {result['score']:.4f}")
                print(f"   Source: {result.get('metadata', {}).get('source', 'unknown')}")
                
                # Get snippet of content (first 200 chars)
                content = result['content']
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"   Content: {content}")
                print()
                
        except Exception as e:
            print(f"Error searching knowledge base: {str(e)}")


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
    
    # Vector store and knowledge base configuration
    parser.add_argument("--enable-knowledge-base", action="store_true",
                        help="Enable knowledge base tools")
    parser.add_argument("--knowledge-base-path", type=str,
                        help="Path to knowledge base data")
    parser.add_argument("--knowledge-base-name", type=str, default="default",
                        help="Name of the knowledge base")
    parser.add_argument("--vector-store-type", type=str, default="chroma",
                        choices=["chroma", "faiss"],
                        help="Type of vector store to use")
    parser.add_argument("--embedder-type", type=str, default="openai",
                        choices=["openai", "local"],
                        help="Type of embedder to use")
    parser.add_argument("--retrieval-k", type=int, default=3,
                        help="Number of results to retrieve")
    parser.add_argument("--enable-memory-search", action="store_true",
                        help="Enable semantic search over conversation history")
    
    # Agent mode
    subparsers = parser.add_subparsers(dest="mode", help="Agent operation mode")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run agent in interactive mode")
    
    # Run a specific task
    task_parser = subparsers.add_parser("task", help="Run agent on a specific task")
    task_parser.add_argument("--task", type=str, required=True, help="Task to run")
    task_parser.add_argument("--input", type=str, help="Input for the task")
    task_parser.add_argument("--input-file", type=str, help="File containing input for the task")
    
    # Knowledge base management
    kb_parser = subparsers.add_parser("knowledge", help="Manage knowledge base")
    kb_subparsers = kb_parser.add_subparsers(dest="kb_action", help="Knowledge base action")
    
    # Import knowledge
    import_parser = kb_subparsers.add_parser("import", help="Import knowledge from files")
    import_parser.add_argument("--directory", type=str, required=True, help="Directory to import from")
    import_parser.add_argument("--recursive", action="store_true", help="Search directories recursively")
    import_parser.add_argument("--file-types", type=str, nargs="+", default=[".txt", ".md", ".pdf"],
                             help="File types to import")
    
    # Search knowledge
    search_parser = kb_subparsers.add_parser("search", help="Search knowledge base")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument("--k", type=int, default=3, help="Number of results to return")
    search_parser.add_argument("--filter", type=str, help="JSON-formatted metadata filter")
    
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
    elif args.mode == "knowledge":
        if args.kb_action == "import":
            await cli.import_knowledge(
                directory=args.directory,
                file_types=args.file_types,
                recursive=args.recursive
            )
        elif args.kb_action == "search":
            filter_dict = None
            if args.filter:
                import json
                try:
                    filter_dict = json.loads(args.filter)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON filter: {args.filter}")
                    return
            
            await cli.search_knowledge(
                query=args.query,
                k=args.k,
                filter=filter_dict
            )
    else:
        # Default to interactive if no mode specified
        await cli.run_interactive_mode()


def cli_entry_point():
    """Entry point for setuptools console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())