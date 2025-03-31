#!/usr/bin/env python3
# ai_agent_framework/main.py (Updated for Org ID)

"""
AI Agent Framework - Main Entry Point (Updated)
Includes loading OpenAI Organization ID from settings.
"""

import argparse
import asyncio
import logging
import os
import sys
import json
from typing import Dict, Optional, Any, Callable, List

# Use absolute imports assuming execution via `python -m ai_agent_framework.main`
try:
    from ai_agent_framework.config.settings import Settings
    from ai_agent_framework.config.logging_config import setup_logging
    from ai_agent_framework.core.llm.factory import LLMFactory
    from ai_agent_framework.core.llm.base import BaseLLM
    from ai_agent_framework.core.tools.registry import ToolRegistry
    from ai_agent_framework.agents.workflow_agent import WorkflowAgent
    from ai_agent_framework.agents.autonomous_agent import AutonomousAgent
    from ai_agent_framework.agents.base_agent import BaseAgent
    # Import filesystem tools for setup_tools
    from ai_agent_framework.tools.file_system.read import FileReadTool
    from ai_agent_framework.tools.file_system.write import FileWriteTool
    from ai_agent_framework.tools.file_system.list_dir import ListDirectoryTool
    # Import other tools as needed by setup_tools
    from ai_agent_framework.tools.apis.web_search import WebSearchTool
    from ai_agent_framework.tools.apis.connector import APIConnectorTool
    from ai_agent_framework.tools.data_analysis import DataAnalysisTool
    # Import RAG tools if used in setup_tools
    from ai_agent_framework.core.memory.embeddings import get_embedder
    from ai_agent_framework.core.memory.vector_store import get_vector_store
    from ai_agent_framework.tools.memory.retrieval_tool import RetrievalTool

except ImportError as e:
     print(f"Error importing framework components: {e}")
     print("Please ensure the script is run correctly (e.g., `python -m ai_agent_framework.main ...`) and all dependencies are installed.")
     sys.exit(1)


logger: Optional[logging.Logger] = None

# --- Argparse, Tool Setup, Agent Creation (Keep Previous Updated Versions) ---

def parse_arguments() -> argparse.Namespace:
    # (Keep the implementation from the previous response)
    parser = argparse.ArgumentParser(description="AI Agent Framework CLI")
    parser.add_argument("--config", type=str, help="Path to configuration file (e.g., config.yaml)")
    parser.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level (overrides settings)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging output for the agent (sets agent verbose flag).")
    parser.add_argument("--agent-type", type=str, choices=["workflow", "autonomous"], help="Type of agent to run (overrides settings)")
    parser.add_argument("--llm-provider", type=str, help="LLM provider to use (e.g., claude, openai). Overrides settings.")
    parser.add_argument("--model", type=str, help="Specific LLM model name to use (overrides settings).")
    parser.add_argument("--temperature", type=float, help="LLM temperature (overrides settings).")
    parser.add_argument("--enable-filesystem", action="store_true", help="Enable filesystem tools (read, list). Write access controlled by settings.")
    parser.add_argument("--enable-web", action="store_true", help="Enable web tools (search).")
    parser.add_argument("--enable-data-analysis", action="store_true", help="Enable data analysis tools (requires pandas).")
    subparsers = parser.add_subparsers(dest="mode", help="Agent operation mode", required=True)
    interactive_parser = subparsers.add_parser("interactive", help="Run agent in interactive mode")
    task_parser = subparsers.add_parser("task", help="Run agent on a specific task")
    task_parser.add_argument("--task", type=str, required=True, help="Task description for the agent")
    task_parser.add_argument("--input", type=str, help="Optional additional input string for the task")
    task_parser.add_argument("--input-file", type=str, help="File containing additional input for the task")
    return parser.parse_args()

def setup_tools(args: argparse.Namespace, settings: Settings) -> ToolRegistry:
    # (Keep the implementation from the previous response, including ListDirectoryTool registration)
    registry = ToolRegistry()
    if logger: logger.info("Configuring tools...")
    filesystem_enabled = args.enable_filesystem or settings.get("tools.filesystem.enabled", False)
    if filesystem_enabled:
        try:
            allowed_dirs_config = settings.get("tools.filesystem.allowed_directories", ["."]); allowed_dirs = allowed_dirs_config if isinstance(allowed_dirs_config, list) else ["."]
            if logger: logger.info(f"Filesystem tools allowed directories: {allowed_dirs}")
            registry.register_tool(FileReadTool(allowed_directories=allowed_dirs), categories=["filesystem"])
            registry.register_tool(ListDirectoryTool(allowed_directories=allowed_dirs), categories=["filesystem"])
            if logger: logger.info("Registered FileReadTool and ListDirectoryTool.")
            if settings.get("tools.filesystem.allow_write", False):
                registry.register_tool(FileWriteTool(allowed_directories=allowed_dirs), categories=["filesystem"])
                if logger: logger.warning("FileWriteTool registered via settings. Ensure allowed_directories is secure.")
            else:
                 if logger: logger.info("FileWriteTool is disabled based on settings.")
        except ImportError as e: logger.error(f"Failed to import filesystem tools: {e}.", exc_info=True) if logger else None
        except Exception as e: logger.error(f"Error initializing filesystem tools: {e}", exc_info=True) if logger else None
    web_enabled = args.enable_web or settings.get("tools.web_search.enabled", False)
    if web_enabled:
        try: registry.register_tool(WebSearchTool(settings=settings), categories=["web", "search"]); logger.info("Registered WebSearchTool.") if logger else None
        except ImportError as e: logger.warning(f"WebSearchTool dependencies missing: {e}. Web search may not function.") if logger else None
        except Exception as e: logger.error(f"Error initializing WebSearchTool: {e}", exc_info=True) if logger else None
    api_connectors_config = settings.get("tools.apis.connectors", {}); connector_available = False
    if isinstance(api_connectors_config, dict):
         try: from ai_agent_framework.tools.apis.connector import APIConnectorTool; connector_available = True
         except ImportError as e: logger.warning(f"APIConnectorTool not available: {e}") if logger else None
         if connector_available:
              for name, config in api_connectors_config.items():
                    if not isinstance(config, dict) or "base_url" not in config: logger.warning(f"Skipping invalid API connector '{name}'.") if logger else None; continue
                    try: description = config.get("description", f"Connects to {name} API"); registry.register_tool(APIConnectorTool(name=name, description=description, **config), categories=["api", name]); logger.info(f"Registered APIConnectorTool: {name}") if logger else None
                    except Exception as e: logger.error(f"Error initializing APIConnectorTool '{name}': {e}", exc_info=True) if logger else None
    data_analysis_enabled = args.enable_data_analysis or settings.get("tools.data_analysis.enabled", False)
    if data_analysis_enabled:
        try:
            if DataAnalysisTool: registry.register_tool(DataAnalysisTool(), categories=["data", "analysis"]); logger.info("Registered DataAnalysisTool.") if logger else None
            else: logger.warning("Data analysis enabled, but DataAnalysisTool failed import (pandas?).") if logger else None
        except Exception as e: logger.error(f"Error initializing DataAnalysisTool: {e}", exc_info=True) if logger else None
    kb_enabled = settings.get("tools.knowledge_base.enabled", False)
    if kb_enabled:
        try:
             embedder_type = settings.get("vector_store.embedding_type", "openai"); embedder = get_embedder(embedder_type)
             kb_path = settings.get("tools.knowledge_base.persist_path", "./data/kb"); vector_store_type = settings.get("vector_store.type", "chroma")
             kb_collection_name = settings.get("tools.knowledge_base.collection_name", "main_kb"); os.makedirs(kb_path, exist_ok=True) if vector_store_type in ["chroma", "faiss"] and kb_path else None
             vector_store = get_vector_store(vector_store_type=vector_store_type, embedder=embedder, collection_name=kb_collection_name, persist_directory=kb_path)
             retrieval_k = settings.get("tools.knowledge_base.retrieval_k", 3)
             retrieval_tool = RetrievalTool(name="knowledge_retrieve", description="Retrieve info from knowledge base.", vector_store=vector_store, embedder=embedder, default_k=retrieval_k)
             registry.register_tool(retrieval_tool, categories=["knowledge", "rag"]); logger.info(f"Registered Knowledge Retrieval tool (k={retrieval_k}).") if logger else None
        except ImportError as e: logger.error(f"Failed import RAG components: {e}. KB tools unavailable.") if logger else None
        except Exception as e: logger.error(f"Error initializing KB tools: {e}", exc_info=True) if logger else None
    if logger: logger.info(f"Tool setup complete. Tools: {list(registry.get_tool_names())}")
    return registry

def create_workflow_agent(name: str, llm: BaseLLM, tools: ToolRegistry, settings: Settings, args: argparse.Namespace) -> BaseAgent:
     # (Keep the implementation from the previous response)
     verbose_flag = args.verbose or settings.get("system.verbose", False)
     agent = WorkflowAgent(name=name, llm=llm, tools=tools, system_prompt=settings.get("agents.workflow.system_prompt"), max_iterations=settings.get("agents.workflow.max_iterations", 10), verbose=verbose_flag)
     if logger: logger.info(f"Created WorkflowAgent: {name}")
     return agent

def create_autonomous_agent(name: str, llm: BaseLLM, tools: ToolRegistry, settings: Settings, args: argparse.Namespace) -> BaseAgent:
     # (Keep the implementation from the previous response)
     verbose_flag = args.verbose or settings.get("system.verbose", False)
     agent = AutonomousAgent(name=name, llm=llm, tools=tools, system_prompt=settings.get("agents.autonomous.system_prompt"), max_iterations=settings.get("agents.autonomous.max_iterations", 15), reflection_threshold=settings.get("agents.autonomous.reflection_threshold", 3), verbose=verbose_flag)
     if logger: logger.info(f"Created AutonomousAgent: {name}")
     return agent

AGENT_FACTORIES: Dict[str, AgentCreator] = { "workflow": create_workflow_agent, "autonomous": create_autonomous_agent }

def create_agent(args: argparse.Namespace, llm: BaseLLM, tool_registry: ToolRegistry, settings: Settings) -> BaseAgent:
    # (Keep the implementation from the previous response)
    agent_type_arg = args.agent_type; agent_type_setting = settings.get("agent.default_type", "workflow"); effective_agent_type = (agent_type_arg or agent_type_setting).lower()
    factory = AGENT_FACTORIES.get(effective_agent_type)
    if factory: agent_name = f"{effective_agent_type}_cli_agent"; return factory(agent_name, llm, tool_registry, settings, args)
    else: raise ValueError(f"Unknown agent type specified: {effective_agent_type}")

async def run_interactive_mode(agent: BaseAgent):
    # (Keep the implementation from the previous response)
    print(f"\nStarting interactive session with {agent.name} (type 'exit' or 'quit' to end)"); print(f"Tools available: {', '.join(agent.tools.get_tool_names()) if agent.tools else 'None'}"); print("-" * 50)
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nYou: "); user_input_lower = user_input.lower()
            if user_input_lower in ["exit", "quit"]: print("\nEnding session."); break
            if not user_input.strip(): continue
            print("\nAgent working..."); result = await agent.run(user_input); print(f"\nAgent: {result.get('response', 'No response received.')}")
            if result.get("error"): print(f"Error during run: {result['error']}")
            if result.get("tool_calls"):
                 print("--- Tool Calls ---")
                 for i, call in enumerate(result["tool_calls"], 1):
                       tool_name = call.get('tool', call.get('name', 'Unknown')); output = call.get('output', {}); status = "Success" if isinstance(output,dict) and 'error' not in output else "Error"
                       result_data = output.get("result") if isinstance(output, dict) else output; error_data = output.get("error") if isinstance(output, dict) else None; result_preview = str(result_data if status=="Success" else error_data)[:100]
                       print(f"{i}. {tool_name}: Status={status}, Output={result_preview}...")
        except (KeyboardInterrupt, EOFError): print("\nEnding session due to interrupt."); break
        except Exception as e: logger.exception(f"Error in interactive loop: {e}") if logger else None; print(f"\nAn unexpected error occurred: {str(e)}")

async def run_task_mode(agent: BaseAgent, task: str, input_data: Optional[str]):
    # (Keep the implementation from the previous response)
    print(f"\nRunning task for {agent.name}: {task}"); print(f"Tools available: {', '.join(agent.tools.get_tool_names()) if agent.tools else 'None'}"); print("-" * 50)
    run_input: Union[str, Dict[str, Any]] = task; run_kwargs: Dict[str, Any] = {}
    if input_data: run_input = {"input": task, "additional_data": input_data}; logger.info(f"Providing additional input data for task.") if logger else None
    try:
        result = await agent.run(run_input, **run_kwargs); print("\n--- Task Result ---"); print(f"Status: {'Success' if result.get('success') else 'Failed'}")
        if result.get("error"): print(f"Error: {result['error']}")
        print("\nResponse:"); print(result.get("response", "No response content."))
        if result.get("tool_calls"):
            print("\n--- Tool Calls ---")
            for i, call in enumerate(result["tool_calls"], 1):
                  tool_name = call.get('tool', call.get('name', 'Unknown')); output = call.get('output', {}); status = "Success" if isinstance(output,dict) and 'error' not in output else "Error"
                  result_data = output.get("result") if isinstance(output, dict) else output; error_data = output.get("error") if isinstance(output, dict) else None; result_preview = str(result_data if status=="Success" else error_data)[:100]
                  print(f"{i}. {tool_name}: Status={status}, Output={result_preview}...")
    except Exception as e: logger.exception(f"Error running task '{task}': {e}") if logger else None; print(f"\nAn unexpected error occurred while running the task: {str(e)}")

async def main():
    """Main entry point function."""
    global logger

    args = parse_arguments()

    # Setup Logging
    log_dir = os.path.join(os.getcwd(), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        # Load settings temporarily just to get log level if specified there
        temp_settings = Settings(config_path=args.config)
        log_level_setting = temp_settings.get("system.log_level", "INFO")
        log_level = (args.log_level or log_level_setting).upper()
        setup_logging(log_level=log_level, log_dir=log_dir)
        logger = logging.getLogger(__name__) # Assign after setup
        logger.info(f"Logging initialized. Level: {log_level}")
    except Exception as e:
         print(f"Error setting up logging: {e}. Continuing without file logging.")
         logging.basicConfig(level=(args.log_level or "INFO").upper())
         logger = logging.getLogger(__name__)

    # Load Settings properly
    try:
        settings = Settings(config_path=args.config)
        logger.info(f"Settings loaded. Default Provider: {settings.get('llm.provider', 'Not Set')}")
    except Exception as e:
         logger.error(f"Failed to load settings: {e}. Using minimal defaults.", exc_info=True)
         print(f"Warning: Could not load settings file. Using defaults and environment variables.")
         settings = type('obj', (object,), {'get': lambda _, key, default=None: default})()

    # Determine Effective Config
    llm_provider = args.llm_provider or settings.get("llm.provider", "openai")
    llm_model = args.model or settings.get("llm.model")
    llm_temperature = args.temperature if args.temperature is not None else settings.get("llm.temperature", 0.7)
    agent_verbose = args.verbose or settings.get("system.verbose", False)
    # --- Load Org ID from settings ---
    openai_org_id = settings.get(f"llm.{llm_provider}.organization_id", settings.get("llm.organization_id"))
    # ---------------------------------


    # Initialize LLM
    try:
        llm = LLMFactory.create_llm(
            provider=llm_provider,
            model_name=llm_model,
            temperature=llm_temperature,
            organization_id=openai_org_id # <-- Pass loaded Org ID
            # timeout=settings.get(f"llm.{llm_provider}.timeout", 60) # Example timeout load
        )
    except (ValueError, ImportError) as e: logger.error(f"{e}", exc_info=True); print(f"Error: {e}"); sys.exit(1)
    except Exception as e: logger.error(f"LLM Init Error '{llm_provider}': {e}", exc_info=True); print(f"Error: Check LLM config/keys."); sys.exit(1)

    # Set up Tools
    try: tool_registry = setup_tools(args, settings)
    except Exception as e: logger.error(f"Tool setup failed: {e}", exc_info=True); print(f"Error: Tool setup failed."); sys.exit(1)

    # Create Agent
    try: args.verbose = agent_verbose; agent = create_agent(args, llm, tool_registry, settings)
    except Exception as e: logger.error(f"Agent creation failed: {e}", exc_info=True); print(f"Error: Could not create agent."); sys.exit(1)

    # Get Input for Task Mode
    # (Implementation remains the same as previous version)
    input_data_for_task: Optional[str] = None
    if args.mode == "task":
        if args.input_file:
            try:
                input_file_path = os.path.abspath(args.input_file); logger.info(f"Reading input from: {input_file_path}")
                with open(input_file_path, 'r', encoding='utf-8') as f: input_data_for_task = f.read()
            except FileNotFoundError: logger.error(f"Input file not found: {input_file_path}"); print(f"Error: Input file not found."); sys.exit(1)
            except Exception as e: logger.error(f"Failed read input file {input_file_path}: {e}", exc_info=True); print(f"Error: Could read input file."); sys.exit(1)
        else: input_data_for_task = args.input

    # Run Mode
    if args.mode == "interactive": await run_interactive_mode(agent)
    elif args.mode == "task": await run_task_mode(agent, args.task, input_data_for_task)
    else: logger.error(f"Invalid mode: {args.mode}"); print(f"Error: Invalid mode."); sys.exit(1)

if __name__ == "__main__":
    # (Keep checks and main execution block from previous version)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"): print("Warning: No default LLM API key found (checked ANTHROPIC_API_KEY, OPENAI_API_KEY).")
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\nOperation cancelled by user."); sys.exit(0)
    except Exception as e: print(f"\nAn unexpected critical error occurred: {e}"); logger.critical(f"Unhandled exception: {e}", exc_info=True) if logger else None; sys.exit(1)