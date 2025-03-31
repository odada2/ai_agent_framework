# ai_agent_framework/interfaces/cli.py
# Final Updated Version incorporating all changes

"""
Command Line Interface

This module provides a command-line interface for interacting with the
AI Agent Framework, including running agents and managing the knowledge base.
"""

import argparse
import asyncio
import logging
import os
import sys
import json # For parsing filters
from typing import Dict, List, Optional, Union, Any, Callable

# Use absolute imports assuming execution via setuptools entry point `ai-agent`
# or `python -m ai_agent_framework.interfaces.cli`
try:
    from ai_agent_framework.core.llm.factory import LLMFactory
    from ai_agent_framework.core.llm.base import BaseLLM # Import BaseLLM type
    from ai_agent_framework.core.tools.registry import ToolRegistry
    from ai_agent_framework.agents.workflow_agent import WorkflowAgent
    from ai_agent_framework.agents.autonomous_agent import AutonomousAgent
    from ai_agent_framework.config.settings import Settings
    from ai_agent_framework.config.logging_config import setup_logging
    from ai_agent_framework.agents.base_agent import BaseAgent # Import BaseAgent type
    # Import PromptChain only if needed for default workflow creation
    # from ai_agent_framework.core.workflow.chain import PromptChain
except ImportError as e:
     # Provide specific error context if run directly
     print(f"Error importing framework components: {e}", file=sys.stderr)
     print("Ensure the framework is installed (`pip install .` or `pip install -e .` from project root) or PYTHONPATH is set.", file=sys.stderr)
     sys.exit(1)


# --- Conditional Tool/KB Imports ---
# Allow CLI to partially function even if optional dependencies are missing
# Define placeholder types for use when imports fail
KnowledgeBase = type('KnowledgeBase', (object,), {})
Embedder = type('Embedder', (object,), {})
RetrievalTool = type('RetrievalTool', (object,), {})
ConversationMemoryTool = type('ConversationMemoryTool', (object,), {})
FileReadTool = type('FileReadTool', (object,), {})
FileWriteTool = type('FileWriteTool', (object,), {})
ListDirectoryTool = type('ListDirectoryTool', (object,), {})
WebSearchTool = type('WebSearchTool', (object,), {})
DataAnalysisTool = type('DataAnalysisTool', (object,), {})
APIConnectorTool = type('APIConnectorTool', (object,), {})

KB_AVAILABLE = False
FILESYSTEM_AVAILABLE = False
WEB_SEARCH_AVAILABLE = False
DATA_ANALYSIS_AVAILABLE = False
APICONNECTOR_AVAILABLE = False

try:
    from ai_agent_framework.core.memory.knowledge_base import KnowledgeBase
    from ai_agent_framework.core.memory.embeddings import get_embedder, Embedder
    from ai_agent_framework.tools.memory.retrieval_tool import RetrievalTool, ConversationMemoryTool
    KB_AVAILABLE = True
except ImportError: pass
try:
    from ai_agent_framework.tools.file_system.read import FileReadTool
    from ai_agent_framework.tools.file_system.write import FileWriteTool
    from ai_agent_framework.tools.file_system.list_dir import ListDirectoryTool
    FILESYSTEM_AVAILABLE = True
except ImportError: pass
try:
    from ai_agent_framework.tools.apis.web_search import WebSearchTool
    WEB_SEARCH_AVAILABLE = True
except ImportError: pass
try:
    # Data analysis tool requires pandas
    from ai_agent_framework.tools.data_analysis import DataAnalysisTool
    DATA_ANALYSIS_AVAILABLE = True
except ImportError: pass
try:
    from ai_agent_framework.tools.apis.connector import APIConnectorTool
    APICONNECTOR_AVAILABLE = True
except ImportError: pass

# --- Logger Setup ---
# Logger is assigned in main() after setup_logging is called
logger: Optional[logging.Logger] = None

# --- CLI Class Definition ---

class CLI:
    """Command-line interface logic for the AI Agent Framework."""

    def __init__(self):
        """Initialize the CLI, loading settings."""
        try:
            self.settings = Settings() # Loads defaults, config file, .env
        except Exception as e:
             print(f"Warning: Error loading settings: {e}. Using minimal defaults.", file=sys.stderr)
             self.settings = type('obj', (object,), {'get': lambda _, key, default=None: default})() # Dummy settings

        self.agent: Optional[BaseAgent] = None
        self.llm: Optional[BaseLLM] = None # Store LLM instance
        self.tools = ToolRegistry()
        # Use 'Any' type hint for KB if KB_AVAILABLE might be False
        self.knowledge_base: Optional[Any] = None # Initialized in setup if enabled

        # Store availability flags for conditional setup
        self.kb_available = KB_AVAILABLE
        self.filesystem_available = FILESYSTEM_AVAILABLE
        self.web_search_available = WEB_SEARCH_AVAILABLE
        self.data_analysis_available = DATA_ANALYSIS_AVAILABLE
        self.api_connector_available = APICONNECTOR_AVAILABLE

    async def setup(self, args: argparse.Namespace):
        """
        Set up CLI components based on args and settings. Initializes LLM,
        Tools, and optionally KnowledgeBase. Agent is created later if needed.
        """
        global logger
        if not logger: # Safety check
            logging.basicConfig(level=(args.log_level or "INFO").upper())
            logger = logging.getLogger(__name__)
        logger.info("CLI setup starting...")

        # --- Determine Effective Config (Args > Settings > Defaults) ---
        # Use getattr safely as args object structure depends on subparser
        llm_provider = getattr(args, 'llm_provider', None) or self.settings.get("llm.provider", "openai")
        llm_model = getattr(args, 'model', None) or self.settings.get("llm.model")
        llm_temperature = args.temperature if hasattr(args, 'temperature') and args.temperature is not None else self.settings.get("llm.temperature", 0.7)
        # Load Org ID from settings (env var loaded by Settings class)
        openai_org_id = self.settings.get(f"llm.{llm_provider}.organization_id", self.settings.get("llm.organization_id"))

        # --- Set up LLM ---
        try:
            logger.info(f"Initializing LLM: Provider={llm_provider}, Model={llm_model or 'default'}")
            self.llm = LLMFactory.create_llm(
                provider=llm_provider,
                model_name=llm_model,
                temperature=llm_temperature,
                organization_id=openai_org_id # Pass Org ID loaded from settings
                # API key loaded from env by factory/constructor
            )
        except (ValueError, ImportError) as e: logger.error(f"{e}", exc_info=True); print(f"Error initializing LLM: {e}. Check provider, model, dependencies, and API keys.", file=sys.stderr); sys.exit(1)
        except Exception as e: logger.error(f"Unexpected error initializing LLM '{llm_provider}': {e}", exc_info=True); print(f"Error: Unexpected problem initializing LLM.", file=sys.stderr); sys.exit(1)

        # --- Set up Tools and KnowledgeBase (if enabled) ---
        await self._setup_tools(args) # Populates self.tools and self.knowledge_base

        logger.info("CLI setup complete.")


    async def _setup_tools(self, args: argparse.Namespace):
        """Set up ToolRegistry and KnowledgeBase based on args and settings."""
        logger.info("Setting up tools...")
        # Check flags from potentially different subparsers safely
        # Prioritize arg flag > settings flag > default (False)
        enable_fs = getattr(args, 'enable_filesystem', False) or self.settings.get("tools.filesystem.enabled", False)
        enable_web = getattr(args, 'enable_web', False) or self.settings.get("tools.web_search.enabled", False)
        enable_data = getattr(args, 'enable_data_analysis', False) or self.settings.get("tools.data_analysis.enabled", False)
        # KB needed if agent needs it OR if knowledge command is run
        enable_kb_for_agent = getattr(args, 'enable_knowledge_base', False) or self.settings.get("tools.knowledge_base.enabled", False)
        enable_mem_search = getattr(args, 'enable_memory_search', False) or self.settings.get("tools.conversation_memory.search_enabled", False)
        kb_needed = enable_kb_for_agent or args.command == "knowledge"

        # Filesystem Tools
        if enable_fs:
            if self.filesystem_available:
                try:
                    allowed_dirs = self.settings.get("tools.filesystem.allowed_directories", ["."]); allowed_dirs = allowed_dirs if isinstance(allowed_dirs, list) else ["."]
                    self.tools.register_tool(FileReadTool(allowed_directories=allowed_dirs), categories=["filesystem"])
                    self.tools.register_tool(ListDirectoryTool(allowed_directories=allowed_dirs), categories=["filesystem"])
                    logger.info("Registered FileReadTool, ListDirectoryTool.")
                    if self.settings.get("tools.filesystem.allow_write", False):
                        self.tools.register_tool(FileWriteTool(allowed_directories=allowed_dirs), categories=["filesystem"])
                        logger.warning("FileWriteTool registered via settings.")
                except Exception as e: logger.error(f"Error registering filesystem tools: {e}", exc_info=True)
            else: logger.warning("Filesystem tools enabled but dependencies missing.")

        # Web Tools
        if enable_web:
            if self.web_search_available:
                try: self.tools.register_tool(WebSearchTool(settings=self.settings), categories=["web", "search"]); logger.info("Registered WebSearchTool.")
                except Exception as e: logger.error(f"Failed init WebSearchTool: {e}")
            else: logger.warning("Web search enabled but dependencies missing.")

        # Data Analysis Tools
        if enable_data:
            if self.data_analysis_available:
                try: self.tools.register_tool(DataAnalysisTool(), categories=["data", "analysis"]); logger.info("Registered DataAnalysisTool.")
                except Exception as e: logger.error(f"Failed init DataAnalysisTool: {e}")
            else: logger.warning("Data analysis enabled but pandas/tool missing.")

        # API Connectors (from settings)
        if self.api_connector_available:
            api_connectors_config = self.settings.get("tools.apis.connectors", {})
            if isinstance(api_connectors_config, dict):
                for name, config in api_connectors_config.items():
                    if not isinstance(config, dict) or "base_url" not in config: logger.warning(f"Skipping invalid API connector '{name}'."); continue
                    try: desc = config.get("description", f"Connects to {name} API"); self.tools.register_tool(APIConnectorTool(name=name, description=desc, **config), categories=["api", name]); logger.info(f"Registered APIConnectorTool: {name}")
                    except Exception as e: logger.error(f"Error initializing APIConnectorTool '{name}': {e}", exc_info=True)

        # Knowledge Base Tools & Instance Initialization
        if kb_needed:
            if self.kb_available:
                try:
                    # Use getattr to safely access args that might belong to different subparsers
                    # Prioritize CLI args > settings > defaults
                    embedder_type = getattr(args, 'embedder_type', None) or self.settings.get("vector_store.embedding_type", "openai")
                    embedder = get_embedder(embedder_type) # Factory handles LLM API keys via env/settings if needed

                    kb_path = getattr(args, 'knowledge_base_path', None) or self.settings.get("tools.knowledge_base.persist_path", "./data/kb")
                    vector_store_type = getattr(args, 'vector_store_type', None) or self.settings.get("vector_store.type", "chroma")
                    kb_name = getattr(args, 'knowledge_base_name', None) or self.settings.get("tools.knowledge_base.collection_name", "cli_kb")

                    if vector_store_type in ["chroma", "faiss"] and kb_path: os.makedirs(kb_path, exist_ok=True)

                    # Initialize KB instance - needed for agent tools AND knowledge commands
                    self.knowledge_base = KnowledgeBase( name=kb_name, embedder=embedder, vector_store_type=vector_store_type, persist_path=kb_path )
                    logger.info(f"KnowledgeBase instance created/loaded: '{kb_name}' at '{kb_path}'")

                    # Register retrieval tool ONLY if agent command needs it
                    if args.command == "agent" and enable_kb_for_agent:
                        retrieval_k_arg = getattr(args, 'retrieval_k', None) # Check if k set by search subparser (won't exist here)
                        retrieval_k_default = self.settings.get("tools.knowledge_base.retrieval_k", 3)
                        retrieval_k = retrieval_k_default # Use default for agent runs, search cmd uses its own -k arg

                        retrieval_tool = RetrievalTool(
                            name="knowledge_retrieve", description="Retrieve relevant information from the knowledge base.",
                            vector_store=self.knowledge_base.vector_store, embedder=embedder, default_k=retrieval_k
                        )
                        self.tools.register_tool(retrieval_tool, categories=["knowledge"])
                        logger.info("Registered Knowledge Retrieval tool for agent.")

                    # Register Conversation Memory Search Tool (optional, if agent needs it)
                    if args.command == "agent" and enable_mem_search and ConversationMemoryTool:
                        mem_embedder = embedder # Reuse embedder or load specific one
                        memory_tool = ConversationMemoryTool(
                            name="conversation_search", description="Search past conversation messages.",
                            vector_store_type=self.settings.get("tools.conversation_memory.vector_store_type", "faiss"),
                            embedder=mem_embedder
                        )
                        self.tools.register_tool(memory_tool, categories=["memory"])
                        logger.info("Registered Conversation Search tool for agent.")

                except Exception as e:
                    logger.error(f"Failed to initialize knowledge base components: {e}", exc_info=True)
                    self.knowledge_base = None # Ensure KB is None if setup failed
            else: logger.warning("Knowledge base tools/commands requested but dependencies missing.")

    # --- Agent Creation Factories ---
    def _create_workflow_agent(self, llm: BaseLLM, args: argparse.Namespace, system_prompt: Optional[str]) -> BaseAgent:
        """Creates a WorkflowAgent, prioritizing args for agent params."""
        # Augment system prompt if RAG tool is available
        if self.tools.has_tool("knowledge_retrieve"):
            rag_guidance = "\nUse the 'knowledge_retrieve' tool for relevant information."
            system_prompt = (system_prompt or "You are a helpful assistant.") + rag_guidance

        # Prioritize CLI args over settings for agent parameters
        max_iterations_arg = getattr(args, 'max_iterations', None) # Safely get arg
        max_iterations = max_iterations_arg if max_iterations_arg is not None else self.settings.get("agents.workflow.max_iterations", 10)
        verbose_flag = getattr(args, 'verbose', False)

        # Add logic here to load specific workflow definitions from settings if desired
        logger.info("Initializing WorkflowAgent (using default internal chain if no workflows configured).")
        return WorkflowAgent(
            name="cli_workflow_agent", llm=llm, tools=self.tools,
            system_prompt=system_prompt, max_iterations=max_iterations, verbose=verbose_flag
        )

    def _create_autonomous_agent(self, llm: BaseLLM, args: argparse.Namespace, system_prompt: Optional[str]) -> BaseAgent:
        """Creates an AutonomousAgent, prioritizing args for agent params."""
        # Augment system prompt if RAG tool is available
        if self.tools.has_tool("knowledge_retrieve"):
            rag_guidance = "\nUse the 'knowledge_retrieve' tool for relevant information."
            system_prompt = (system_prompt or "You are an autonomous AI assistant.") + rag_guidance

        # Prioritize CLI args over settings
        max_iterations_arg = getattr(args, 'max_iterations', None)
        max_iterations = max_iterations_arg if max_iterations_arg is not None else self.settings.get("agents.autonomous.max_iterations", 15)
        reflection_arg = getattr(args, 'reflection_threshold', None)
        reflection_threshold = reflection_arg if reflection_arg is not None else self.settings.get("agents.autonomous.reflection_threshold", 3)
        verbose_flag = getattr(args, 'verbose', False)

        logger.info("Initializing AutonomousAgent.")
        return AutonomousAgent(
            name="cli_autonomous_agent", llm=llm, tools=self.tools,
            system_prompt=system_prompt, max_iterations=max_iterations,
            reflection_threshold=reflection_threshold, verbose=verbose_flag
        )

    # --- Mode Execution Methods ---
    async def run_interactive_mode(self):
         """Runs the interactive agent loop."""
         if not self.agent: print("Error: Agent not initialized.", file=sys.stderr); return
         print(f"\nStarting interactive session with {self.agent.name} (type 'exit' or 'quit' to end)")
         print(f"Tools available: {', '.join(self.agent.tools.get_tool_names()) if self.agent.tools else 'None'}")
         print("-" * 50)
         while True:
             try:
                 user_input = await asyncio.to_thread(input, "\nYou: ")
                 user_input_lower = user_input.lower()
                 if user_input_lower in ["exit", "quit"]: print("\nEnding session."); break
                 if not user_input.strip(): continue
                 print("\nAgent working...")
                 result = await self.agent.run(user_input)
                 print(f"\nAgent: {result.get('response', 'No response received.')}")
                 if result.get("error"): print(f"Error: {result['error']}", file=sys.stderr)
                 if result.get("tool_calls"):
                     print("--- Tools Called ---")
                     for i, call in enumerate(result["tool_calls"], 1):
                         tool_name = call.get('tool', call.get('name', 'Unknown'))
                         output = call.get('output', {})
                         status = "Success" if isinstance(output, dict) and 'error' not in output else "Error"
                         result_data = output.get("result") if isinstance(output, dict) else output
                         error_data = output.get("error") if isinstance(output, dict) else None
                         result_preview = str(result_data if status=="Success" else error_data)[:100]
                         print(f"{i}. {tool_name}: Status={status}, Output={result_preview}...")
             except (KeyboardInterrupt, EOFError): print("\nEnding session."); break
             except Exception as e:
                 logger.exception(f"Interactive loop error: {e}") if logger else None
                 print(f"\nAn unexpected error occurred: {str(e)}", file=sys.stderr)

    async def run_task_mode(self, task_description: str, input_data: Optional[str]):
         """Runs the agent for a single task."""
         if not self.agent: print("Error: Agent not initialized.", file=sys.stderr); return
         print(f"\nRunning task for {self.agent.name}: {task_description}")
         print(f"Tools available: {', '.join(self.agent.tools.get_tool_names()) if self.agent.tools else 'None'}")
         print("-" * 50)
         run_input: Union[str, Dict[str, Any]] = task_description; run_kwargs: Dict[str, Any] = {}
         if input_data:
              run_input = {"input": task_description, "additional_data": input_data}
              if logger: logger.info(f"Providing additional input data for task.")
         try:
             result = await self.agent.run(run_input, **run_kwargs)
             print("\n--- Task Result ---")
             print(f"Status: {'Success' if result.get('success') else 'Failed'}")
             if result.get("error"): print(f"Error: {result['error']}", file=sys.stderr)
             print("\nResponse:"); print(result.get("response", "No response content."))
             if result.get("tool_calls"):
                 print("\n--- Tool Calls ---")
                 for i, call in enumerate(result["tool_calls"], 1):
                      tool_name = call.get('tool', call.get('name', 'Unknown'))
                      output = call.get('output', {})
                      status = "Success" if isinstance(output, dict) and 'error' not in output else "Error"
                      result_data = output.get("result") if isinstance(output, dict) else output
                      error_data = output.get("error") if isinstance(output, dict) else None
                      result_preview = str(result_data if status=="Success" else error_data)[:100]
                      print(f"{i}. {tool_name}: Status={status}, Output={result_preview}...")
         except Exception as e:
             logger.exception(f"Error running task '{task_description}': {e}") if logger else None
             print(f"\nAn unexpected error occurred while running the task: {str(e)}", file=sys.stderr)

    # --- Knowledge Management Methods ---
    async def import_knowledge(self, directory: str, file_types: Optional[List[str]], recursive: bool):
         """Imports documents into the initialized knowledge base."""
         if not self.knowledge_base: print("Error: Knowledge base not initialized.", file=sys.stderr); return
         print(f"\nImporting knowledge from: {directory}"); print(f"Recursive: {recursive}, Types: {file_types or 'Default'}"); print("-" * 50)
         try:
             # Use the instance created during setup
             results = await self.knowledge_base.add_documents_from_directory(
                 directory=directory, source_name=os.path.basename(directory) or "cli_import",
                 file_types=file_types, recursive=recursive, chunk=True
             )
             print("\n--- Import Summary ---"); print(f"  Added (Chunks): {results.get('added', 0)}"); print(f"  Skipped: {results.get('skipped', 0)}"); print(f"  Failed: {results.get('failed', 0)}")
             if results.get('failed', 0) > 0 and results.get('errors'):
                 print("\n--- Errors ---")
                 for err in results['errors'][:10]: print(f"  - {err.get('file', 'N/A')}: {err.get('error', '?')}", file=sys.stderr)
                 if len(results['errors']) > 10: print(f"  ... and {len(results['errors']) - 10} more errors.", file=sys.stderr)
             stats = await self.knowledge_base.get_stats()
             print(f"\nKB ('{self.knowledge_base.name}') approx vectors: {stats.get('vector_count', 0)}")
         except FileNotFoundError: print(f"Error: Directory not found: {directory}", file=sys.stderr)
         except Exception as e: logger.exception(f"Error importing KB from {directory}: {e}") if logger else None; print(f"\nImport error: {str(e)}", file=sys.stderr)

    async def search_knowledge(self, query: str, k: int, filter_dict: Optional[Dict]):
         """Searches the initialized knowledge base."""
         if not self.knowledge_base: print("Error: Knowledge base not initialized.", file=sys.stderr); return
         print(f"\nSearching KB ('{self.knowledge_base.name}') for: \"{query}\" (k={k})"); print(f"Filter: {filter_dict}") if filter_dict else None; print("-" * 50)
         try:
             # Use the instance created during setup
             results_data = await self.knowledge_base.query(query=query, k=k, filter=filter_dict)
             results = results_data.get("results", []); print(f"\nFound {len(results)} relevant chunks:")
             if not results: print("No matches found."); return
             for i, result in enumerate(results, 1):
                 print(f"\n{i}. Score: {result.get('score', 0.0):.4f}"); metadata = result.get('metadata', {}); print(f"   Source: {metadata.get('filepath', metadata.get('source', '?'))}")
                 if "chunk_index" in metadata: print(f"   Chunk: {metadata['chunk_index'] + 1}/{metadata.get('chunk_count', '?')}")
                 print(f"   Content: {result.get('content', '')[:300]}...")
         except Exception as e: logger.exception(f"Error searching KB: {e}") if logger else None; print(f"\nSearch error: {str(e)}", file=sys.stderr)


# --- Argument Parser Definition (Refined) ---
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments using subparsers."""
    parser = argparse.ArgumentParser(
        description="AI Agent Framework Command Line Interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help
    )
    # Top-level arguments
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file (overrides defaults).")
    parser.add_argument("--log-level", type=str, default=argparse.SUPPRESS, # Let Settings handle default unless overridden
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging level.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose agent execution logging.")

    # --- Subcommands: agent | knowledge ---
    command_subparsers = parser.add_subparsers(dest="command", help="Choose command", required=True)

    # --- Agent Subcommand Parser ---
    agent_parser = command_subparsers.add_parser("agent", help="Run an AI agent.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Agent arguments group
    agent_group = agent_parser.add_argument_group('Agent Configuration (Overrides Settings)')
    agent_group.add_argument("--agent-type", type=str, default=argparse.SUPPRESS, choices=["workflow", "autonomous"], help="Type of agent to run.")
    agent_group.add_argument("--system-prompt", type=str, default=argparse.SUPPRESS, help="Override system prompt for the agent.")
    agent_group.add_argument("--max-iterations", type=int, default=argparse.SUPPRESS, help="Override max agent execution iterations.")
    agent_group.add_argument("--reflection-threshold", type=int, default=argparse.SUPPRESS, help="Override reflection threshold (autonomous only).")

    # LLM arguments group
    llm_group = agent_parser.add_argument_group('LLM Configuration (Overrides Settings)')
    llm_group.add_argument("--llm-provider", type=str, default=argparse.SUPPRESS, help="LLM provider (e.g., openai, claude).")
    llm_group.add_argument("--model", type=str, default=argparse.SUPPRESS, help="Specific LLM model name.")
    llm_group.add_argument("--temperature", type=float, default=argparse.SUPPRESS, help="LLM generation temperature (0.0-1.0).")

    # Tool arguments group
    tool_group = agent_parser.add_argument_group('Tool Enablement Flags (Overrides Settings)')
    tool_group.add_argument("--enable-filesystem", action="store_true", help="Enable filesystem tools (read, list). Write controlled by settings.")
    tool_group.add_argument("--enable-web", action="store_true", help="Enable web search tools.")
    tool_group.add_argument("--enable-data-analysis", action="store_true", help="Enable data analysis tools (requires pandas).")
    tool_group.add_argument("--enable-knowledge-base", action="store_true", help="Enable knowledge base retrieval tool for agent.")
    tool_group.add_argument("--enable-memory-search", action="store_true", help="Enable conversation memory search tool.")

    # KB config flags (relevant only if --enable-knowledge-base is active)
    kb_group_agent = agent_parser.add_argument_group('Knowledge Base Configuration (Used if --enable-knowledge-base)')
    kb_group_agent.add_argument("--knowledge-base-path", type=str, default=argparse.SUPPRESS, help="Path to KB data directory.")
    kb_group_agent.add_argument("--knowledge-base-name", type=str, default=argparse.SUPPRESS, help="KB collection name.")
    kb_group_agent.add_argument("--vector-store-type", type=str, choices=["chroma", "faiss"], default=argparse.SUPPRESS, help="Vector store type.")
    kb_group_agent.add_argument("--embedder-type", type=str, choices=["openai", "local"], default=argparse.SUPPRESS, help="Embedder type.")
    # Note: retrieval_k for agent is set by settings, not usually a direct agent flag

    # Agent operation mode subparsers
    agent_mode_subparsers = agent_parser.add_subparsers(dest="mode", help="Agent operation mode", required=True)
    interactive_parser = agent_mode_subparsers.add_parser("interactive", help="Run agent in interactive mode.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    task_parser = agent_mode_subparsers.add_parser("task", help="Run agent on a specific task.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    task_parser.add_argument("--task", type=str, required=True, help="Task description for the agent.")
    task_parser.add_argument("--input", type=str, help="Optional additional input string.")
    task_parser.add_argument("--input-file", type=str, help="File containing additional input.")

    # --- Knowledge Subcommand Parser ---
    kb_parser = command_subparsers.add_parser("knowledge", help="Manage knowledge base.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # KB config flags (needed for KB commands)
    kb_config_group = kb_parser.add_argument_group('Knowledge Base Configuration (Overrides Settings)')
    kb_config_group.add_argument("--knowledge-base-path", type=str, default=argparse.SUPPRESS, help="Path to KB data directory.")
    kb_config_group.add_argument("--knowledge-base-name", type=str, default=argparse.SUPPRESS, help="KB collection name.")
    kb_config_group.add_argument("--vector-store-type", type=str, choices=["chroma", "faiss"], default=argparse.SUPPRESS, help="Vector store type.")
    kb_config_group.add_argument("--embedder-type", type=str, choices=["openai", "local"], default=argparse.SUPPRESS, help="Embedder type.")

    kb_action_subparsers = kb_parser.add_subparsers(dest="kb_action", help="Knowledge base action", required=True)
    # Import knowledge
    import_parser = kb_action_subparsers.add_parser("import", help="Import knowledge from files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    import_parser.add_argument("--directory", type=str, required=True, help="Directory to import from.")
    import_parser.add_argument("--recursive", action="store_true", help="Search directories recursively.")
    import_parser.add_argument("--file-types", type=str, nargs="+", help="File extensions to import (e.g., .txt .md). Default: .txt, .md, .pdf")

    # Search knowledge
    search_parser = kb_action_subparsers.add_parser("search", help="Search knowledge base.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    search_parser.add_argument("--query", type=str, required=True, help="Search query.")
    # Use dest to store in args.retrieval_k
    search_parser.add_argument("-k", type=int, dest="retrieval_k", default=argparse.SUPPRESS, help="Number of results to return.")
    search_parser.add_argument("--filter", type=str, help="Metadata filter as JSON string (e.g., '{\"topic\":\"ai\"}')")

    return parser.parse_args()


# --- Main Execution Logic ---
async def main():
    """Main entry point function for the CLI."""
    global logger
    args = parse_arguments()

    # Setup Logging (needs to happen before settings potentially use logger)
    log_dir = os.path.join(os.getcwd(), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        # Peek at settings/args for log level
        temp_settings = Settings(config_path=args.config)
        log_level_setting = temp_settings.get("system.log_level", "INFO")
        # Command line arg overrides settings/default
        log_level = (args.log_level or log_level_setting).upper()
        setup_logging(log_level=log_level, log_dir=log_dir)
        logger = logging.getLogger(__name__) # Assign global logger
        logger.info(f"Logging initialized. Level: {log_level}")
    except Exception as e:
         print(f"Error setting up logging: {e}. Continuing with basic console logging.", file=sys.stderr)
         logging.basicConfig(level=(args.log_level or "INFO").upper())
         logger = logging.getLogger(__name__)

    # Initialize CLI Class (loads settings)
    cli = CLI()

    # --- Setup necessary components based on command ---
    try:
        # Setup initializes LLM, Tools, and KB (if enabled/needed)
        await cli.setup(args)
    except (ValueError, ImportError) as e: logger.error(f"CLI setup failed: {e}", exc_info=False); print(f"Error during setup: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e: logger.error(f"Unexpected error during CLI setup: {e}", exc_info=True); print(f"Unexpected error during setup: {e}", file=sys.stderr); sys.exit(1)

    # --- Execute Command ---
    if args.command == "agent":
        if not cli.agent: logger.critical("Agent command selected, but agent failed to initialize during setup."); print("Error: Agent setup failed.", file=sys.stderr); sys.exit(1)
        if args.mode == "interactive": await cli.run_interactive_mode()
        elif args.mode == "task":
            input_data_for_task: Optional[str] = None
            task_desc = getattr(args, 'task', None) # Task is required by parser
            if not task_desc: logger.critical("Task mode selected but --task argument missing parser state."); print("Error: --task is required.", file=sys.stderr); sys.exit(1)
            if hasattr(args, 'input_file') and args.input_file:
                try: input_file_path = os.path.abspath(args.input_file); logger.info(f"Reading task input from: {input_file_path}"); f=open(input_file_path, 'r', encoding='utf-8'); input_data_for_task = f.read(); f.close()
                except Exception as e: logger.error(f"Read task input file error {args.input_file}: {e}", exc_info=True); print(f"Error reading input file.", file=sys.stderr); sys.exit(1)
            elif hasattr(args, 'input'): input_data_for_task = args.input
            await cli.run_task_mode(task_desc, input_data_for_task)

    elif args.command == "knowledge":
        if not cli.knowledge_base:
            print("Error: Knowledge base components not available or failed to initialize.", file=sys.stderr)
            logger.error("Knowledge command selected, but KB instance is not available (check dependencies/config).")
            sys.exit(1)

        if args.kb_action == "import":
            file_types = args.file_types or cli.settings.get("knowledge.import.default_file_types", [".txt", ".md", ".pdf"])
            await cli.import_knowledge(directory=args.directory, file_types=file_types, recursive=args.recursive)
        elif args.kb_action == "search":
            filter_dict = None
            if args.filter:
                try: filter_dict = json.loads(args.filter); assert isinstance(filter_dict, dict)
                except Exception as e: print(f"Invalid JSON filter: {e}", file=sys.stderr); sys.exit(1)
            # Use retrieval_k from args if provided, else default
            k_value_setting = cli.settings.get("tools.knowledge_base.retrieval_k", 3)
            # args.retrieval_k might be None if not provided, check this
            k_value = args.retrieval_k if args.retrieval_k is not None else k_value_setting
            await cli.search_knowledge(query=args.query, k=k_value, filter_dict=filter_dict)
    else:
        # Should be caught by argparse
        logger.error(f"Unknown command provided: {args.command}")
        print(f"Error: Unknown command '{args.command}'. Use --help.", file=sys.stderr)
        sys.exit(1)


def cli_entry_point():
    """Entry point for setuptools console_scripts."""
    # Check for essential API Keys after potentially loading .env via Settings()
    # This check might need adjustment depending on final Settings implementation
    # temp_settings = Settings() # Quick load to check env after dotenv potentially ran
    # if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    #      print("Warning: No common LLM API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) found.", file=sys.stderr)

    try:
        asyncio.run(main())
    except KeyboardInterrupt: print("\nOperation cancelled by user."); sys.exit(0)
    except Exception as e:
         # Log critical error if logger was initialized, otherwise print
         if logger: logger.critical(f"Unhandled CLI exception: {e}", exc_info=True)
         print(f"\nCritical error: {e}", file=sys.stderr)
         sys.exit(1)


if __name__ == "__main__":
    # This allows running `python -m ai_agent_framework.interfaces.cli ...`
    cli_entry_point()