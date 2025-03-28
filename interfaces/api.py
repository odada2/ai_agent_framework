# ai_agent_framework/interfaces/api.py

"""
FastAPI Interface for AI Agent Framework

Provides an HTTP API for interacting with different types of agents configured
via the application settings. Handles request validation, agent execution,
and response formatting.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union

from fastapi import FastAPI, HTTPException, Depends, Body, status
from pydantic import BaseModel, Field

# Core framework components (using absolute imports)
from ai_agent_framework.config.settings import Settings
from ai_agent_framework.config.logging_config import setup_logging
from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.agents.autonomous_agent import AutonomousAgent
from ai_agent_framework.agents.base_agent import BaseAgent
# Import necessary exceptions if specific handling is needed beyond HTTPException
from ai_agent_framework.core.exceptions import AgentFrameworkError, ConfigurationError

# --- Application Setup ---

# Configure logging early
# Consider loading log_level from settings if needed before full agent setup
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Load settings globally for the API module
try:
    settings = Settings()
    API_CONFIG = settings.get("api", {}) # Get API specific config section
    SYSTEM_CONFIG = settings.get("system", {})
    TOOL_CONFIG = settings.get("tools", {}) # Get tool config section
except ConfigurationError as e:
    logger.exception(f"Fatal Error: Could not load settings. API cannot start. Error: {e}")
    # In a real app, this might prevent startup entirely
    settings = None
    API_CONFIG = {}
    SYSTEM_CONFIG = {}
    TOOL_CONFIG = {}
except Exception as e:
     logger.exception(f"Unexpected error loading settings: {e}")
     settings = None
     API_CONFIG = {}
     SYSTEM_CONFIG = {}
     TOOL_CONFIG = {}

# --- Agent Factory / Management (Simplified Startup Initialization) ---
# For robust state management (memory per user/session), consider FastAPI Dependencies
# or a more sophisticated agent management strategy.

_agents: Dict[str, BaseAgent] = {}

def _setup_api_tools() -> ToolRegistry:
    """Creates and configures the ToolRegistry for API agents."""
    registry = ToolRegistry()
    logger.info("Configuring tools for API...")

    # Example: Enable Web Search tool based on config
    if TOOL_CONFIG.get("enable_web_access", False) or API_CONFIG.get("tools.enable_web_search", False):
        try:
            from ai_agent_framework.tools.apis.web_search import WebSearchTool
            # Pass settings if the tool needs them for provider config
            registry.register_tool(WebSearchTool(settings=settings))
            logger.info("WebSearchTool enabled for API.")
        except ImportError:
            logger.warning("WebSearchTool configured but could not be imported.")
        except Exception as e:
            logger.error(f"Failed to initialize WebSearchTool: {e}", exc_info=True)

    # Example: Enable Filesystem Read tool (use cautiously in API)
    if TOOL_CONFIG.get("enable_filesystem", False) and API_CONFIG.get("tools.enable_filesystem_read", False):
         try:
              from ai_agent_framework.tools.file_system.read import FileReadTool
              allowed_dirs = TOOL_CONFIG.get("filesystem.allowed_directories", ["./api_readable_data"]) # API specific allowed dirs
              registry.register_tool(FileReadTool(allowed_directories=allowed_dirs))
              logger.info(f"FileReadTool enabled for API (Allowed Dirs: {allowed_dirs}).")
         except ImportError:
              logger.warning("FileReadTool configured but could not be imported.")
         except Exception as e:
              logger.error(f"Failed to initialize FileReadTool: {e}", exc_info=True)

    # TODO: Add setup for other tools based on API_CONFIG or TOOL_CONFIG as needed
    # e.g., RAG Tool, Data Analysis Tool

    logger.info(f"API Tool Registry initialized with {len(registry)} tools.")
    return registry


def setup_agents():
    """Initialize agent instances based on configuration at startup."""
    global _agents
    if not settings:
         logger.error("Settings not loaded, cannot setup agents.")
         return

    logger.info("Setting up API agents...")
    try:
        # --- LLM Setup (using API config or fallback to global) ---
        llm_provider = API_CONFIG.get("llm.provider", settings.get("llm.provider", "claude"))
        llm_model = API_CONFIG.get("llm.model", settings.get("llm.model", None))
        llm_temp = API_CONFIG.get("llm.temperature", settings.get("llm.temperature", 0.7))
        llm = LLMFactory.create_llm(provider=llm_provider, model_name=llm_model, temperature=llm_temp)

        # --- Tool Setup ---
        api_tools = _setup_api_tools()

        # --- Create Agents ---
        agent_configs = API_CONFIG.get("agents", {})
        agent_verbose = agent_configs.get("verbose", False)

        if agent_configs.get("enable_workflow", True):
             try:
                 # TODO: Define/Load specific workflows for the API agent
                 # For now, relies on the default chain workflow within WorkflowAgent if none provided
                 _agents["workflow"] = WorkflowAgent(
                     name="api_workflow_agent", llm=llm, tools=api_tools,
                     max_iterations=agent_configs.get("workflow.max_iterations", 10),
                     verbose=agent_verbose
                 )
                 logger.info("WorkflowAgent initialized for API.")
             except Exception as e: logger.error(f"Failed to initialize WorkflowAgent: {e}", exc_info=True)

        if agent_configs.get("enable_autonomous", True):
             try:
                 _agents["autonomous"] = AutonomousAgent(
                     name="api_autonomous_agent", llm=llm, tools=api_tools,
                     max_iterations=agent_configs.get("autonomous.max_iterations", 15),
                     reflection_threshold=agent_configs.get("autonomous.reflection_threshold", 3),
                     verbose=agent_verbose
                 )
                 logger.info("AutonomousAgent initialized for API.")
             except Exception as e: logger.error(f"Failed to initialize AutonomousAgent: {e}", exc_info=True)

        if not _agents: logger.error("No agents were successfully initialized for the API!")

    except Exception as e:
        logger.exception(f"Fatal error during agent setup for API: {e}")


# --- FastAPI App ---

app = FastAPI(
    title="AI Agent Framework API",
    description="HTTP API for interacting with configured AI agents.",
    version=SYSTEM_CONFIG.get("version", "0.1.0"),
)

@app.on_event("startup")
async def startup_event():
    """Run agent setup logic on application startup."""
    setup_agents()
    if not _agents:
         logger.warning("API starting without any active agents due to setup errors.")

@app.on_event("shutdown")
async def shutdown_event():
     """Run cleanup logic on application shutdown."""
     logger.info("Shutting down API and agents...")
     # Add agent shutdown logic if agents have specific cleanup needs
     shutdown_tasks = []
     for agent in _agents.values():
          if hasattr(agent, 'shutdown') and asyncio.iscoroutinefunction(agent.shutdown):
               shutdown_tasks.append(agent.shutdown())
          # Add specific cleanup for tools if needed (e.g., closing web search sessions)
          if hasattr(agent, 'tools'):
                for tool in agent.tools.get_all_tools():
                     if hasattr(tool, 'close') and asyncio.iscoroutinefunction(tool.close):
                          shutdown_tasks.append(tool.close())
     if shutdown_tasks:
          await asyncio.gather(*shutdown_tasks, return_exceptions=True)
     logger.info("API shutdown complete.")


# --- Pydantic Models ---

class AgentRunRequest(BaseModel):
    input: Union[str, Dict[str, Any]] = Field(..., description="The input prompt or structured data for the agent.")
    agent_type: Optional[str] = Field(default="workflow", description="Type of agent ('workflow' or 'autonomous').", pattern="^(workflow|autonomous)$")
    conversation_id: Optional[str] = Field(None, description="Optional ID to maintain conversation context across requests.")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional runtime configuration overrides for the agent run.")

class AgentRunResponse(BaseModel):
    response: Union[str, Dict[str, Any]] = Field(..., description="The agent's final response.")
    agent_type_used: str = Field(..., description="The type of agent that handled the request.")
    conversation_id: Optional[str] = Field(None, description="Conversation ID used or generated.")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Details of tools called during execution.")
    iterations: Optional[int] = Field(None, description="Number of iterations taken.")
    success: bool = Field(..., description="Indicates if the agent run completed without errors.")
    error: Optional[str] = Field(None, description="Error message if the run failed.")


# --- API Endpoints ---

@app.get("/", summary="Health Check", tags=["General"])
async def read_root():
    """Provides basic API status and lists active agents."""
    active_agent_types = list(_agents.keys())
    return {"status": "ok", "message": "AI Agent Framework API is running.", "active_agents": active_agent_types}

@app.post("/run", response_model=AgentRunResponse, summary="Run an agent task", tags=["Agent Execution"])
async def run_agent_task(request: AgentRunRequest = Body(...)):
    """
    Accepts input data and runs it through the specified agent type.
    Manages conversation state based on `conversation_id`.
    """
    agent_type = request.agent_type.lower()
    input_data = request.input
    conv_id = request.conversation_id
    runtime_config = request.config or {}

    logger.info(f"Received /run request for agent: {agent_type}, ConvID: {conv_id}")

    # Select agent instance
    agent = _agents.get(agent_type)
    if not agent:
        logger.error(f"Requested agent type '{agent_type}' is not available.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent type '{agent_type}' not found.")

    # --- Conversation Management (Basic Example) ---
    # A real implementation would use a proper session store keyed by conv_id
    # to retrieve/update the agent's memory. This is a simplified placeholder.
    if conv_id:
         # Placeholder: Ideally, load agent state/memory associated with conv_id
         logger.debug(f"Attempting to use conversation context: {conv_id} (basic implementation)")
         # agent.load_memory(conv_id) # Requires agent method to load memory state
    else:
         # Placeholder: Start a new conversation context
         agent.reset() # Reset agent state for new conversation (includes memory clear)
         conv_id = agent.id # Use agent ID as a pseudo conversation ID for this run

    # --- Agent Execution ---
    try:
        logger.debug(f"Running agent '{agent.name}' with input: {str(input_data)[:100]}...")
        result = await agent.run(input_data, **runtime_config)

        # Check agent result for success/failure
        agent_success = result.get("success", True) # Assume success if not specified
        agent_error = result.get("error")

        if not agent_success:
             logger.warning(f"Agent run indicated failure for ConvID {conv_id}. Error: {agent_error}")

        # Prepare response
        response_payload = AgentRunResponse(
            response=result.get("response", "" if agent_success else "Agent run failed."),
            agent_type_used=agent_type,
            conversation_id=conv_id, # Return the conversation ID used
            tool_calls=result.get("tool_calls"),
            iterations=result.get("iterations"),
            success=agent_success,
            error=agent_error
        )

        # Placeholder: Ideally, save agent state/memory associated with conv_id
        # agent.save_memory(conv_id)

        return response_payload

    except HTTPException:
         raise # Re-raise HTTPException directly
    except AgentFrameworkError as e:
         # Catch specific framework errors
         logger.error(f"Agent Framework Error during run for ConvID {conv_id}: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent error: {e}")
    except Exception as e:
        # Catch unexpected errors during agent run
        logger.exception(f"Internal server error processing request for agent '{agent_type}', ConvID {conv_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")