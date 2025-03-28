# ai_agent_framework/interfaces/api.py

import logging
import asyncio
import uuid # For generating conversation IDs
from typing import Dict, Any, Optional, List, Union, Tuple

from fastapi import FastAPI, HTTPException, Depends, Body, status, Request

# Core framework components
from ai_agent_framework.config.settings import Settings
from ai_agent_framework.config.logging_config import setup_logging
from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.agents.autonomous_agent import AutonomousAgent
from ai_agent_framework.agents.base_agent import BaseAgent
from ai_agent_framework.core.memory.conversation import ConversationMemory
from ai_agent_framework.core.state.redis_manager import RedisStateManager # Import State Manager
from ai_agent_framework.core.exceptions import AgentFrameworkError, ConfigurationError

# --- Application Setup ---

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Load settings globally - needed for state manager and dependencies
try:
    settings = Settings()
    API_CONFIG = settings.get("api", {})
    SYSTEM_CONFIG = settings.get("system", {})
    TOOL_CONFIG = settings.get("tools", {})
    REDIS_ENABLED = settings.get("storage.redis.enabled", False) # Check if Redis is enabled
except Exception as e:
    logger.exception(f"Fatal Error: Could not load settings. API cannot start. Error: {e}")
    # Handle missing settings appropriately (e.g., exit or run with defaults)
    settings = None
    API_CONFIG = {}
    SYSTEM_CONFIG = {}
    TOOL_CONFIG = {}
    REDIS_ENABLED = False

# --- Global Resources (Initialized at startup, shared across requests) ---

# Global Tool Registry (Tools are generally stateless)
global_tools = ToolRegistry()

# State Manager Instance (conditionally initialized)
state_manager: Optional[RedisStateManager] = None

# Global LLM Factory/Instance (LLMs can often be shared)
# Consider if different endpoints need different LLMs later
global_llm: Optional[BaseLLM] = None

def setup_global_resources():
    """Initialize shared resources like Tools, LLM, State Manager at startup."""
    global global_tools, state_manager, global_llm

    # --- Setup Tools ---
    logger.info("Configuring global tools...")
    # Example: Enable Web Search tool based on config
    if TOOL_CONFIG.get("enable_web_access", False) or API_CONFIG.get("tools.enable_web_search", False):
        try:
            from ai_agent_framework.tools.apis.web_search import WebSearchTool
            global_tools.register_tool(WebSearchTool(settings=settings)) # Pass settings if needed
            logger.info("Global WebSearchTool enabled.")
        except ImportError: logger.warning("WebSearchTool configured but could not be imported.")
        except Exception as e: logger.error(f"Failed to initialize WebSearchTool: {e}", exc_info=True)
    # Add setup for other globally available tools...
    logger.info(f"Global Tool Registry initialized with {len(global_tools)} tools.")

    # --- Setup LLM ---
    if settings:
         try:
              llm_provider = API_CONFIG.get("llm.provider", settings.get("llm.provider", "claude"))
              llm_model = API_CONFIG.get("llm.model", settings.get("llm.model", None))
              llm_temp = API_CONFIG.get("llm.temperature", settings.get("llm.temperature", 0.7))
              global_llm = LLMFactory.create_llm(provider=llm_provider, model_name=llm_model, temperature=llm_temp)
              logger.info(f"Global LLM ({llm_provider}) initialized.")
         except Exception as e: logger.error(f"Failed to initialize global LLM: {e}", exc_info=True)

    # --- Setup State Manager ---
    if REDIS_ENABLED and settings:
        try:
            state_manager = RedisStateManager(settings=settings)
            # Test connection during setup (optional, handled lazily in _get_client)
            # asyncio.run(state_manager._get_client()) # Cannot run async here directly
            logger.info("Redis State Manager initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Redis State Manager: {e}. State persistence disabled.", exc_info=True)
            # state_manager remains None if init fails

    if not global_llm:
         logger.error("API cannot function without a configured LLM.")
         # Potentially raise an error to prevent startup

# --- FastAPI App ---

app = FastAPI(
    title="AI Agent Framework API",
    description="HTTP API for interacting with configured AI agents.",
    version=SYSTEM_CONFIG.get("version", "0.1.0"),
)

@app.on_event("startup")
async def startup_event():
    """Run setup logic on application startup."""
    setup_global_resources()
    # Test redis connection async at startup
    if state_manager:
         try:
             await state_manager._get_client()
         except ConnectionError as e:
             logger.error(f"Redis connection test failed on startup: {e}")
             # Decide if this should prevent startup

@app.on_event("shutdown")
async def shutdown_event():
     """Run cleanup logic on application shutdown."""
     logger.info("Shutting down API...")
     if state_manager and hasattr(state_manager, 'close'):
         await state_manager.close()
     # Add cleanup for other resources if needed (e.g., global_llm if it has sessions)
     logger.info("API shutdown complete.")

# --- Pydantic Models (Request/Response) ---
# (Keep AgentRunRequest and AgentRunResponse as they were)
class AgentRunRequest(BaseModel):
    input: Union[str, Dict[str, Any]] = Field(..., description="The input prompt or structured data for the agent.")
    agent_type: str = Field(default="workflow", description="Type of agent ('workflow' or 'autonomous').", pattern="^(workflow|autonomous)$")
    conversation_id: Optional[str] = Field(None, description="Optional ID to maintain conversation context across requests. If None, a new conversation starts.")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional runtime configuration overrides for the agent run.")

class AgentRunResponse(BaseModel):
    response: Union[str, Dict[str, Any]] = Field(..., description="The agent's final response.")
    agent_type_used: str = Field(..., description="The type of agent that handled the request.")
    conversation_id: str = Field(..., description="Conversation ID used or generated for this request.") # Made required
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Details of tools called during execution.")
    iterations: Optional[int] = Field(None, description="Number of iterations taken.")
    success: bool = Field(..., description="Indicates if the agent run completed without errors.")
    error: Optional[str] = Field(None, description="Error message if the run failed.")


# --- FastAPI Dependencies ---

async def get_conversation_state(request: AgentRunRequest = Body(...)) -> Tuple[str, ConversationMemory]:
    """
    Dependency to load or create conversation memory based on conversation_id.
    Handles state persistence via the state_manager.
    """
    conversation_id = request.conversation_id
    is_new_conversation = False

    if not state_manager:
         # Fallback to in-memory if Redis is disabled or failed
         logger.warning("State manager not available. Using transient in-memory conversation.")
         if not conversation_id:
             conversation_id = str(uuid.uuid4())
             is_new_conversation = True
         # Need a global dict for fallback, but this has limitations mentioned before
         # For simplicity here, just create a new memory each time if no Redis
         memory = ConversationMemory()
         return conversation_id, memory

    # --- Using Redis State Manager ---
    if not conversation_id:
        conversation_id = str(uuid.uuid4()) # Generate new ID for new conversations
        is_new_conversation = True
        logger.info(f"Starting new conversation with ID: {conversation_id}")
        memory = ConversationMemory() # Create fresh memory
        # Optional: Add a system message or welcome message here if desired
        # memory.add_system_message("Welcome!")
    else:
        logger.info(f"Attempting to load memory for conversation ID: {conversation_id}")
        memory = await state_manager.load_memory(conversation_id)
        if memory is None:
            logger.warning(f"No existing memory found for ID {conversation_id}. Starting fresh.")
            is_new_conversation = True
            memory = ConversationMemory() # Create fresh memory if load fails or ID is unknown
            # Optionally handle cases where ID was provided but not found differently

    return conversation_id, memory

# Dependency to get the correct agent instance based on request
# This now creates the agent per-request, injecting the conversation memory
async def get_agent(
    request: AgentRunRequest = Body(...),
    state: Tuple[str, ConversationMemory] = Depends(get_conversation_state)
) -> BaseAgent:
    """
    Dependency to instantiate the requested agent with loaded/new memory.
    """
    agent_type = request.agent_type.lower()
    conversation_id, memory = state

    if not global_llm: # Check if global LLM was initialized
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM service not available.")

    # Agent config can be loaded from settings based on agent_type
    agent_configs = API_CONFIG.get("agents", {})
    agent_verbose = agent_configs.get("verbose", False) # Example config loading

    try:
        if agent_type == "workflow":
            agent = WorkflowAgent(
                name=f"api_workflow_{conversation_id}", # Make name unique per conversation
                llm=global_llm,
                tools=global_tools,
                memory=memory, # Inject loaded/new memory
                max_iterations=agent_configs.get("workflow.max_iterations", 10),
                verbose=agent_verbose
                # Pass other necessary config from settings
            )
        elif agent_type == "autonomous":
            agent = AutonomousAgent(
                name=f"api_autonomous_{conversation_id}", # Make name unique
                llm=global_llm,
                tools=global_tools,
                memory=memory, # Inject loaded/new memory
                max_iterations=agent_configs.get("autonomous.max_iterations", 15),
                reflection_threshold=agent_configs.get("autonomous.reflection_threshold", 3),
                verbose=agent_verbose
                # Pass other necessary config from settings
            )
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid agent type: {agent_type}")

        logger.debug(f"Instantiated agent '{agent.name}' for conversation {conversation_id}")
        return agent

    except Exception as e:
         logger.error(f"Failed to instantiate agent type {agent_type}: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create agent instance: {e}")


# --- API Endpoints ---

@app.get("/", summary="Health Check", tags=["General"])
async def read_root():
    """Provides basic API status."""
    redis_status = "disabled"
    if state_manager:
        try:
            client = await state_manager._get_client()
            if client:
                 await client.ping()
                 redis_status = "connected"
            else:
                 redis_status = "init_failed"
        except Exception as e:
            redis_status = f"error: {e}"

    return {
        "status": "ok",
        "message": "AI Agent Framework API is running.",
        "llm_available": global_llm is not None,
        "redis_state_manager": redis_status,
        "global_tools_count": len(global_tools),
        }

@app.post("/run", response_model=AgentRunResponse, summary="Run an agent task", tags=["Agent Execution"])
async def run_agent_task(
    request: AgentRunRequest = Body(...), # Keep request body for potential future use
    state: Tuple[str, ConversationMemory] = Depends(get_conversation_state),
    agent: BaseAgent = Depends(get_agent) # Inject agent with correct memory
):
    """
    Accepts input data, runs it through the specified agent type using
    persistent conversation state managed by Redis.
    """
    conversation_id, memory = state # Unpack state from dependency
    input_data = request.input
    runtime_config = request.config or {}

    logger.info(f"Handling /run request for agent: {agent.name}, ConvID: {conversation_id}")

    # Agent Execution
    try:
        result = await agent.run(input_data, **runtime_config) # Agent has correct memory injected

        # --- Persist State ---
        if state_manager:
             await state_manager.save_memory(conversation_id, agent.memory) # Save updated memory
        # --------------------

        agent_success = result.get("success", True)
        agent_error = result.get("error")

        if not agent_success:
             logger.warning(f"Agent run indicated failure for ConvID {conversation_id}. Error: {agent_error}")

        # Prepare response
        response_payload = AgentRunResponse(
            response=result.get("response", "" if agent_success else "Agent run failed."),
            agent_type_used=request.agent_type.lower(),
            conversation_id=conversation_id, # Return the definite conversation ID
            tool_calls=result.get("tool_calls"),
            iterations=result.get("iterations"),
            success=agent_success,
            error=agent_error
        )
        return response_payload

    except HTTPException:
         raise # Re-raise HTTPException directly
    except AgentFrameworkError as e:
         logger.error(f"Agent Framework Error during run for ConvID {conversation_id}: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent error: {e}")
    except Exception as e:
        logger.exception(f"Internal server error processing request for ConvID {conversation_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")

# Example endpoint to clear memory for a conversation (for testing/debugging)
@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete Conversation State", tags=["Admin"])
async def delete_conversation(conversation_id: str):
     if not state_manager:
          raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="State manager (Redis) is not enabled or available.")
     deleted = await state_manager.delete_memory(conversation_id)
     if not deleted:
          # This could mean Redis error or key not found, either way it's gone or wasn't there
          logger.warning(f"Deletion command executed for conversation {conversation_id}, but Redis reported 0 keys deleted (might not have existed or error occurred).")
          # Optionally raise 404 if key must exist, but DELETE is often idempotent
          # raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation ID not found")
     logger.info(f"Requested deletion for conversation state: {conversation_id}")
     # No response body for 204