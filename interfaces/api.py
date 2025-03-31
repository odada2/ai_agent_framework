# ai_agent_framework/interfaces/api.py
# Updated for Org ID Handling

import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple

from fastapi import FastAPI, HTTPException, Depends, Body, status, Request

# Core framework components (use absolute imports)
try:
    from ai_agent_framework.config.settings import Settings
    from ai_agent_framework.config.logging_config import setup_logging
    from ai_agent_framework.core.llm.factory import LLMFactory
    from ai_agent_framework.core.llm.base import BaseLLM
    from ai_agent_framework.core.tools.registry import ToolRegistry
    from ai_agent_framework.agents.workflow_agent import WorkflowAgent
    from ai_agent_framework.agents.autonomous_agent import AutonomousAgent
    from ai_agent_framework.agents.base_agent import BaseAgent
    from ai_agent_framework.core.memory.conversation import ConversationMemory
    # Import state manager only if Redis support is compiled/intended
    REDIS_ENABLED_FLAG = False # Default flag
    try:
        from ai_agent_framework.core.state.redis_manager import RedisStateManager
        REDIS_ENABLED_FLAG = True
    except ImportError:
        RedisStateManager = None # Define as None if import fails
    from ai_agent_framework.core.exceptions import AgentFrameworkError, ConfigurationError
except ImportError as e:
     print(f"CRITICAL API ERROR: Failed to import core framework components: {e}")
     # API cannot start without core components
     sys.exit(1)


# --- Application Setup ---
# Setup logging early, before loading settings that might use it
# Configure log dir path appropriately for deployment context
log_dir = os.environ.get("LOG_DIR", "./api_logs")
os.makedirs(log_dir, exist_ok=True)
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO").upper(), log_dir=log_dir)
logger = logging.getLogger(__name__)

# Load settings globally - needed for state manager and dependencies
try:
    settings = Settings() # Assumes settings can find config (e.g., default path or env var)
    API_CONFIG = settings.get("api", {}) # Get API specific config section
    SYSTEM_CONFIG = settings.get("system", {})
    TOOL_CONFIG = settings.get("tools", {})
    LLM_CONFIG = settings.get("llm", {}) # Get default LLM config
    # Determine if Redis is configured *and* library was imported
    REDIS_ENABLED = settings.get("storage.redis.enabled", False) and REDIS_ENABLED_FLAG
except Exception as e:
    logger.exception(f"Fatal Error: Could not load settings. API cannot start. Error: {e}")
    settings = None; API_CONFIG = {}; SYSTEM_CONFIG = {}; TOOL_CONFIG = {}; LLM_CONFIG = {}; REDIS_ENABLED = False

# --- Global Resources (Initialized at startup) ---
global_tools = ToolRegistry()
state_manager: Optional[RedisStateManager] = None
global_llm: Optional[BaseLLM] = None

def setup_global_resources():
    """Initialize shared resources like Tools, LLM, State Manager at startup."""
    global global_tools, state_manager, global_llm

    if not settings:
         logger.error("Settings not loaded, cannot setup global resources.")
         return

    # --- Setup Tools ---
    logger.info("Configuring global tools for API...")
    # Example: Enable Web Search tool based on config
    if TOOL_CONFIG.get("enable_web_search", API_CONFIG.get("tools.enable_web_search", False)): # Check both tool and API specific config
        try:
            from ai_agent_framework.tools.apis.web_search import WebSearchTool
            global_tools.register_tool(WebSearchTool(settings=settings))
            logger.info("Global WebSearchTool enabled.")
        except ImportError: logger.warning("WebSearchTool configured but could not be imported.")
        except Exception as e: logger.error(f"Failed to initialize WebSearchTool: {e}", exc_info=True)

    # Example: Enable Filesystem tools based on config (Use API config section)
    if TOOL_CONFIG.get("enable_filesystem", API_CONFIG.get("tools.enable_filesystem", False)):
        try:
            from ai_agent_framework.tools.file_system.read import FileReadTool
            from ai_agent_framework.tools.file_system.write import FileWriteTool
            from ai_agent_framework.tools.file_system.list_dir import ListDirectoryTool

            # IMPORTANT: Define allowed_directories VERY carefully for API security
            allowed_dirs_api = API_CONFIG.get("tools.filesystem.allowed_directories") # Explicit API config needed
            if allowed_dirs_api and isinstance(allowed_dirs_api, list):
                 global_tools.register_tool(FileReadTool(allowed_directories=allowed_dirs_api))
                 global_tools.register_tool(ListDirectoryTool(allowed_directories=allowed_dirs_api))
                 logger.info(f"Global Filesystem Read/List tools enabled for API (Allowed: {allowed_dirs_api}).")
                 # Only allow write if explicitly configured for API
                 if API_CONFIG.get("tools.filesystem.allow_write", False):
                      global_tools.register_tool(FileWriteTool(allowed_directories=allowed_dirs_api))
                      logger.warning("Global FileWriteTool enabled for API (EXTREME CAUTION REQUIRED).")
            else:
                 logger.error("Filesystem tools enabled for API but 'api.tools.filesystem.allowed_directories' is not set or invalid in config. Tools not registered.")

        except ImportError: logger.warning("Filesystem tools configured for API but could not be imported.")
        except Exception as e: logger.error(f"Failed to initialize Filesystem tools for API: {e}", exc_info=True)

    # Add setup for other globally available tools based on TOOL_CONFIG or API_CONFIG...
    logger.info(f"Global Tool Registry initialized with {len(global_tools)} tools.")

    # --- Setup Global LLM ---
    try:
        # Use defaults from top-level llm config section
        llm_provider = LLM_CONFIG.get("provider", "openai") # Default to openai
        llm_model = LLM_CONFIG.get("model")
        llm_temp = LLM_CONFIG.get("temperature", 0.7)
        # --- Load Org ID from settings ---
        llm_org_id = settings.get(f"llm.{llm_provider}.organization_id", settings.get("llm.organization_id"))
        # ---------------------------------

        global_llm = LLMFactory.create_llm(
             provider=llm_provider,
             model_name=llm_model,
             temperature=llm_temp,
             organization_id=llm_org_id # <-- Pass Org ID
             # API key loaded from env by factory/constructor
        )
        logger.info(f"Global LLM ({llm_provider}, model: {global_llm.model_name if global_llm else 'N/A'}) initialized for API.")
    except (ValueError, ImportError) as e:
         logger.error(f"Failed to initialize global LLM: {e}")
         global_llm = None # Ensure it's None if failed
    except Exception as e:
         logger.error(f"Unexpected error initializing global LLM: {e}", exc_info=True)
         global_llm = None

    if not global_llm:
         logger.critical("API cannot function without a configured Global LLM.")
         # Consider preventing FastAPI startup if LLM fails?

    # --- Setup State Manager ---
    if REDIS_ENABLED and RedisStateManager: # Check flag and successful import
        try:
            state_manager = RedisStateManager(settings=settings)
            # Optional: Add async connection test here if needed at startup
            # asyncio.create_task(state_manager._get_client()) # Fire-and-forget test
            logger.info("Redis State Manager initialized for API.")
        except Exception as e:
            logger.error(f"Failed to initialize Redis State Manager: {e}. State persistence disabled.", exc_info=True)
            state_manager = None # Ensure it's None on failure
    else:
         logger.info("Redis State Manager is disabled or library unavailable.")


# --- FastAPI App ---
app = FastAPI(
    title="AI Agent Framework API",
    description="HTTP API for interacting with configured AI agents.",
    version=SYSTEM_CONFIG.get("version", "0.1.0"),
)

@app.on_event("startup")
async def startup_event():
    """Run setup logic on application startup."""
    logger.info("API Startup...")
    setup_global_resources()
    # Test redis connection async at startup if manager exists
    if state_manager:
         try:
             client = await state_manager._get_client()
             if client: await client.ping(); logger.info("Redis connection test successful.")
             else: logger.warning("State manager initialized but failed to get Redis client on startup.")
         except Exception as e:
             logger.error(f"Redis connection test failed on startup: {e}")
             # Decide if this should prevent startup

@app.on_event("shutdown")
async def shutdown_event():
     """Run cleanup logic on application shutdown."""
     logger.info("Shutting down API...")
     if state_manager and hasattr(state_manager, 'close'):
         await state_manager.close()
     if global_llm and hasattr(global_llm, 'close'): # If LLM clients need closing
          if asyncio.iscoroutinefunction(global_llm.close): await global_llm.close()
          else: global_llm.close()
     # Add cleanup for other resources if needed
     logger.info("API shutdown complete.")

# --- Pydantic Models (Keep from previous version) ---
class AgentRunRequest(BaseModel):
    input: Union[str, Dict[str, Any]] = Body(..., description="Input prompt or data.")
    agent_type: str = Body(default="workflow", description="Agent type ('workflow'/'autonomous').", pattern="^(workflow|autonomous)$")
    conversation_id: Optional[str] = Body(None, description="ID for conversation context.")
    config: Optional[Dict[str, Any]] = Body(default_factory=dict, description="Runtime config overrides.")
class AgentRunResponse(BaseModel):
    response: Union[str, Dict[str, Any]] = Field(..., description="Agent's final response.")
    agent_type_used: str = Field(..., description="Agent type used.")
    conversation_id: str = Field(..., description="Conversation ID used/generated.")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tools called.")
    iterations: Optional[int] = Field(None, description="Iterations taken.")
    success: bool = Field(..., description="If agent run succeeded.")
    error: Optional[str] = Field(None, description="Error message on failure.")


# --- FastAPI Dependencies (Keep from previous version, uses state_manager) ---
async def get_conversation_state(request: AgentRunRequest = Body(...)) -> Tuple[str, ConversationMemory]:
    """Loads/creates conversation memory, using Redis if available."""
    conversation_id = request.conversation_id
    is_new_conversation = False
    memory: Optional[ConversationMemory] = None

    if state_manager: # Use Redis if available
        if not conversation_id:
            conversation_id = str(uuid.uuid4()); is_new_conversation = True
            logger.info(f"API: Starting new Redis conversation: {conversation_id}")
            memory = ConversationMemory() # Create fresh memory to be saved later
        else:
            logger.info(f"API: Attempting to load memory from Redis for {conversation_id}")
            memory = await state_manager.load_memory(conversation_id)
            if memory is None:
                logger.warning(f"API: No memory found in Redis for {conversation_id}. Starting fresh.")
                is_new_conversation = True
                memory = ConversationMemory()
    else: # Fallback to transient in-memory
        logger.warning("API: Redis State Manager not available. Using transient in-memory conversation.")
        if not conversation_id: conversation_id = str(uuid.uuid4()); is_new_conversation = True
        # !! WARNING: In-memory state is lost between requests in multi-worker deployments !!
        # A simple global dict cache could be used for single-worker dev, but is not robust.
        # For this example, we just create a new memory each time if Redis isn't used.
        memory = ConversationMemory()
        if not is_new_conversation: logger.warning(f"API: Received existing conv ID {conversation_id} but Redis unavailable; starting fresh in memory.")

    # Ensure conversation_id and memory are always valid
    conversation_id = conversation_id or str(uuid.uuid4()) # Should be set above, but safety check
    memory = memory or ConversationMemory() # Safety check

    return conversation_id, memory

async def get_agent(request: AgentRunRequest = Body(...), state: Tuple[str, ConversationMemory] = Depends(get_conversation_state)) -> BaseAgent:
    """Dependency to instantiate the requested agent with loaded/new memory."""
    agent_type = request.agent_type.lower()
    conversation_id, memory = state

    if not global_llm: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM service not available.")

    # Use agent-specific config from settings if available
    agent_config_base_key = f"agents.{agent_type}"
    agent_sys_prompt = settings.get(f"{agent_config_base_key}.system_prompt") if settings else None
    agent_max_iter = settings.get(f"{agent_config_base_key}.max_iterations", 10) if settings else 10
    agent_verbose = settings.get("system.verbose", False) if settings else False
    agent_reflect_thresh = settings.get("agents.autonomous.reflection_threshold", 3) if settings else 3

    try:
        if agent_type == "workflow":
            agent = WorkflowAgent(
                name=f"api_workflow_{conversation_id}", llm=global_llm, tools=global_tools,
                memory=memory, max_iterations=agent_max_iter, verbose=agent_verbose, system_prompt=agent_sys_prompt
            )
        elif agent_type == "autonomous":
            agent = AutonomousAgent(
                name=f"api_autonomous_{conversation_id}", llm=global_llm, tools=global_tools,
                memory=memory, max_iterations=agent_max_iter, verbose=agent_verbose, system_prompt=agent_sys_prompt,
                reflection_threshold=agent_reflect_thresh # Pass specific param
            )
        else:
            # Should be caught by AgentRunRequest pattern, but handle defensively
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid agent type: {agent_type}")

        logger.debug(f"Instantiated agent '{agent.name}' for conversation {conversation_id}")
        return agent

    except Exception as e:
         logger.error(f"Failed to instantiate agent type {agent_type}: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create agent instance: {e}")


# --- API Endpoints (Keep from previous version, uses dependencies) ---

@app.get("/", summary="Health Check", tags=["General"])
async def read_root():
    """Provides basic API status."""
    redis_status = "disabled"
    if state_manager:
        client = await state_manager._get_client() # Use internal getter for check
        if client:
             try: await client.ping(); redis_status = "connected"
             except Exception as e: redis_status = f"error ({e})"
        else: redis_status = "init_failed"

    return {
        "status": "ok", "message": "AI Agent Framework API is running.",
        "llm_available": global_llm is not None, "llm_provider": getattr(global_llm,'provider','N/A') if global_llm else 'N/A',
        "redis_state_manager": redis_status, "global_tools_count": len(global_tools),
        }

@app.post("/run", response_model=AgentRunResponse, summary="Run an agent task", tags=["Agent Execution"])
async def run_agent_task(
    request: AgentRunRequest = Body(...), # Get request body for agent type etc.
    state: Tuple[str, ConversationMemory] = Depends(get_conversation_state),
    agent: BaseAgent = Depends(get_agent) # Inject agent with correct memory
):
    """Accepts input, runs through specified agent type with conversation state."""
    conversation_id, memory = state # Unpack state from dependency
    input_data = request.input
    runtime_config = request.config or {}

    logger.info(f"Handling /run request for agent: {agent.name}, ConvID: {conversation_id}")

    try:
        result = await agent.run(input_data, **runtime_config)

        if state_manager: # Persist memory if Redis is enabled
             try: await state_manager.save_memory(conversation_id, agent.memory)
             except Exception as redis_e: logger.error(f"Failed to save memory to Redis for {conversation_id}: {redis_e}")

        agent_success = result.get("success", True) # Assume success if key missing
        agent_error = result.get("error")
        if not agent_success: logger.warning(f"Agent run failed for ConvID {conversation_id}. Error: {agent_error}")

        return AgentRunResponse(
            response=result.get("response", "Agent run finished, but no response content found."),
            agent_type_used=request.agent_type.lower(), conversation_id=conversation_id,
            tool_calls=result.get("tool_calls"), iterations=result.get("iterations"),
            success=agent_success, error=agent_error
        )

    except HTTPException: raise # Re-raise validation/dependency errors
    except AgentFrameworkError as e: logger.error(f"Agent Error for ConvID {conversation_id}: {e}", exc_info=True); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent error: {e}")
    except Exception as e: logger.exception(f"Internal server error for ConvID {conversation_id}: {e}"); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")

@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete Conversation State", tags=["Admin"])
async def delete_conversation(conversation_id: str):
     if not state_manager: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis state manager is not enabled.")
     try:
         deleted_count = await state_manager.delete_memory(conversation_id)
         logger.info(f"Requested deletion for conversation state: {conversation_id}. Keys deleted: {deleted_count}")
     except Exception as e:
          logger.error(f"Error deleting memory for conversation {conversation_id} from Redis: {e}")
          # Don't raise 500, just log it, deletion is often best-effort
     # Return 204 No Content regardless of whether key existed

# --- Example: Add entry point for running with uvicorn ---
# Remove this if deploying differently
if __name__ == "__main__":
     import uvicorn
     # Load host/port from settings or use defaults
     host = API_CONFIG.get("host", "127.0.0.1")
     port = API_CONFIG.get("port", 8000)
     reload = SYSTEM_CONFIG.get("environment", "development") == "development"
     # Note: Startup event runs when Uvicorn starts the app
     uvicorn.run("ai_agent_framework.interfaces.api:app", host=host, port=port, reload=reload)