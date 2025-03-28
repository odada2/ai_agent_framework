# ai_agent_framework/examples/simple_agent.py

"""
Simple Agent Example

This script demonstrates the basic setup and usage of a WorkflowAgent
for handling a simple task using the AI Agent Framework.
"""

import asyncio
import logging
import os
import sys

# Add parent directory to path to allow running as standalone script
# Assumes the script is run from the 'examples' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Framework components (using absolute imports based on assumed root 'ai_agent_framework')
from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.config.logging_config import setup_logging
# Optional: Import a simple tool for demonstration
# from ai_agent_framework.tools.apis.web_search import WebSearchTool

# Configure logging
# Consider loading log level from settings or args in a more complex app
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

async def main():
    """Sets up and runs a simple agent interaction."""

    logger.info("--- Simple Agent Example ---")

    # 1. Initialize LLM
    # Ensure your environment has the necessary API key (e.g., ANTHROPIC_API_KEY)
    try:
        llm_provider = "claude" # Or load from settings/env
        logger.info(f"Initializing LLM provider: {llm_provider}")
        llm = LLMFactory.create_llm(provider=llm_provider)
    except (ValueError, ImportError, Exception) as e:
        logger.error(f"Failed to initialize LLM: {e}. Please check configuration and API keys.")
        return

    # 2. Initialize Tool Registry (can be empty or add tools)
    tools = ToolRegistry()
    logger.info("Initialized empty Tool Registry.")
    # Example: Add a tool if needed
    # try:
    #     web_tool = WebSearchTool() # Ensure API keys (e.g., SERPER_API_KEY) are set
    #     tools.register_tool(web_tool)
    #     logger.info("Added WebSearchTool to registry.")
    # except Exception as e:
    #     logger.warning(f"Could not initialize WebSearchTool: {e}. Proceeding without it.")


    # 3. Initialize Agent (Using WorkflowAgent for simplicity)
    # It will use its default internal workflow if none is provided.
    try:
        agent = WorkflowAgent(
            name="simple_agent",
            llm=llm,
            tools=tools,
            verbose=True # Set to True for more detailed logging during execution
        )
        logger.info(f"Initialized agent: {agent.name}")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return

    # 4. Define the task/query
    user_query = "Explain the concept of prompt engineering in simple terms."
    logger.info(f"Running agent with query: \"{user_query}\"")

    # 5. Run the agent
    try:
        # Using agent.run() provides more detailed output dictionary
        result = await agent.run(user_query)

        # Or use agent.chat() for just the text response (less detail)
        # response_text = await agent.chat(user_query)

        logger.info("--- Agent Run Complete ---")

        # 6. Display results
        if result.get("success"):
            print("\nAgent Response:")
            print(result.get("response", "No response content found."))

            if result.get("tool_calls"):
                print("\nTools Called:")
                for call in result["tool_calls"]:
                    # Format tool call details nicely
                    tool_name = call.get('tool', call.get('name', 'Unknown Tool'))
                    params = call.get('parameters', call.get('input', {}))
                    tool_result_preview = str(call.get('result', 'No result'))[:150] # Preview
                    print(f"- Tool: {tool_name}")
                    print(f"  Params: {json.dumps(params, indent=2)}")
                    print(f"  Result: {tool_result_preview}{'...' if len(str(call.get('result', ''))) > 150 else ''}")
        else:
            print(f"\nAgent run failed: {result.get('error', 'Unknown error')}")
            print(f"Partial Response: {result.get('response', 'N/A')}")

        # Print summary info
        print(f"\nWorkflow Used: {result.get('workflow', {}).get('name', 'N/A')}")
        print(f"Workflow Success: {result.get('workflow', {}).get('success', 'N/A')}")


    except Exception as e:
        logger.exception(f"An error occurred during agent execution: {e}")
        print(f"\nAn unexpected error occurred: {e}")

    # Optional: Add shutdown logic if needed (e.g., for tools with sessions)
    # if hasattr(agent, 'shutdown'):
    #     await agent.shutdown()

if __name__ == "__main__":
    # Ensure necessary environment variables (like ANTHROPIC_API_KEY) are set
    # before running the script.
    # Example: export ANTHROPIC_API_KEY='your_key_here'
    if "ANTHROPIC_API_KEY" not in os.environ:
         print("Warning: ANTHROPIC_API_KEY environment variable not set. LLM calls may fail.")
         # Add checks for other needed keys (e.g., SERPER_API_KEY if using web search)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Example execution interrupted by user.")
    except Exception as e:
         logger.critical(f"Script failed to run: {e}", exc_info=True)