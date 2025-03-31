import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import framework components
from ai_agent_framework.core.llm.claude import ClaudeLLM
from ai_agent_framework.agents.autonomous_agent import AutonomousAgent
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.tools.file_system.read import FileReadTool

async def main():
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Make sure ANTHROPIC_API_KEY is set in your .env file.")

    # Initialize LLM
    llm = ClaudeLLM(
        model_name="claude-3-7-sonnet-20250219",  # Use an available Claude model 
        api_key=api_key
    )
    
    # Create a tool registry
    tools = ToolRegistry()
    
    # Register tools
    file_tool = FileReadTool(
        allowed_directories=[os.getcwd()]  # Allow reading files from current directory
    )
    tools.register_tool(file_tool)
    
    # Create the agent
    agent = AutonomousAgent(
        name="research_assistant",
        llm=llm,
        tools=tools,
        system_prompt="You are a helpful assistant that can read files and provide information.",
        max_iterations=5,  # Limit the number of reasoning steps
        verbose=True  # Show detailed logs
    )
    
    # Run the agent
    print("Running agent. This may take a moment...")
    result = await agent.run("Please help me understand what files are in the current directory and what they contain.")
    
    # Print the result
    print("\n=== AGENT RESPONSE ===\n")
    print(result["response"])

if __name__ == "__main__":
    asyncio.run(main())