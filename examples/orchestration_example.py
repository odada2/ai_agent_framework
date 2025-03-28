"""
Orchestration System Example

This script demonstrates how to use the Agent Orchestration System to
coordinate multiple specialized agents for complex tasks.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, List

# Add parent directory to path to run as a standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.agents.autonomous_agent import AutonomousAgent
from ai_agent_framework.agents.supervisor_agent import SupervisorAgent
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.tools.file_system.read import FileReadTool
from ai_agent_framework.tools.apis.web_search import WebSearchTool
from ai_agent_framework.core.workflow.orchestrator import OrchestratorWorkflow
from ai_agent_framework.core.communication.agent_protocol import (
    AgentCommunicator, MessageType, global_message_queue
)
from ai_agent_framework.core.workflow.task_queue import schedule_task
from ai_agent_framework.config.logging_config import setup_logging
from ai_agent_framework.config.settings import Settings


# Set up logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


async def create_specialized_agents():
    """Create a set of specialized agents for different domains."""
    # Create LLM
    llm = LLMFactory.create_llm(provider="claude")
    
    # Create specialized agents
    agents = {}
    
    # 1. Research Agent
    research_tools = ToolRegistry()
    research_tools.register_tool(WebSearchTool())
    
    research_agent = AutonomousAgent(
        name="research_agent",
        llm=llm,
        tools=research_tools,
        system_prompt=(
            "You are a Research Agent specialized in finding and synthesizing information "
            "from the web. Your goal is to provide accurate, comprehensive, and up-to-date "
            "information on any topic requested."
        ),
        max_iterations=5,
        verbose=True
    )
    agents["research"] = research_agent
    
    # 2. Writer Agent
    writer_agent = WorkflowAgent(
        name="writer_agent",
        llm=llm,
        system_prompt=(
            "You are a Writing Agent specialized in creating clear, engaging, and well-structured "
            "content. Your goal is to produce high-quality written material based on the information "
            "and specifications provided."
        ),
        verbose=True
    )
    agents["writer"] = writer_agent
    
    # 3. Analytics Agent
    analytics_tools = ToolRegistry()
    analytics_agent = AutonomousAgent(
        name="analytics_agent",
        llm=llm,
        tools=analytics_tools,
        system_prompt=(
            "You are an Analytics Agent specialized in analyzing data, identifying patterns, "
            "and drawing insights. Your goal is to provide clear, data-driven analysis and "
            "recommendations."
        ),
        max_iterations=3,
        verbose=True
    )
    agents["analytics"] = analytics_agent
    
    # 4. Code Agent
    code_tools = ToolRegistry()
    code_tools.register_tool(FileReadTool())
    
    code_agent = AutonomousAgent(
        name="code_agent",
        llm=llm,
        tools=code_tools,
        system_prompt=(
            "You are a Code Agent specialized in software development and programming. "
            "Your goal is to write clean, efficient, and well-documented code based on "
            "the requirements provided."
        ),
        max_iterations=5,
        verbose=True
    )
    agents["code"] = code_agent
    
    return agents


async def setup_communication_for_agents(agents: Dict[str, Any]):
    """Set up communication between agents."""
    # Create communicators for each agent
    for agent_id, agent in agents.items():
        agent.communicator = AgentCommunicator(
            agent_id=agent_id,
            queue=global_message_queue,
            auto_register=True
        )
        
        # Add simple message handler
        async def message_handler(agent=agent):
            while True:
                message = await agent.communicator.receive(timeout=1.0)
                if message:
                    logger.info(f"Agent {agent.name} received: {message.content[:50]}...")
                    
                    # Simple automatic response to queries
                    if message.message_type == MessageType.QUERY:
                        await agent.communicator.send(
                            content=f"I'll help with that: {message.content}",
                            message_type=MessageType.RESPONSE,
                            receiver=message.sender,
                            reference_id=message.id
                        )
                        
                await asyncio.sleep(0.1)
                
        # Start message handler task
        asyncio.create_task(message_handler())


async def example_complex_task():
    """Example of using the orchestration system for a complex task."""
    logger.info("=== Complex Task Orchestration Example ===")
    
    # Create specialized agents
    specialized_agents = await create_specialized_agents()
    
    # Set up communication
    await setup_communication_for_agents(specialized_agents)
    
    # Create LLM for supervisor
    llm = LLMFactory.create_llm(provider="claude")
    
    # Create supervisor agent
    supervisor = SupervisorAgent(
        name="supervisor_agent",
        llm=llm,
        specialized_agents=specialized_agents,
        parallel_execution=True,
        max_parallel_agents=2,
        verbose=True
    )
    
    # Define a complex task that requires multiple specialized agents
    complex_task = (
        "Create a market analysis report for electric vehicles in the United States. "
        "The report should include:\n"
        "1. Current market trends and statistics\n"
        "2. Analysis of major players and their market share\n"
        "3. Consumer adoption patterns and barriers\n"
        "4. Future projections and growth potential\n"
        "5. A simple Python script to visualize the market share data"
    )
    
    logger.info(f"Submitting complex task to supervisor:\n{complex_task}")
    
    # Execute the task
    result = await supervisor.run(complex_task)
    
    logger.info("=== Task Complete ===")
    logger.info(f"Response preview: {result['response'][:500]}...")
    
    return result


async def example_direct_orchestration():
    """Example of using the orchestrator workflow directly."""
    logger.info("=== Direct Orchestration Example ===")
    
    # Create LLM
    llm = LLMFactory.create_llm(provider="claude")
    
    # Define worker functions
    async def research_worker(input_data):
        task = input_data.get("input", "")
        logger.info(f"Research worker executing: {task[:50]}...")
        # Simulate research work
        await asyncio.sleep(2)
        return f"Research findings for: {task}"
    
    async def writing_worker(input_data):
        task = input_data.get("input", "")
        dependencies = input_data.get("dependencies", {})
        logger.info(f"Writing worker executing: {task[:50]}...")
        # Use research results if available
        research_data = ""
        for dep_id, dep_result in dependencies.items():
            research_data += f"\n{dep_result}"
        # Simulate writing work
        await asyncio.sleep(3)
        return f"Written content based on: {task}\nIncorporating: {research_data}"
    
    async def code_worker(input_data):
        task = input_data.get("input", "")
        logger.info(f"Code worker executing: {task[:50]}...")
        # Simulate coding work
        await asyncio.sleep(2)
        return f"```python\n# Code for: {task}\nprint('Hello, world!')\n```"
    
    # Create orchestrator with worker functions
    orchestrator = OrchestratorWorkflow(
        name="direct_orchestrator",
        llm=llm,
        workers={
            "research": research_worker,
            "writing": writing_worker,
            "code": code_worker
        },
        parallel=True,
        max_parallel_workers=2,
        verbose=True
    )
    
    # Define a task
    task = (
        "Create a report on climate change mitigation strategies, "
        "including data visualization code."
    )
    
    logger.info(f"Submitting task to orchestrator:\n{task}")
    
    # Execute the task
    result = await orchestrator.execute({"input": task})
    
    logger.info("=== Task Complete ===")
    logger.info(f"Final result: {result['final_result'][:500]}...")
    
    return result


async def example_task_queue_usage():
    """Example of using the task queue for parallel execution."""
    logger.info("=== Task Queue Example ===")
    
    # Define some async tasks
    async def task_1():
        logger.info("Task 1 started")
        await asyncio.sleep(2)
        logger.info("Task 1 completed")
        return "Result from Task 1"
    
    async def task_2():
        logger.info("Task 2 started")
        await asyncio.sleep(3)
        logger.info("Task 2 completed")
        return "Result from Task 2"
    
    async def task_3(dependency_result):
        logger.info(f"Task 3 started with input: {dependency_result}")
        await asyncio.sleep(1)
        logger.info("Task 3 completed")
        return f"Result from Task 3 based on: {dependency_result}"
    
    # Schedule independent tasks
    task1_id = await schedule_task(task_1, priority=1)
    task2_id = await schedule_task(task_2, priority=2)
    
    logger.info(f"Scheduled tasks: {task1_id}, {task2_id}")
    
    # Wait for task 1 to complete
    task1_result = await wait_for_task(task1_id)
    logger.info(f"Task 1 result: {task1_result}")
    
    # Schedule dependent task
    task3_id = await schedule_task(
        task_3, 
        task1_result['result'],
        dependencies=[task2_id]
    )
    
    logger.info(f"Scheduled dependent task: {task3_id}")
    
    # Wait for all tasks to complete
    task2_result = await wait_for_task(task2_id)
    task3_result = await wait_for_task(task3_id)
    
    logger.info(f"All tasks completed:")
    logger.info(f"Task 2 result: {task2_result}")
    logger.info(f"Task 3 result: {task3_result}")
    
    return [task1_result, task2_result, task3_result]


async def main():
    """Run the orchestration examples."""
    # Set up logging and config
    settings = Settings()
    
    # Choose which examples to run
    examples = {
        "complex_task": True,
        "direct_orchestration": True,
        "task_queue": True
    }
    
    results = {}
    
    # Run selected examples
    if examples["task_queue"]:
        results["task_queue"] = await example_task_queue_usage()
    
    if examples["direct_orchestration"]:
        results["direct_orchestration"] = await example_direct_orchestration()
    
    if examples["complex_task"]:
        results["complex_task"] = await example_complex_task()
    
    # Allow time for any remaining tasks to complete
    logger.info("Waiting for any remaining tasks to complete...")
    await asyncio.sleep(5)
    
    logger.info("All examples completed successfully!")
    return results


if __name__ == "__main__":
    asyncio.run(main())