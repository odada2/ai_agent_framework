# AI Agent Framework

A flexible, composable framework for building AI agents using simple patterns rather than complex abstractions. This framework is inspired by Anthropic's research on building effective AI agents and emphasizes simplicity, transparency, and targeted complexity only when needed.

## Overview

AI Agent Framework provides a set of building blocks for creating AI agents that can:

- Interact with language models like Claude
- Use tools to accomplish tasks
- Maintain conversation context
- Follow structured workflows
- Make decisions and adapt to user requests

The framework follows these core principles:
1. Maintain simplicity in agent design
2. Prioritize transparency through explicit planning steps
3. Carefully craft agent-computer interfaces with thorough documentation

## Architecture

The framework is organized around these core components:

### Core Components

- **LLM Integration**: Connect to language models with a unified interface
- **Tool System**: Define, use, and manage tools for agent capabilities
- **Memory Management**: Store conversation history and knowledge
- **Workflow Patterns**: Structured patterns for agent execution flows

### Agent Types

- **Workflow Agent**: Follows predefined workflows for predictable execution
- **Autonomous Agent**: Self-directed agent that makes its own decisions
- **Custom Agents**: Extend the framework for specific use cases

### Workflow Patterns

- **Prompt Chaining**: Sequential steps passing outputs forward
- **Routing**: Classify and direct requests to specialized handlers
- **Parallelization**: Concurrent execution for speed or diverse outputs
- **Orchestrator-Workers**: Centralized coordination of specialized workers
- **Evaluator-Optimizer**: Iterative improvement through feedback loops

## Getting Started

### Installation

```bash
# Basic installation
pip install -r requirements.txt

# Development installation
pip install -e ".[dev]"

### Quick Start
import asyncio
from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.tools.file_system.read import FileReadTool

async def main():
    # Create LLM
    llm = LLMFactory.create_llm(provider="claude")
    
    # Create tool registry
    tools = ToolRegistry()
    tools.register_tool(FileReadTool())
    
    # Create agent
    agent = WorkflowAgent(
        name="my_agent",
        llm=llm,
        tools=tools
    )
    
    # Use the agent
    response = await agent.run("What's in the README.md file?")
    print(response["response"])

asyncio.run(main())
```

## Command Line Interface
The framework includes a command-line interface for interacting with agents:

```bash
python -m ai_agent_framework.main interactive --agent-type workflow --enable-filesystem
```

### Creating Custom Workflows

```bash
from ai_agent_framework.core.workflow.chain import PromptChain

custom_workflow = PromptChain(
    name="my_workflow",
    llm=llm,
    steps=[
        {
            "name": "step1",
            "prompt_template": "Process this input: {input}"
        },
        {
            "name": "step2",
            "prompt_template": "Build on the previous step: {input}",
            "use_tools": True
        }
    ]
)

agent.add_workflow("custom", custom_workflow)
```

## Development
### Running Tests

```bash
pytest ai_agent_framework/tests/
```

### Code Style
The project follows PEP 8 style guidelines with a line length of 88 characters, using Black for formatting and isort for import sorting.

### Format code

```bash
black ai_agent_framework/
isort ai_agent_framework/

```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
This framework is inspired by Anthropic's research on building effective AI agents, emphasizing simple composable patterns over complex frameworks.