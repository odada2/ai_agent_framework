# Vector Store Integration

This document provides an overview of the vector store integration in the AI Agent Framework, including how to set up and use the retrieval-augmented generation (RAG) capabilities.

## Overview

The vector store integration allows agents to access external knowledge through semantic search, enhancing their capabilities beyond what's contained in the model's training data. The integration includes:

- **Vector Databases**: Store document embeddings for efficient similarity search
- **Document Processing**: Tools for chunking and processing text
- **Embedding Generation**: Convert text to vector representations
- **Retrieval**: Find relevant information based on semantic similarity
- **Agent Integration**: Tools for agents to access the knowledge

## Components

### Vector Stores

The framework supports multiple vector database backends:

- **Chroma**: Lightweight, open-source embedding database
- **FAISS**: High-performance vector similarity search from Facebook AI

Each implementation follows a consistent interface defined in `VectorStore`, making it easy to switch between different backends.

### Document Representation

Documents are represented by the `Document` class, which includes:

- **Text Content**: The document text
- **Metadata**: Additional information about the document
- **ID**: Unique identifier
- **Embedding**: Optional vector representation

### Embedders

Text embeddings can be generated using:

- **LLMEmbedder**: Uses OpenAI or Claude APIs for embeddings
- **LocalEmbedder**: Uses local models like sentence-transformers

### Knowledge Base

The `KnowledgeBase` class provides a high-level interface for managing documents:

- Add, update, and remove documents
- Manage document sources
- Process and chunk documents
- Query for relevant information

## Usage Examples

### Basic RAG Setup

```python
from ai_agent_framework.core.memory.knowledge_base import KnowledgeBase
from ai_agent_framework.core.memory.embeddings import get_embedder
from ai_agent_framework.tools.memory.retrieval_tool import RetrievalTool
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.core.llm.factory import LLMFactory

# Initialize embedder
embedder = get_embedder(embedder_type="openai")

# Create knowledge base
kb = KnowledgeBase(
    name="my_knowledge_base",
    embedder=embedder,
    vector_store_type="chroma",
    persist_path="./kb_data"
)

# Add documents
await kb.add_document(
    content="Mars is the fourth planet from the Sun.",
    metadata={"topic": "astronomy"}
)

# Create retrieval tool
retrieval_tool = RetrievalTool(
    vector_store=kb.vector_store,
    embedder=embedder
)

# Set up agent with RAG
llm = LLMFactory.create_llm(provider="claude")
tools = ToolRegistry()
tools.register_tool(retrieval_tool)

agent = WorkflowAgent(
    name="rag_agent",
    llm=llm,
    tools=tools,
    system_prompt="Use the retrieve tool to find relevant information."
)

# Use the agent
response = await agent.run("What can you tell me about Mars?")
```

### Loading Documents from Files

```python
# Load documents from a directory
results = await kb.add_documents_from_directory(
    directory="./documents",
    source_name="research_papers",
    file_types=[".pdf", ".txt", ".md"],
    recursive=True
)

print(f"Added {results['added']} documents")
```

### Querying the Knowledge Base

```python
# Query directly
results = await kb.query(
    query="What is machine learning?",
    k=3,
    filter={"topic": "AI"}
)

for result in results["results"]:
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']}")
    print(f"Metadata: {result['metadata']}")
```

## Integration with Agents

The `RetrievalTool` provides a simple interface for agents to query the knowledge base:

```python
# Agent prompt
prompt = """
To answer the user's question, search the knowledge base using the retrieve tool.
Always cite your sources from the knowledge base.
"""

# Execute
response = await agent.run("What are the key concepts in reinforcement learning?")
```

## Customization

### Custom Chunking

```python
from ai_agent_framework.core.memory.text_utils import MarkdownTextSplitter

# Create custom text splitter
splitter = MarkdownTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Use with knowledge base
kb.default_splitter = splitter
```

### Custom Embedder

```python
from ai_agent_framework.core.memory.embeddings import LocalEmbedder

# Create custom embedder
embedder = LocalEmbedder(
    model_name="all-mpnet-base-v2",
    use_gpu=True
)

# Use with knowledge base
kb = KnowledgeBase(
    name="custom_kb",
    embedder=embedder
)
```

## Best Practices

1. **Document Chunking**: Choose appropriate chunk sizes for your content (smaller for precise retrieval, larger for more context)
2. **Metadata Enrichment**: Add useful metadata to documents for filtering
3. **Query Formulation**: Be specific in retrieval queries
4. **Model Guidance**: Add clear instructions for the agent on when and how to use retrieved information
5. **Persistence**: Use persistent storage for production systems

## Performance Considerations

- FAISS is more performant for large collections (100K+ documents)
- Chroma is simpler to set up and supports more metadata filtering
- Local embedders are faster but may be less accurate than API-based ones
- Consider using a cross-encoder for reranking results when precision is critical

## Limitations

- Document capacity is limited by available memory (especially for FAISS)
- API-based embedders require network access and may have rate limits
- Local embedders require additional dependencies

## Future Development

- Additional vector store backends (Pinecone, Weaviate, etc.)
- Support for hybrid search (combining semantic and keyword search)
- Improved document processing pipeline with image support
- Integration with document loaders for more file formats