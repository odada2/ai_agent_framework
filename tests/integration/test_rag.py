"""
Integration Tests for Retrieval-Augmented Generation

This module provides tests for the complete RAG pipeline, including embedding,
retrieval, and agent integration.
"""

import pytest
import asyncio
import os
import tempfile
from typing import Dict, List, Any, Optional

from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.tools.memory.retrieval_tool import RetrievalTool
from ai_agent_framework.core.memory.embeddings import get_embedder, LocalEmbedder
from ai_agent_framework.core.memory.vector_store import Document
from ai_agent_framework.core.memory.knowledge_base import KnowledgeBase


# Test Documents
TEST_KNOWLEDGE = [
    {
        "text": "Mars is the fourth planet from the Sun in our solar system. " 
                "It is often called the 'Red Planet' due to its reddish appearance. " 
                "Mars has two small moons: Phobos and Deimos.",
        "metadata": {"topic": "astronomy", "subject": "Mars"}
    },
    {
        "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. " 
                "It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair. " 
                "It is named after engineer Gustave Eiffel.",
        "metadata": {"topic": "landmarks", "subject": "Eiffel Tower"}
    },
    {
        "text": "Artificial Neural Networks are computing systems inspired by the biological " 
                "neural networks that constitute animal brains. They are a subset of machine learning " 
                "and are at the heart of deep learning algorithms.",
        "metadata": {"topic": "computer science", "subject": "neural networks"}
    },
    {
        "text": "Python is a high-level, interpreted programming language known for its readability. " 
                "It was created by Guido van Rossum and first released in 1991. " 
                "Python's design philosophy emphasizes code readability.",
        "metadata": {"topic": "computer science", "subject": "Python"}
    },
    {
        "text": "The Great Barrier Reef is the world's largest coral reef system. " 
                "It is located off the coast of Queensland, Australia. " 
                "It can be seen from outer space and is the world's biggest " 
                "single structure made by living organisms.",
        "metadata": {"topic": "nature", "subject": "Great Barrier Reef"}
    }
]


@pytest.fixture
def local_embedder():
    """Create a local embedder for testing."""
    return LocalEmbedder(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for vector store persistence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
async def knowledge_base(local_embedder, temp_dir):
    """Create a knowledge base with test documents."""
    kb = KnowledgeBase(
        name="test_kb",
        embedder=local_embedder,
        vector_store_type="faiss",
        persist_path=os.path.join(temp_dir, "kb_test")
    )
    
    # Add test documents
    for doc in TEST_KNOWLEDGE:
        await kb.add_document(
            content=doc["text"],
            metadata=doc["metadata"],
            chunk=False  # Don't chunk for simplicity
        )
    
    return kb


@pytest.fixture
async def retrieval_tool(knowledge_base):
    """Create a retrieval tool using the test knowledge base."""
    return RetrievalTool(
        name="retrieve",
        vector_store=knowledge_base.vector_store,
        embedder=knowledge_base.embedder,
        default_k=3
    )


@pytest.fixture
async def agent_with_rag(retrieval_tool):
    """Create an agent with the retrieval tool."""
    # Mock LLM that returns a fixed response for testing
    class MockLLM:
        async def generate(self, prompt, system_prompt=None, **kwargs):
            # Just echo back the prompt for testing
            return {"content": f"Response based on: {prompt}"}
            
        async def generate_with_tools(self, prompt, tools, system_prompt=None, **kwargs):
            # Simulate using the tool
            return {
                "content": "I'll use the retrieve tool",
                "tool_calls": [
                    {"name": "retrieve", "parameters": {"query": "What is Mars?"}}
                ]
            }
    
    # Create agent with RAG
    tools = ToolRegistry()
    tools.register_tool(retrieval_tool)
    
    agent = WorkflowAgent(
        name="test_rag_agent",
        llm=MockLLM(),
        tools=tools,
        system_prompt="You are a helpful assistant with knowledge base access."
    )
    
    return agent


@pytest.mark.asyncio
async def test_knowledge_base_retrieval(knowledge_base):
    """Test retrieving information from the knowledge base."""
    # Query about Mars
    results = await knowledge_base.query("What is Mars?", k=1)
    
    assert len(results["results"]) == 1
    assert "Mars" in results["results"][0]["content"]
    assert results["results"][0]["metadata"]["subject"] == "Mars"
    
    # Query about programming
    results = await knowledge_base.query("Tell me about Python programming", k=2)
    
    assert len(results["results"]) == 2
    # First result should be about Python
    assert "Python" in results["results"][0]["content"]
    assert results["results"][0]["metadata"]["subject"] == "Python"


@pytest.mark.asyncio
async def test_retrieval_tool_execution(retrieval_tool):
    """Test executing the retrieval tool directly."""
    # Execute the tool
    result = await retrieval_tool.execute(
        query="What is the Eiffel Tower?",
        k=1
    )
    
    # Check the results
    assert "results" in result
    assert len(result["results"]) == 1
    assert "Eiffel Tower" in result["results"][0]["content"]
    assert result["results"][0]["metadata"]["subject"] == "Eiffel Tower"
    
    # Check formatting
    assert "formatted_results" in result
    assert "Eiffel Tower" in result["formatted_results"]


@pytest.mark.asyncio
async def test_filtered_retrieval(retrieval_tool):
    """Test retrieving with metadata filters."""
    # Filter by topic
    result = await retrieval_tool.execute(
        query="What can you tell me about science?",
        filter={"topic": "computer science"},
        k=2
    )
    
    # Check the results
    assert len(result["results"]) == 2
    assert all(r["metadata"]["topic"] == "computer science" for r in result["results"])
    
    # More specific filter
    result = await retrieval_tool.execute(
        query="Tell me about programming",
        filter={"subject": "Python"},
        k=1
    )
    
    assert len(result["results"]) == 1
    assert result["results"][0]["metadata"]["subject"] == "Python"


@pytest.mark.asyncio
async def test_agent_with_rag_integration(agent_with_rag):
    """Test the full agent with RAG integration."""
    # This is a basic integration test to ensure components work together
    # We're using a mock LLM to avoid actual API calls
    
    response = await agent_with_rag.run("What is Mars?")
    
    # Check that we got a response
    assert "response" in response
    
    # In a real system, the agent would use the retrieved information
    # to answer the question. With our mock, we're just checking that
    # the tool was called correctly.
    assert "tool_calls" in response
    assert len(response["tool_calls"]) > 0
    
    # First tool call should be to retrieve
    assert response["tool_calls"][0]["tool"] == "retrieve"

    
@pytest.mark.asyncio
async def test_knowledge_base_document_management(knowledge_base):
    """Test document management in the knowledge base."""
    # Get initial stats
    initial_stats = await knowledge_base.get_stats()
    initial_count = initial_stats["documents"]
    
    # Add a new document
    doc_id = await knowledge_base.add_document(
        content="JavaScript is a programming language commonly used for web development.",
        metadata={"topic": "computer science", "subject": "JavaScript"},
        chunk=False
    )
    
    # Check document was added
    updated_stats = await knowledge_base.get_stats()
    assert updated_stats["documents"] == initial_count + 1
    
    # Test document retrieval
    results = await knowledge_base.query("JavaScript web development", k=1)
    assert len(results["results"]) == 1
    assert "JavaScript" in results["results"][0]["content"]
    
    # Update document
    success = await knowledge_base.update_document(
        doc_id=doc_id,
        content="JavaScript is a high-level programming language primarily used for web development. It was created by Brendan Eich.",
        chunk=False
    )
    assert success
    
    # Verify update
    results = await knowledge_base.query("Who created JavaScript?", k=1)
    assert len(results["results"]) == 1
    assert "Brendan Eich" in results["results"][0]["content"]
    
    # Remove document
    success = await knowledge_base.remove_document(doc_id)
    assert success
    
    # Verify removal
    final_stats = await knowledge_base.get_stats()
    assert final_stats["documents"] == initial_count


@pytest.mark.asyncio
async def test_knowledge_base_clear_and_rebuild(knowledge_base):
    """Test clearing and rebuilding the knowledge base."""
    # Clear the knowledge base
    success = await knowledge_base.clear()
    assert success
    
    # Verify it's empty
    stats = await knowledge_base.get_stats()
    assert stats["documents"] == 0
    
    # Add documents back
    for doc in TEST_KNOWLEDGE:
        await knowledge_base.add_document(
            content=doc["text"],
            metadata=doc["metadata"],
            chunk=False
        )
    
    # Verify documents were added
    stats = await knowledge_base.get_stats()
    assert stats["documents"] == len(TEST_KNOWLEDGE)
    
    # Test retrieval still works
    results = await knowledge_base.query("What is the Great Barrier Reef?", k=1)
    assert len(results["results"]) == 1
    assert "Great Barrier Reef" in results["results"][0]["content"]


@pytest.mark.asyncio
async def test_chunking_functionality(local_embedder, temp_dir):
    """Test document chunking functionality."""
    # Create a knowledge base with chunking enabled
    kb = KnowledgeBase(
        name="chunking_test",
        embedder=local_embedder,
        vector_store_type="faiss",
        persist_path=os.path.join(temp_dir, "chunk_test"),
        config={"chunk_size": 100, "chunk_overlap": 20}
    )
    
    # Create a longer document that will be chunked
    long_document = (
        "Artificial Intelligence (AI) is intelligence demonstrated by machines. "
        "Machine Learning is a subset of AI that focuses on data and algorithms. "
        "Deep Learning is a subset of Machine Learning using neural networks with multiple layers. "
        "Natural Language Processing (NLP) is a field of AI focused on interactions between computers and human language. "
        "Computer Vision involves teaching computers to interpret and understand visual information. "
        "Reinforcement Learning is about training agents to make sequences of decisions."
    )
    
    # Add the document with chunking enabled
    doc_id = await kb.add_document(
        content=long_document,
        metadata={"topic": "AI overview"},
        chunk=True
    )
    
    # The document should be split into multiple chunks
    doc_info = kb.document_index[doc_id]
    assert doc_info["chunks"] > 1
    
    # Test retrieving specific information
    results = await kb.query("What is NLP?", k=1)
    assert len(results["results"]) == 1
    assert "Natural Language Processing" in results["results"][0]["content"]
    
    # Test retrieving different information
    results = await kb.query("Tell me about reinforcement learning", k=1)
    assert len(results["results"]) == 1
    assert "Reinforcement Learning" in results["results"][0]["content"]