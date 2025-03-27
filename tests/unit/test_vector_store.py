"""
Unit Tests for Vector Store Implementation

This module provides tests for the vector store interfaces and implementations.
"""

import pytest
import asyncio
import os
import tempfile
from typing import Dict, List, Any, Optional

from ai_agent_framework.core.memory.vector_store import (
    Document,
    VectorStore,
    get_vector_store
)
from ai_agent_framework.core.memory.vector_store.base import Document
from ai_agent_framework.core.memory.vector_store.faiss import FAISSVectorStore
from ai_agent_framework.core.memory.embeddings import get_embedder, LocalEmbedder


# Test Documents
TEST_DOCS = [
    Document(
        text="The quick brown fox jumps over the lazy dog",
        metadata={"source": "test", "animal": "fox"}
    ),
    Document(
        text="Machine learning is a subset of artificial intelligence",
        metadata={"source": "test", "topic": "ai"}
    ),
    Document(
        text="Python is a programming language with clean syntax",
        metadata={"source": "test", "topic": "programming"}
    ),
    Document(
        text="Neural networks are inspired by the human brain",
        metadata={"source": "test", "topic": "ai"}
    ),
    Document(
        text="The Transformer architecture revolutionized NLP",
        metadata={"source": "test", "topic": "ai"}
    )
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
async def faiss_store(local_embedder, temp_dir):
    """Create a FAISS vector store for testing."""
    store = FAISSVectorStore(
        embedder=local_embedder,
        index_path=os.path.join(temp_dir, "faiss_test")
    )
    
    # Add test documents
    await store.add_documents(TEST_DOCS)
    
    return store


@pytest.mark.asyncio
async def test_document_creation():
    """Test document creation and properties."""
    doc = Document(
        text="Test document",
        metadata={"source": "test"},
        id="test-doc"
    )
    
    assert doc.text == "Test document"
    assert doc.metadata == {"source": "test"}
    assert doc.id == "test-doc"
    assert doc.embedding is None
    
    # Test to_dict and from_dict
    doc_dict = doc.to_dict()
    restored_doc = Document.from_dict(doc_dict)
    
    assert restored_doc.text == doc.text
    assert restored_doc.metadata == doc.metadata
    assert restored_doc.id == doc.id


@pytest.mark.asyncio
async def test_faiss_vector_store_basics(faiss_store):
    """Test basic FAISS vector store operations."""
    # Check we have the right number of documents
    stats = faiss_store.get_collection_stats()
    assert stats["count"] == len(TEST_DOCS)
    
    # Test get_document_by_id
    doc = await faiss_store.get_document_by_id(TEST_DOCS[0].id)
    assert doc is not None
    assert doc.text == TEST_DOCS[0].text
    
    # Test delete_document
    success = await faiss_store.delete_document(TEST_DOCS[0].id)
    assert success
    
    # Verify deletion
    doc = await faiss_store.get_document_by_id(TEST_DOCS[0].id)
    assert doc is None
    
    # Check count was updated
    stats = faiss_store.get_collection_stats()
    assert stats["count"] == len(TEST_DOCS) - 1


@pytest.mark.asyncio
async def test_faiss_similarity_search(faiss_store):
    """Test similarity search functionality in FAISS store."""
    # Search for AI-related content
    results = await faiss_store.similarity_search(
        query="artificial intelligence and neural networks",
        k=2
    )
    
    # Should find the ML and neural network documents
    assert len(results) == 2
    assert any("artificial intelligence" in doc.text.lower() for doc in results)
    assert any("neural networks" in doc.text.lower() for doc in results)
    
    # Test with filter
    results = await faiss_store.similarity_search(
        query="artificial intelligence",
        k=3,
        filter={"topic": "ai"}
    )
    
    # Should only return docs with topic=ai
    assert len(results) > 0
    assert all(doc.metadata.get("topic") == "ai" for doc in results)


@pytest.mark.asyncio
async def test_factory_function(local_embedder, temp_dir):
    """Test the vector store factory function."""
    # Create FAISS store
    faiss_store = get_vector_store(
        vector_store_type="faiss",
        embedder=local_embedder,
        index_path=os.path.join(temp_dir, "factory_test")
    )
    
    assert isinstance(faiss_store, FAISSVectorStore)
    
    # Test with invalid type
    with pytest.raises(ValueError):
        get_vector_store(
            vector_store_type="invalid_type",
            embedder=local_embedder
        )


@pytest.mark.asyncio
async def test_add_texts(faiss_store):
    """Test adding texts instead of documents."""
    texts = [
        "Vectors stores are efficient for similarity search",
        "Embeddings capture semantic meaning of text"
    ]
    metadatas = [
        {"category": "databases"},
        {"category": "nlp"}
    ]
    
    # Add texts
    ids = await faiss_store.add_texts(
        texts=texts,
        metadatas=metadatas
    )
    
    assert len(ids) == 2
    
    # Verify they were added
    stats = faiss_store.get_collection_stats()
    assert stats["count"] == len(TEST_DOCS) + 2
    
    # Test retrieval
    results = await faiss_store.similarity_search(
        query="semantic meaning",
        k=1
    )
    
    assert len(results) == 1
    assert "embeddings" in results[0].text.lower()
    assert results[0].metadata.get("category") == "nlp"


@pytest.mark.asyncio
async def test_similarity_search_with_score(faiss_store):
    """Test similarity search with scores."""
    # Search with scores
    results = await faiss_store.similarity_search_with_score(
        query="programming language",
        k=2
    )
    
    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)
    
    # Scores should be between 0 and 1
    for _, score in results:
        assert 0 <= score <= 1
    
    # First result should be about Python
    assert "python" in results[0][0].text.lower()
    
    # Scores should be in descending order
    assert results[0][1] >= results[1][1]


@pytest.mark.asyncio
async def test_batch_operations(local_embedder, temp_dir):
    """Test batch operations on vector store."""
    store = FAISSVectorStore(
        embedder=local_embedder,
        index_path=os.path.join(temp_dir, "batch_test")
    )
    
    # Add documents in batch
    ids = await store.add_documents(TEST_DOCS)
    assert len(ids) == len(TEST_DOCS)
    
    # Verify they were added
    stats = store.get_collection_stats()
    assert stats["count"] == len(TEST_DOCS)
    
    # Delete with filter
    success = await store.delete(filter={"topic": "ai"})
    assert success
    
    # Verify deletion
    stats = store.get_collection_stats()
    assert stats["count"] == len(TEST_DOCS) - 3  # 3 documents have topic=ai
    
    # Delete all
    success = await store.delete()
    assert success
    
    # Verify deletion
    stats = store.get_collection_stats()
    assert stats["count"] == 0