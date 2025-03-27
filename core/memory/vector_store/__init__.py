"""
Vector Store Package

This package provides integrations with various vector databases for
storing and retrieving vector embeddings used in retrieval systems.
"""

from typing import Any, Dict, Optional

from .base import VectorStore, Document
from .chroma import ChromaVectorStore
from .faiss import FAISSVectorStore

__all__ = [
    'VectorStore',
    'Document',
    'ChromaVectorStore',
    'FAISSVectorStore',
    'get_vector_store'
]


def get_vector_store(
    vector_store_type: str,
    embedder: Optional[Any] = None,
    **kwargs
) -> VectorStore:
    """
    Factory function to create a vector store instance.
    
    Args:
        vector_store_type: Type of vector store ('chroma', 'faiss')
        embedder: Embedder to use with the vector store
        **kwargs: Additional parameters for the vector store
        
    Returns:
        Initialized VectorStore instance
        
    Raises:
        ValueError: If vector_store_type is not supported
    """
    vector_store_type = vector_store_type.lower()
    
    if vector_store_type == "chroma":
        # For Chroma, we need to convert our embedder to their format
        if embedder is not None:
            embedding_function = get_chroma_embedding_function(embedder)
            return ChromaVectorStore(embedding_function=embedding_function, **kwargs)
        else:
            return ChromaVectorStore(**kwargs)
            
    elif vector_store_type == "faiss":
        return FAISSVectorStore(embedder=embedder, **kwargs)
        
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")


def get_chroma_embedding_function(embedder: Any) -> Any:
    """
    Convert our embedder to a Chroma-compatible embedding function.
    
    Args:
        embedder: Our embedder object
        
    Returns:
        Chroma-compatible embedding function
    """
    # Define a wrapper class that satisfies Chroma's interface
    class ChromaEmbeddingFunction:
        def __init__(self, embedder):
            self.embedder = embedder
            
        def __call__(self, texts):
            import asyncio
            # Convert async to sync using event loop
            embeddings = asyncio.run(self.embedder.embed_documents(texts))
            return embeddings
    
    return ChromaEmbeddingFunction(embedder)