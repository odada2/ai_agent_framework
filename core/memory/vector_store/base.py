"""
Base Vector Database Interface

This module defines the abstract base class for all vector database integrations,
providing a standardized interface for storing, retrieving, and searching embeddings.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class Document:
    """
    Document class representing text with metadata and optional embedding.
    
    This class provides a standardized format for documents that can be 
    stored in and retrieved from vector databases.
    """
    
    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ):
        """
        Initialize a document.
        
        Args:
            text: The document text content
            metadata: Optional metadata associated with the document
            id: Optional unique identifier
            embedding: Optional pre-computed embedding vector
        """
        self.text = text
        self.metadata = metadata or {}
        self.id = id
        self.embedding = embedding
    
    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document(id={self.id}, text='{self.text[:50]}...', metadata={self.metadata})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a Document from dictionary data."""
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            id=data.get("id"),
            embedding=data.get("embedding")
        )


class VectorStore(ABC):
    """
    Abstract base class for vector database integrations.
    
    This class defines the interface that all vector database
    implementations must follow to ensure consistent behavior.
    """
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings for the documents
            
        Returns:
            List of document IDs generated or used
        """
        pass
    
    @abstractmethod
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add text strings to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts for each text
            ids: Optional list of IDs for each text
            embeddings: Optional pre-computed embeddings for the texts
            
        Returns:
            List of document IDs generated or used
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for documents similar to the query string.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Document objects most similar to the query
        """
        pass
    
    @abstractmethod
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query string with similarity scores.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples most similar to the query
        """
        pass
    
    @abstractmethod
    async def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for documents similar to the embedding vector.
        
        Args:
            embedding: Embedding vector to search for
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Document objects most similar to the embedding
        """
        pass
    
    @abstractmethod
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, filter: Optional[Dict[str, Any]] = None) -> bool:
        """
        Delete documents matching the filter criteria.
        
        Args:
            filter: Metadata filter to match documents for deletion
            
        Returns:
            True if operation succeeded
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary of collection statistics
        """
        pass