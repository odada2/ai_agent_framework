"""
Chroma Vector Database Integration

This module provides integration with the Chroma vector database,
a lightweight, open-source embedding database optimized for developer experience.
"""

import logging
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union, cast

from .base import VectorStore, Document

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """
    Vector store implementation backed by Chroma DB.
    
    Chroma is an open-source embedding database designed for
    AI applications with Python and JavaScript clients.
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        embedding_function: Any = None,
        client_settings: Optional[Dict[str, Any]] = None,
        client=None,
        **kwargs
    ):
        """
        Initialize the Chroma vector store.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist data
            embedding_function: Function to use for embeddings
            client_settings: Optional settings for Chroma client
            client: Optional pre-configured Chroma client
            **kwargs: Additional arguments for Chroma
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Set up embedding function
        self.embedding_function = embedding_function
        
        # Set up Chroma client
        if client is not None:
            self.client = client
        elif client_settings is not None:
            self.client = chromadb.Client(client_settings)
        elif persist_directory is not None:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Initialized Chroma vector store with collection '{collection_name}'")
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            import chromadb
            
            # Try to get existing collection
            try:
                return self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            except ValueError:
                # Collection doesn't exist, create it
                return self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                
        except Exception as e:
            logger.error(f"Error getting/creating Chroma collection: {str(e)}")
            raise
    
    async def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the Chroma vector store.
        
        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of document IDs
        """
        # Generate IDs if not already present
        ids = []
        for doc in documents:
            if doc.id is None:
                doc.id = str(uuid.uuid4())
            ids.append(doc.id)
        
        # Extract texts and metadata
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        try:
            # Add to Chroma
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {str(e)}")
            raise
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add texts to the Chroma vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of document IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Create default metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        try:
            # Add to Chroma
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return ids
            
        except Exception as e:
            logger.error(f"Error adding texts to Chroma: {str(e)}")
            raise
    
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
        # Get search results without scores
        results_with_scores = await self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        
        # Return just the documents
        return [doc for doc, _ in results_with_scores]
    
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
            List of (Document, score) tuples
        """
        try:
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]
            
            # Create Document objects with scores
            # Note: Chroma returns distances (lower is better), convert to scores (higher is better)
            doc_score_pairs = []
            
            for i in range(len(documents)):
                doc = Document(
                    text=documents[i],
                    metadata=metadatas[i],
                    id=ids[i]
                )
                
                # Convert distance to similarity score (1 - normalized_distance)
                # Chroma uses L2 distance or cosine distance, so we normalize and invert
                distance = distances[i]
                similarity = 1.0 - min(distance, 2.0) / 2.0  # Normalize and invert
                
                doc_score_pairs.append((doc, similarity))
            
            return doc_score_pairs
            
        except Exception as e:
            logger.error(f"Error performing similarity search in Chroma: {str(e)}")
            raise
    
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
        try:
            # Perform vector search
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=filter,
                include=["documents", "metadatas"]
            )
            
            # Extract results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]
            
            # Create Document objects
            docs = []
            
            for i in range(len(documents)):
                doc = Document(
                    text=documents[i],
                    metadata=metadatas[i],
                    id=ids[i]
                )
                docs.append(doc)
            
            return docs
            
        except Exception as e:
            logger.error(f"Error performing vector search in Chroma: {str(e)}")
            raise
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        try:
            # Get document by ID
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            # Check if document was found
            if not result["documents"]:
                return None
            
            # Create Document object
            return Document(
                text=result["documents"][0],
                metadata=result["metadatas"][0],
                id=doc_id
            )
            
        except Exception as e:
            logger.error(f"Error getting document from Chroma: {str(e)}")
            raise
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        try:
            # Delete document by ID
            self.collection.delete(ids=[doc_id])
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document from Chroma: {str(e)}")
            return False
    
    async def delete(self, filter: Optional[Dict[str, Any]] = None) -> bool:
        """
        Delete documents matching the filter criteria.
        
        Args:
            filter: Metadata filter to match documents for deletion
            
        Returns:
            True if operation succeeded
        """
        try:
            if filter is None:
                # Delete all documents
                self.collection.delete()
            else:
                # Get matching document IDs
                results = self.collection.get(where=filter, include=[])
                ids = results.get("ids", [])
                
                # Delete matching documents
                if ids:
                    self.collection.delete(ids=ids)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from Chroma: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary of collection statistics
        """
        try:
            # Get collection info
            count = self.collection.count()
            
            # Build stats
            stats = {
                "name": self.collection_name,
                "count": count,
                "type": "chroma",
                "persistent": self.persist_directory is not None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats from Chroma: {str(e)}")
            return {
                "name": self.collection_name,
                "error": str(e),
                "type": "chroma"
            }