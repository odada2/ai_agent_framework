"""
FAISS Vector Database Integration

This module provides integration with the FAISS vector database,
a library for efficient similarity search and dense vector clustering.
"""

import logging
import os
import pickle
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union, Set

import numpy as np

from .base import VectorStore, Document

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    Vector store implementation backed by FAISS.
    
    FAISS is a library for efficient similarity search and 
    clustering of dense vectors developed by Facebook AI Research.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedder: Optional[Any] = None,
        dimension: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the FAISS vector store.
        
        Args:
            index_path: Path to load/save the FAISS index
            embedder: Embedding function to use
            dimension: Dimension of the embeddings
            **kwargs: Additional FAISS-specific parameters
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss python package. "
                "Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
            )
            
        self.faiss = faiss
        self.index_path = index_path
        self.embedder = embedder
        
        # Map from IDs to document data
        self.docstore: Dict[str, Document] = {}
        
        # Map from index to ID
        self.index_to_id: Dict[int, str] = {}
        self.id_to_index: Dict[str, int] = {}
        
        # Initialize or load index
        if index_path and os.path.exists(index_path):
            self._load_index()
        else:
            if dimension is None and embedder is not None:
                if hasattr(embedder, "embedding_dimension"):
                    dimension = embedder.embedding_dimension
                else:
                    # Default dimension
                    dimension = 1536
                    logger.warning(f"Dimension not specified, using default: {dimension}")
            
            if dimension is None:
                raise ValueError("Either dimension or an embedder with embedding_dimension must be provided")
                
            self.dimension = dimension
            
            # Create a new index
            self.index = faiss.IndexFlatL2(dimension)
            logger.info(f"Created new FAISS index with dimension {dimension}")
    
    def _load_index(self):
        """Load the FAISS index and document store from disk."""
        if not self.index_path:
            raise ValueError("No index path specified")
            
        try:
            # Load FAISS index
            self.index = self.faiss.read_index(f"{self.index_path}.index")
            self.dimension = self.index.d
            
            # Load document store and mappings
            with open(f"{self.index_path}.pickle", "rb") as f:
                saved_data = pickle.load(f)
                self.docstore = saved_data["docstore"]
                self.index_to_id = saved_data["index_to_id"]
                self.id_to_index = saved_data["id_to_index"]
            
            logger.info(f"Loaded FAISS index from {self.index_path} with {len(self.docstore)} documents")
                
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise
    
    def _save_index(self):
        """Save the FAISS index and document store to disk."""
        if not self.index_path:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            self.faiss.write_index(self.index, f"{self.index_path}.index")
            
            # Save document store and mappings
            with open(f"{self.index_path}.pickle", "wb") as f:
                pickle.dump(
                    {
                        "docstore": self.docstore,
                        "index_to_id": self.index_to_id,
                        "id_to_index": self.id_to_index
                    },
                    f
                )
                
            logger.info(f"Saved FAISS index to {self.index_path}")
                
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    async def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the FAISS vector store.
        
        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        # Generate embeddings if not provided
        if embeddings is None:
            if self.embedder is None:
                raise ValueError("No embedder provided and no pre-computed embeddings given")
                
            # Generate embeddings in batches
            texts = [doc.text for doc in documents]
            embeddings = await self.embedder.embed_documents(texts)
        
        # Convert embeddings to numpy array
        vectors = np.array(embeddings).astype("float32")
        
        # Generate IDs if not already present
        ids = []
        for i, doc in enumerate(documents):
            if doc.id is None:
                doc.id = str(uuid.uuid4())
            ids.append(doc.id)
            
            # Store document
            self.docstore[doc.id] = doc
        
        # Update index
        start_index = len(self.index_to_id)
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Update mappings
        for i, doc_id in enumerate(ids):
            index = start_index + i
            self.index_to_id[index] = doc_id
            self.id_to_index[doc_id] = index
        
        # Save index if path is specified
        if self.index_path:
            self._save_index()
        
        return ids
    
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
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of document IDs
        """
        # Create Document objects
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        documents = [
            Document(text=text, metadata=metadata, id=id_)
            for text, metadata, id_ in zip(texts, metadatas, ids)
        ]
        
        # Add documents
        return await self.add_documents(documents, embeddings)
    
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
        # Get query embedding
        if self.embedder is None:
            raise ValueError("No embedder provided")
            
        query_embedding = await self.embedder.embed_query(query)
        
        # Search by vector
        return await self.similarity_search_by_vector_with_score(
            embedding=query_embedding,
            k=k,
            filter=filter
        )
    
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
        # Get search results without scores
        results_with_scores = await self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            filter=filter
        )
        
        # Return just the documents
        return [doc for doc, _ in results_with_scores]
    
    async def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the embedding vector with similarity scores.
        
        Args:
            embedding: Embedding vector to search for
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.docstore:
            return []
            
        # Convert embedding to numpy array
        query_vector = np.array([embedding]).astype("float32")
        
        # Determine if we need to filter results
        if filter is not None:
            # Get all document IDs that match the filter
            filtered_ids = set()
            for doc_id, doc in self.docstore.items():
                if self._matches_filter(doc.metadata, filter):
                    filtered_ids.add(doc_id)
            
            if not filtered_ids:
                return []
                
            # Get all index positions for these IDs
            filtered_indices = [
                self.id_to_index[doc_id] 
                for doc_id in filtered_ids 
                if doc_id in self.id_to_index
            ]
            
            # If no valid indices, return empty list
            if not filtered_indices:
                return []
                
            # Create a subset index with only filtered documents
            subset_index = self._create_subset_index(filtered_indices)
            
            # Search in subset index
            distances, indices = subset_index.search(query_vector, k)
            
            # Map subset indices back to original indices
            original_indices = [filtered_indices[i] for i in indices[0] if i >= 0 and i < len(filtered_indices)]
        else:
            # Search in full index
            distances, indices = self.index.search(query_vector, k)
            original_indices = [int(i) for i in indices[0] if i >= 0]
        
        # Create result list
        results = []
        
        for i, index in enumerate(original_indices):
            if index in self.index_to_id:
                doc_id = self.index_to_id[index]
                if doc_id in self.docstore:
                    doc = self.docstore[doc_id]
                    
                    # Convert distance to similarity score
                    # FAISS uses L2 distance, so we convert to similarity
                    distance = float(distances[0][i])
                    
                    # Normalize and invert: larger distances -> lower similarity
                    # Using a simple exponential decay formula
                    similarity = np.exp(-distance / 10)
                    
                    results.append((doc, similarity))
        
        return results
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        return self.docstore.get(doc_id)
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        # FAISS doesn't support direct removal, so we:
        # 1. Remove from docstore
        # 2. Mark index as invalid in mappings
        # 3. Rebuild index if too many deleted items
        
        if doc_id not in self.docstore:
            return False
            
        # Remove from docstore
        del self.docstore[doc_id]
        
        # Remove from mappings if present
        if doc_id in self.id_to_index:
            index = self.id_to_index[doc_id]
            del self.id_to_index[doc_id]
            
            if index in self.index_to_id:
                del self.index_to_id[index]
        
        # Save changes if path is specified
        if self.index_path:
            self._save_index()
            
        return True
    
    async def delete(self, filter: Optional[Dict[str, Any]] = None) -> bool:
        """
        Delete documents matching the filter criteria.
        
        Args:
            filter: Metadata filter to match documents for deletion
            
        Returns:
            True if operation succeeded
        """
        if filter is None:
            # Delete everything
            self.docstore = {}
            self.index_to_id = {}
            self.id_to_index = {}
            
            # Reset index
            self.index = self.faiss.IndexFlatL2(self.dimension)
        else:
            # Find documents matching filter
            to_delete = []
            
            for doc_id, doc in self.docstore.items():
                if self._matches_filter(doc.metadata, filter):
                    to_delete.append(doc_id)
            
            # Delete matching documents
            for doc_id in to_delete:
                await self.delete_document(doc_id)
        
        # Save changes if path is specified
        if self.index_path:
            self._save_index()
            
        return True
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary of collection statistics
        """
        return {
            "count": len(self.docstore),
            "dimension": self.dimension,
            "type": "faiss",
            "persistent": self.index_path is not None
        }
    
    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the filter criteria.
        
        Args:
            metadata: Document metadata
            filter: Filter criteria
            
        Returns:
            True if metadata matches filter
        """
        for key, value in filter.items():
            # Handle nested keys with dot notation
            if "." in key:
                parts = key.split(".")
                current = metadata
                
                # Navigate through nested structure
                for part in parts[:-1]:
                    if part not in current or not isinstance(current[part], dict):
                        return False
                    current = current[part]
                
                # Check final key
                final_key = parts[-1]
                if final_key not in current or current[final_key] != value:
                    return False
            
            # Simple key check
            elif key not in metadata or metadata[key] != value:
                return False
                
        return True
    
    def _create_subset_index(self, indices: List[int]) -> Any:
        """
        Create a temporary FAISS index containing only the specified indices.
        
        Args:
            indices: List of indices to include
            
        Returns:
            FAISS index containing only the requested vectors
        """
        if not indices:
            return self.faiss.IndexFlatL2(self.dimension)
            
        # Extract vectors for these indices
        vectors = []
        
        for index in indices:
            # FAISS doesn't have a simple API to extract vectors by index
            # Here we're using a reconstruction approach
            vector = self.index.reconstruct(int(index))
            vectors.append(vector)
        
        # Create new index with these vectors
        subset_index = self.faiss.IndexFlatL2(self.dimension)
        subset_index.add(np.array(vectors).astype("float32"))
        
        return subset_index