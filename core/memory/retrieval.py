"""
Memory Retrieval System

This module provides high-level functionality for retrieving information from
various memory sources, including vector stores and conversation history.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple

from .vector_store import VectorStore, Document, get_vector_store
from .embeddings import Embedder, get_embedder
from .text_utils import chunk_text

logger = logging.getLogger(__name__)


class RetrievalSystem:
    """
    Unified system for retrieving information from multiple sources.
    
    This class provides high-level methods for retrieving information from
    various sources, including documents, knowledge bases, and conversation
    history, using vector similarity search.
    """
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_stores: Optional[Dict[str, VectorStore]] = None,
        default_k: int = 5,
        reranker: Optional[Callable] = None
    ):
        """
        Initialize the retrieval system.
        
        Args:
            embedder: Embedder to use for queries (will create default if None)
            vector_stores: Dictionary of named vector stores
            default_k: Default number of results to retrieve
            reranker: Optional function to rerank retrieved results
        """
        # Initialize embedder
        self.embedder = embedder or get_embedder(embedder_type="openai")
        
        # Initialize vector stores
        self.vector_stores = vector_stores or {}
        
        # Set defaults
        self.default_k = default_k
        
        # Set up reranker if provided
        self.reranker = reranker
    
    def add_vector_store(
        self,
        name: str,
        vector_store: VectorStore
    ) -> None:
        """
        Add a vector store to the retrieval system.
        
        Args:
            name: Name to identify the vector store
            vector_store: The vector store instance
        """
        self.vector_stores[name] = vector_store
        logger.info(f"Added vector store '{name}' to retrieval system")
    
    def create_vector_store(
        self,
        name: str,
        vector_store_type: str = "chroma",
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        **kwargs
    ) -> VectorStore:
        """
        Create and add a new vector store.
        
        Args:
            name: Name to identify the vector store
            vector_store_type: Type of vector store to create
            collection_name: Name of collection (for supported stores)
            persist_directory: Directory to persist data (for supported stores)
            **kwargs: Additional vector store parameters
            
        Returns:
            The created vector store
        """
        # Create the vector store
        vector_store = get_vector_store(
            vector_store_type=vector_store_type,
            embedder=self.embedder,
            collection_name=collection_name or name,
            persist_directory=persist_directory,
            **kwargs
        )
        
        # Add to the registry
        self.add_vector_store(name, vector_store)
        
        return vector_store
    
    async def query(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Query all or specified vector stores for relevant information.
        
        Args:
            query: Query to search for
            sources: Optional list of vector store names to search (None = all)
            k: Number of results per source
            filter: Optional metadata filter
            rerank: Whether to rerank results across sources
            
        Returns:
            Dictionary with search results
        """
        # Set defaults
        k = k or self.default_k
        sources_to_query = sources or list(self.vector_stores.keys())
        
        # Check if we have any sources
        if not self.vector_stores or not sources_to_query:
            return {
                "query": query,
                "results": [],
                "sources": [],
                "execution_time_seconds": 0
            }
        
        # Start timing
        start_time = time.time()
        
        # Query each source
        all_results = []
        source_results = {}
        
        for source_name in sources_to_query:
            if source_name not in self.vector_stores:
                logger.warning(f"Vector store '{source_name}' not found")
                continue
                
            vector_store = self.vector_stores[source_name]
            
            try:
                # Perform the search
                results = await vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter
                )
                
                # Store results by source
                source_results[source_name] = results
                
                # Add source information to results
                for doc, score in results:
                    doc.metadata["source_store"] = source_name
                    all_results.append((doc, score))
                    
            except Exception as e:
                logger.error(f"Error querying vector store '{source_name}': {str(e)}")
        
        # Sort all results by score (descending)
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply reranker if available and requested
        if rerank and self.reranker and len(all_results) > k:
            try:
                all_results = self.reranker(query, all_results)
            except Exception as e:
                logger.error(f"Error reranking results: {str(e)}")
        
        # Format the results
        formatted_results = []
        
        for doc, score in all_results[:k]:
            formatted_results.append({
                "content": doc.text,
                "metadata": doc.metadata,
                "score": score
            })
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Return the results
        return {
            "query": query,
            "results": formatted_results,
            "sources": list(source_results.keys()),
            "execution_time_seconds": execution_time,
            "total_results_found": len(all_results)
        }
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        store_name: str = "default",
        ids: Optional[List[str]] = None,
        chunk: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        Add texts to a vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional metadata for each text
            store_name: Name of vector store to add to
            ids: Optional IDs for each text
            chunk: Whether to chunk texts
            chunk_size: Size of chunks if chunking
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document IDs
        """
        # Ensure vector store exists
        if store_name not in self.vector_stores:
            raise ValueError(f"Vector store '{store_name}' not found")
            
        vector_store = self.vector_stores[store_name]
        
        # Handle chunking if requested
        if chunk:
            chunked_texts = []
            chunked_metadatas = []
            
            for i, text in enumerate(texts):
                # Create chunks
                chunks = chunk_text(
                    text=text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Add chunks to lists
                for j, chunk in enumerate(chunks):
                    chunked_texts.append(chunk.text)
                    
                    # Create metadata for chunk
                    chunk_metadata = {}
                    if metadatas and i < len(metadatas):
                        chunk_metadata = metadatas[i].copy()
                    
                    # Add chunk information
                    chunk_metadata.update({
                        "chunk_index": j,
                        "chunk_count": len(chunks),
                        "original_index": i
                    })
                    
                    chunked_metadatas.append(chunk_metadata)
            
            # Update texts and metadatas
            texts = chunked_texts
            metadatas = chunked_metadatas
        
        # Add to vector store
        return await vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    async def add_documents(
        self,
        documents: List[Document],
        store_name: str = "default"
    ) -> List[str]:
        """
        Add documents to a vector store.
        
        Args:
            documents: List of documents to add
            store_name: Name of vector store to add to
            
        Returns:
            List of document IDs
        """
        # Ensure vector store exists
        if store_name not in self.vector_stores:
            raise ValueError(f"Vector store '{store_name}' not found")
            
        vector_store = self.vector_stores[store_name]
        
        # Add to vector store
        return await vector_store.add_documents(documents)
    
    async def delete(
        self,
        store_name: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete documents from vector stores.
        
        Args:
            store_name: Name of specific vector store (None = all)
            filter: Metadata filter for documents to delete
            
        Returns:
            True if successful
        """
        if store_name is not None:
            # Delete from specific store
            if store_name not in self.vector_stores:
                logger.warning(f"Vector store '{store_name}' not found")
                return False
                
            return await self.vector_stores[store_name].delete(filter)
        else:
            # Delete from all stores
            success = True
            
            for name, store in self.vector_stores.items():
                try:
                    result = await store.delete(filter)
                    if not result:
                        success = False
                except Exception as e:
                    logger.error(f"Error deleting from vector store '{name}': {str(e)}")
                    success = False
            
            return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all vector stores.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "vector_stores": {},
            "total_stores": len(self.vector_stores)
        }
        
        # Get stats for each store
        for name, store in self.vector_stores.items():
            try:
                store_stats = store.get_collection_stats()
                stats["vector_stores"][name] = store_stats
            except Exception as e:
                logger.error(f"Error getting stats for vector store '{name}': {str(e)}")
                stats["vector_stores"][name] = {"error": str(e)}
        
        # Add total document count
        total_docs = sum(
            stats["vector_stores"][name].get("count", 0) 
            for name in stats["vector_stores"]
        )
        stats["total_documents"] = total_docs
        
        return stats


class CrossEncoder:
    """
    Simple cross-encoder for reranking retrieval results.
    
    Cross-encoders take both query and document as input and produce
    a relevance score, which is often more accurate than pure embedding
    similarity for reranking purposes.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        """
        Initialize the cross-encoder.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on ('cpu' or 'cuda')
        """
        try:
            # Conditional import to avoid hard dependency
            from sentence_transformers import CrossEncoder as STCrossEncoder
            
            self.model = STCrossEncoder(model_name, device=device)
            self.initialized = True
            
        except ImportError:
            logger.warning("sentence-transformers not installed; cross-encoder reranking unavailable")
            self.initialized = False
    
    def __call__(
        self,
        query: str,
        documents: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using the cross-encoder.
        
        Args:
            query: Query string
            documents: List of (Document, score) tuples
            
        Returns:
            Reranked list of (Document, score) tuples
        """
        if not self.initialized:
            logger.warning("Cross-encoder not initialized, returning documents in original order")
            return documents
        
        # Prepare input for cross-encoder
        sentence_pairs = [(query, doc.text) for doc, _ in documents]
        
        # Compute scores
        try:
            scores = self.model.predict(sentence_pairs)
            
            # Create new document-score pairs
            reranked = [(documents[i][0], float(scores[i])) for i in range(len(documents))]
            
            # Sort by new scores (descending)
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            return documents