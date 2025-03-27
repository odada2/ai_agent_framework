"""
Retrieval-Augmented Generation (RAG) Tool

This module provides a tool for retrieving relevant documents from a vector store
to augment the LLM's context for improved generation.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any, Union

from ...core.tools.base import BaseTool
from ...core.memory.vector_store import VectorStore, Document
from ...core.memory.vector_store import get_vector_store
from ...core.memory.embeddings import get_embedder, Embedder

logger = logging.getLogger(__name__)


class RetrievalTool(BaseTool):
    """
    Tool for retrieving relevant information from a knowledge base.
    
    This tool performs semantic search on a vector store to find
    information relevant to the user's query.
    """
    
    def __init__(
        self,
        name: str = "retrieve",
        description: str = "Retrieve relevant information from the knowledge base.",
        vector_store: Optional[VectorStore] = None,
        vector_store_type: str = "chroma",
        vector_store_path: Optional[str] = None,
        collection_name: str = "default",
        embedder: Optional[Embedder] = None,
        embedder_type: str = "openai",
        default_k: int = 5,
        **kwargs
    ):
        """
        Initialize the retrieval tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            vector_store: Optional pre-configured vector store
            vector_store_type: Type of vector store to create if none provided
            vector_store_path: Path to vector store data
            collection_name: Name of the collection in the vector store
            embedder: Optional pre-configured embedder
            embedder_type: Type of embedder to create if none provided
            default_k: Default number of results to return
            **kwargs: Additional tool parameters
        """
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for in the knowledge base"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": default_k
                },
                "filter": {
                    "type": "object",
                    "description": "Optional metadata filter to apply"
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include document metadata in the results",
                    "default": False
                }
            },
            "required": ["query"]
        }
        
        examples = [
            {
                "description": "Retrieve information about machine learning",
                "parameters": {
                    "query": "What is machine learning?",
                    "k": 3
                }
            },
            {
                "description": "Search with metadata filter",
                "parameters": {
                    "query": "neural networks",
                    "filter": {"category": "deep_learning"}
                }
            }
        ]
        
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            examples=examples,
            required_permissions=["knowledge_base"],
            **kwargs
        )
        
        # Set up embedder
        self.embedder = embedder
        if self.embedder is None:
            self.embedder = get_embedder(embedder_type)
        
        # Set up vector store
        self.vector_store = vector_store
        if self.vector_store is None:
            self.vector_store = self._initialize_vector_store(
                vector_store_type=vector_store_type,
                vector_store_path=vector_store_path,
                collection_name=collection_name
            )
        
        self.default_k = default_k
    
    def _initialize_vector_store(
        self,
        vector_store_type: str,
        vector_store_path: Optional[str],
        collection_name: str
    ) -> VectorStore:
        """
        Initialize vector store based on configuration.
        
        Args:
            vector_store_type: Type of vector store
            vector_store_path: Path to store/load data
            collection_name: Name of collection
            
        Returns:
            Configured VectorStore instance
        """
        # Special handling for different vector store types
        if vector_store_type.lower() == "chroma":
            return get_vector_store(
                vector_store_type="chroma",
                embedder=self.embedder,
                collection_name=collection_name,
                persist_directory=vector_store_path
            )
        
        elif vector_store_type.lower() == "faiss":
            return get_vector_store(
                vector_store_type="faiss",
                embedder=self.embedder,
                index_path=vector_store_path
            )
        
        else:
            return get_vector_store(
                vector_store_type=vector_store_type,
                embedder=self.embedder
            )
    
    async def _run(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Run the retrieval tool to find relevant documents.
        
        Args:
            query: The query to search for
            k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Whether to include document metadata
            
        Returns:
            Dictionary containing search results
        """
        # Set default k if not provided
        k = k or self.default_k
        
        try:
            # Start timing
            start_time = time.time()
            
            # Perform the search
            results = await self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            # Format the results
            formatted_docs = []
            
            for doc, score in results:
                formatted_doc = {
                    "content": doc.text,
                    "score": score
                }
                
                # Include metadata if requested
                if include_metadata:
                    formatted_doc["metadata"] = doc.metadata
                
                formatted_docs.append(formatted_doc)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Return the results
            return {
                "query": query,
                "results": formatted_docs,
                "count": len(formatted_docs),
                "execution_time_seconds": execution_time
            }
            
        except Exception as e:
            logger.error(f"Error in retrieval tool: {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "results": []
            }
    
    async def add_documents(
        self,
        documents: List[Union[Document, Dict[str, Any]]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents to add
            ids: Optional list of IDs for the documents
            
        Returns:
            List of document IDs
        """
        # Convert dict documents to Document objects
        docs = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, Document):
                docs.append(doc)
            elif isinstance(doc, dict):
                doc_obj = Document(
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                    id=doc.get("id", ids[i] if ids and i < len(ids) else None)
                )
                docs.append(doc_obj)
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")
        
        # Add to vector store
        return await self.vector_store.add_documents(docs)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing stats
        """
        stats = self.vector_store.get_collection_stats()
        
        # Add embedder stats if available
        if hasattr(self.embedder, "get_stats"):
            stats["embedder"] = self.embedder.get_stats()
        
        return stats


class ConversationMemoryTool(BaseTool):
    """
    Tool for semantic search over conversation history.
    
    This tool allows agents to retrieve relevant parts of conversation
    history based on semantic similarity to the current context.
    """
    
    def __init__(
        self,
        name: str = "memory",
        description: str = "Search conversation history for relevant information.",
        vector_store: Optional[VectorStore] = None,
        vector_store_type: str = "faiss",
        vector_store_path: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        embedder_type: str = "openai",
        default_k: int = 5,
        **kwargs
    ):
        """
        Initialize the conversation memory tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            vector_store: Optional pre-configured vector store
            vector_store_type: Type of vector store to create if none provided
            vector_store_path: Path to vector store data
            embedder: Optional pre-configured embedder
            embedder_type: Type of embedder to create if none provided
            default_k: Default number of results to return
            **kwargs: Additional tool parameters
        """
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for in conversation history"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": default_k
                },
                "filter": {
                    "type": "object",
                    "description": "Optional metadata filter to apply"
                },
                "include_timestamps": {
                    "type": "boolean",
                    "description": "Whether to include message timestamps",
                    "default": True
                }
            },
            "required": ["query"]
        }
        
        examples = [
            {
                "description": "Find previous mentions of a topic",
                "parameters": {
                    "query": "What did we discuss about neural networks?",
                    "k": 3
                }
            },
            {
                "description": "Search messages from a specific role",
                "parameters": {
                    "query": "project timeline",
                    "filter": {"role": "user"}
                }
            }
        ]
        
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            examples=examples,
            required_permissions=["memory_search"],
            **kwargs
        )
        
        # Set up embedder
        self.embedder = embedder
        if self.embedder is None:
            self.embedder = get_embedder(embedder_type)
        
        # Set up vector store
        self.vector_store = vector_store
        if self.vector_store is None:
            # Default to in-memory FAISS for conversation history
            self.vector_store = get_vector_store(
                vector_store_type=vector_store_type,
                embedder=self.embedder,
                index_path=vector_store_path,
                dimension=self.embedder.embedding_dimension
            )
        
        self.default_k = default_k
        
        # Track indexed messages to avoid duplicates
        self.indexed_message_ids = set()
    
    async def _run(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Search conversation history for relevant messages.
        
        Args:
            query: The query to search for
            k: Number of results to return
            filter: Optional metadata filter
            include_timestamps: Whether to include message timestamps
            
        Returns:
            Dictionary containing search results
        """
        # Set default k if not provided
        k = k or self.default_k
        
        try:
            # Start timing
            start_time = time.time()
            
            # Perform the search
            results = await self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            # Format the results
            formatted_messages = []
            
            for doc, score in results:
                message = {
                    "content": doc.text,
                    "role": doc.metadata.get("role", "unknown"),
                    "score": score
                }
                
                # Include timestamp if requested and available
                if include_timestamps and "timestamp" in doc.metadata:
                    message["timestamp"] = doc.metadata["timestamp"]
                
                # Include message index if available
                if "index" in doc.metadata:
                    message["index"] = doc.metadata["index"]
                
                formatted_messages.append(message)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Return the results
            return {
                "query": query,
                "messages": formatted_messages,
                "count": len(formatted_messages),
                "execution_time_seconds": execution_time
            }
            
        except Exception as e:
            logger.error(f"Error in memory search tool: {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "messages": []
            }
    
    async def add_message(
        self,
        content: str,
        role: str,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a message to the searchable conversation history.
        
        Args:
            content: Message content
            role: Message role (user, assistant, system)
            message_id: Optional unique ID for the message
            metadata: Optional additional metadata
            
        Returns:
            Message ID
        """
        # Skip if this message was already indexed
        if message_id and message_id in self.indexed_message_ids:
            return message_id
        
        # Generate message ID if not provided
        if message_id is None:
            import uuid
            message_id = str(uuid.uuid4())
        
        # Prepare metadata
        meta = metadata or {}
        meta["role"] = role
        meta["message_id"] = message_id
        meta["timestamp"] = meta.get("timestamp", time.time())
        
        # Create document
        doc = Document(
            text=content,
            metadata=meta,
            id=message_id
        )
        
        # Add to vector store
        await self.vector_store.add_documents([doc])
        
        # Track indexed message
        self.indexed_message_ids.add(message_id)
        
        return message_id
    
    async def clear(self) -> bool:
        """
        Clear all messages from memory.
        
        Returns:
            True if successful
        """
        await self.vector_store.delete()
        self.indexed_message_ids.clear()
        return True