"""
Conversation Memory

This module provides functionality for storing and managing conversation history,
with optional vector storage for semantic search capabilities.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from .vector_store import Document, VectorStore, get_vector_store
from .embeddings import get_embedder, Embedder

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history for an agent.
    
    This class handles storing, retrieving, and managing messages in a conversation,
    with support for summarization, selective retention, and optional vector-based
    semantic search for conversation history.
    """
    
    def __init__(
        self,
        max_messages: int = 100,
        max_tokens: Optional[int] = None,
        include_timestamps: bool = True,
        use_vector_storage: bool = False,
        vector_store: Optional[VectorStore] = None,
        vector_store_type: str = "faiss",
        embedder: Optional[Embedder] = None,
        embedder_type: str = "openai",
        persist_path: Optional[str] = None
    ):
        """
        Initialize the conversation memory.
        
        Args:
            max_messages: Maximum number of messages to store
            max_tokens: Optional maximum total tokens to store (requires token counting)
            include_timestamps: Whether to include timestamps with messages
            use_vector_storage: Whether to use vector storage for semantic search
            vector_store: Optional pre-configured vector store
            vector_store_type: Type of vector store to create if none provided
            embedder: Optional pre-configured embedder
            embedder_type: Type of embedder to create if none provided
            persist_path: Path to persist vector storage data
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.include_timestamps = include_timestamps
        
        self.messages: List[Dict[str, Any]] = []
        self.summaries: List[Dict[str, Any]] = []
        self.total_tokens: int = 0
        
        # Set up vector storage if enabled
        self.use_vector_storage = use_vector_storage
        self.vector_store = None
        self.embedder = None
        
        if self.use_vector_storage:
            # Initialize embedder
            self.embedder = embedder or get_embedder(embedder_type)
            
            # Initialize vector store
            if vector_store:
                self.vector_store = vector_store
            else:
                self.vector_store = get_vector_store(
                    vector_store_type=vector_store_type,
                    embedder=self.embedder,
                    persist_directory=persist_path,
                    collection_name="conversation_memory"
                )
        
        # Track indexed message IDs
        self.indexed_message_ids = set()
    
    def add_message(
        self,
        content: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            content: The message content
            role: The role of the message sender (e.g., 'user', 'assistant')
            metadata: Optional additional metadata for the message
        """
        # Generate a unique ID for the message
        message_id = f"{role}_{int(time.time() * 1000)}_{len(self.messages)}"
        
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        if self.include_timestamps:
            message["timestamp"] = time.time()
        
        # Calculate approximate tokens
        # In production, use a proper tokenizer here
        approximate_tokens = len(content.split())
        
        # Check if we need to make room by pruning old messages
        if len(self.messages) >= self.max_messages or (
            self.max_tokens and self.total_tokens + approximate_tokens > self.max_tokens
        ):
            self._prune_history()
        
        # Add message to list
        self.messages.append(message)
        self.total_tokens += approximate_tokens
        
        # Add to vector store if enabled
        if self.use_vector_storage and self.vector_store:
            self._index_message(message)
    
    async def _index_message(self, message: Dict[str, Any]) -> None:
        """
        Index a message in the vector store.
        
        Args:
            message: Message dict to index
        """
        try:
            # Create a document
            doc = Document(
                text=message["content"],
                metadata={
                    "role": message["role"],
                    "message_id": message["id"],
                    "timestamp": message.get("timestamp", time.time()),
                    "index": len(self.messages) - 1
                },
                id=message["id"]
            )
            
            # Add to vector store
            await self.vector_store.add_documents([doc])
            
            # Track indexed message
            self.indexed_message_ids.add(message["id"])
            
        except Exception as e:
            logger.error(f"Error indexing message in vector store: {str(e)}")
    
    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a system message to the conversation history.
        
        Args:
            content: The system message content
            metadata: Optional additional metadata for the message
        """
        self.add_message(content, role="system", metadata=metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            content: The assistant message content
            metadata: Optional additional metadata for the message
        """
        self.add_message(content, role="assistant", metadata=metadata)
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            content: The user message content
            metadata: Optional additional metadata for the message
        """
        self.add_message(content, role="user", metadata=metadata)
    
    def get_messages(
        self,
        count: Optional[int] = None,
        roles: Optional[List[str]] = None,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages from the conversation history.
        
        Args:
            count: Maximum number of messages to return (from most recent)
            roles: Optional filter for specific roles
            include_metadata: Whether to include metadata in the returned messages
            
        Returns:
            List of messages
        """
        # Filter by role if specified
        if roles:
            filtered_messages = [m for m in self.messages if m["role"] in roles]
        else:
            filtered_messages = self.messages.copy()
        
        # Limit by count if specified
        if count:
            filtered_messages = filtered_messages[-count:]
        
        # Remove metadata if not requested
        if not include_metadata:
            result_messages = []
            for message in filtered_messages:
                message_copy = message.copy()
                if "metadata" in message_copy:
                    del message_copy["metadata"]
                result_messages.append(message_copy)
            return result_messages
        
        return filtered_messages
    
    def get_conversation_history(
        self,
        format_type: str = "string",
        count: Optional[int] = None,
        include_roles: bool = True
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Get the conversation history in a specified format.
        
        Args:
            format_type: Format type ('string', 'list', or 'dict')
            count: Optional limit on number of messages to include
            include_roles: Whether to include roles in string format
            
        Returns:
            Conversation history in the requested format
        """
        messages = self.get_messages(count)
        
        # Return as a formatted string
        if format_type.lower() == "string":
            lines = []
            for msg in messages:
                if include_roles:
                    lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
                else:
                    lines.append(msg['content'])
            return "\n\n".join(lines)
        
        # Return as a list of dictionaries
        elif format_type.lower() in ("list", "dict"):
            result = []
            for msg in messages:
                result.append({"role": msg["role"], "content": msg["content"]})
            return result
        
        # Unsupported format
        else:
            logger.warning(f"Unsupported format type: {format_type}, defaulting to string")
            return self.get_conversation_history(format_type="string", count=count)
    
    async def search_history(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of matching messages with similarity scores
        """
        if not self.use_vector_storage or not self.vector_store:
            logger.warning("Vector storage not enabled, returning empty results")
            return []
        
        try:
            # Search vector store
            results = await self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            # Format results
            messages = []
            for doc, score in results:
                # Get index from metadata
                index = doc.metadata.get("index")
                message_id = doc.metadata.get("message_id")
                
                # Try to find original message
                original_message = None
                if index is not None and 0 <= index < len(self.messages):
                    original_message = self.messages[index]
                elif message_id:
                    for msg in self.messages:
                        if msg.get("id") == message_id:
                            original_message = msg
                            break
                
                # Use document content if original message not found
                if original_message:
                    message = original_message.copy()
                    message["score"] = score
                else:
                    message = {
                        "role": doc.metadata.get("role", "unknown"),
                        "content": doc.text,
                        "metadata": doc.metadata,
                        "score": score
                    }
                
                messages.append(message)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error searching conversation history: {str(e)}")
            return []
    
    def _prune_history(self) -> None:
        """
        Prune the conversation history when it exceeds limits.
        
        This implementation keeps the most recent messages and creates a summary
        of older messages that are removed.
        """
        if not self.messages:
            return
        
        # Simple strategy: keep half of max_messages, focusing on most recent
        if len(self.messages) > self.max_messages:
            keep_count = self.max_messages // 2
            
            # Create a summary of pruned messages
            pruned = self.messages[:-keep_count]
            summary = self._create_summary(pruned)
            self.summaries.append(summary)
            
            # Update messages and token count
            self.messages = self.messages[-keep_count:]
            
            # Recalculate tokens
            self.total_tokens = sum(len(m["content"].split()) for m in self.messages)
    
    def _create_summary(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of a set of messages.
        
        In a production system, this would use an LLM to generate a summary.
        For now, this is a simple placeholder.
        
        Args:
            messages: The messages to summarize
            
        Returns:
            A summary object
        """
        # In a real implementation, you would use an LLM to generate a summary
        # For now, just create a basic summary object
        start_time = messages[0].get("timestamp") if messages and "timestamp" in messages[0] else None
        end_time = messages[-1].get("timestamp") if messages and "timestamp" in messages[-1] else None
        
        message_count = len(messages)
        role_counts = {}
        for msg in messages:
            role = msg["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Simple text summary for now
        summary_text = f"Conversation segment with {message_count} messages"
        
        return {
            "type": "summary",
            "message_count": message_count,
            "role_counts": role_counts,
            "start_time": start_time,
            "end_time": end_time,
            "summary": summary_text
        }
    
    async def clear(self) -> None:
        """Clear all messages and summaries in the conversation memory."""
        self.messages = []
        self.summaries = []
        self.total_tokens = 0
        
        # Clear vector store if enabled
        if self.use_vector_storage and self.vector_store:
            try:
                await self.vector_store.delete()
                self.indexed_message_ids.clear()
            except Exception as e:
                logger.error(f"Error clearing vector store: {str(e)}")
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the conversation history to a file.
        
        Args:
            filepath: Path to save the file to
        """
        data = {
            "messages": self.messages,
            "summaries": self.summaries,
            "total_tokens": self.total_tokens,
            "vector_storage_enabled": self.use_vector_storage,
            "indexed_messages": list(self.indexed_message_ids)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load conversation history from a file.
        
        Args:
            filepath: Path to load the file from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.messages = data.get("messages", [])
        self.summaries = data.get("summaries", [])
        self.total_tokens = data.get("total_tokens", 0)
        
        # Update indexed message tracking
        if "indexed_messages" in data:
            self.indexed_message_ids = set(data["indexed_messages"])
        
        # Index messages in vector store if needed
        if self.use_vector_storage and self.vector_store:
            for message in self.messages:
                if message.get("id") not in self.indexed_message_ids:
                    self._index_message(message)