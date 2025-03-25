"""
Conversation Memory

This module provides functionality for storing and managing conversation history.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history for an agent.
    
    This class handles storing, retrieving, and managing messages in a conversation,
    with support for summarization and selective retention.
    """
    
    def __init__(
        self,
        max_messages: int = 100,
        max_tokens: Optional[int] = None,
        include_timestamps: bool = True
    ):
        """
        Initialize the conversation memory.
        
        Args:
            max_messages: Maximum number of messages to store
            max_tokens: Optional maximum total tokens to store (requires token counting)
            include_timestamps: Whether to include timestamps with messages
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.include_timestamps = include_timestamps
        
        self.messages: List[Dict[str, Any]] = []
        self.summaries: List[Dict[str, Any]] = []
        self.total_tokens: int = 0
    
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
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        if self.include_timestamps:
            message["timestamp"] = time.time()
        
        # Calculate tokens if max_tokens is set
        # In production, you would use a proper tokenizer here
        approximate_tokens = len(content.split())
        
        # Check if we need to make room by pruning old messages
        if len(self.messages) >= self.max_messages or (
            self.max_tokens and self.total_tokens + approximate_tokens > self.max_tokens
        ):
            self._prune_history()
        
        self.messages.append(message)
        self.total_tokens += approximate_tokens
    
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
            for message in filtered_messages:
                if "metadata" in message:
                    message = message.copy()
                    del message["metadata"]
        
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
    
    def clear(self) -> None:
        """Clear all messages and summaries in the conversation memory."""
        self.messages = []
        self.summaries = []
        self.total_tokens = 0
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the conversation history to a file.
        
        Args:
            filepath: Path to save the file to
        """
        data = {
            "messages": self.messages,
            "summaries": self.summaries,
            "total_tokens": self.total_tokens
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