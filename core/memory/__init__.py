"""
Memory Package

This package provides memory management for AI agents, including:
- Conversation history tracking and management
- Knowledge base and vector storage for retrieval-augmented generation
- Text processing utilities for document chunking and embedding
"""

from .conversation import ConversationMemory
from .vector_store import Document, get_vector_store
from .embeddings import get_embedder, Embedder, LLMEmbedder, LocalEmbedder
from .text_utils import TextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter, chunk_text

__all__ = [
    'ConversationMemory',
    'Document',
    'get_vector_store',
    'get_embedder',
    'Embedder',
    'LLMEmbedder',
    'LocalEmbedder',
    'TextSplitter',
    'RecursiveCharacterTextSplitter',
    'MarkdownTextSplitter',
    'chunk_text'
]