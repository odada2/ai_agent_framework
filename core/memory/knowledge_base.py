"""
Knowledge Base Management

This module provides a comprehensive knowledge base system for storing,
retrieving, and managing document-based knowledge.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from pathlib import Path

from .vector_store import VectorStore, Document, get_vector_store
from .embeddings import Embedder, get_embedder
from .retrieval import RetrievalSystem
from .text_utils import (
    TextSplitter, 
    RecursiveCharacterTextSplitter, 
    MarkdownTextSplitter, 
    HTMLTextSplitter,
    chunk_text,
    extract_metadata_from_text
)

logger = logging.getLogger(__name__)


class DocumentSource:
    """
    Represents a source of documents for the knowledge base.
    
    This class tracks metadata about a document source and provides
    methods for loading, updating, and managing documents.
    """
    
    def __init__(
        self,
        name: str,
        source_type: str,
        location: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a document source.
        
        Args:
            name: Name of the source
            source_type: Type of source (file, directory, database, api)
            location: Location of the source
            metadata: Additional metadata for the source
        """
        self.name = name
        self.source_type = source_type
        self.location = location
        self.metadata = metadata or {}
        
        # Track document stats
        self.document_count = 0
        self.last_updated = None
        self.last_sync = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "source_type": self.source_type,
            "location": self.location,
            "metadata": self.metadata,
            "document_count": self.document_count,
            "last_updated": self.last_updated,
            "last_sync": self.last_sync
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentSource':
        """Create from dictionary representation."""
        source = cls(
            name=data["name"],
            source_type=data["source_type"],
            location=data["location"],
            metadata=data.get("metadata", {})
        )
        
        source.document_count = data.get("document_count", 0)
        source.last_updated = data.get("last_updated")
        source.last_sync = data.get("last_sync")
        
        return source


class KnowledgeBase:
    """
    Comprehensive knowledge base for storing and retrieving information.
    
    The knowledge base manages documents from various sources, handles
    document processing, embedding, storage, and retrieval using vector
    databases for semantic search.
    """
    
    def __init__(
        self,
        name: str = "default_kb",
        vector_store: Optional[VectorStore] = None,
        vector_store_type: str = "chroma",
        embedder: Optional[Embedder] = None,
        embedder_type: str = "openai",
        persist_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the knowledge base.
        
        Args:
            name: Name of the knowledge base
            vector_store: Optional pre-configured vector store
            vector_store_type: Type of vector store to create if none provided
            embedder: Optional pre-configured embedder
            embedder_type: Type of embedder to create if none provided
            persist_path: Path to persist knowledge base data
            config: Additional configuration options
        """
        self.name = name
        self.config = config or {}
        self.persist_path = persist_path
        
        # Initialize embedder
        self.embedder = embedder
        if self.embedder is None:
            self.embedder = get_embedder(embedder_type)
        
        # Initialize vector store
        self.vector_store = vector_store
        if self.vector_store is None:
            if self.persist_path:
                persist_dir = os.path.join(self.persist_path, "vectors")
            else:
                persist_dir = None
                
            self.vector_store = get_vector_store(
                vector_store_type=vector_store_type,
                embedder=self.embedder,
                collection_name=name,
                persist_directory=persist_dir
            )
        
        # Create retrieval system
        self.retrieval = RetrievalSystem(
            embedder=self.embedder,
            vector_stores={"default": self.vector_store}
        )
        
        # Initialize document sources registry
        self.sources: Dict[str, DocumentSource] = {}
        
        # Initialize document registry
        self.document_index: Dict[str, Dict[str, Any]] = {}
        
        # Set default text splitter
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200)
        )
        
        # Load existing data if available
        if self.persist_path:
            self._load_state()
    
    def _get_state_path(self) -> str:
        """Get path to state file."""
        if not self.persist_path:
            raise ValueError("No persist path specified")
            
        return os.path.join(self.persist_path, "kb_state.json")
    
    def _load_state(self) -> None:
        """Load knowledge base state from disk."""
        if not self.persist_path:
            return
            
        state_path = self._get_state_path()
        
        if not os.path.exists(state_path):
            return
            
        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
                
            # Load document sources
            if "sources" in data:
                for source_data in data["sources"]:
                    source = DocumentSource.from_dict(source_data)
                    self.sources[source.name] = source
            
            # Load document index
            if "document_index" in data:
                self.document_index = data["document_index"]
                
            logger.info(f"Loaded knowledge base state with {len(self.sources)} sources and {len(self.document_index)} documents")
                
        except Exception as e:
            logger.error(f"Error loading knowledge base state: {str(e)}")
    
    def _save_state(self) -> None:
        """Save knowledge base state to disk."""
        if not self.persist_path:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_path, exist_ok=True)
            
            # Prepare data
            state_data = {
                "name": self.name,
                "sources": [source.to_dict() for source in self.sources.values()],
                "document_index": self.document_index,
                "last_updated": datetime.now().isoformat()
            }
            
            # Save to file
            with open(self._get_state_path(), 'w') as f:
                json.dump(state_data, f, indent=2)
                
            logger.debug(f"Saved knowledge base state to {self._get_state_path()}")
                
        except Exception as e:
            logger.error(f"Error saving knowledge base state: {str(e)}")
    
    async def query(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the knowledge base for relevant information.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional query parameters
            
        Returns:
            Query results
        """
        return await self.retrieval.query(
            query=query,
            sources=["default"],
            k=k,
            filter=filter,
            **kwargs
        )
    
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        source_name: Optional[str] = None,
        chunk: bool = True,
        content_type: Optional[str] = None
    ) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID
            source_name: Name of the source
            chunk: Whether to chunk the document
            content_type: Content type (markdown, html, text)
            
        Returns:
            Document ID
        """
        # Generate doc ID if not provided
        if doc_id is None:
            import uuid
            doc_id = str(uuid.uuid4())
        
        # Set up metadata
        metadata = metadata or {}
        
        # Extract metadata from content if not provided
        if not metadata:
            extracted_metadata = extract_metadata_from_text(content)
            metadata.update(extracted_metadata)
        
        # Add source information to metadata
        if source_name:
            metadata["source"] = source_name
            
            # Update source document count
            if source_name in self.sources:
                self.sources[source_name].document_count += 1
                self.sources[source_name].last_updated = datetime.now().isoformat()
        
        # Add document to index
        self.document_index[doc_id] = {
            "metadata": metadata,
            "added_at": datetime.now().isoformat(),
            "content_hash": hash(content),
            "content_type": content_type or metadata.get("content_type", "text")
        }
        
        # Process document based on content type
        if chunk:
            # Select appropriate splitter
            if content_type == "markdown" or metadata.get("content_type") == "markdown":
                splitter = MarkdownTextSplitter(
                    chunk_size=self.config.get("chunk_size", 1000),
                    chunk_overlap=self.config.get("chunk_overlap", 200)
                )
            elif content_type == "html" or metadata.get("content_type") == "html":
                splitter = HTMLTextSplitter(
                    chunk_size=self.config.get("chunk_size", 1000),
                    chunk_overlap=self.config.get("chunk_overlap", 200)
                )
            else:
                splitter = self.default_splitter
                
            # Split content
            chunks = splitter.split_text(content)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "document_id": doc_id
                })
                
                doc = Document(
                    text=chunk,
                    metadata=chunk_metadata,
                    id=f"{doc_id}_chunk_{i}"
                )
                documents.append(doc)
            
            # Add to vector store
            await self.vector_store.add_documents(documents)
            
            # Update document index
            self.document_index[doc_id]["chunks"] = len(chunks)
            
        else:
            # Add full document
            doc = Document(
                text=content,
                metadata=metadata,
                id=doc_id
            )
            
            await self.vector_store.add_documents([doc])
            
            # Update document index
            self.document_index[doc_id]["chunks"] = 1
        
        # Save state
        self._save_state()
        
        return doc_id
    
    async def add_documents_from_directory(
        self,
        directory: str,
        source_name: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        recursive: bool = True,
        chunk: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add documents from a directory.
        
        Args:
            directory: Directory path
            source_name: Source name
            file_types: List of file extensions to include
            recursive: Whether to search subdirectories
            chunk: Whether to chunk documents
            metadata: Additional metadata to add to all documents
            
        Returns:
            Dictionary with results
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
            
        # Create source if not exists
        if source_name:
            if source_name not in self.sources:
                source = DocumentSource(
                    name=source_name,
                    source_type="directory",
                    location=directory,
                    metadata={"file_types": file_types}
                )
                self.sources[source_name] = source
            
            source = self.sources[source_name]
            source.last_sync = datetime.now().isoformat()
        
        # Set up default file types
        if file_types is None:
            file_types = [".txt", ".md", ".markdown", ".rst", ".html", ".htm"]
        
        # Walk directory
        results = {
            "added": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }
        
        for root, _, files in os.walk(directory):
            # Skip further directories if not recursive
            if not recursive and root != directory:
                continue
                
            for file in files:
                # Check file type
                if not any(file.endswith(ext) for ext in file_types):
                    results["skipped"] += 1
                    continue
                    
                file_path = os.path.join(root, file)
                
                try:
                    # Read file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Determine content type
                    if file.endswith((".md", ".markdown")):
                        content_type = "markdown"
                    elif file.endswith((".html", ".htm")):
                        content_type = "html"
                    else:
                        content_type = "text"
                    
                    # Create document metadata
                    doc_metadata = metadata.copy() if metadata else {}
                    doc_metadata.update({
                        "filename": file,
                        "filepath": file_path,
                        "content_type": content_type
                    })
                    
                    # Generate ID from path
                    relative_path = os.path.relpath(file_path, directory)
                    doc_id = f"{source_name or 'dir'}_{relative_path}"
                    
                    # Add document
                    await self.add_document(
                        content=content,
                        metadata=doc_metadata,
                        doc_id=doc_id,
                        source_name=source_name,
                        chunk=chunk,
                        content_type=content_type
                    )
                    
                    results["added"] += 1
                    
                except Exception as e:
                    logger.error(f"Error adding document {file_path}: {str(e)}")
                    results["failed"] += 1
                    results["errors"].append({
                        "file": file_path,
                        "error": str(e)
                    })
        
        # Update source
        if source_name and source_name in self.sources:
            self.sources[source_name].document_count = results["added"]
            self.sources[source_name].last_updated = datetime.now().isoformat()
            self._save_state()
        
        return results
    
    async def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the knowledge base.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if document was removed
        """
        # Check if document exists
        if doc_id not in self.document_index:
            return False
        
        # Get chunk count
        chunk_count = self.document_index[doc_id].get("chunks", 1)
        
        # Remove chunks from vector store
        if chunk_count > 1:
            # Remove all chunks
            for i in range(chunk_count):
                chunk_id = f"{doc_id}_chunk_{i}"
                await self.vector_store.delete_document(chunk_id)
        else:
            # Remove single document
            await self.vector_store.delete_document(doc_id)
        
        # Remove from document index
        source_name = self.document_index[doc_id].get("metadata", {}).get("source")
        del self.document_index[doc_id]
        
        # Update source document count
        if source_name and source_name in self.sources:
            self.sources[source_name].document_count -= 1
            self.sources[source_name].last_updated = datetime.now().isoformat()
        
        # Save state
        self._save_state()
        
        return True
    
    async def clear(self, source_name: Optional[str] = None) -> bool:
        """
        Clear the knowledge base.
        
        Args:
            source_name: Optional name of specific source to clear
            
        Returns:
            True if successful
        """
        try:
            if source_name:
                # Clear specific source
                if source_name not in self.sources:
                    return False
                
                # Build filter
                filter = {"source": source_name}
                
                # Remove from vector store
                await self.vector_store.delete(filter)
                
                # Remove from document index
                to_remove = []
                for doc_id, doc_info in self.document_index.items():
                    doc_source = doc_info.get("metadata", {}).get("source")
                    if doc_source == source_name:
                        to_remove.append(doc_id)
                
                for doc_id in to_remove:
                    del self.document_index[doc_id]
                
                # Update source
                self.sources[source_name].document_count = 0
                self.sources[source_name].last_updated = datetime.now().isoformat()
                
            else:
                # Clear entire knowledge base
                await self.vector_store.delete()
                self.document_index = {}
                
                # Reset source document counts
                for source in self.sources.values():
                    source.document_count = 0
                    source.last_updated = datetime.now().isoformat()
            
            # Save state
            self._save_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            return False
    
    async def update_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: Optional[bool] = None
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            content: New content
            metadata: New metadata (None = keep existing)
            chunk: Whether to chunk (None = use original setting)
            
        Returns:
            True if document was updated
        """
        # Check if document exists
        if doc_id not in self.document_index:
            return False
        
        # Get existing information
        doc_info = self.document_index[doc_id]
        existing_metadata = doc_info.get("metadata", {})
        
        # Check if content has changed
        content_hash = hash(content)
        if content_hash == doc_info.get("content_hash") and metadata is None:
            # No changes needed
            return True
        
        # Remove existing document
        await self.remove_document(doc_id)
        
        # Merge metadata
        if metadata is None:
            metadata = existing_metadata
        else:
            # Keep some original metadata fields
            for key in ["source", "filename", "filepath"]:
                if key in existing_metadata and key not in metadata:
                    metadata[key] = existing_metadata[key]
        
        # Determine chunking
        if chunk is None:
            chunk = doc_info.get("chunks", 1) > 1
        
        # Add updated document
        await self.add_document(
            content=content,
            metadata=metadata,
            doc_id=doc_id,
            source_name=metadata.get("source"),
            chunk=chunk,
            content_type=metadata.get("content_type")
        )
        
        return True
    
    async def sync_directory(
        self,
        directory: str,
        source_name: str,
        file_types: Optional[List[str]] = None,
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronize a directory with the knowledge base.
        
        This updates existing documents, adds new ones, and removes deleted ones.
        
        Args:
            directory: Directory path
            source_name: Source name
            file_types: List of file extensions to include
            recursive: Whether to search subdirectories
            metadata: Additional metadata to add to all documents
            
        Returns:
            Dictionary with results
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        # Create source if not exists
        if source_name not in self.sources:
            source = DocumentSource(
                name=source_name,
                source_type="directory",
                location=directory,
                metadata={"file_types": file_types}
            )
            self.sources[source_name] = source
        
        # Set up default file types
        if file_types is None:
            file_types = [".txt", ".md", ".markdown", ".rst", ".html", ".htm"]
        
        # Track results
        results = {
            "added": 0,
            "updated": 0,
            "removed": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }
        
        # Find all files in directory
        current_files = set()
        for root, _, files in os.walk(directory):
            # Skip further directories if not recursive
            if not recursive and root != directory:
                continue
                
            for file in files:
                # Check file type
                if not any(file.endswith(ext) for ext in file_types):
                    continue
                    
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                doc_id = f"{source_name}_{relative_path}"
                
                current_files.add(doc_id)
        
        # Find existing documents for this source
        existing_files = set()
        for doc_id, doc_info in self.document_index.items():
            doc_source = doc_info.get("metadata", {}).get("source")
            if doc_source == source_name:
                existing_files.add(doc_id)
        
        # Find files to add, update, and remove
        files_to_add = current_files - existing_files
        files_to_update = current_files & existing_files
        files_to_remove = existing_files - current_files
        
        # Remove deleted files
        for doc_id in files_to_remove:
            try:
                await self.remove_document(doc_id)
                results["removed"] += 1
            except Exception as e:
                logger.error(f"Error removing document {doc_id}: {str(e)}")
                results["failed"] += 1
                results["errors"].append({
                    "doc_id": doc_id,
                    "error": str(e)
                })
        
        # Add new files
        for doc_id in files_to_add:
            try:
                # Extract relative path
                relative_path = doc_id[len(source_name) + 1:]
                file_path = os.path.join(directory, relative_path)
                
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Determine content type
                if file_path.endswith((".md", ".markdown")):
                    content_type = "markdown"
                elif file_path.endswith((".html", ".htm")):
                    content_type = "html"
                else:
                    content_type = "text"
                
                # Create document metadata
                doc_metadata = metadata.copy() if metadata else {}
                doc_metadata.update({
                    "filename": os.path.basename(file_path),
                    "filepath": file_path,
                    "content_type": content_type
                })
                
                # Add document
                await self.add_document(
                    content=content,
                    metadata=doc_metadata,
                    doc_id=doc_id,
                    source_name=source_name,
                    chunk=True,
                    content_type=content_type
                )
                
                results["added"] += 1
                
            except Exception as e:
                logger.error(f"Error adding document {doc_id}: {str(e)}")
                results["failed"] += 1
                results["errors"].append({
                    "doc_id": doc_id,
                    "error": str(e)
                })
        
        # Update existing files
        for doc_id in files_to_update:
            try:
                # Extract relative path
                relative_path = doc_id[len(source_name) + 1:]
                file_path = os.path.join(directory, relative_path)
                
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get existing document info
                doc_info = self.document_index[doc_id]
                content_hash = hash(content)
                
                # Check if content has changed
                if content_hash == doc_info.get("content_hash"):
                    results["skipped"] += 1
                    continue
                
                # Update document
                await self.update_document(
                    doc_id=doc_id,
                    content=content,
                    chunk=None  # Use original chunking setting
                )
                
                results["updated"] += 1
                
            except Exception as e:
                logger.error(f"Error updating document {doc_id}: {str(e)}")
                results["failed"] += 1
                results["errors"].append({
                    "doc_id": doc_id,
                    "error": str(e)
                })
        
        # Update source
        self.sources[source_name].document_count = len(current_files)
        self.sources[source_name].last_updated = datetime.now().isoformat()
        self.sources[source_name].last_sync = datetime.now().isoformat()
        
        # Save state
        self._save_state()
        
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        # Get vector store stats
        vector_stats = self.vector_store.get_collection_stats()
        
        # Count documents by source
        source_counts = {}
        for source_name, source in self.sources.items():
            source_counts[source_name] = source.document_count
        
        # Build stats
        stats = {
            "name": self.name,
            "documents": len(self.document_index),
            "vector_count": vector_stats.get("count", 0),
            "sources": {
                "count": len(self.sources),
                "details": source_counts
            }
        }
        
        return stats