"""
Text Embedding Utilities

This module provides utilities for generating and working with text embeddings,
including integrations with various embedding models and services.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """
    Abstract base class for text embedding models.
    
    This class defines the interface that all embedding model
    implementations must follow to ensure consistent behavior.
    """
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding model usage."""
        pass


class LLMEmbedder(Embedder):
    """
    Embedding implementation that uses LLM APIs for text embeddings.
    
    This implementation supports Anthropic, OpenAI, and other LLM
    embedding endpoints through a unified interface.
    """
    
    def __init__(
        self,
        provider: str = "claude",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 8,
        dimensions: Optional[int] = None,
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize the LLM embedder.
        
        Args:
            provider: LLM provider ('claude', 'openai')
            model_name: Name of the specific embedding model
            api_key: API key for the provider
            batch_size: Maximum number of texts to embed in a single request
            dimensions: Embedding dimensions (if supported by provider)
            timeout: Timeout for API requests in seconds
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider.lower()
        self.timeout = timeout
        self.batch_size = batch_size
        self.dimensions = dimensions
        
        # Set up provider-specific configuration
        if self.provider == "claude":
            from anthropic import Anthropic
            self.model_name = model_name or "claude-3-sonnet-20240229"
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            
            if not self.api_key:
                raise ValueError("Anthropic API key required")
                
            self.client = Anthropic(api_key=self.api_key)
            self._embedding_dimension = 1536  # Claude default
            
        elif self.provider == "openai":
            import openai
            self.model_name = model_name or "text-embedding-3-small"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            
            if not self.api_key:
                raise ValueError("OpenAI API key required")
                
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Model-specific dimensions
            dimension_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            self._embedding_dimension = dimension_map.get(self.model_name, 1536)
            
            # Override with specified dimensions if supported
            if self.dimensions and self.model_name.startswith("text-embedding-3"):
                self._embedding_dimension = self.dimensions
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Initialize stats
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_texts": 0,
            "errors": 0,
            "last_usage_time": None
        }
        
        logger.info(f"Initialized {self.provider} embedder with model {self.model_name}")
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Single queries are just wrapped calls to embed_documents
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        self.stats["total_texts"] += len(texts)
        self.stats["total_requests"] += 1
        self.stats["last_usage_time"] = time.time()
        
        try:
            # Process in batches
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                
                if self.provider == "claude":
                    # Anthropic implementation
                    embeddings = await self._embed_with_anthropic(batch)
                    
                elif self.provider == "openai":
                    # OpenAI implementation
                    embeddings = await self._embed_with_openai(batch)
                
                all_embeddings.extend(embeddings)
            
            return all_embeddings
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    async def _embed_with_anthropic(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Anthropic's API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Note: This assumes a future Anthropic embeddings API
        # Currently, Anthropic doesn't offer a public embeddings API
        # but this implementation is future-proofed for when they do
        
        # Temporary implementation: Use OpenAI instead
        logger.warning("Anthropic embeddings not available, falling back to OpenAI")
        return await self._embed_with_openai(texts)
    
    async def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI's API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        loop = asyncio.get_event_loop()
        
        # Create the request
        params = {
            "model": self.model_name,
            "input": texts
        }
        
        # Add dimensions if specified and supported
        if self.dimensions and self.model_name.startswith("text-embedding-3"):
            params["dimensions"] = self.dimensions
        
        try:
            # Perform the request
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embeddings.create(**params)
            )
            
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            
            # Update token usage stats
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                self.stats["total_tokens"] += response.usage.total_tokens
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {str(e)}")
            self.stats["errors"] += 1
            raise
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self._embedding_dimension
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding model usage."""
        return self.stats.copy()


class LocalEmbedder(Embedder):
    """
    Embedding implementation that uses local models for text embeddings.
    
    This implementation supports various open-source embedding models 
    that can be run locally without API calls.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        use_gpu: bool = False,
        **kwargs
    ):
        """
        Initialize the local embedder.
        
        Args:
            model_name: Name of the model to use
            cache_dir: Directory to cache models
            use_gpu: Whether to use GPU for inference
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        self.model = None
        self._embedding_dimension = 384  # Default
        
        # Initialize stats
        self.stats = {
            "total_requests": 0,
            "total_texts": 0,
            "errors": 0,
            "last_usage_time": None
        }
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            # Conditionally import to avoid hard dependency
            from sentence_transformers import SentenceTransformer
            
            # Load the model
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name,
                cache_folder=self.cache_dir
            )
            
            # Set device if GPU is requested
            if self.use_gpu:
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to(torch.device("cuda"))
                else:
                    logger.warning("GPU requested but not available, using CPU instead")
            
            # Get embedding dimension
            self._embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Loaded local embedding model: {self.model_name} ({self._embedding_dimension} dimensions)")
            
        except ImportError:
            logger.error("sentence-transformers package not installed. Run 'pip install sentence-transformers'")
            raise ImportError("sentence-transformers package required for LocalEmbedder")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self._load_model()
        
        self.stats["total_texts"] += len(texts)
        self.stats["total_requests"] += 1
        self.stats["last_usage_time"] = time.time()
        
        loop = asyncio.get_event_loop()
        
        try:
            # Run embedding in executor to avoid blocking
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                ).tolist()
            )
            
            return embeddings
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self._embedding_dimension
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding model usage."""
        return self.stats.copy()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Similarity score between -1 and 1
    """
    if not a or not b:
        return 0.0
        
    # Convert to numpy arrays for efficient computation
    a_np = np.array(a)
    b_np = np.array(b)
    
    # Calculate cosine similarity
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a_np, b_np) / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Distance value (lower is more similar)
    """
    if not a or not b:
        return float('inf')
        
    # Convert to numpy arrays for efficient computation
    a_np = np.array(a)
    b_np = np.array(b)
    
    # Calculate Euclidean distance
    return float(np.linalg.norm(a_np - b_np))


def dot_product(a: List[float], b: List[float]) -> float:
    """
    Calculate dot product between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Dot product value
    """
    if not a or not b:
        return 0.0
        
    # Convert to numpy arrays for efficient computation
    a_np = np.array(a)
    b_np = np.array(b)
    
    # Calculate dot product
    return float(np.dot(a_np, b_np))


def get_embedder(embedder_type: str = "openai", **kwargs) -> Embedder:
    """
    Factory function to get an embedder instance.
    
    Args:
        embedder_type: Type of embedder ('openai', 'local', etc.)
        **kwargs: Additional parameters for the embedder
        
    Returns:
        Configured Embedder instance
    """
    embedder_type = embedder_type.lower()
    
    if embedder_type in ["openai", "claude"]:
        return LLMEmbedder(provider=embedder_type, **kwargs)
    elif embedder_type == "local":
        return LocalEmbedder(**kwargs)
    else:
        raise ValueError(f"Unsupported embedder type: {embedder_type}")