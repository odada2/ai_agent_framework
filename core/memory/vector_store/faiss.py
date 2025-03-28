"""
FAISS Vector Database Integration

This module provides integration with the FAISS vector database,
a library for efficient similarity search and dense vector clustering.
Uses IndexIDMap2 for efficient additions and deletions.
"""

import logging
import os
import pickle
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union, Set, cast

import numpy as np

# Try importing FAISS and handle error
try:
    import faiss
except ImportError:
    raise ImportError(
        "Could not import faiss python package. "
        "Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
    )

from .base import VectorStore, Document
# Assuming Embedder type hint is available, adjust import if needed
# from ..embeddings import Embedder # Example import path

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    Vector store implementation backed by FAISS using IndexIDMap2.

    FAISS is a library for efficient similarity search and
    clustering of dense vectors developed by Facebook AI Research.
    This implementation uses IndexIDMap2 to allow efficient addition
    and deletion of vectors by ID.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedder: Optional[Any] = None, # Use actual Embedder type hint if available
        dimension: Optional[int] = None,
        index_factory: str = "Flat", # FAISS index factory string (e.g., "Flat", "IVF4096,Flat")
        metric: int = faiss.METRIC_L2, # faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        normalize_l2: bool = False, # Normalize vectors for cosine similarity via inner product
        **kwargs # Catch-all for future args
    ):
        """
        Initialize the FAISS vector store.

        Args:
            index_path: Path to load/save the FAISS index and associated data.
            embedder: Embedding object with embed_query, embed_documents methods
                      and embedding_dimension property.
            dimension: Dimension of the embeddings (required if embedder has no dimension).
            index_factory: FAISS index factory string for the underlying index. Defaults to "Flat".
            metric: Distance metric (faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT).
                    Use METRIC_INNER_PRODUCT with normalize_l2=True for cosine similarity.
            normalize_l2: Normalize vectors before adding/searching. Required for cosine
                          similarity when using METRIC_INNER_PRODUCT.
            **kwargs: Additional unused arguments.
        """
        self.faiss = faiss
        self.index_path = index_path
        self.embedder = embedder
        self.normalize_l2 = normalize_l2

        # Core data structures
        self.docstore: Dict[str, Document] = {} # Stores Document objects keyed by string doc_id
        self._doc_id_to_faiss_id: Dict[str, int] = {} # Map string doc_id -> int faiss_id
        self._faiss_id_to_doc_id: Dict[int, str] = {} # Map int faiss_id -> string doc_id
        self._next_faiss_id: int = 0 # Counter for generating unique int faiss_ids
        self.index: Optional[faiss.Index] = None # The FAISS index (will be IndexIDMap2)
        self.dimension: Optional[int] = dimension
        self.metric = metric

        # Determine dimension
        if self.dimension is None:
            if self.embedder and hasattr(self.embedder, 'embedding_dimension'):
                 self.dimension = self.embedder.embedding_dimension
            else:
                 # Try loading first if path exists, maybe dimension is stored
                 pass # Dimension will be set during load or index creation

        # Initialize or load index
        loaded_from_file = False
        if index_path and os.path.exists(f"{index_path}.index"):
            try:
                self._load_index()
                loaded_from_file = True
            except Exception as e:
                 logger.error(f"Failed to load index from {index_path}, will create new one. Error: {e}")

        if not loaded_from_file:
            if self.dimension is None:
                raise ValueError("Cannot create new index: embedding dimension must be provided "
                                 "either via 'dimension' argument or 'embedder.embedding_dimension'.")

            # Create a new index
            logger.info(f"Creating new FAISS index with factory='{index_factory}', metric={metric}, dimension={self.dimension}")
            # Create the underlying index (e.g., IndexFlatL2)
            base_index = faiss.index_factory(self.dimension, index_factory, metric)
            # Wrap with IndexIDMap2 for ID management (supports 64-bit IDs)
            self.index = faiss.IndexIDMap2(base_index)
            # Reset mappings for new index
            self.docstore = {}
            self._doc_id_to_faiss_id = {}
            self._faiss_id_to_doc_id = {}
            self._next_faiss_id = 0

        logger.info(f"FAISS vector store initialized. Documents: {len(self.docstore)}, Next FAISS ID: {self._next_faiss_id}")


    def _get_persist_files(self) -> Tuple[str, str]:
        """Get the paths for index and metadata files."""
        if not self.index_path:
            raise ValueError("index_path must be set for persistence.")
        index_file = f"{self.index_path}.index"
        metadata_file = f"{self.index_path}.metadata.pkl"
        return index_file, metadata_file

    def _load_index(self):
        """Load the FAISS index and associated metadata from disk."""
        index_file, metadata_file = self._get_persist_files()
        logger.info(f"Loading FAISS index from {index_file} and metadata from {metadata_file}")

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(f"FAISS index or metadata file not found at {self.index_path}")

        try:
            # Load FAISS index
            self.index = self.faiss.read_index(index_file)
            self.dimension = self.index.d # Get dimension from loaded index

            # Load metadata store
            with open(metadata_file, "rb") as f:
                saved_data = pickle.load(f)
                self.docstore = saved_data["docstore"]
                self._doc_id_to_faiss_id = saved_data["doc_id_to_faiss_id"]
                self._faiss_id_to_doc_id = saved_data["faiss_id_to_doc_id"]
                self._next_faiss_id = saved_data["next_faiss_id"]
                # Load metric type if saved, otherwise assume default L2
                self.metric = saved_data.get("metric", faiss.METRIC_L2)
                self.normalize_l2 = saved_data.get("normalize_l2", False)

            logger.info(f"Loaded FAISS index. Documents: {len(self.docstore)}, Next FAISS ID: {self._next_faiss_id}, Metric: {self.metric}")

        except Exception as e:
            logger.error(f"Error loading FAISS index and metadata: {e}", exc_info=True)
            raise IOError(f"Failed to load FAISS data from {self.index_path}") from e

    def _save_index(self):
        """Save the FAISS index and associated metadata to disk."""
        if not self.index_path:
             logger.debug("No index_path specified, skipping save.")
             return
        if self.index is None:
             logger.warning("Index is not initialized, cannot save.")
             return

        index_file, metadata_file = self._get_persist_files()
        logger.debug(f"Saving FAISS index to {index_file} and metadata to {metadata_file}")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(index_file), exist_ok=True)

            # Save FAISS index
            self.faiss.write_index(cast(faiss.Index, self.index), index_file)

            # Save metadata store
            metadata_to_save = {
                "docstore": self.docstore,
                "doc_id_to_faiss_id": self._doc_id_to_faiss_id,
                "faiss_id_to_doc_id": self._faiss_id_to_doc_id,
                "next_faiss_id": self._next_faiss_id,
                "metric": self.metric,
                "normalize_l2": self.normalize_l2,
            }
            with open(metadata_file, "wb") as f:
                pickle.dump(metadata_to_save, f)

            logger.info(f"Saved FAISS index ({len(self.docstore)} docs) and metadata to {self.index_path}")

        except Exception as e:
            logger.error(f"Error saving FAISS index and metadata: {e}", exc_info=True)
            # Decide if failure to save should raise an error or just warn
            # raise IOError(f"Failed to save FAISS data to {self.index_path}") from e


    async def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any # Allow extra kwargs
    ) -> List[str]:
        """
        Add documents to the FAISS vector store.

        Args:
            documents: List of Document objects to add.
            embeddings: Optional pre-computed embeddings. If None, computes using embedder.
            **kwargs: Unused arguments.

        Returns:
            List of document IDs added/updated.
        """
        if not documents:
            return []
        if self.index is None:
             raise RuntimeError("FAISS index is not initialized.")

        # Compute embeddings if not provided
        if embeddings is None:
            if self.embedder is None:
                raise ValueError("Embedder must be set if embeddings are not provided.")
            texts = [doc.text for doc in documents]
            embeddings = await self.embedder.embed_documents(texts)
            if len(embeddings) != len(documents):
                 raise ValueError("Number of embeddings generated does not match number of documents.")

        # Prepare data for FAISS
        vectors = np.array(embeddings).astype("float32")
        if self.normalize_l2:
             faiss.normalize_L2(vectors)

        added_doc_ids = []
        faiss_ids_to_add = []
        vectors_to_add = []

        for i, doc in enumerate(documents):
            # Assign string doc_id if missing
            doc_id = doc.id or str(uuid.uuid4())
            doc.id = doc_id # Ensure document object has the ID

            # Check if document already exists (for potential update)
            if doc_id in self._doc_id_to_faiss_id:
                logger.warning(f"Document ID '{doc_id}' already exists. Skipping add. Use update_document instead.")
                continue # Or implement update logic here if desired

            # Assign a unique integer FAISS ID
            faiss_id = self._next_faiss_id
            self._next_faiss_id += 1

            # Store mappings and document
            self.docstore[doc_id] = doc
            self._doc_id_to_faiss_id[doc_id] = faiss_id
            self._faiss_id_to_doc_id[faiss_id] = doc_id

            faiss_ids_to_add.append(faiss_id)
            vectors_to_add.append(vectors[i])
            added_doc_ids.append(doc_id)

        # Add vectors with IDs to FAISS index
        if vectors_to_add:
            faiss_id_array = np.array(faiss_ids_to_add).astype('int64')
            vector_array = np.array(vectors_to_add).astype('float32')
            self.index.add_with_ids(vector_array, faiss_id_array)
            logger.debug(f"Added {len(added_doc_ids)} vectors to FAISS index.")

        # Persist changes
        if added_doc_ids:
            self._save_index()

        return added_doc_ids

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Add text strings to the vector store.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dicts for each text.
            ids: Optional list of string IDs for each text.
            embeddings: Optional pre-computed embeddings.
            **kwargs: Unused arguments.

        Returns:
            List of document IDs added/updated.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if len(texts) != len(metadatas) or len(texts) != len(ids):
            raise ValueError("texts, metadatas, and ids must have the same length.")

        documents = [
            Document(text=text, metadata=metadata, id=doc_id)
            for text, metadata, doc_id in zip(texts, metadatas, ids)
        ]

        return await self.add_documents(documents, embeddings=embeddings, **kwargs)

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Search for documents similar to the query string.

        Args:
            query: Query string to search for.
            k: Number of results to return *after* filtering.
            filter: Optional metadata filter.
            **kwargs: Additional arguments for embedding or search.

        Returns:
            List of Document objects most similar to the query that match the filter.
        """
        results_with_scores = await self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
        return [doc for doc, _score in results_with_scores]

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20, # Number of results to fetch initially for filtering
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query string with similarity scores.

        Args:
            query: Query string to search for.
            k: Number of results to return *after* filtering.
            filter: Optional metadata filter.
            fetch_k: Number of candidates to fetch from FAISS before filtering.
                     Increase if filtering is strict and you need more candidates.
            **kwargs: Additional arguments for embedding or search.

        Returns:
            List of (Document, score) tuples most similar to the query that match the filter.
        """
        if self.embedder is None:
            raise ValueError("Embedder must be set for similarity search.")

        query_embedding = await self.embedder.embed_query(query)

        results = await self.similarity_search_by_vector_with_score(
            embedding=query_embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs
        )
        return results

    async def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any
    ) -> List[Document]:
        """
        Search for documents similar to the embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of results to return *after* filtering.
            filter: Optional metadata filter.
            fetch_k: Number of candidates to fetch from FAISS before filtering.
            **kwargs: Additional search arguments.

        Returns:
            List of Document objects most similar to the embedding that match the filter.
        """
        results_with_scores = await self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs
        )
        return [doc for doc, _score in results_with_scores]

    async def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20, # Number of results to fetch initially for filtering
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to embedding, with filtering applied after search.

        Args:
            embedding: Embedding vector.
            k: Number of results to return *after* filtering.
            filter: Metadata filter dictionary.
            fetch_k: Number of candidates to fetch initially before filtering.
            **kwargs: Unused arguments.

        Returns:
            List of (Document, score) tuples.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.debug("Index is empty, returning no results.")
            return []
        if k <= 0 or fetch_k <= 0:
             return []

        # Fetch more results initially if filtering is applied
        num_to_fetch = max(k, fetch_k) if filter else k
        num_to_fetch = min(num_to_fetch, self.index.ntotal) # Cannot fetch more than exist

        query_vector = np.array([embedding]).astype("float32")
        if self.normalize_l2:
            faiss.normalize_L2(query_vector)

        logger.debug(f"Searching FAISS index for {num_to_fetch} neighbors...")
        # Search returns distances and the corresponding int faiss_ids
        distances, faiss_ids = self.index.search(query_vector, num_to_fetch)

        matched_results: List[Tuple[Document, float]] = []
        faiss_ids_list = faiss_ids[0]
        distances_list = distances[0]

        for i, faiss_id_int in enumerate(faiss_ids_list):
            if faiss_id_int == -1: # FAISS uses -1 for invalid/empty indices
                continue

            # Map FAISS int ID back to string doc_id
            doc_id = self._faiss_id_to_doc_id.get(faiss_id_int)
            if not doc_id:
                logger.warning(f"FAISS ID {faiss_id_int} not found in mapping.")
                continue

            doc = self.docstore.get(doc_id)
            if not doc:
                logger.warning(f"Document '{doc_id}' (FAISS ID {faiss_id_int}) not found in docstore.")
                continue

            # Apply filter if provided
            if filter and not self._matches_filter(doc.metadata, filter):
                continue

            # Calculate score based on metric
            distance = distances_list[i]
            if self.metric == faiss.METRIC_INNER_PRODUCT:
                # Higher inner product is better (more similar for normalized vectors)
                score = float(distance)
            else: # Default L2 distance
                # Convert L2 distance to similarity score (0-1), closer to 1 is better
                # Simple inversion: score = 1.0 / (1.0 + distance)
                # Exponential decay: score = np.exp(-distance)
                # Choose one method consistently
                score = 1.0 / (1.0 + float(distance))

            matched_results.append((doc, score))

            # Stop if we have collected k filtered results
            if len(matched_results) >= k:
                break

        # Sort final results by score (descending - higher is better)
        matched_results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Found {len(matched_results)} results after filtering.")
        return matched_results

    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by its string ID."""
        # No lock needed for read-only dict access if updates are locked
        return self.docstore.get(doc_id)

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by its string ID using IndexIDMap2.

        Args:
            doc_id: String document ID to delete.

        Returns:
            True if document was found and deletion was attempted, False otherwise.
        """
        if self.index is None: return False

        faiss_id_to_remove = self._doc_id_to_faiss_id.get(doc_id)

        if faiss_id_to_remove is not None:
            logger.debug(f"Attempting to remove document '{doc_id}' (FAISS ID: {faiss_id_to_remove})")
            # Prepare ID selector for removal
            id_selector = self.faiss.IDSelectorArray([faiss_id_to_remove])
            try:
                 # Remove from the FAISS index
                 removed_count = self.index.remove_ids(id_selector) # remove_ids expects a selector
                 if removed_count == 0:
                      logger.warning(f"FAISS ID {faiss_id_to_remove} for doc '{doc_id}' not found in index for removal.")
                 else:
                      logger.debug(f"Removed {removed_count} vector(s) for FAISS ID {faiss_id_to_remove}.")

                 # Remove from mappings and docstore
                 if doc_id in self.docstore: del self.docstore[doc_id]
                 if doc_id in self._doc_id_to_faiss_id: del self._doc_id_to_faiss_id[doc_id]
                 if faiss_id_to_remove in self._faiss_id_to_doc_id: del self._faiss_id_to_doc_id[faiss_id_to_remove]

                 # Persist changes
                 self._save_index()
                 return True

            except Exception as e:
                 # Catch potential errors during remove_ids
                 logger.error(f"Error removing FAISS ID {faiss_id_to_remove} for doc '{doc_id}': {e}", exc_info=True)
                 return False # Indicate failure
        else:
            logger.warning(f"Document ID '{doc_id}' not found in mappings for deletion.")
            return False # Document ID not found in our mappings

    async def delete(self, filter: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bool:
        """
        Delete documents matching the filter criteria or all documents if filter is None.

        Args:
            filter: Metadata filter to match documents for deletion.
            **kwargs: Unused arguments.

        Returns:
            True if deletion process completed (even if some IDs failed), False on major error.
        """
        if self.index is None: return False

        faiss_ids_to_remove = []
        doc_ids_to_remove = []

        if filter is None:
            # Delete everything
            logger.info("Deleting all documents from FAISS store.")
            doc_ids_to_remove = list(self.docstore.keys())
            faiss_ids_to_remove = list(self._faiss_id_to_doc_id.keys())
        else:
            # Find documents matching filter
            for doc_id, doc in self.docstore.items():
                if self._matches_filter(doc.metadata, filter):
                    doc_ids_to_remove.append(doc_id)
                    faiss_id = self._doc_id_to_faiss_id.get(doc_id)
                    if faiss_id is not None:
                        faiss_ids_to_remove.append(faiss_id)
                    else:
                         logger.warning(f"Doc ID '{doc_id}' found matching filter but missing from FAISS ID map.")

        if not faiss_ids_to_remove:
            logger.info("No documents matched filter for deletion.")
            return True # Nothing to delete is considered success

        logger.info(f"Attempting to remove {len(faiss_ids_to_remove)} documents matching filter.")
        # Prepare ID selector for batch removal
        id_selector = self.faiss.IDSelectorBatch(np.array(faiss_ids_to_remove).astype('int64'))
        try:
            # Remove from the FAISS index in batch
            removed_count = self.index.remove_ids(id_selector)
            logger.info(f"Removed {removed_count} vector(s) from FAISS index.")

            # Remove from mappings and docstore
            for doc_id in doc_ids_to_remove:
                if doc_id in self.docstore: del self.docstore[doc_id]
                faiss_id = self._doc_id_to_faiss_id.pop(doc_id, None)
                if faiss_id is not None:
                    self._faiss_id_to_doc_id.pop(faiss_id, None)

            # Persist changes
            self._save_index()
            return True

        except Exception as e:
            logger.error(f"Error during batch removal from FAISS: {e}", exc_info=True)
            # Consider potential partial success - state might be inconsistent
            # Attempting to save might be risky. May need recovery logic.
            return False # Indicate failure

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        count = self.index.ntotal if self.index else 0
        dimension = self.index.d if self.index else self.dimension or 0
        return {
            "count": count, # Number of vectors currently in the index
            "docstore_count": len(self.docstore), # Number of documents tracked
            "dimension": dimension,
            "type": "faiss",
            "metric": self.metric,
            "normalize_l2": self.normalize_l2,
            "persistent": self.index_path is not None
        }

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the filter criteria. Supports simple key-value equality.
        Assumes filter_dict contains key-value pairs to match exactly.
        """
        if not filter_dict: # No filter always matches
            return True
        if not metadata: # No metadata cannot match a filter
            return False

        for key, value in filter_dict.items():
            # Handle potential nested keys later if needed
            if key not in metadata or metadata[key] != value:
                return False
        return True