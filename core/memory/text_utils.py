"""
Text Processing Utilities

This module provides utilities for processing text documents, including:
- Text splitting and chunking
- Metadata extraction
- Text normalization
- Context window management
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterator, Union

from .vector_store.base import Document

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any]


class TextSplitter:
    """
    Base text splitter class for chunking text documents.
    
    This class provides the foundation for all text splitting strategies,
    handling common functionality like overlap and metadata preservation.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
        keep_separator: bool = False,
        strip_whitespace: bool = True
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Overlap between adjacent chunks (in characters)
            separator: String to use as separator for splitting
            keep_separator: Whether to keep the separator in chunks
            strip_whitespace: Whether to strip whitespace from chunk edges
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size)
        self.separator = separator
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on separator.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split by separator
        splits = text.split(self.separator)
        
        # Remove empty splits and strip whitespace if needed
        if self.strip_whitespace:
            splits = [s.strip() for s in splits if s.strip()]
        else:
            splits = [s for s in splits if s]
        
        # Merge splits into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # If adding this split would exceed chunk_size, store the current chunk
            if current_chunk and current_length + len(self.separator) + split_length > self.chunk_size:
                # Join with separator if we're keeping it
                if self.keep_separator:
                    chunks.append(self.separator.join(current_chunk))
                else:
                    chunks.append(" ".join(current_chunk))
                
                # Handle overlap: keep the last few splits that fit within chunk_overlap
                overlap_length = 0
                overlap_splits = []
                
                for split_to_keep in reversed(current_chunk):
                    if overlap_length + len(split_to_keep) + len(self.separator) <= self.chunk_overlap:
                        overlap_splits.insert(0, split_to_keep)
                        overlap_length += len(split_to_keep) + len(self.separator)
                    else:
                        break
                
                # Start a new chunk with the overlap content
                current_chunk = overlap_splits
                current_length = overlap_length
            
            # Add the current split to the chunk
            current_chunk.append(split)
            current_length += split_length + len(self.separator)
        
        # Add the last chunk if non-empty
        if current_chunk:
            if self.keep_separator:
                chunks.append(self.separator.join(current_chunk))
            else:
                chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        Create Document objects from text chunks.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        for i, text in enumerate(texts):
            # Split text into chunks
            chunks = self.split_text(text)
            
            # Create documents from chunks
            for j, chunk in enumerate(chunks):
                # Clone metadata and add chunk info
                chunk_metadata = metadatas[i].copy()
                chunk_metadata.update({
                    "chunk_index": j,
                    "chunk_count": len(chunks),
                    "source_index": i
                })
                
                # Create document
                doc = Document(
                    text=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
        
        return documents


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Recursive character-based text splitter.
    
    This splitter attempts to split text on a hierarchy of separators,
    ensuring that semantically meaningful chunks are preserved.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", ". ", " ", ""],
        strip_whitespace: bool = True
    ):
        """
        Initialize the recursive character text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators to use, in order of preference
            strip_whitespace: Whether to strip whitespace from chunk edges
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separators[0] if separators else "\n",
            strip_whitespace=strip_whitespace
        )
        self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively using the hierarchy of separators.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Final separator is empty string, which splits on each character
        final_separator = self.separators[-1] if self.separators else ""
        
        # If text fits in a single chunk, return it
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in turn
        for separator in self.separators:
            if separator == "":
                # For final empty separator, just split by char count
                return self._split_by_size(text, self.chunk_size, self.chunk_overlap)
            
            # Try splitting with this separator
            if separator in text:
                self.separator = separator
                return super().split_text(text)
        
        # Fallback: split by size
        return self._split_by_size(text, self.chunk_size, self.chunk_overlap)
    
    def _split_by_size(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Find chunk end
            end = start + chunk_size
            
            # If this is the last chunk, just use the rest of the text
            if end >= text_len:
                chunks.append(text[start:])
                break
            
            # Otherwise, try to break at a word boundary
            # Look backward from end to find a space
            while end > start and not text[end].isspace():
                end -= 1
            
            # If we couldn't find a space, just use the chunk size
            if end == start:
                end = start + chunk_size
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Set next start position, accounting for overlap
            start = end - overlap
        
        return chunks


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """
    Markdown-specific text splitter.
    
    This splitter understands Markdown structure and attempts to split text
    at meaningful boundaries like headers and paragraphs.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the Markdown text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        separators = [
            # Headers
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n###### ",
            # Paragraphs
            "\n\n",
            "\n",
            # Sentences
            ". ",
            "! ",
            "? ",
            # Words and characters
            " ",
            ""
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        Split Markdown text preserving structure.
        
        Args:
            text: Markdown text to split
            
        Returns:
            List of text chunks
        """
        # Process headers to include them with their content
        processed_text = text
        
        # Extract metadata from Markdown frontmatter if present
        metadata = {}
        frontmatter_match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            # Parse YAML-like frontmatter
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
            
            # Remove frontmatter from text
            processed_text = text[frontmatter_match.end():]
        
        # Split text using parent method
        chunks = super().split_text(processed_text)
        
        # Preserve header hierarchy in each chunk
        result_chunks = []
        current_headers = ["", "", "", "", "", ""]  # h1 to h6
        
        for chunk in chunks:
            # Check for headers in this chunk
            chunk_with_headers = chunk
            
            # Check for headers being split from their content
            for i, header_prefix in enumerate(["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### "]):
                if chunk.startswith(header_prefix[1:]):  # Header at start of chunk
                    # Update current header at this level and reset lower levels
                    current_headers[i] = chunk.split("\n")[0]
                    for j in range(i + 1, 6):
                        current_headers[j] = ""
            
            # Prepend relevant headers if chunk doesn't start with one
            if not any(chunk.startswith(prefix[1:]) for prefix in ["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### "]):
                # Find the most specific relevant header
                relevant_headers = []
                for i, header in enumerate(current_headers):
                    if header:
                        relevant_headers.append(f"{'#' * (i + 1)} {header}")
                
                if relevant_headers:
                    chunk_with_headers = relevant_headers[-1] + "\n\n" + chunk
            
            result_chunks.append(chunk_with_headers)
        
        return result_chunks


class HTMLTextSplitter(RecursiveCharacterTextSplitter):
    """
    HTML-specific text splitter.
    
    This splitter attempts to preserve HTML structure when splitting text.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strip_tags: bool = True
    ):
        """
        Initialize the HTML text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            strip_tags: Whether to strip HTML tags from the output
        """
        # Define separators based on HTML structure
        separators = [
            # Major sections
            "</div>",
            "</section>",
            "</article>",
            # Block-level elements
            "</p>",
            "</h1>",
            "</h2>",
            "</h3>",
            "</li>",
            "<br>",
            "<br/>",
            "<br />",
            # Sentences
            ". ",
            "! ",
            "? ",
            # Words and characters
            " ",
            ""
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        
        self.strip_tags = strip_tags
    
    def split_text(self, text: str) -> List[str]:
        """
        Split HTML text preserving structure where possible.
        
        Args:
            text: HTML text to split
            
        Returns:
            List of text chunks
        """
        # First split by parent method
        chunks = super().split_text(text)
        
        # Strip HTML tags if requested
        if self.strip_tags:
            chunks = [self._strip_html_tags(chunk) for chunk in chunks]
        
        return chunks
    
    def _strip_html_tags(self, text: str) -> str:
        """
        Strip HTML tags from text, preserving content.
        
        Args:
            text: HTML text to process
            
        Returns:
            Text with HTML tags removed
        """
        # Simple regex-based HTML tag stripping
        # Note: This is not a perfect solution; for production use,
        # consider using a proper HTML parser like BeautifulSoup
        clean_text = re.sub(r'<[^>]*>', ' ', text)
        
        # Clean up whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    text_format: str = "text"
) -> List[TextChunk]:
    """
    Chunk text based on format.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        text_format: Format of text ('text', 'markdown', 'html')
        
    Returns:
        List of TextChunk objects
    """
    # Select appropriate splitter based on format
    if text_format.lower() == "markdown":
        splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif text_format.lower() == "html":
        splitter = HTMLTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:  # Default to standard text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    # Split text
    chunks = splitter.split_text(text)
    
    # Create TextChunk objects
    return [
        TextChunk(
            text=chunk,
            metadata={
                "index": i,
                "count": len(chunks),
                "format": text_format
            }
        )
        for i, chunk in enumerate(chunks)
    ]


def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Extract metadata from text content using heuristics.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}
    
    # Try to identify a title (first line or first heading)
    lines = text.split("\n")
    if lines:
        # Check for Markdown heading
        if lines[0].startswith("# "):
            metadata["title"] = lines[0][2:].strip()
        # Otherwise use first non-empty line
        else:
            for line in lines:
                if line.strip():
                    metadata["title"] = line.strip()
                    break
    
    # Estimate reading time
    word_count = len(text.split())
    metadata["word_count"] = word_count
    metadata["estimated_reading_time_minutes"] = max(1, round(word_count / 200))
    
    # Check content type based on patterns
    if re.search(r'<[a-z]+[^>]*>', text):
        metadata["content_type"] = "html"
    elif re.search(r'^#+ ', text, re.MULTILINE) or re.search(r'\*\*.*\*\*', text):
        metadata["content_type"] = "markdown"
    else:
        metadata["content_type"] = "text"
    
    return metadata