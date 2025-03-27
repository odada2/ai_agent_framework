"""
Utility Functions for Web Search

This module provides utility functions for:
- Query sanitization
- Result scoring
- Domain validation
- Rate limit handling
"""

import re
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from urllib.parse import quote_plus, urlparse

from .base import SearchResult

logger = logging.getLogger(__name__)


def sanitize_query(query: str, max_length: int = 256) -> str:
    """
    Sanitize a search query by removing dangerous characters and limiting length.
    
    Args:
        query: The search query to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized query
    """
    # Remove potentially dangerous characters
    # Allow alphanumeric, spaces, and common punctuation
    safe_query = re.sub(r'[^\w\s.,!?"\'-]', '', query.strip())
    
    # Limit length
    safe_query = safe_query[:max_length]
    
    # URL encode if needed
    if any(c in safe_query for c in '&%+#'):
        safe_query = quote_plus(safe_query)
        
    return safe_query


def is_valid_domain(domain: str) -> bool:
    """
    Validate that a domain string is properly formatted.
    
    Args:
        domain: Domain to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Basic domain validation pattern
    if not domain:
        return False
    
    # Remove protocol if present
    if '://' in domain:
        domain = domain.split('://', 1)[1]
    
    # Remove path if present
    if '/' in domain:
        domain = domain.split('/', 1)[0]
    
    # Basic domain pattern
    pattern = r'^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$'
    return bool(re.match(pattern, domain.lower()))


def extract_domain(url: str) -> str:
    """
    Extract domain from a URL.
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def calculate_score(result: SearchResult, query_keywords: Optional[List[str]] = None) -> SearchResult:
    """
    Score a search result based on relevance to query, domain authority, etc.
    
    Args:
        result: Search result to score
        query_keywords: List of keywords from the query (optional)
        
    Returns:
        The same result with updated score
    """
    score = 0.0
    content = f"{result.title} {result.snippet or ''}".lower()
    
    # Domain authority scoring
    domain = extract_domain(result.url).lower()
    
    # Educational and government sites get a boost
    if domain.endswith(('.edu', '.gov')):
        score += 0.25
    # Academic and knowledge sites get a boost
    elif any(site in domain for site in ('scholar.', 'research.', 'science.', 'academic.')):
        score += 0.2
    # Well-known reference sites get a boost
    elif any(site in domain for site in ('wikipedia.org', 'britannica.com', 'arxiv.org')):
        score += 0.2
    # Technical domains get a boost for technical queries
    elif any(site in domain for site in ('github.com', 'stackoverflow.com', 'docs.')):
        score += 0.15
        
    # Match query keywords if provided
    if query_keywords:
        content_words = set(content.split())
        keyword_matches = sum(1 for kw in query_keywords if kw in content_words)
        score += min(0.5, 0.1 * keyword_matches)  # Cap at 0.5
        
        # Exact title match gives a big boost
        if all(kw in result.title.lower() for kw in query_keywords if len(kw) > 3):
            score += 0.2
    
    # Check for complete sentences in snippet (indicates higher quality)
    if result.snippet and re.search(r'[A-Z][^.!?]*[.!?]', result.snippet):
        score += 0.05
    
    # Penalize extremely short snippets
    if not result.snippet or len(result.snippet) < 20:
        score -= 0.1
    
    # URL quality factors
    url_lower = result.url.lower()
    
    # Penalize URLs with many query parameters (often lower quality)
    if url_lower.count('?') > 0 and url_lower.count('&') > 3:
        score -= 0.05
    
    # Penalize very deep URLs
    if url_lower.count('/') > 5:
        score -= 0.05
    
    # Penalize temporary/session URLs
    if re.search(r'(sess|tmp|temp|session|cache)[=_-]', url_lower):
        score -= 0.1
        
    # Cap score between 0 and 1
    result.score = max(0.0, min(1.0, score))
    return result


def format_search_results(results: List[SearchResult], query: str, highlight: bool = False) -> str:
    """
    Format search results as a readable text string.
    
    Args:
        results: List of search results
        query: Original query
        highlight: Whether to highlight search terms in output
        
    Returns:
        Formatted results as a string
    """
    if not results:
        return f"No results found for query: {query}"
    
    lines = [f"Search results for: {query}"]
    
    keywords = [kw.lower() for kw in query.split() if len(kw) > 3] if highlight else []
    
    for i, result in enumerate(results, 1):
        title = result.title
        snippet = result.snippet or "No description available"
        
        # Highlight keywords if requested
        if highlight and keywords:
            for kw in keywords:
                # Bold keywords in title and snippet
                if kw in title.lower():
                    pattern = re.compile(re.escape(kw), re.IGNORECASE)
                    title = pattern.sub(f"**{pattern.group(0)}**", title)
                if kw in snippet.lower():
                    pattern = re.compile(re.escape(kw), re.IGNORECASE)
                    snippet = pattern.sub(f"**{pattern.group(0)}**", snippet)
        
        # Add result to output
        lines.extend([
            f"\n{i}. {title}",
            f"   URL: {result.url}",
            f"   {snippet}",
            f"   Source: {result.source.capitalize()}"
        ])
    
    return "\n".join(lines)


def parse_retry_after(header_value: Optional[str]) -> int:
    """
    Parse the Retry-After header value.
    
    Args:
        header_value: Value of the Retry-After header
        
    Returns:
        Number of seconds to wait
    """
    if not header_value:
        return 60  # Default to 60 seconds
    
    try:
        # If it's a number of seconds
        return int(header_value)
    except ValueError:
        # If it's an HTTP date
        try:
            retry_date = datetime.strptime(header_value, "%a, %d %b %Y %H:%M:%S %Z")
            now = datetime.now()
            wait_seconds = (retry_date - now).total_seconds()
            return max(1, int(wait_seconds))
        except ValueError:
            return 60  # Default if parsing fails


def get_url_with_params(base_url: str, params: Dict[str, Any]) -> str:
    """
    Build a URL with query parameters.
    
    Args:
        base_url: Base URL
        params: Dictionary of parameters
        
    Returns:
        Full URL with parameters
    """
    query_parts = []
    
    for key, value in params.items():
        if value is None:
            continue
            
        if isinstance(value, (list, tuple)):
            for v in value:
                query_parts.append(f"{quote_plus(key)}={quote_plus(str(v))}")
        else:
            query_parts.append(f"{quote_plus(key)}={quote_plus(str(value))}")
    
    query_string = "&".join(query_parts)
    
    if query_string:
        if "?" in base_url:
            return f"{base_url}&{query_string}"
        else:
            return f"{base_url}?{query_string}"
    
    return base_url