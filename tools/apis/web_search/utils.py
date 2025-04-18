"""
Utility Functions for Web Search

This module provides utility functions for the web search implementation, including:
- Query sanitization
- Result scoring
- Domain validation
- URL processing
- Rate limit handling
"""

import re
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Union
from urllib.parse import quote_plus, urlparse, parse_qs

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
    # Empty domain is invalid
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
        
    # Freshness boost for recent content if date is available in metadata
    if 'published' in result.metadata:
        pub_date_str = result.metadata['published']
        try:
            if isinstance(pub_date_str, str):
                # Handle different date formats
                for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
                    try:
                        pub_date = datetime.strptime(pub_date_str[:19], fmt)  # Truncate to avoid timezone issues
                        break
                    except ValueError:
                        continue
                else:
                    # If no format worked, try ISO format with various timezone handling
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                
                # Calculate days since publication
                days_old = (datetime.now() - pub_date).days
                if days_old < 1:
                    score += 0.3  # Today
                elif days_old < 7:
                    score += 0.2  # Within a week
                elif days_old < 30:
                    score += 0.1  # Within a month
                elif days_old > 365:
                    score -= 0.1  # Older than a year
        except (ValueError, TypeError):
            # Ignore date parsing errors
            pass
        
    # Cap score between 0 and 1
    result.score = max(0.0, min(1.0, score))
    return result


def extract_query_parameters(url: str) -> Dict[str, str]:
    """
    Extract query parameters from a URL.
    
    Args:
        url: URL to extract from
        
    Returns:
        Dictionary of query parameters
    """
    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Convert lists to single values
        return {k: v[0] if v else '' for k, v in query_params.items()}
    except Exception as e:
        logger.warning(f"Error extracting query parameters: {str(e)}")
        return {}


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
                    title = re.sub(
                        rf'\b{re.escape(kw)}\b', 
                        lambda m: f"**{m.group(0)}**", 
                        title, 
                        flags=re.IGNORECASE
                    )
                if kw in snippet.lower():
                    snippet = re.sub(
                        rf'\b{re.escape(kw)}\b', 
                        lambda m: f"**{m.group(0)}**", 
                        snippet, 
                        flags=re.IGNORECASE
                    )
        
        # Add result to output
        lines.extend([
            f"\n{i}. {title}",
            f"   URL: {result.url}",
            f"   {snippet}",
            f"   Source: {result.source.capitalize()}"
        ])
        
        # Add metadata type if available
        if "type" in result.metadata:
            lines.append(f"   Type: {result.metadata['type'].replace('_', ' ').title()}")
    
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


def truncate_text(text: str, max_length: int = 200, append_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length, optionally appending an ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        append_ellipsis: Whether to append an ellipsis
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Try to truncate at a word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # Only if space is reasonably far in
        truncated = truncated[:last_space]
    
    if append_ellipsis:
        truncated += "..."
    
    return truncated


def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing common tracking parameters and fragments.
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL
    """
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Remove common tracking parameters
        if parsed.query:
            query_dict = parse_qs(parsed.query)
            # List of tracking parameters to remove
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'ocid', 'msclkid', 'zanpid', 'dclid', '_hsenc', '_hsmi'
            }
            
            # Filter out tracking parameters
            filtered_query = {k: v for k, v in query_dict.items() if k.lower() not in tracking_params}
            
            # Rebuild query string
            if filtered_query:
                query_string = '&'.join(f"{k}={v[0]}" for k, v in filtered_query.items())
            else:
                query_string = ''
        else:
            query_string = ''
        
        # Remove fragment (hash) if present
        fragment = ''
        
        # Rebuild the URL
        normalized = parsed._replace(query=query_string, fragment=fragment).geturl()
        
        # Ensure URL has a protocol
        if not normalized.startswith(('http://', 'https://')):
            normalized = 'https://' + normalized
        
        return normalized
        
    except Exception as e:
        logger.warning(f"Error normalizing URL: {str(e)}")
        return url