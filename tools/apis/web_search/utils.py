"""
Utility Functions for Web Search

Includes:
- Query sanitization
- Result scoring
- Domain validation
"""

import re
from urllib.parse import quote_plus
from typing import Optional
from .base import SearchResult

def sanitize_query(query: str, max_length: int = 256) -> str:
    """
    Sanitizes search queries:
    - Removes dangerous characters
    - Limits length
    - URL encodes
    """
    safe_query = re.sub(r'[^\w\s-]', '', query.strip())
    return quote_plus(safe_query[:max_length])

def calculate_score(result: SearchResult, query_keywords: list[str] = None) -> SearchResult:
    """
    Scores results based on:
    - Domain authority
    - Keyword matches
    - Result freshness
    """
    score = 0.0
    content = f"{result.title} {result.snippet}".lower()
    
    # Domain authority
    if "arxiv.org" in result.url:
        score += 0.3
    elif "github.com" in result.url:
        score += 0.2
        
    # Keyword matches
    if query_keywords:
        score += 0.1 * sum(kw in content for kw in query_keywords)
        
    # Freshness
    if (datetime.now() - result.timestamp).days < 7:
        score += 0.05
        
    result.score = min(score, 1.0)
    return result

def is_valid_domain(domain: Optional[str]) -> bool:
    """Validates domain filters"""
    if not domain:
        return True
    return bool(re.match(r'^[a-z0-9-]+(\.[a-z]{2,})+$', domain.lower()))