import html
from typing import Optional
from tenacity import retry, stop_after_attempt

class QuerySanitizer:
    BLACKLISTED_TERMS = [
        "password", "secret", "api_key", 
        "credentials", "private"
    ]

    @classmethod
    def sanitize(cls, query: str) -> str:
        clean_query = html.escape(query.strip())
        if len(clean_query) > 256:
            clean_query = clean_query[:256]
        return clean_query

    @classmethod
    def validate(cls, query: str) -> bool:
        lower_query = query.lower()
        return not any(term in lower_query for term in cls.BLACKLISTED_TERMS)

@retry(stop=stop_after_attempt(3))
async def safe_search(provider, query: str) -> list:
    if not QuerySanitizer.validate(query):
        raise ValueError("Query contains blocked terms")
    return await provider.search(QuerySanitizer.sanitize(query))