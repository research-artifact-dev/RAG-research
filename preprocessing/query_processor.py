# preprocessing/query_processor.py
"""
Simplified Query Preprocessing Module.

Purpose: Only enhance queries when retrieval confidence is low.
Approach:
  1. Try original query first
  2. Check retrieval confidence (similarity scores)
  3. If low confidence → extract keywords and enhance query
  4. Retry with enhanced query
"""

import spacy
from typing import List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize spaCy (lazy loading)
_nlp = None


def get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        print("Loading spaCy model (en_core_web_sm)...")
        try:
            _nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded")
        except OSError:
            print("⚠️  spaCy model not found. Installing...")
            print("Run: python -m spacy download en_core_web_sm")
            raise
    return _nlp


def extract_keywords(query: str) -> List[str]:
    """
    Extract keywords from query using spaCy noun phrase extraction.
    Used only when we need to enhance a low-confidence query.

    Args:
        query: User query string

    Returns:
        List of keywords (noun phrases and entities)
    """
    nlp = get_nlp()
    doc = nlp(query)

    # Extract noun chunks (noun phrases)
    keywords = [chunk.text.lower() for chunk in doc.noun_chunks]

    # Also extract named entities
    entities = [ent.text.lower() for ent in doc.ents]

    # Combine and deduplicate
    all_keywords = list(set(keywords + entities))

    return all_keywords


def enhance_query(query: str, keywords: List[str] = None) -> str:
    """
    Enhance a query by adding relevant keywords.
    Used when retrieval confidence is low.

    Args:
        query: Original query string
        keywords: Extracted keywords (if None, will extract them)

    Returns:
        Enhanced query string
    """
    if keywords is None:
        keywords = extract_keywords(query)

    # Add top 2-3 keywords to the query
    if keywords:
        # Filter out keywords already in the query
        new_keywords = [kw for kw in keywords[:3] if kw.lower() not in query.lower()]

        if new_keywords:
            enhanced = f"{query} {' '.join(new_keywords[:2])}"
            return enhanced.strip()

    # If no new keywords, return original
    return query


def is_compound_query(query: str) -> bool:
    """
    Check if query is compound (has multiple tasks).

    Examples:
        "read CSV and sort data" → True
        "sort dictionary" → False

    Args:
        query: User query string

    Returns:
        True if compound query
    """
    conjunctions = [" and then ", " and ", " then "]
    return any(conj in query.lower() for conj in conjunctions)


def decompose_query(query: str) -> List[str]:
    """
    Split compound queries into separate sub-queries.
    Only call this if is_compound_query() returns True.

    Examples:
        "read CSV and sort by column" → ["read CSV", "sort by column"]
        "connect to MySQL then insert data" → ["connect to MySQL", "insert data"]

    Args:
        query: Compound query string

    Returns:
        List of sub-queries
    """
    conjunctions = [" and then ", " and ", " then ", ", then ", ", and "]

    parts = [query]

    # Split by each conjunction
    for conj in conjunctions:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(conj))
        parts = new_parts

    # Filter out very short parts
    sub_queries = [p.strip() for p in parts if len(p.strip()) > 5]

    return sub_queries if len(sub_queries) > 1 else [query]


if __name__ == "__main__":
    # Test the simplified preprocessing
    print("=" * 70)
    print("SIMPLIFIED QUERY PREPROCESSING - TEST")
    print("=" * 70)

    test_queries = [
        "sort dictionary",  # Vague
        "read a CSV file and process the data",  # Compound
        "pandas DataFrame filter rows",  # Clear
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print(f"{'='*70}")

        # Extract keywords
        keywords = extract_keywords(query)
        print(f"Keywords: {keywords}")

        # Enhance if needed
        enhanced = enhance_query(query, keywords)
        print(f"Enhanced: '{enhanced}'")

        # Check if compound
        if is_compound_query(query):
            sub_queries = decompose_query(query)
            print(f"Compound query detected!")
            print(f"Sub-queries: {sub_queries}")
        else:
            print(f"Simple query (no decomposition needed)")
