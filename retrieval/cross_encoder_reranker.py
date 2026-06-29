#!/usr/bin/env python3
# retrieval/cross_encoder_reranker.py
"""
Cross-Encoder Reranking Module

Hierarchical Retrieval for Hallucination Mitigation.
Uses cross-encoder model to rerank retrieved examples based on query-document relevance.
Cross-encoders are more accurate than bi-encoders but slower, making them ideal for
reranking a small candidate set.
"""

from typing import Dict, List, Tuple
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Cross-encoder reranking unavailable.")
    CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker:
    """
    Rerank retrieved examples using cross-encoder model.

    Cross-encoders process query+document together for more accurate relevance scoring
    compared to bi-encoders that encode them separately.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Hugging Face model name
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (balanced, 80M params)
                - cross-encoder/ms-marco-TinyBERT-L-2-v2 (faster, 4M params)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (best quality, 134M params)
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers required for cross-encoder reranking")

        print(f"Loading cross-encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print("✓ Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 8
    ) -> List[Dict]:
        """
        Rerank candidates by query-document relevance using cross-encoder.

        Args:
            query: User query
            candidates: List of candidate examples with 'query' and 'code' fields
            top_k: Number of top candidates to return

        Returns:
            Reranked list of candidates (best first) with 'rerank_score' added
        """
        if not candidates:
            return []

        # Prepare query-document pairs
        # Use candidate's query field (docstring/description) for comparison
        pairs = []
        for candidate in candidates:
            doc_text = candidate.get("query", "")
            # If no query field, use first 200 chars of code
            if not doc_text:
                doc_text = candidate.get("code", "")[:200]
            pairs.append([query, doc_text])

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores to candidates
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(score)
            scored_candidates.append(candidate_copy)

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Return top-k
        return scored_candidates[:top_k]

    def rerank_with_stats(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 8
    ) -> Tuple[List[Dict], Dict]:
        """
        Rerank and return statistics.

        Args:
            query: User query
            candidates: List of candidates
            top_k: Number to return

        Returns:
            Tuple of (reranked_candidates, statistics)
        """
        reranked = self.rerank(query, candidates, top_k)

        if not reranked:
            return [], {"error": "No candidates"}

        scores = [c["rerank_score"] for c in reranked]

        stats = {
            "input_count": len(candidates),
            "output_count": len(reranked),
            "top_score": max(scores) if scores else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "score_range": max(scores) - min(scores) if scores else 0
        }

        return reranked, stats


if __name__ == "__main__":
    # Test Cross-Encoder Reranker
    print("=" * 70)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 70)

    if not CROSS_ENCODER_AVAILABLE:
        print("\n✗ sentence-transformers not available. Cannot test.")
        exit(1)

    # Initialize reranker
    reranker = CrossEncoderReranker()

    # Test query
    query = "sort a pandas dataframe by multiple columns"

    # Mock candidates (simulate retrieval results)
    candidates = [
        {
            "id": "1",
            "query": "filter pandas dataframe rows",
            "code": "df[df['col'] > 10]",
            "similarity_score": 0.75
        },
        {
            "id": "2",
            "query": "sort pandas dataframe by column ascending",
            "code": "df.sort_values('column_name')",
            "similarity_score": 0.82
        },
        {
            "id": "3",
            "query": "sort pandas dataframe by multiple columns with custom order",
            "code": "df.sort_values(['col1', 'col2'], ascending=[True, False])",
            "similarity_score": 0.78
        },
        {
            "id": "4",
            "query": "merge two pandas dataframes",
            "code": "pd.merge(df1, df2, on='key')",
            "similarity_score": 0.65
        },
        {
            "id": "5",
            "query": "reset index of pandas dataframe after sorting",
            "code": "df.sort_values('col').reset_index(drop=True)",
            "similarity_score": 0.80
        }
    ]

    print(f"\nQuery: {query}")
    print(f"\nOriginal candidates (by bi-encoder similarity):")
    for i, c in enumerate(sorted(candidates, key=lambda x: x['similarity_score'], reverse=True), 1):
        print(f"  {i}. [{c['id']}] {c['query'][:50]}... (score: {c['similarity_score']:.3f})")

    # Rerank with cross-encoder
    print("\n" + "=" * 70)
    print("Reranking with Cross-Encoder...")
    print("=" * 70)

    reranked, stats = reranker.rerank_with_stats(query, candidates, top_k=3)

    print(f"\nReranked candidates (by cross-encoder):")
    for i, c in enumerate(reranked, 1):
        print(f"  {i}. [{c['id']}] {c['query'][:50]}...")
        print(f"     Cross-Encoder Score: {c['rerank_score']:.3f}")
        print(f"     Original Bi-Encoder: {c['similarity_score']:.3f}")
        print()

    print("=" * 70)
    print("Reranking Statistics:")
    print("=" * 70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("\nNotice how cross-encoder reordering differs from bi-encoder:")
    print("- Cross-encoder sees full query+document context together")
    print("- More accurate semantic matching")
    print("- Better understands 'sort by multiple columns' intent")
    print("\nThis hierarchical approach (bi-encoder → cross-encoder) is")
    print("a proven pattern for reducing hallucinations through better retrieval.")
