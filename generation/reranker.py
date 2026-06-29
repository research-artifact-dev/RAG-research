#!/usr/bin/env python3
# generation/reranker.py
"""
Candidate Reranker for selecting the best generated code.

Scores each candidate by comparing embeddings with retrieved examples
and selects the candidate with highest similarity.
"""

import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, NORMALIZE_EMBEDDINGS


class CandidateReranker:
    """
    Rerank generated code candidates by similarity to retrieved examples.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize reranker with embedding model.

        Args:
            model_name: Name of sentence transformer model
        """
        print(f"Loading embedding model for reranking: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✓ Reranker ready")

    def score_candidate(
        self,
        candidate_code: str,
        retrieved_examples: List[Dict],
        top_n_examples: int = 5
    ) -> float:
        """
        Score a candidate by comparing with retrieved examples.

        Args:
            candidate_code: Generated code to score
            retrieved_examples: Retrieved example documents
            top_n_examples: Number of top examples to compare against

        Returns:
            Average similarity score (0.0-1.0)
        """
        # Embed the candidate
        candidate_emb = self.model.encode(
            [candidate_code],
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True
        ).astype('float32')

        # Embed retrieved example codes (top N only)
        example_codes = [
            ex["code"] for ex in retrieved_examples[:top_n_examples]
        ]

        example_embs = self.model.encode(
            example_codes,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True
        ).astype('float32')

        # Compute cosine similarities (already normalized)
        similarities = np.dot(example_embs, candidate_emb.T).flatten()

        # Return average similarity
        avg_score = float(np.mean(similarities))
        return avg_score

    def rerank(
        self,
        candidates: List[str],
        retrieved_examples: List[Dict]
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Rerank all candidates and return the best one.

        Args:
            candidates: List of generated code candidates
            retrieved_examples: Retrieved example documents

        Returns:
            Tuple of (best_code, best_score, all_scores)
            where all_scores is [(code, score), ...]
        """
        print(f"\n🏆 Reranking {len(candidates)} candidates...")

        # Score each candidate
        scored_candidates = []
        for i, candidate in enumerate(candidates, 1):
            score = self.score_candidate(candidate, retrieved_examples)
            scored_candidates.append((candidate, score))
            print(f"   Candidate {i}: score = {score:.4f}")

        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Extract best
        best_code, best_score = scored_candidates[0]

        print(f"✓ Best candidate: score = {best_score:.4f}")

        return best_code, best_score, scored_candidates


if __name__ == "__main__":
    # Test reranker
    print("=" * 70)
    print("CANDIDATE RERANKER TEST")
    print("=" * 70)

    # Mock retrieved examples
    retrieved_examples = [
        {
            "code": "sorted(dict.items(), key=lambda x: x[1])",
            "query": "sort dictionary",
            "similarity_score": 0.85
        },
        {
            "code": "{k: v for k, v in sorted(d.items(), key=lambda x: x[1])}",
            "query": "sort dict values",
            "similarity_score": 0.82
        },
        {
            "code": "dict(sorted(my_dict.items(), key=lambda item: item[1]))",
            "query": "sort dict by value",
            "similarity_score": 0.80
        }
    ]

    # Mock candidates
    candidates = [
        # Good candidate (similar to examples)
        "def sort_dict(d):\n    return dict(sorted(d.items(), key=lambda x: x[1]))",

        # Okay candidate (different approach)
        "def sort_dict(d):\n    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}",

        # Poor candidate (completely different)
        "def sort_dict(d):\n    import operator\n    return sorted(d.items(), key=operator.itemgetter(1))"
    ]

    # Initialize reranker
    reranker = CandidateReranker()

    # Rerank candidates
    best_code, best_score, all_scores = reranker.rerank(
        candidates,
        retrieved_examples
    )

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nBest candidate (score: {best_score:.4f}):")
    print(best_code)

    print(f"\nAll candidates ranked:")
    for i, (code, score) in enumerate(all_scores, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   {code[:60]}...")

    print(f"\n{'='*70}")
