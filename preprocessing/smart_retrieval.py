#!/usr/bin/env python3
# preprocessing/smart_retrieval.py
"""
Smart Retrieval with Confidence Gate.

This is the main interface that combines:
  1. Retrieval from vector DB
  2. Confidence checking
  3. Query enhancement (only if needed)
  4. Retry with enhanced query

Flow:
    User Query → Try Retrieval → Check Confidence
                        ↓
                  Good? → Return results ✅
                        ↓
                   Bad? → Enhance query → Retry → Return results
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectordb.retriever import Retriever
from preprocessing.query_processor import extract_keywords, enhance_query, is_compound_query, decompose_query
from config import RETRIEVAL_CONFIDENCE_THRESHOLD, MAX_REFORMULATION_ATTEMPTS, DEFAULT_TOP_K
from typing import List, Dict, Tuple


class SmartRetriever:
    """
    Smart retrieval that only enhances queries when confidence is low.
    """

    def __init__(self, index_path: str = None, docs_path: str = None):
        """
        Initialize smart retriever with vector DB.

        Args:
            index_path: Path to FAISS index (optional, uses config default)
            docs_path: Path to documents JSON (optional, uses config default)
        """
        from config import FAISS_INDEX_PATH, DOCUMENTS_PATH

        index_path = index_path or FAISS_INDEX_PATH
        docs_path = docs_path or DOCUMENTS_PATH

        print("Initializing Smart Retriever...")
        self.retriever = Retriever(index_path, docs_path)
        print("✓ Smart Retriever ready!\n")

    def check_confidence(self, results: List[Dict]) -> Tuple[bool, float]:
        """
        Check if retrieval results have good confidence.

        Args:
            results: List of retrieved documents with similarity_score

        Returns:
            (is_confident, top_score)
        """
        if not results:
            return False, 0.0

        top_score = results[0]["similarity_score"]
        avg_top3 = sum(r["similarity_score"] for r in results[:3]) / min(3, len(results))

        # Good confidence if:
        # 1. Top score is above threshold (0.65)
        # 2. Average of top 3 is reasonable (0.55+)
        is_confident = (
            top_score >= RETRIEVAL_CONFIDENCE_THRESHOLD
            and avg_top3 >= 0.55
        )

        return is_confident, top_score

    def retrieve_smart(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        use_hybrid: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Smart retrieval with confidence gate and automatic query enhancement.

        Flow:
            1. Try original query
            2. Check confidence
            3. If low → enhance and retry (max 2 attempts)
            4. Return best results

        Args:
            query: User's natural language query
            top_k: Number of results to return
            use_hybrid: Use hybrid search (vector + BM25)
            verbose: Print progress messages

        Returns:
            Dict with:
                - results: List of retrieved documents
                - query_used: Final query that was used
                - confidence_score: Top similarity score
                - attempts: Number of attempts made
                - enhanced: Whether query was enhanced
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"SMART RETRIEVAL")
            print(f"{'='*70}")
            print(f"Query: '{query}'")
            print(f"{'='*70}\n")

        # Handle compound queries separately
        if is_compound_query(query):
            if verbose:
                print("⚠️  Detected compound query - decomposing...")
            return self._retrieve_compound(query, top_k, use_hybrid, verbose)

        # Attempt 1: Try original query
        if verbose:
            print("📊 Attempt 1: Original query")

        results = self.retriever.retrieve(query, top_k=top_k, use_hybrid=use_hybrid)
        is_confident, top_score = self.check_confidence(results)

        if verbose:
            print(f"   Top score: {top_score:.4f}")
            print(f"   Confidence: {'✅ GOOD' if is_confident else '⚠️  LOW'}")

        if is_confident:
            if verbose:
                print(f"\n✅ Confident retrieval! Returning {len(results)} results.\n")
            return {
                "results": results,
                "query_used": query,
                "confidence_score": top_score,
                "attempts": 1,
                "enhanced": False
            }

        # Confidence is low - enhance query and retry
        if verbose:
            print(f"\n⚠️  Low confidence ({top_score:.4f} < {RETRIEVAL_CONFIDENCE_THRESHOLD})")
            print("🔄 Enhancing query with keywords...\n")

        # Extract keywords and enhance
        keywords = extract_keywords(query)
        if verbose and keywords:
            print(f"   Extracted keywords: {keywords[:3]}")

        enhanced_query = enhance_query(query, keywords)

        if verbose:
            print(f"   Enhanced query: '{enhanced_query}'\n")

        # Attempt 2: Try enhanced query
        if verbose:
            print("📊 Attempt 2: Enhanced query")

        results = self.retriever.retrieve(enhanced_query, top_k=top_k, use_hybrid=use_hybrid)
        is_confident, top_score = self.check_confidence(results)

        if verbose:
            print(f"   Top score: {top_score:.4f}")
            print(f"   Confidence: {'✅ GOOD' if is_confident else '⚠️  STILL LOW'}")

        if verbose:
            confidence_text = "✅ Confident" if is_confident else "⚠️  Low confidence"
            print(f"\n{confidence_text} retrieval after enhancement.")
            print(f"Returning {len(results)} results.\n")

        return {
            "results": results,
            "query_used": enhanced_query,
            "confidence_score": top_score,
            "attempts": 2,
            "enhanced": True
        }

    def _retrieve_compound(
        self,
        query: str,
        top_k: int,
        use_hybrid: bool,
        verbose: bool
    ) -> Dict:
        """
        Handle compound queries by decomposing and merging results.

        Args:
            query: Compound query (e.g., "read CSV and sort data")
            top_k: Number of results per sub-query
            use_hybrid: Use hybrid search
            verbose: Print messages

        Returns:
            Merged results from all sub-queries
        """
        sub_queries = decompose_query(query)

        if verbose:
            print(f"Sub-queries detected:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")
            print()

        all_results = []
        seen_ids = set()

        for i, sub_query in enumerate(sub_queries, 1):
            if verbose:
                print(f"📊 Retrieving for sub-query {i}/{len(sub_queries)}: '{sub_query}'")

            results = self.retriever.retrieve(sub_query, top_k=top_k//len(sub_queries) + 1, use_hybrid=use_hybrid)

            # Deduplicate
            for result in results:
                if result["id"] not in seen_ids:
                    seen_ids.add(result["id"])
                    all_results.append(result)

        # Sort by score and take top_k
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        final_results = all_results[:top_k]

        top_score = final_results[0]["similarity_score"] if final_results else 0.0

        if verbose:
            print(f"\n✅ Merged results from {len(sub_queries)} sub-queries")
            print(f"Returning top {len(final_results)} results.\n")

        return {
            "results": final_results,
            "query_used": query,
            "confidence_score": top_score,
            "attempts": 1,
            "enhanced": False,
            "compound": True,
            "sub_queries": sub_queries
        }

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, **kwargs) -> List[Dict]:
        """
        Simplified interface that returns just the results list.

        Args:
            query: User query
            top_k: Number of results

        Returns:
            List of retrieved documents
        """
        result = self.retrieve_smart(query, top_k=top_k, **kwargs)
        return result["results"]


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SMART RETRIEVAL - DEMO")
    print("=" * 70)

    smart_retriever = SmartRetriever()

    # Test queries with different confidence levels
    test_queries = [
        "read CSV file",  # Clear query - should be confident
        "sort dict",  # Vague - should enhance
        "read file and sort data",  # Compound - should decompose
    ]

    for query in test_queries:
        result = smart_retriever.retrieve_smart(query, top_k=5, verbose=True)

        print(f"{'='*70}")
        print(f"RESULT SUMMARY:")
        print(f"  Query used: '{result['query_used']}'")
        print(f"  Confidence: {result['confidence_score']:.4f}")
        print(f"  Attempts: {result['attempts']}")
        print(f"  Enhanced: {result['enhanced']}")
        print(f"  Results returned: {len(result['results'])}")
        print(f"{'='*70}\n")

        # Show top 3 results
        print("Top 3 Results:")
        for i, doc in enumerate(result['results'][:3], 1):
            print(f"  {i}. Score: {doc['similarity_score']:.4f} | {doc['query'][:60]}...")
        print()
