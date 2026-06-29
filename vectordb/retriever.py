# vectordb/retriever.py
"""
Retriever interface for querying the FAISS vector database.
Provides semantic search over code examples with hybrid search support.
"""

import json
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    DOCUMENTS_PATH,
    NORMALIZE_EMBEDDINGS,
    DEFAULT_TOP_K,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    MAX_REFORMULATION_ATTEMPTS,
    # Hierarchical Retrieval
    HYBRID_SEARCH_TOP_K,
    CROSS_ENCODER_TOP_K,
    ENABLE_CROSS_ENCODER,
    CROSS_ENCODER_MODEL
)

# Import cross-encoder reranker
from retrieval.cross_encoder_reranker import CrossEncoderReranker


class Retriever:
    """
    Semantic code retrieval using FAISS vector search.
    """

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        docs_path: str = DOCUMENTS_PATH,
        model_name: str = EMBEDDING_MODEL,
        enable_cross_encoder: bool = ENABLE_CROSS_ENCODER
    ):
        """
        Initialize retriever with FAISS index and documents.

        Args:
            index_path: Path to FAISS index file
            docs_path: Path to documents JSON file
            model_name: Name of embedding model
            enable_cross_encoder: Enable cross-encoder reranking
        """
        print(f"Initializing Retriever...")

        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded FAISS index from {index_path}")

        # Load documents
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Documents not found at {docs_path}")
        with open(docs_path, 'r') as f:
            self.documents = json.load(f)
        print(f"✓ Loaded {len(self.documents)} documents from {docs_path}")

        # Load embedding model
        self.model = SentenceTransformer(model_name)
        print(f"✓ Loaded embedding model: {model_name}")

        # Initialize BM25 for hybrid search
        print("Building BM25 index for hybrid search...")
        self._build_bm25_index()
        print("✓ BM25 index ready")

        # Initialize cross-encoder reranker
        self.enable_cross_encoder = enable_cross_encoder
        if self.enable_cross_encoder:
            try:
                print("Initializing cross-encoder reranker...")
                self.cross_encoder = CrossEncoderReranker(model_name=CROSS_ENCODER_MODEL)
                print("✓ Cross-encoder ready")
            except ImportError as e:
                print(f"⚠ Cross-encoder unavailable: {e}")
                self.enable_cross_encoder = False
                self.cross_encoder = None
        else:
            self.cross_encoder = None
            print("⚠ Cross-encoder reranking disabled")

        print("Retriever ready!\n")

    def _build_bm25_index(self):
        """Build BM25 index from document queries for keyword search."""
        # Tokenize all queries
        tokenized_queries = [
            doc["query"].lower().split()
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_queries)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Basic retrieval using hybrid search (dense vector + BM25).

        This is the base method that returns results with retrieval similarity scores.
        Confidence checking should be done on these scores BEFORE applying
        metadata filtering or cross-encoder reranking.

        Args:
            query: Natural language query string
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (dense vector + BM25)

        Returns:
            List of documents with retrieval similarity scores
        """
        if use_hybrid:
            return self._hybrid_search(query, top_k)
        else:
            return self._vector_search(query, top_k)

    def retrieve_with_hierarchical_refinement(
        self,
        query: str,
        candidates: List[Dict],
        final_top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        Apply hierarchical refinement to retrieved candidates.

        Pipeline:
        1. Metadata Filtering (if enabled)
        2. Cross-Encoder Reranking (if enabled)

        Args:
            query: Original query
            candidates: Retrieved candidates with bi-encoder scores
            final_top_k: Final number to return

        Returns:
            Refined list of candidates
        """
        refined = candidates

        # Cross-Encoder Reranking
        if self.enable_cross_encoder and self.cross_encoder:
            refined = self.cross_encoder.rerank(
                query=query,
                candidates=refined,
                top_k=final_top_k
            )
            print(f"   Cross-Encoder: {len(refined)} candidates (reranked)")
        else:
            # Just take top-k if no cross-encoder
            refined = refined[:final_top_k]

        return refined

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Pure vector similarity search."""
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True
        ).astype('float32')

        # Search index
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = {
                    **self.documents[idx],
                    "similarity_score": float(score)
                }
                results.append(result)

        return results

    def _hybrid_search(self, query: str, top_k: int, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining vector similarity and BM25 keyword matching.

        Args:
            query: Query string
            top_k: Number of results
            alpha: Weight for vector search (1-alpha for BM25)
                   0.7 = 70% vector, 30% keyword

        Returns:
            List of documents with combined scores
        """
        # 1. Vector search (get more candidates for reranking)
        candidate_k = min(top_k * 3, len(self.documents))
        vector_results = self._vector_search(query, candidate_k)

        # 2. BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 3. Combine scores
        combined_scores = {}
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        # Add vector scores
        for result in vector_results:
            doc_id = result["id"]
            # Normalize vector score (already 0-1 from cosine)
            vector_score = result["similarity_score"]
            combined_scores[doc_id] = alpha * vector_score

        # Add BM25 scores
        for idx, bm25_score in enumerate(bm25_scores):
            doc_id = self.documents[idx]["id"]
            # Normalize BM25 score (0-1)
            normalized_bm25 = bm25_score / max_bm25
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * normalized_bm25
            else:
                combined_scores[doc_id] = (1 - alpha) * normalized_bm25

        # 4. Sort by combined score and get top-k
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]

        # 5. Build final results
        id_to_doc = {doc["id"]: doc for doc in self.documents}
        results = []
        for doc_id in sorted_ids:
            if doc_id in id_to_doc:
                result = {
                    **id_to_doc[doc_id],
                    "similarity_score": float(combined_scores[doc_id]),
                    "search_method": "hybrid"
                }
                results.append(result)

        return results

    def retrieve_with_reformulation(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        confidence_threshold: float = RETRIEVAL_CONFIDENCE_THRESHOLD,
        max_attempts: int = MAX_REFORMULATION_ATTEMPTS,
        reformulator=None
    ) -> Dict:
        """
        Smart retrieval with automatic query reformulation on low confidence.

        Flow:
        1. Hybrid Search (dense vector + BM25) → get initial candidates
        2. Check RETRIEVAL CONFIDENCE on hybrid scores
        3. If retrieval confidence < threshold → Reformulate → Retry step 1
        4. If retrieval confidence >= threshold → Apply hierarchical refinement:
           - Metadata Filtering
           - Cross-Encoder Reranking

        Args:
            query: Natural language query
            top_k: Final number of results to return
            confidence_threshold: Minimum retrieval confidence required (0.65)
            max_attempts: Maximum reformulation attempts
            reformulator: QueryReformulator instance (optional)

        Returns:
            Dict with:
                - results: Retrieved documents (after hierarchical refinement)
                - confidence: Retrieval confidence score (hybrid)
                - reformulated_query: Final query used (if reformulated)
                - attempts: Number of attempts made
                - success: Whether threshold was met
        """
        current_query = query
        best_results = None
        best_confidence = 0.0
        best_query = query
        best_attempt = 1

        for attempt in range(1, max_attempts + 1):
            # Step 1: Hybrid Search (dense vector + BM25) - get MORE candidates for filtering
            print(f"\n🔍 Retrieval Attempt {attempt}:")
            print(f"   Query: {current_query}")

            # Get broader set for hierarchical refinement
            search_top_k = HYBRID_SEARCH_TOP_K if (self.enable_cross_encoder) else top_k
            results = self.retrieve(current_query, top_k=search_top_k)

            if not results:
                print(f"   ⚠️ No results found")
                if attempt < max_attempts and reformulator:
                    current_query = reformulator.reformulate(query, attempt=attempt)
                    continue
                else:
                    # Return best so far or empty
                    if best_results:
                        print(f"\n✓ Using best attempt ({best_attempt}) with confidence {best_confidence:.3f}")
                        return {
                            "results": best_results,
                            "confidence": best_confidence,
                            "reformulated_query": best_query if best_query != query else None,
                            "attempts": attempt,
                            "success": best_confidence >= confidence_threshold
                        }
                    else:
                        return {
                            "results": [],
                            "confidence": 0.0,
                            "reformulated_query": current_query,
                            "attempts": attempt,
                            "success": False
                        }

            # Step 2: Calculate RETRIEVAL CONFIDENCE from hybrid scores
            avg_confidence = sum(r["similarity_score"] for r in results) / len(results)
            print(f"   Retrieval Confidence: {avg_confidence:.3f} (threshold: {confidence_threshold})")

            # Track best result
            if avg_confidence > best_confidence:
                best_results = results
                best_confidence = avg_confidence
                best_query = current_query
                best_attempt = attempt

            # Step 3: Check if confidence meets threshold
            if avg_confidence >= confidence_threshold:
                print(f"   ✓ Confidence threshold met!")

                # Step 4: Apply hierarchical refinement (Metadata + Cross-Encoder)
                if self.enable_cross_encoder:
                    print(f"\n   Applying hierarchical refinement:")
                    refined_results = self.retrieve_with_hierarchical_refinement(
                        query=current_query,
                        candidates=results,
                        final_top_k=top_k
                    )
                else:
                    refined_results = results[:top_k]

                # Return successful result
                return {
                    "results": refined_results,
                    "confidence": avg_confidence,  # Retrieval confidence (hybrid)
                    "reformulated_query": current_query if current_query != query else None,
                    "attempts": attempt,
                    "success": True
                }

            # Step 5: Confidence too low - try reformulation
            print(f"   ✗ Confidence below threshold")
            if attempt < max_attempts and reformulator:
                print(f"   🔄 Reformulating query...")
                current_query = reformulator.reformulate(query, attempt=attempt)
            else:
                # Max attempts reached - return best result with hierarchical refinement
                print(f"\n⚠️  Max attempts reached. Using best result (confidence: {best_confidence:.3f})")

                if self.enable_cross_encoder:
                    print(f"   Applying hierarchical refinement to best result:")
                    refined_results = self.retrieve_with_hierarchical_refinement(
                        query=best_query,
                        candidates=best_results,
                        final_top_k=top_k
                    )
                else:
                    refined_results = best_results[:top_k]

                return {
                    "results": refined_results,
                    "confidence": best_confidence,
                    "reformulated_query": best_query if best_query != query else None,
                    "attempts": attempt,
                    "success": False
                }

    def retrieve_by_source(
        self,
        query: str,
        source: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        Retrieve examples from a specific dataset source.

        Args:
            query: Natural language query
            source: Dataset source (apps, codesearchnet, humaneval, github_code)
            top_k: Number of results to return

        Returns:
            List of documents from specified source
        """
        # Get more results initially to filter by source
        all_results = self.retrieve(query, top_k=top_k * 5)

        # Filter by source
        filtered = [r for r in all_results if r["source"] == source]

        # Return top-k from filtered results
        return filtered[:top_k]

    def get_statistics(self) -> Dict:
        """
        Get statistics about the retriever's document collection.

        Returns:
            Dictionary with statistics
        """
        sources = {}
        languages = {}

        for doc in self.documents:
            source = doc.get("source", "unknown")
            language = doc.get("language", "unknown")

            sources[source] = sources.get(source, 0) + 1
            languages[language] = languages.get(language, 0) + 1

        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "sources": sources,
            "languages": languages
        }

    def print_statistics(self):
        """Print retriever statistics."""
        stats = self.get_statistics()

        print("=" * 60)
        print("RETRIEVER STATISTICS")
        print("=" * 60)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Index Size: {stats['index_size']}")
        print("\nDocuments by Source:")
        for source, count in stats["sources"].items():
            print(f"  - {source}: {count}")
        print("\nDocuments by Language:")
        for lang, count in stats["languages"].items():
            print(f"  - {lang}: {count}")
        print("=" * 60)


def demo_retrieval():
    """
    Demonstration of retriever functionality.
    """
    print("\n" + "=" * 60)
    print("RETRIEVER DEMO")
    print("=" * 60)

    # Initialize retriever
    retriever = Retriever()

    # Print statistics
    retriever.print_statistics()

    # Test queries
    test_queries = [
        "sort a list of dictionaries by a specific key",
        "read a CSV file and process data",
        "implement binary search algorithm",
        "convert string to uppercase"
    ]

    for query in test_queries:
        print("\n" + "-" * 60)
        print(f"Query: '{query}'")
        print("-" * 60)

        results = retriever.retrieve(query, top_k=3)

        for rank, result in enumerate(results, 1):
            print(f"\n{rank}. Score: {result['similarity_score']:.4f}")
            print(f"   Source: {result['source']}")
            print(f"   ID: {result['id']}")
            print(f"   Query: {result['query'][:80]}...")

            code_preview = result['code'][:150].replace('\n', ' ')
            print(f"   Code: {code_preview}...")


if __name__ == "__main__":
    demo_retrieval()
