#!/usr/bin/env python3
# generation/pipeline.py
"""
Complete RAG Pipeline for Code Generation.

Orchestrates the full flow:
1. Query understanding
2. Example retrieval from vector DB
3. Prompt construction
4. LLM generation (multiple candidates)
5. Candidate reranking
6. Best code selection
"""

import sys
import os
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectordb.retriever import Retriever
from retrieval.query_reformulator import QueryReformulator
from generation.prompt_builder import build_prompt, extract_code_from_response
from generation.llm_generator import LLMGenerator
# Removed: from generation.reranker import CandidateReranker (no longer needed)
from validation.validator import CodeValidator
from feedback.feedback_enrichment import FeedbackEnricher
from evaluation.metrics_tracker import MetricsTracker
from config import (
    FAISS_INDEX_PATH,
    DOCUMENTS_PATH,
    LLM_MODEL_NAME,
    NUM_CANDIDATES,
    DEFAULT_TOP_K,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    MAX_PIPELINE_ATTEMPTS,
    MAX_REFORMULATION_ATTEMPTS
)


class RAGPipeline:
    """
    Complete RAG pipeline for hallucination-mitigated code generation.
    """

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        docs_path: str = DOCUMENTS_PATH,
        model_name: str = LLM_MODEL_NAME
    ):
        """
        Initialize RAG pipeline with all components.

        Args:
            index_path: Path to FAISS index
            docs_path: Path to documents JSON
            model_name: LLM model name in SAP AI Core
        """
        print("=" * 70)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 70)

        # Initialize retriever
        print("\n📚 Loading vector database...")
        self.retriever = Retriever(index_path, docs_path)
        print(f"✓ Loaded {len(self.retriever.documents)} documents")

        # Initialize query reformulator
        print("\n🔄 Initializing query reformulator...")
        self.reformulator = QueryReformulator(model_name)
        print("✓ Reformulator ready")

        # Initialize LLM generator
        print("\n🤖 Connecting to SAP AI Core...")
        self.generator = LLMGenerator(model_name)

        # Removed: Initialize reranker (no longer needed with cross-encoder)
        # With hierarchical retrieval (cross-encoder), we get high-quality examples
        # LLM generates 1 code directly instead of multiple candidates

        # Initialize validator
        print("\n✅ Initializing validation layers...")
        self.validator = CodeValidator()
        print("✓ Validator ready")

        # Initialize feedback enricher
        print("\n🔄 Initializing feedback enrichment...")
        self.feedback_enricher = FeedbackEnricher()
        print("✓ Feedback enricher ready")

        # Initialize metrics tracker
        print("\n📊 Initializing metrics tracker...")
        self.metrics_tracker = MetricsTracker()
        print("✓ Metrics tracker ready")

        print("\n✓ Pipeline ready!")
        print("=" * 70)

    def generate_code(
        self,
        user_query: str,
        top_k: int = DEFAULT_TOP_K,
        num_candidates: int = NUM_CANDIDATES,
        max_attempts: int = MAX_PIPELINE_ATTEMPTS,
        error_feedback: Optional[str] = None,
        test_cases: list = None
    ) -> Dict:
        """
        Generate code using RAG pipeline.

        Args:
            user_query: Natural language description of desired code
            top_k: Number of examples to retrieve
            num_candidates: Number of candidates to generate
            max_attempts: Maximum retry attempts
            error_feedback: Error message from previous attempt (for retry)
            test_cases: Optional test case assertions for validation

        Returns:
            Dict with:
                - best_code: Generated code
                - retrieved_examples: Examples used
                - confidence: Retrieval confidence
        """
        print("\n" + "=" * 70)
        print(f"GENERATING CODE (Attempt {max_attempts - MAX_PIPELINE_ATTEMPTS + 1})")
        print("=" * 70)
        print(f"\n📝 Query: {user_query}")

        # Step 1: Smart retrieval with hierarchical retrieval pipeline
        print(f"\n🔍 Hierarchical retrieval (Hybrid → Metadata → Cross-Encoder)...")
        retrieval_result = self.retriever.retrieve_with_reformulation(
            query=user_query,
            top_k=top_k,
            confidence_threshold=RETRIEVAL_CONFIDENCE_THRESHOLD,
            max_attempts=MAX_REFORMULATION_ATTEMPTS,
            reformulator=self.reformulator
        )

        results = retrieval_result["results"]
        avg_confidence = retrieval_result["confidence"]
        reformulated_query = retrieval_result.get("reformulated_query")
        retrieval_attempts = retrieval_result["attempts"]

        if not results:
            print("✗ No examples retrieved even after reformulation!")
            return {
                "best_code": None,
                "retrieved_examples": [],
                "confidence": 0.0,
                "reformulated_query": reformulated_query,
                "retrieval_attempts": retrieval_attempts
            }

        # Show reformulation results
        if reformulated_query:
            print(f"\n✓ Query was reformulated after {retrieval_attempts} attempt(s)")
            print(f"   Final query: {reformulated_query}")
        else:
            print(f"✓ Original query worked (confidence: {avg_confidence:.3f})")

        print(f"✓ Retrieved {len(results)} examples")
        print(f"   Final confidence: {avg_confidence:.3f}")

        # Step 2: Build prompt with retrieved examples
        print("\n📝 Building prompt with examples...")
        attempt = MAX_PIPELINE_ATTEMPTS - max_attempts + 1
        system_prompt, messages = build_prompt(
            user_query=user_query,
            retrieved_examples=results,
            max_examples=5,
            attempt=attempt,
            error_feedback=error_feedback,
            test_cases=test_cases
        )
        print(f"✓ Prompt built with {min(5, len(results))} examples")

        # Step 3: Generate code (n=1, rely on high-quality retrieval)
        print(f"\n🤖 Generating code with {self.generator.model_name}...")
        print(f"   (n={num_candidates} - relying on hierarchical retrieval for quality)")
        try:
            raw_candidates = self.generator.generate_candidates(
                messages=messages,
                n=num_candidates
            )

            # Extract clean code from response(s)
            if num_candidates == 1:
                generated_code = extract_code_from_response(raw_candidates[0])
                print(f"✓ Generated code ({len(generated_code)} chars)")
            else:
                # Multiple candidates (legacy support)
                candidates = [extract_code_from_response(c) for c in raw_candidates]
                generated_code = candidates[0]  # Take first
                print(f"✓ Generated {len(candidates)} candidates, using first")

        except Exception as e:
            print(f"✗ Error generating code: {e}")
            return {
                "best_code": None,
                "retrieved_examples": results,
                "confidence": avg_confidence,
                "error": str(e)
            }

        # Step 4: Validate generated code through 4 validation layers
        print("\n🔬 Running 4 validation layers...")
        validation_result = self.validator.validate(
            code=generated_code,
            retrieved_examples=results,
            test_input=None,  # TODO: Extract from examples if available
            expected_output=None,
            test_cases=test_cases  # Pass test cases for MBPP evaluation
        )

        return {
            "best_code": generated_code,
            "retrieved_examples": results,
            "confidence": avg_confidence,
            "reformulated_query": reformulated_query,
            "retrieval_attempts": retrieval_attempts,
            "validation": validation_result
        }

    def generate_with_retry(
        self,
        user_query: str,
        top_k: int = DEFAULT_TOP_K,
        num_candidates: int = NUM_CANDIDATES,
        test_cases: list = None
    ) -> Dict:
        """
        Generate code with automatic retry using validation-based feedback and metrics tracking.

        Args:
            user_query: Natural language description
            top_k: Number of examples to retrieve
            num_candidates: Number of candidates per attempt
            test_cases: Optional list of test case assertions for validation

        Returns:
            Final generation result with metrics
        """
        # ⭐ START METRICS TRACKING
        self.metrics_tracker.start_run(user_query)

        original_query = user_query
        current_query = user_query
        error_feedback = None
        feedback_data = None

        for attempt in range(MAX_PIPELINE_ATTEMPTS):
            print(f"\n{'#' * 70}")
            print(f"# ATTEMPT {attempt + 1}/{MAX_PIPELINE_ATTEMPTS}")
            print(f"{'#' * 70}")

            # Generate code
            result = self.generate_code(
                user_query=current_query,
                top_k=top_k,
                num_candidates=num_candidates,
                max_attempts=MAX_PIPELINE_ATTEMPTS - attempt,
                error_feedback=error_feedback,
                test_cases=test_cases  # Pass test cases
            )

            validation = result.get("validation", {})
            best_code = result.get("best_code")

            # ⭐ RECORD ATTEMPT METRICS
            self.metrics_tracker.record_attempt(
                attempt=attempt + 1,
                retrieval_confidence=result.get("confidence", 0),
                reformulated=result.get("reformulated_query") is not None,
                targeted_query=current_query if current_query != original_query else None,
                rerank_score=0.0,  # No candidate reranking anymore
                validation_result=validation,
                num_candidates=num_candidates,
                feedback_triggered=(attempt > 0 and feedback_data is not None)
            )

            # Check if validation passed
            if validation.get("passed", False):
                print("\n✅ SUCCESS - All validation layers passed!")

                # ⭐ FINALIZE METRICS
                metrics = self.metrics_tracker.finalize(success=True, best_code=best_code)
                self.metrics_tracker.print_summary()

                return {**result, "metrics": metrics}

            # Validation failed
            failed_layers = validation.get("failed_layers", [])
            print(f"\n❌ Validation failed - {len(failed_layers)}/4 layers failed: {', '.join(failed_layers)}")

            # If more attempts remain, use feedback to improve
            if attempt < MAX_PIPELINE_ATTEMPTS - 1:
                print(f"\n🔄 Building targeted feedback for retry...")

                # ⭐ BUILD FEEDBACK FROM VALIDATION ERRORS
                feedback_data = self.feedback_enricher.build_feedback(
                    original_query=original_query,
                    validation_result=validation,
                    previous_code=best_code,
                    attempt=attempt + 1
                )

                # Update query and feedback for retry
                current_query = feedback_data["targeted_query"]
                error_feedback = feedback_data  # Pass entire feedback dict, not just error_context string

                print(f"   Feedback Type: {feedback_data['feedback_type']}")
                print(f"   Targeted Query: {current_query}")

                if feedback_data.get("hallucinated_apis"):
                    print(f"   Hallucinated APIs: {', '.join(feedback_data['hallucinated_apis'])}")

            else:
                print(f"\n⚠️  Max attempts reached ({MAX_PIPELINE_ATTEMPTS})")

        # All attempts failed - return best attempt with metrics
        print("\n⚠️  Final code did not pass all validation layers")

        # ⭐ FINALIZE METRICS (failed)
        metrics = self.metrics_tracker.finalize(success=False, best_code=best_code)
        self.metrics_tracker.print_summary()

        return {**result, "metrics": metrics}


if __name__ == "__main__":
    # Test RAG pipeline
    print("\n" + "=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)

    try:
        # Initialize pipeline
        pipeline = RAGPipeline()

        # Test queries
        test_queries = [
            "Write a Python function to merge two sorted lists into one sorted list",
            "Create a function to find the longest common substring between two strings",
            "Implement a function to check if a string is a valid palindrome"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'=' * 70}")
            print(f"TEST {i}/{len(test_queries)}")
            print(f"{'=' * 70}")

            # Generate code
            result = pipeline.generate_with_retry(
                user_query=query,
                top_k=8,
                num_candidates=1  # Generate 1 code directly with high-quality retrieval
            )

            # Display result
            if result["best_code"]:
                print(f"\n{'=' * 70}")
                print("FINAL RESULT:")
                print(f"{'=' * 70}")
                print(f"\n{result['best_code']}")

                print(f"\n{'=' * 70}")
                print("QUALITY SCORES:")
                print(f"{'=' * 70}")
                print(f"Retrieval Confidence: {result['confidence']:.3f}")

                # Show validation results
                validation = result.get("validation", {})
                if validation:
                    print(f"\n{'-' * 70}")
                    print("VALIDATION RESULTS:")
                    print(f"{'-' * 70}")
                    if validation["passed"]:
                        print("   ✅ ALL 4 LAYERS PASSED")
                    else:
                        print(f"   ❌ {len(validation['failed_layers'])}/4 LAYERS FAILED")
                        print(f"   Failed: {', '.join(validation['failed_layers'])}")
                        print(f"   Summary: {validation['error_summary']}")

                # Show metrics
                metrics = result.get("metrics", {})
                if metrics:
                    print(f"\n{'-' * 70}")
                    print("METRICS:")
                    print(f"{'-' * 70}")
                    print(f"Success: {'✓' if metrics['success'] else '✗'}")
                    print(f"Attempts: {metrics['total_attempts']}")
                    print(f"Pass@1: {'✓' if metrics['pass@1'] else '✗'}")
                    print(f"Pass@{metrics['total_attempts']}: {'✓' if metrics['success'] else '✗'}")

                    if metrics['hallucination_detected']:
                        print(f"\nHallucination:")
                        print(f"  Detected: Yes")
                        print(f"  Corrected: {'Yes' if metrics['hallucination_corrected'] else 'No'}")
                        if metrics['hallucinated_apis_found']:
                            print(f"  APIs: {', '.join(metrics['hallucinated_apis_found'])}")

                    if metrics['feedback_used']:
                        print(f"\nFeedback Loop:")
                        print(f"  Used: Yes")
                        print(f"  Improved Retrieval: {metrics['feedback_improved_retrieval']}")
                        print(f"  Improved Validation: {metrics['feedback_improved_validation']}")

                    print(f"\nPerformance:")
                    print(f"  Total Time: {metrics['total_time_seconds']}s")

                # Show reformulation info
                if result.get("reformulated_query"):
                    print(f"\n{'-' * 70}")
                    print("QUERY REFORMULATION:")
                    print(f"{'-' * 70}")
                    print(f"  Attempts: {result['retrieval_attempts']}")
                    print(f"  Original: {query}")
                    print(f"  Final: {result['reformulated_query']}")
            else:
                print("\n✗ Failed to generate code")

            print(f"\n{'=' * 70}\n")

    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
