#!/usr/bin/env python3
# evaluation/run_evaluation.py
"""
Complete Evaluation Script

Runs 100-query evaluation with:
- pass@k metrics
- Hallucination rate
- RAG metrics (Faithfulness, Context Precision, Context Recall)

Generates:
- JSON results file
- PDF report with queries and generated codes
- Aggregate statistics
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.pipeline import RAGPipeline
from evaluation.rag_metrics import compute_all_rag_metrics
from config import NUM_CANDIDATES


class PipelineEvaluator:
    """
    Evaluate RAG pipeline on test queries.
    """

    def __init__(self):
        """Initialize evaluator with pipeline."""
        print("=" * 70)
        print("PIPELINE EVALUATOR")
        print("=" * 70)

        # Initialize pipeline
        self.pipeline = RAGPipeline()

        print("\n✓ Evaluator ready!")
        print("=" * 70)

    def sample_test_queries(self, num_samples: int = 100) -> List[Dict]:
        """
        Load MBPP test queries with test cases from file.

        Args:
            num_samples: Number of queries to use (up to 100)

        Returns:
            List of dicts with query, test_cases, task_id
        """
        print(f"\nLoading MBPP test queries with test cases...")

        # Load MBPP queries from file
        mbpp_queries_path = "evaluation/mbpp_queries.json"

        if os.path.exists(mbpp_queries_path):
            with open(mbpp_queries_path, 'r') as f:
                all_queries = json.load(f)

            print(f"✓ Loaded {len(all_queries)} MBPP queries from file")
            print("   (Programming problems with functional test cases)")

            # Take first num_samples
            sampled = all_queries[:num_samples]
        else:
            print(f"⚠️  MBPP queries file not found!")
            print(f"   Run: python evaluation/load_mbpp.py")
            print(f"   Falling back to OOD queries without test cases")

            # Fallback to OOD queries (no test cases)
            ood_queries_path = "evaluation/ood_queries.json"
            if os.path.exists(ood_queries_path):
                with open(ood_queries_path, 'r') as f:
                    all_queries = json.load(f)
                # Convert to dict format without test cases
                sampled = [{"query": q, "test_cases": [], "task_id": i} for i, q in enumerate(all_queries[:num_samples])]
            else:
                print(f"⚠️  No query files found, using document store")
                all_queries = list(set(doc["query"] for doc in self.pipeline.retriever.documents))
                filtered_queries = [q for q in all_queries if len(q.split()) >= 3]
                sampled = [{"query": q, "test_cases": [], "task_id": i} for i, q in enumerate(random.sample(filtered_queries, min(num_samples, len(filtered_queries))))]

        print(f"✓ Using {len(sampled)} queries for evaluation")

        return sampled

    def evaluate_query(self, query_data: Dict, query_idx: int) -> Dict:
        """
        Evaluate pipeline on single query.

        Args:
            query_data: Dict with query, test_cases, task_id
            query_idx: Query index (for progress tracking)

        Returns:
            Dict with complete evaluation results
        """
        query = query_data["query"]
        test_cases = query_data.get("test_cases", [])
        task_id = query_data.get("task_id", query_idx)

        print(f"\n{'='*70}")
        print(f"Query {query_idx + 1} (Task {task_id}): {query[:60]}...")
        if test_cases:
            print(f"   Test cases: {len(test_cases)}")
        print(f"{'='*70}")

        # Run pipeline with retry (pass test_cases for validation)
        result = self.pipeline.generate_with_retry(
            user_query=query,
            top_k=8,
            num_candidates=NUM_CANDIDATES,
            test_cases=test_cases  # Pass test cases to pipeline
        )

        # Extract key results
        generated_code = result.get("best_code", "")
        retrieved_examples = result.get("retrieved_examples", [])
        metrics = result.get("metrics", {})
        validation = result.get("validation", {})

        # Compute RAG metrics
        rag_metrics = compute_all_rag_metrics(
            generated_code=generated_code,
            retrieved_examples=retrieved_examples,
            ground_truth=None  # No ground truth available
        )

        # Compile complete result
        eval_result = {
            "task_id": task_id,
            "query": query,
            "generated_code": generated_code,

            # Primary metrics (for paper)
            "pass@1": metrics.get("pass@1", False),
            "pass@2": metrics.get("pass@2", False),
            "pass@3": metrics.get("pass@3", False),
            "hallucination_detected": metrics.get("hallucination_detected", False),
            "hallucination_corrected": metrics.get("hallucination_corrected", False),
            "hallucinated_apis": metrics.get("hallucinated_apis_found", []),

            # RAG metrics (for analysis)
            "faithfulness": rag_metrics["faithfulness"],
            "context_precision": rag_metrics["context_precision"],
            "context_recall": rag_metrics["context_recall"],

            # Internal metrics (for debugging)
            "retrieval_confidence": result.get("confidence", 0),
            "total_attempts": metrics.get("total_attempts", 0),
            "validation_breakdown": metrics.get("validation_breakdown", {}),
            "feedback_used": metrics.get("feedback_used", False),
            "total_time_seconds": metrics.get("total_time_seconds", 0),

            # Retrieved examples count
            "num_retrieved_examples": len(retrieved_examples),

            # Test case results (if available)
            "test_cases_total": len(test_cases) if test_cases else 0,
            "test_cases_passed": validation.get("tests_passed", 0) if test_cases else None
        }

        # Print summary
        print(f"\nResults:")
        print(f"  pass@1: {'✓' if eval_result['pass@1'] else '✗'}")
        if test_cases:
            print(f"  Tests: {eval_result['test_cases_passed']}/{eval_result['test_cases_total']} passed")
        print(f"  Hallucination: {'Yes' if eval_result['hallucination_detected'] else 'No'}")
        print(f"  Faithfulness: {eval_result['faithfulness']:.3f}")
        print(f"  Context Precision: {eval_result['context_precision']:.3f}")
        print(f"  Attempts: {eval_result['total_attempts']}")

        return eval_result

    def run_evaluation(self, num_samples: int = 100) -> Dict:
        """
        Run complete evaluation on N queries.

        Args:
            num_samples: Number of queries to evaluate

        Returns:
            Dict with all results and aggregate metrics
        """
        print(f"\n{'='*70}")
        print(f"RUNNING EVALUATION ON {num_samples} QUERIES")
        print(f"{'='*70}")

        # Sample test queries
        test_queries = self.sample_test_queries(num_samples)

        # Evaluate each query
        results = []
        for i, query_data in enumerate(test_queries):
            try:
                result = self.evaluate_query(query_data, i)
                results.append(result)
            except Exception as e:
                print(f"\n✗ Error on query {i+1}: {e}")
                # Continue with next query
                continue

        # Compute aggregate metrics
        aggregate = self.compute_aggregate_metrics(results)

        # Print summary
        self.print_evaluation_summary(aggregate)

        return {
            "test_queries": test_queries,
            "detailed_results": results,
            "aggregate_metrics": aggregate,
            "evaluation_date": datetime.now().isoformat(),
            "num_queries": len(results)
        }

    def compute_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """
        Compute aggregate metrics across all results.

        Args:
            results: List of evaluation results

        Returns:
            Dict with aggregate metrics
        """
        if not results:
            return {}

        total = len(results)

        # Primary metrics
        pass_at_1 = sum(1 for r in results if r["pass@1"]) / total
        pass_at_2 = sum(1 for r in results if r["pass@2"]) / total
        pass_at_3 = sum(1 for r in results if r["pass@3"]) / total

        hallucination_rate = sum(1 for r in results if r["hallucination_detected"]) / total

        hallucinations_detected = sum(1 for r in results if r["hallucination_detected"])
        hallucination_correction_rate = (
            sum(1 for r in results if r["hallucination_corrected"]) / hallucinations_detected
            if hallucinations_detected > 0 else 0
        )

        # RAG metrics
        faithfulness_scores = [r["faithfulness"] for r in results]
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)

        precision_scores = [r["context_precision"] for r in results]
        avg_context_precision = sum(precision_scores) / len(precision_scores)

        recall_scores = [r["context_recall"] for r in results if r["context_recall"] is not None]
        avg_context_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None

        # Validation breakdown
        validation_breakdown = {
            "layer1_syntax_pass_rate": sum(
                1 for r in results
                if r["validation_breakdown"].get("layer1_syntax", False)
            ) / total,
            "layer2_api_pass_rate": sum(
                1 for r in results
                if r["validation_breakdown"].get("layer2_api", False)
            ) / total,
            "layer3_types_pass_rate": sum(
                1 for r in results
                if r["validation_breakdown"].get("layer3_types", False)
            ) / total,
            "layer4_execution_pass_rate": sum(
                1 for r in results
                if r["validation_breakdown"].get("layer4_execution", False)
            ) / total
        }

        # Performance metrics
        avg_time = sum(r["total_time_seconds"] for r in results) / total
        avg_attempts = sum(r["total_attempts"] for r in results) / total

        return {
            # Primary metrics (for paper)
            "pass@1": round(pass_at_1, 4),
            "pass@2": round(pass_at_2, 4),
            "pass@3": round(pass_at_3, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "hallucination_correction_rate": round(hallucination_correction_rate, 4),

            # RAG metrics (for analysis)
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_context_precision": round(avg_context_precision, 4),
            "avg_context_recall": round(avg_context_recall, 4) if avg_context_recall else None,

            # Validation breakdown
            "validation_breakdown": {
                k: round(v, 4) for k, v in validation_breakdown.items()
            },

            # Performance
            "avg_time_seconds": round(avg_time, 2),
            "avg_attempts": round(avg_attempts, 2)
        }

    def print_evaluation_summary(self, aggregate: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        print("\n🎯 PRIMARY METRICS (for paper):")
        print(f"   pass@1: {aggregate['pass@1']:.4f} ({aggregate['pass@1']*100:.2f}%)")
        print(f"   pass@2: {aggregate['pass@2']:.4f} ({aggregate['pass@2']*100:.2f}%)")
        print(f"   pass@3: {aggregate['pass@3']:.4f} ({aggregate['pass@3']*100:.2f}%)")
        print(f"   Hallucination Rate: {aggregate['hallucination_rate']:.4f} ({aggregate['hallucination_rate']*100:.2f}%)")
        print(f"   Hallucination Correction: {aggregate['hallucination_correction_rate']:.4f} ({aggregate['hallucination_correction_rate']*100:.2f}%)")

        print("\n📊 RAG METRICS (for analysis):")
        print(f"   Faithfulness: {aggregate['avg_faithfulness']:.4f}")
        print(f"   Context Precision: {aggregate['avg_context_precision']:.4f}")
        if aggregate['avg_context_recall']:
            print(f"   Context Recall: {aggregate['avg_context_recall']:.4f}")

        print("\n✅ VALIDATION BREAKDOWN:")
        breakdown = aggregate['validation_breakdown']
        print(f"   Layer 1 (Syntax):    {breakdown['layer1_syntax_pass_rate']:.4f} ({breakdown['layer1_syntax_pass_rate']*100:.1f}%)")
        print(f"   Layer 2 (API):       {breakdown['layer2_api_pass_rate']:.4f} ({breakdown['layer2_api_pass_rate']*100:.1f}%)")
        print(f"   Layer 3 (Types):     {breakdown['layer3_types_pass_rate']:.4f} ({breakdown['layer3_types_pass_rate']*100:.1f}%)")
        print(f"   Layer 4 (Execution): {breakdown['layer4_execution_pass_rate']:.4f} ({breakdown['layer4_execution_pass_rate']*100:.1f}%)")

        print("\n⚡ PERFORMANCE:")
        print(f"   Avg Time: {aggregate['avg_time_seconds']:.2f}s")
        print(f"   Avg Attempts: {aggregate['avg_attempts']:.2f}")

        print("=" * 70)

    def save_results(self, evaluation_results: Dict, output_path: str = "evaluation/results.json"):
        """Save evaluation results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    # Run evaluation
    print("\n" + "=" * 70)
    print("RAG PIPELINE EVALUATION")
    print("=" * 70)

    # Initialize evaluator
    evaluator = PipelineEvaluator()

    # Run on 100 queries
    results = evaluator.run_evaluation(num_samples=100)

    # Save results
    evaluator.save_results(results, "evaluation/evaluation_results.json")

    print("\n✓ Evaluation complete!")
    print(f"✓ Evaluated {results['num_queries']} queries")
    print(f"✓ Results saved to: evaluation/evaluation_results.json")
