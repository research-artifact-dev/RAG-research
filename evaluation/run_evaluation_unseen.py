#!/usr/bin/env python3
# evaluation/run_evaluation_unseen.py
"""
Run evaluation on UNSEEN 15% test set.
Tests generalization to problems NOT in vector DB.
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.pipeline import RAGPipeline
from evaluation.rag_metrics import compute_all_rag_metrics
from config import NUM_CANDIDATES


class UnseenEvaluator:
    """
    Evaluate RAG pipeline on UNSEEN test set (15%).
    """

    def __init__(self):
        """Initialize evaluator with pipeline."""
        print("=" * 70)
        print("UNSEEN TEST SET EVALUATOR (15% HELD-OUT)")
        print("=" * 70)

        # Initialize pipeline
        self.pipeline = RAGPipeline()

        print(f"\n✓ Pipeline ready with {len(self.pipeline.retriever.documents)} documents")

        # Verify MBPP split
        mbpp_docs = [doc for doc in self.pipeline.retriever.documents
                     if doc.get('source') == 'mbpp_train85']
        print(f"✓ MBPP in DB: {len(mbpp_docs)} (85% for retrieval)")
        print("=" * 70)

    def load_test_queries(self) -> List[Dict]:
        """Load UNSEEN 15% test queries."""
        with open("evaluation/mbpp_test_queries_15.json", 'r') as f:
            queries = json.load(f)
        return queries

    def evaluate_query(self, query_data: Dict, query_idx: int) -> Dict:
        """Evaluate pipeline on single UNSEEN query."""
        query = query_data["query"]
        test_cases = query_data.get("test_cases", [])
        task_id = query_data.get("task_id", query_idx)

        print(f"\n{'='*70}")
        print(f"Query {query_idx + 1} (Task {task_id}): {query[:60]}...")
        print(f"   UNSEEN - NOT in vector DB")
        if test_cases:
            print(f"   Test cases: {len(test_cases)}")
        print(f"{'='*70}")

        # Run pipeline
        result = self.pipeline.generate_with_retry(
            user_query=query,
            top_k=8,
            num_candidates=NUM_CANDIDATES,
            test_cases=test_cases
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
            ground_truth=None
        )

        # Compile result
        eval_result = {
            "task_id": task_id,
            "query": query,
            "generated_code": generated_code,
            "pass@1": metrics.get("pass@1", False),
            "pass@2": metrics.get("pass@2", False),
            "pass@3": metrics.get("pass@3", False),
            "hallucination_detected": metrics.get("hallucination_detected", False),
            "hallucination_corrected": metrics.get("hallucination_corrected", False),
            "hallucinated_apis": metrics.get("hallucinated_apis_found", []),
            "faithfulness": rag_metrics["faithfulness"],
            "context_precision": rag_metrics["context_precision"],
            "context_recall": rag_metrics["context_recall"],
            "retrieval_confidence": result.get("confidence", 0),
            "total_attempts": metrics.get("total_attempts", 0),
            "validation_breakdown": metrics.get("validation_breakdown", {}),
            "feedback_used": metrics.get("feedback_used", False),
            "total_time_seconds": metrics.get("total_time_seconds", 0),
            "num_retrieved_examples": len(retrieved_examples),
            "test_cases_total": len(test_cases) if test_cases else 0,
            "test_cases_passed": validation.get("tests_passed", 0) if test_cases else None
        }

        # Print summary
        print(f"\nResults:")
        print(f"  pass@1: {'✓' if eval_result['pass@1'] else '✗'}")
        if test_cases:
            print(f"  Tests: {eval_result['test_cases_passed']}/{eval_result['test_cases_total']} passed")
        print(f"  Faithfulness: {eval_result['faithfulness']:.3f}")
        print(f"  Attempts: {eval_result['total_attempts']}")

        return eval_result

    def run_evaluation(self) -> Dict:
        """Run evaluation on ALL unseen queries."""
        print(f"\n{'='*70}")
        print(f"RUNNING UNSEEN TEST SET EVALUATION")
        print(f"{'='*70}")

        # Load test queries
        test_queries = self.load_test_queries()
        print(f"\n✓ Loaded {len(test_queries)} UNSEEN test queries")

        # Evaluate each query
        results = []
        for i, query_data in enumerate(test_queries):
            try:
                result = self.evaluate_query(query_data, i)
                results.append(result)
            except Exception as e:
                print(f"\n✗ Error on query {i+1}: {e}")
                import traceback
                traceback.print_exc()
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
            "num_queries": len(results),
            "test_set": "unseen_15_percent"
        }

    def compute_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics."""
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
        avg_faithfulness = sum(r["faithfulness"] for r in results) / total
        avg_context_precision = sum(r["context_precision"] for r in results) / total

        # Validation breakdown
        validation_breakdown = {
            "layer1_syntax_pass_rate": sum(
                1 for r in results if r["validation_breakdown"].get("layer1_syntax", False)
            ) / total,
            "layer2_api_pass_rate": sum(
                1 for r in results if r["validation_breakdown"].get("layer2_api", False)
            ) / total,
            "layer3_types_pass_rate": sum(
                1 for r in results if r["validation_breakdown"].get("layer3_types", False)
            ) / total,
            "layer4_execution_pass_rate": sum(
                1 for r in results if r["validation_breakdown"].get("layer4_execution", False)
            ) / total
        }

        # Performance
        avg_time = sum(r["total_time_seconds"] for r in results) / total
        avg_attempts = sum(r["total_attempts"] for r in results) / total

        return {
            "pass@1": round(pass_at_1, 4),
            "pass@2": round(pass_at_2, 4),
            "pass@3": round(pass_at_3, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "hallucination_correction_rate": round(hallucination_correction_rate, 4),
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_context_precision": round(avg_context_precision, 4),
            "avg_context_recall": None,
            "validation_breakdown": {k: round(v, 4) for k, v in validation_breakdown.items()},
            "avg_time_seconds": round(avg_time, 2),
            "avg_attempts": round(avg_attempts, 2)
        }

    def print_evaluation_summary(self, aggregate: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("UNSEEN TEST SET EVALUATION SUMMARY")
        print("=" * 70)
        print("\n🎯 PRIMARY METRICS:")
        print(f"   pass@1: {aggregate['pass@1']:.4f} ({aggregate['pass@1']*100:.2f}%)")
        print(f"   pass@2: {aggregate['pass@2']:.4f} ({aggregate['pass@2']*100:.2f}%)")
        print(f"   pass@3: {aggregate['pass@3']:.4f} ({aggregate['pass@3']*100:.2f}%)")
        print(f"\n📊 RAG METRICS:")
        print(f"   Faithfulness: {aggregate['avg_faithfulness']:.4f}")
        print(f"   Context Precision: {aggregate['avg_context_precision']:.4f}")
        print(f"\n✅ VALIDATION:")
        print(f"   Layer 4 (Execution): {aggregate['validation_breakdown']['layer4_execution_pass_rate']:.4f}")
        print("=" * 70)

    def save_results(self, evaluation_results: Dict, output_path: str = "evaluation/unseen_results.json"):
        """Save evaluation results."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("UNSEEN TEST SET EVALUATION (15% HELD-OUT)")
    print("=" * 70)

    evaluator = UnseenEvaluator()
    results = evaluator.run_evaluation()
    evaluator.save_results(results, "evaluation/unseen_results.json")

    print("\n✓ Evaluation complete!")
    print(f"✓ Evaluated {results['num_queries']} UNSEEN queries")
    print(f"✓ Expected: 40-50% pass@1 (generalization to unseen)")
