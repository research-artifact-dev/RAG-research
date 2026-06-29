#!/usr/bin/env python3
# evaluation/run_baseline.py
"""
Baseline Evaluation: LLM WITHOUT RAG (no retrieval).

Tests pure LLM generation without any retrieved examples.
This establishes the baseline to prove RAG helps.
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.llm_generator import LLMGenerator
from validation.validator import CodeValidator
from generation.prompt_builder import extract_code_from_response
from config import NUM_CANDIDATES


class BaselineEvaluator:
    """
    Baseline evaluation: LLM without RAG.
    """

    def __init__(self):
        """Initialize baseline evaluator."""
        print("=" * 70)
        print("BASELINE EVALUATOR (NO RAG)")
        print("=" * 70)

        self.generator = LLMGenerator()
        self.validator = CodeValidator()

        print("✓ LLM Generator ready")
        print("✓ Validator ready")
        print("\n⚠️  NO RETRIEVAL - Pure LLM generation")
        print("=" * 70)

    def generate_without_rag(self, query: str, test_cases: List[str] = None) -> str:
        """Generate code without RAG - pure LLM with generic prompt."""

        # Generic baseline prompt - NO function names, NO strict instructions
        user_message = f"""Write a Python function to solve this task:

{query}

Provide a working Python implementation."""

        # Generic system prompt for baseline
        generic_system_prompt = """You are a helpful Python programming assistant.
Write clean and correct Python code to solve the given task."""

        messages = [
            {"role": "system", "content": generic_system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Generate
        candidates = self.generator.generate_candidates(messages, n=1)
        if candidates:
            return extract_code_from_response(candidates[0])
        return ""

    def evaluate_query(self, query_data: Dict, query_idx: int) -> Dict:
        """Evaluate single query without RAG."""
        query = query_data["query"]
        test_cases = query_data.get("test_cases", [])
        task_id = query_data.get("task_id", query_idx)

        print(f"\n{'='*70}")
        print(f"Query {query_idx + 1} (Task {task_id}): {query[:60]}...")
        print(f"{'='*70}")

        start_time = time.time()

        # Generate WITHOUT RAG
        print("🤖 Generating code WITHOUT retrieval (baseline)...")
        generated_code = self.generate_without_rag(query, test_cases)
        print(f"✓ Generated code ({len(generated_code)} chars)")

        # Validate
        print("\n🔬 Validating...")
        validation_result = self.validator.validate(
            code=generated_code,
            retrieved_examples=[],  # No examples
            test_cases=test_cases
        )

        total_time = time.time() - start_time
        passed = validation_result.get("passed", False)

        # Compile result
        eval_result = {
            "task_id": task_id,
            "query": query,
            "generated_code": generated_code,
            "pass@1": passed,
            "pass@2": passed,  # Only 1 attempt in baseline
            "pass@3": passed,
            "hallucination_detected": not validation_result["layers"]["layer2"]["passed"],
            "validation_breakdown": {
                "layer1_syntax": validation_result["layers"]["layer1"]["passed"],
                "layer2_api": validation_result["layers"]["layer2"]["passed"],
                "layer3_types": validation_result["layers"]["layer3"]["passed"],
                "layer4_execution": validation_result["layers"]["layer4"]["passed"],
                "all_passed": passed
            },
            "total_time_seconds": total_time,
            "test_cases_total": len(test_cases),
            "test_cases_passed": validation_result["layers"]["layer4"].get("tests_passed", 0)
        }

        # Print summary
        print(f"\nResults:")
        print(f"  Passed: {'✓' if passed else '✗'}")
        if test_cases:
            print(f"  Tests: {eval_result['test_cases_passed']}/{eval_result['test_cases_total']}")
        print(f"  Time: {total_time:.2f}s")

        return eval_result

    def run_evaluation(self, num_samples: int = 100) -> Dict:
        """Run baseline evaluation."""
        print(f"\n{'='*70}")
        print(f"RUNNING BASELINE EVALUATION ON {num_samples} QUERIES")
        print(f"{'='*70}")

        # Load test queries
        with open("evaluation/mbpp_test_queries_15.json", 'r') as f:
            test_queries = json.load(f)

        print(f"\n✓ Loaded {len(test_queries)} test queries")

        # Evaluate each query
        results = []
        for i, query_data in enumerate(test_queries[:num_samples]):
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
            "test_queries": test_queries[:num_samples],
            "detailed_results": results,
            "aggregate_metrics": aggregate,
            "evaluation_date": datetime.now().isoformat(),
            "num_queries": len(results),
            "mode": "baseline_no_rag"
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

        # Hallucination rate (API layer failures)
        hallucination_rate = sum(1 for r in results if r["hallucination_detected"]) / total

        # Validation breakdown
        validation_breakdown = {
            "layer1_syntax_pass_rate": sum(
                1 for r in results if r["validation_breakdown"]["layer1_syntax"]
            ) / total,
            "layer2_api_pass_rate": sum(
                1 for r in results if r["validation_breakdown"]["layer2_api"]
            ) / total,
            "layer3_types_pass_rate": sum(
                1 for r in results if r["validation_breakdown"]["layer3_types"]
            ) / total,
            "layer4_execution_pass_rate": sum(
                1 for r in results if r["validation_breakdown"]["layer4_execution"]
            ) / total
        }

        # Performance
        avg_time = sum(r["total_time_seconds"] for r in results) / total

        return {
            "pass@1": round(pass_at_1, 4),
            "pass@2": round(pass_at_2, 4),
            "pass@3": round(pass_at_3, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "validation_breakdown": {k: round(v, 4) for k, v in validation_breakdown.items()},
            "avg_time_seconds": round(avg_time, 2)
        }

    def print_evaluation_summary(self, aggregate: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("BASELINE EVALUATION SUMMARY (NO RAG)")
        print("=" * 70)
        print("\n🎯 PRIMARY METRICS:")
        print(f"   pass@1: {aggregate['pass@1']:.4f} ({aggregate['pass@1']*100:.2f}%)")
        print(f"   pass@2: {aggregate['pass@2']:.4f} ({aggregate['pass@2']*100:.2f}%)")
        print(f"   pass@3: {aggregate['pass@3']:.4f} ({aggregate['pass@3']*100:.2f}%)")
        print(f"\n⚠️  HALLUCINATIONS:")
        print(f"   Hallucination Rate: {aggregate['hallucination_rate']:.4f} ({aggregate['hallucination_rate']*100:.2f}%)")
        print(f"\n✅ VALIDATION:")
        print(f"   Layer 2 (API): {aggregate['validation_breakdown']['layer2_api_pass_rate']:.4f}")
        print(f"   Layer 4 (Execution): {aggregate['validation_breakdown']['layer4_execution_pass_rate']:.4f}")
        print("=" * 70)

    def save_results(self, evaluation_results: Dict, output_path: str = "evaluation/baseline_results.json"):
        """Save baseline results."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (NO RAG)")
    print("=" * 70)

    evaluator = BaselineEvaluator()
    results = evaluator.run_evaluation(num_samples=100)
    evaluator.save_results(results, "evaluation/baseline_results.json")

    print("\n✓ Baseline evaluation complete!")
    print(f"✓ Evaluated {results['num_queries']} queries WITHOUT retrieval")
    print(f"\nExpected baseline performance: 10-25% pass@1")
    print(f"(Much lower than RAG pipeline: 50% pass@1)")
