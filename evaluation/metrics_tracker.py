#!/usr/bin/env python3
# evaluation/metrics_tracker.py
"""
Metrics Tracker for RAG Pipeline.

Tracks and computes metrics during code generation:
- Functional correctness (pass@k)
- Hallucination detection
- Validation breakdown
- Performance metrics
"""

from typing import Dict, List, Optional
import time


class MetricsTracker:
    """
    Track metrics during pipeline execution.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.start_time = None
        self.query = None
        self.attempts = []
        self.validation_results = []
        self.hallucinations = []

    def start_run(self, query: str):
        """Start tracking a new run."""
        self.start_time = time.time()
        self.query = query
        self.attempts = []
        self.validation_results = []
        self.hallucinations = []

    def record_attempt(
        self,
        attempt: int,
        retrieval_confidence: float,
        reformulated: bool,
        targeted_query: str,
        rerank_score: float,
        validation_result: Dict,
        num_candidates: int,
        feedback_triggered: bool
    ):
        """Record an attempt with all details."""
        generated_code = validation_result.get("code", "")

        self.attempts.append({
            "attempt": attempt,
            "validation": validation_result,
            "code": generated_code,
            "passed": validation_result.get("passed", False),
            "retrieval_confidence": retrieval_confidence,
            "reformulated": reformulated,
            "rerank_score": rerank_score,
            "num_candidates": num_candidates,
            "feedback_triggered": feedback_triggered
        })

        self.validation_results.append(validation_result)

        # Track hallucinations (API layer failures)
        if not validation_result["layers"]["layer2"]["passed"]:
            self.hallucinations.append({
                "attempt": attempt,
                "apis": validation_result["layers"]["layer2"].get("hallucinated_apis", [])
            })

    def finalize(self, success: bool, best_code: str) -> Dict:
        """
        Finalize and compute metrics.

        Args:
            success: Whether the pipeline succeeded
            best_code: Best generated code

        Returns:
            Dictionary of metrics
        """
        total_time = time.time() - self.start_time if self.start_time else 0
        total_attempts = len(self.attempts)

        # pass@k metrics
        pass_at_1 = self.attempts[0]["passed"] if len(self.attempts) >= 1 else False
        pass_at_2 = any(a["passed"] for a in self.attempts[:2]) if len(self.attempts) >= 2 else pass_at_1
        pass_at_3 = any(a["passed"] for a in self.attempts[:3]) if len(self.attempts) >= 3 else pass_at_2

        # Hallucination metrics
        hallucination_detected = len(self.hallucinations) > 0
        hallucination_corrected = hallucination_detected and (pass_at_2 or pass_at_3)

        # Get final validation
        final_validation = self.validation_results[-1] if self.validation_results else {}

        # Validation breakdown
        validation_breakdown = {
            "layer1_syntax": final_validation.get("layers", {}).get("layer1", {}).get("passed", False),
            "layer2_api": final_validation.get("layers", {}).get("layer2", {}).get("passed", False),
            "layer3_types": final_validation.get("layers", {}).get("layer3", {}).get("passed", False),
            "layer4_execution": final_validation.get("layers", {}).get("layer4", {}).get("passed", False)
        }

        # Get hallucinated APIs if any
        hallucinated_apis_found = []
        if self.hallucinations:
            for h in self.hallucinations:
                hallucinated_apis_found.extend(h.get("apis", []))
            hallucinated_apis_found = list(set(hallucinated_apis_found))  # Unique

        return {
            "pass@1": pass_at_1,
            "pass@2": pass_at_2,
            "pass@3": pass_at_3,
            "hallucination_detected": hallucination_detected,
            "hallucination_corrected": hallucination_corrected,
            "hallucinated_apis_found": hallucinated_apis_found,
            "total_attempts": total_attempts,
            "validation_breakdown": validation_breakdown,
            "feedback_used": total_attempts > 1,
            "total_time_seconds": round(total_time, 2)
        }

    def print_summary(self):
        """Print metrics summary."""
        print("\n" + "=" * 70)
        print("RUN METRICS SUMMARY")
        print("=" * 70)

        if not self.attempts:
            print("\n⚠️  No attempts recorded")
            return

        metrics = self.finalize(success=self.attempts[-1]["passed"], best_code=self.attempts[-1]["code"])

        print("\n🎯 PRIMARY METRICS:")
        print(f"\n   1. Functional Correctness (pass@k):")
        print(f"      Pass@1: {'✓' if metrics['pass@1'] else '✗'}")
        print(f"      Pass@2: {'✓' if metrics['pass@2'] else '✗'}")
        print(f"      Pass@3: {'✓' if metrics['pass@3'] else '✗'}")

        print(f"\n   2. API Correctness (Hallucination Rate):")
        print(f"      Hallucination: {metrics['hallucination_detected']:.4f}")
        print(f"      Detected: {'Yes' if metrics['hallucination_detected'] else 'No'}")

        print(f"\n   3. Validation Breakdown:")
        vb = metrics['validation_breakdown']
        print(f"      Layer 1 (Syntax):    {'✓' if vb['layer1_syntax'] else '✗'}")
        print(f"      Layer 2 (API):       {'✓' if vb['layer2_api'] else '✗'}")
        print(f"      Layer 3 (Types):     {'✓' if vb['layer3_types'] else '✗'}")
        print(f"      Layer 4 (Execution): {'✓' if vb['layer4_execution'] else '✗'}")

        print(f"\n📊 ADDITIONAL INFO:")
        print(f"   Success: {'✓' if self.attempts[-1]['passed'] else '✗'}")
        print(f"   Total Attempts: {metrics['total_attempts']}")
        print(f"   Feedback Used: {'Yes' if metrics['feedback_used'] else 'No'}")
        print(f"   Total Time: {metrics['total_time_seconds']:.2f}s")
        print("=" * 70)

