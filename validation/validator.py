#!/usr/bin/env python3
# validation/validator.py
"""
Validation Orchestrator

Runs all 5 validation layers and collects results.
Continues through all layers even if some fail.
"""

import sys
import os
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.layer1_syntax import check_syntax
from validation.layer2_api import check_api_correctness
from validation.layer3_types import check_types
from validation.layer4_execution import check_execution, check_execution_with_test_cases
# Layer 5 removed from implementation


class CodeValidator:
    """
    Orchestrates all 5 validation layers for generated code.
    """

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(
        self,
        code: str,
        retrieved_examples: List[Dict],
        test_input: str = None,
        expected_output: str = None,
        test_cases: list = None
    ) -> Dict:
        """
        Run all 4 validation layers on generated code.

        Args:
            code: Generated Python code
            retrieved_examples: Retrieved examples (for future use)
            test_input: Optional test input (for Layer 4 basic execution)
            expected_output: Optional expected output (for Layer 4)
            test_cases: Optional list of test case assertions (for MBPP evaluation)

        Returns:
            Dict with:
                - passed: bool (True if ALL layers passed)
                - layers: Dict with results from each layer
                - failed_layers: List[str] (names of failed layers)
                - error_summary: str (summary of all errors)
        """
        print("\n" + "=" * 70)
        print("RUNNING VALIDATION LAYERS")
        print("=" * 70)

        # Layer 1: Syntax Check
        print("\n🔍 Layer 1: Syntax Check")
        layer1 = check_syntax(code)
        if layer1["passed"]:
            print("   ✓ PASSED")
        else:
            print(f"   ✗ FAILED - {layer1['error']}")

        # Layer 2: API Correctness (only if syntax passed)
        print("\n🔍 Layer 2: API/Library Correctness")
        if layer1["passed"]:
            layer2 = check_api_correctness(code)
            if layer2["passed"]:
                print("   ✓ PASSED")
            else:
                print(f"   ✗ FAILED - Found {len(layer2['hallucinated_apis'])} hallucinated API(s)")
                for api in layer2["hallucinated_apis"]:
                    print(f"      - {api}")
        else:
            layer2 = {
                "passed": False,
                "error": "Skipped (syntax check failed)",
                "errors": [],
                "hallucinated_apis": []
            }
            print("   ⚠️  SKIPPED (syntax failed)")

        # Layer 3: Type Checking (only if syntax passed)
        print("\n🔍 Layer 3: Type Checking")
        if layer1["passed"]:
            layer3 = check_types(code)
            if layer3["passed"]:
                print("   ✓ PASSED")
            else:
                print(f"   ✗ FAILED - {layer3['error_count']} type error(s)")
        else:
            layer3 = {
                "passed": False,
                "error": "Skipped (syntax check failed)",
                "error_count": 0
            }
            print("   ⚠️  SKIPPED (syntax failed)")

        # Layer 4: Execution Test (only if syntax passed)
        print("\n🔍 Layer 4: Execution Test")
        if layer1["passed"]:
            # Use test cases if provided (MBPP), otherwise basic execution
            if test_cases:
                layer4 = check_execution_with_test_cases(code, test_cases)
                if layer4["passed"]:
                    print(f"   ✓ PASSED - All {layer4['tests_total']} test cases passed")
                else:
                    print(f"   ✗ FAILED - {layer4['tests_passed']}/{layer4['tests_total']} tests passed")
                    if layer4['failed_test']:
                        print(f"      Failed test: {layer4['failed_test'][:80]}...")
            else:
                layer4 = check_execution(code, test_input, expected_output)
                if layer4["passed"]:
                    print("   ✓ PASSED")
                    if layer4["actual_output"]:
                        print(f"      Output: {layer4['actual_output'][:100]}")
                else:
                    print(f"   ✗ FAILED - {layer4['error']}")
        else:
            layer4 = {
                "passed": False,
                "error": "Skipped (syntax check failed)",
                "actual_output": None,
                "output_matches": None,
                "tests_passed": 0,
                "tests_total": 0
            }
            print("   ⚠️  SKIPPED (syntax failed)")

        # Layer 5: Output Similarity (DISABLED - removed from implementation)
        # print("\n🔍 Layer 5: Output Embedding Similarity")
        layer5 = {
            "passed": True,  # Always pass (layer disabled)
            "score": None,
            "error": None,
            "similar_to": None,
            "disabled": True
        }
        # print("   ⚠️  DISABLED (layer removed from implementation)")

        # Collect results (only first 4 layers)
        layers_result = {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4
            # layer5 removed
        }

        # Determine which layers failed
        failed_layers = [
            name for name, result in layers_result.items()
            if not result["passed"]
        ]

        # Check if all layers passed
        all_passed = len(failed_layers) == 0

        # Build error summary
        error_summary = self._build_error_summary(layers_result, failed_layers)

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL 4 LAYERS PASSED")
        else:
            print(f"❌ {len(failed_layers)}/{len(layers_result)} LAYERS FAILED: {', '.join(failed_layers)}")
        print("=" * 70)

        return {
            "passed": all_passed,
            "layers": layers_result,
            "failed_layers": failed_layers,
            "error_summary": error_summary
        }

    def _build_error_summary(self, layers: Dict, failed_layers: List[str]) -> str:
        """Build a human-readable error summary."""
        if not failed_layers:
            return "All validation layers passed"

        summary_parts = []

        for layer_name in failed_layers:
            layer = layers[layer_name]

            if layer_name == "layer1":
                summary_parts.append(f"Syntax: {layer['error']}")

            elif layer_name == "layer2":
                apis = ", ".join(layer["hallucinated_apis"])
                summary_parts.append(f"Hallucinated APIs: {apis}")

            elif layer_name == "layer3":
                summary_parts.append(f"Type errors: {layer['error_count']} found")

            elif layer_name == "layer4":
                summary_parts.append(f"Execution: {layer['error']}")

            # layer5 removed

        return "; ".join(summary_parts)


if __name__ == "__main__":
    # Test validator
    print("=" * 70)
    print("CODE VALIDATOR TEST")
    print("=" * 70)

    validator = CodeValidator()

    # Mock retrieved examples
    retrieved_examples = [
        {
            "id": "ex1",
            "query": "merge sorted lists",
            "code": "def merge(a, b): return sorted(a + b)",
            "expected_output": "[1, 2, 3, 4, 5, 6]"
        }
    ]

    test_cases = [
        # Valid code
        {
            "name": "Valid code",
            "code": """
def merge_sorted_lists(list1, list2):
    return sorted(list1 + list2)
""",
            "test_input": "[1, 3, 5], [2, 4, 6]"
        },
        # Syntax error
        {
            "name": "Syntax error",
            "code": """
def merge_sorted_lists(list1, list2)
    return sorted(list1 + list2)
""",
            "test_input": None
        },
        # Hallucinated API
        {
            "name": "Hallucinated pandas API",
            "code": """
import pandas as pd

def process_data(data):
    df = pd.DataFrame(data)
    return df.super_transform()  # Doesn't exist!
""",
            "test_input": None
        },
        # Runtime error
        {
            "name": "Runtime error",
            "code": """
def divide(a, b):
    return a / b
""",
            "test_input": "10, 0"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'#' * 70}")
        print(f"# TEST {i}: {test['name']}")
        print(f"{'#' * 70}")

        result = validator.validate(
            test["code"],
            retrieved_examples,
            test["test_input"]
        )

        print(f"\nFinal Result:")
        print(f"   Passed: {result['passed']}")
        print(f"   Failed Layers: {result['failed_layers']}")
        print(f"   Error Summary: {result['error_summary']}")

    print("\n" + "=" * 70)
