#!/usr/bin/env python3
# feedback/feedback_enrichment.py
"""
Feedback Enrichment Module (Phase 5)

Analyzes validation failures and builds targeted feedback for retry attempts.
Extracts specific errors to guide retrieval and prompt construction.
"""

from typing import Dict, List, Optional


class FeedbackEnricher:
    """
    Extract validation errors and build targeted feedback for retry.
    """

    def __init__(self):
        """Initialize feedback enricher."""
        pass

    def build_feedback(
        self,
        original_query: str,
        validation_result: Dict,
        previous_code: str,
        attempt: int
    ) -> Dict:
        """
        Build targeted feedback based on validation failures.

        Args:
            original_query: Original user query
            validation_result: Validation results from validator
            previous_code: Code that failed validation
            attempt: Current attempt number

        Returns:
            Dict with:
                - targeted_query: Refined query for re-retrieval
                - error_context: Error description for prompt
                - feedback_type: Type of error detected
                - failed_layers: List of failed layer names
        """
        layers = validation_result.get("layers", {})
        failed_layers = validation_result.get("failed_layers", [])

        if not failed_layers:
            # No failures - shouldn't happen but handle gracefully
            return {
                "targeted_query": original_query,
                "error_context": None,
                "feedback_type": "none",
                "failed_layers": []
            }

        # Priority order: Layer 2 (hallucination) > Layer 4 (execution) > Layer 3 (types) > Layer 1 (syntax)

        # Check Layer 2: Hallucinated APIs (highest priority)
        if "layer2" in failed_layers:
            return self._build_hallucination_feedback(
                original_query,
                layers["layer2"],
                previous_code,
                attempt
            )

        # Check Layer 4: Execution errors
        if "layer4" in failed_layers:
            return self._build_execution_feedback(
                original_query,
                layers["layer4"],
                previous_code,
                attempt
            )

        # Check Layer 3: Type errors
        if "layer3" in failed_layers:
            return self._build_type_feedback(
                original_query,
                layers["layer3"],
                previous_code,
                attempt
            )

        # Check Layer 1: Syntax errors
        if "layer1" in failed_layers:
            return self._build_syntax_feedback(
                original_query,
                layers["layer1"],
                previous_code,
                attempt
            )

        # Fallback
        return {
            "targeted_query": original_query,
            "error_context": "Previous attempt failed validation.",
            "feedback_type": "generic",
            "failed_layers": failed_layers
        }

    def _build_hallucination_feedback(
        self,
        original_query: str,
        layer2_result: Dict,
        previous_code: str,
        attempt: int
    ) -> Dict:
        """
        Build feedback for hallucinated APIs (Layer 2 failures).
        This is the most important type - LLM invented non-existent APIs.
        """
        hallucinated_apis = layer2_result.get("hallucinated_apis", [])

        if hallucinated_apis:
            # Extract the API name for targeted search
            first_api = hallucinated_apis[0]  # Focus on first hallucination

            # Build targeted query
            # Example: "sort pandas dataframe" + "correct usage of sort_values"
            if "." in first_api:
                # Extract method name from full path
                # e.g., "pandas.DataFrame.transform_apply" -> "transform_apply"
                method_name = first_api.split(".")[-1]
                module_path = ".".join(first_api.split(".")[:-1])

                targeted_query = f"{original_query} correct usage instead of {method_name} in {module_path}"
            else:
                targeted_query = f"{original_query} correct API usage"

            # Build error context for prompt
            error_context = f"""Previous attempt failed because it used non-existent API(s): {', '.join(hallucinated_apis)}.

These APIs do not exist in the library. Search for and use the CORRECT, EXISTING APIs instead.
Do not invent or hallucinate function names."""

            return {
                "targeted_query": targeted_query,
                "error_context": error_context,
                "feedback_type": "hallucinated_api",
                "failed_layers": ["layer2"],
                "hallucinated_apis": hallucinated_apis
            }

        else:
            # Layer 2 failed but no specific APIs extracted
            error_msg = layer2_result.get("error", "Unknown API error")

            return {
                "targeted_query": f"{original_query} with correct library APIs",
                "error_context": f"Previous attempt had API errors: {error_msg}",
                "feedback_type": "api_error",
                "failed_layers": ["layer2"]
            }

    def _build_execution_feedback(
        self,
        original_query: str,
        layer4_result: Dict,
        previous_code: str,
        attempt: int
    ) -> Dict:
        """
        Build feedback for execution failures (Layer 4 failures).
        Code runs but crashes or produces errors.
        """
        error = layer4_result.get("error", "Unknown execution error")

        # Analyze error type
        if "ZeroDivisionError" in error:
            targeted_query = f"{original_query} with proper error handling and edge cases"
            error_context = "Previous code crashed with division by zero. Handle edge cases properly."

        elif "IndexError" in error or "KeyError" in error:
            targeted_query = f"{original_query} with bounds checking and safe access"
            error_context = f"Previous code had index/key access error: {error}. Check bounds and existence."

        elif "TypeError" in error:
            targeted_query = f"{original_query} with correct data types"
            error_context = f"Previous code had type error: {error}. Ensure correct types."

        elif "NameError" in error:
            targeted_query = f"{original_query} with all variables defined"
            error_context = f"Previous code referenced undefined variable: {error}. Define all variables."

        elif "timeout" in error.lower() or "infinite loop" in error.lower():
            targeted_query = f"{original_query} efficient implementation without loops"
            error_context = "Previous code timed out (likely infinite loop). Use efficient algorithms."

        else:
            # Generic execution error
            targeted_query = f"{original_query} working example with test cases"
            error_context = f"Previous code crashed during execution: {error}. Provide working code."

        return {
            "targeted_query": targeted_query,
            "error_context": error_context,
            "feedback_type": "execution_error",
            "failed_layers": ["layer4"],
            "execution_error": error
        }

    def _build_type_feedback(
        self,
        original_query: str,
        layer3_result: Dict,
        previous_code: str,
        attempt: int
    ) -> Dict:
        """
        Build feedback for type checking failures (Layer 3 failures).
        """
        error_count = layer3_result.get("error_count", 0)

        targeted_query = f"{original_query} with proper type hints and type safety"

        error_context = f"""Previous code had {error_count} type error(s).

Ensure:
- Function parameters have correct type hints
- Return types match declarations
- Operations are type-safe"""

        return {
            "targeted_query": targeted_query,
            "error_context": error_context,
            "feedback_type": "type_error",
            "failed_layers": ["layer3"],
            "error_count": error_count
        }

    def _build_syntax_feedback(
        self,
        original_query: str,
        layer1_result: Dict,
        previous_code: str,
        attempt: int
    ) -> Dict:
        """
        Build feedback for syntax errors (Layer 1 failures).
        """
        error = layer1_result.get("error", "Unknown syntax error")
        error_line = layer1_result.get("error_line")

        targeted_query = f"{original_query} syntactically correct Python code"

        if error_line:
            error_context = f"Previous code had syntax error on line {error_line}: {error}. Generate valid Python syntax."
        else:
            error_context = f"Previous code had syntax error: {error}. Generate valid Python syntax."

        return {
            "targeted_query": targeted_query,
            "error_context": error_context,
            "feedback_type": "syntax_error",
            "failed_layers": ["layer1"],
            "syntax_error": error
        }


if __name__ == "__main__":
    # Test Feedback Enricher
    print("=" * 70)
    print("FEEDBACK ENRICHER TEST")
    print("=" * 70)

    enricher = FeedbackEnricher()

    # Test case 1: Hallucinated API
    print("\n" + "=" * 70)
    print("Test 1: Hallucinated API (Layer 2 failure)")
    print("=" * 70)

    validation_result = {
        "passed": False,
        "failed_layers": ["layer2", "layer4"],
        "layers": {
            "layer1": {"passed": True},
            "layer2": {
                "passed": False,
                "hallucinated_apis": ["pandas.DataFrame.transform_apply"],
                "error": "'transform_apply' does not exist on 'pandas.DataFrame'"
            },
            "layer3": {"passed": True},
            "layer4": {
                "passed": False,
                "error": "AttributeError: 'DataFrame' object has no attribute 'transform_apply'"
            }
        }
    }

    feedback = enricher.build_feedback(
        original_query="sort pandas dataframe by column",
        validation_result=validation_result,
        previous_code="df.transform_apply()",
        attempt=1
    )

    print(f"\nOriginal Query: sort pandas dataframe by column")
    print(f"Targeted Query: {feedback['targeted_query']}")
    print(f"Feedback Type: {feedback['feedback_type']}")
    print(f"Error Context:\n{feedback['error_context']}")

    # Test case 2: Execution error
    print("\n" + "=" * 70)
    print("Test 2: Execution Error (Layer 4 failure)")
    print("=" * 70)

    validation_result = {
        "passed": False,
        "failed_layers": ["layer4"],
        "layers": {
            "layer1": {"passed": True},
            "layer2": {"passed": True},
            "layer3": {"passed": True},
            "layer4": {
                "passed": False,
                "error": "ZeroDivisionError: division by zero"
            }
        }
    }

    feedback = enricher.build_feedback(
        original_query="calculate average of list",
        validation_result=validation_result,
        previous_code="sum(lst) / 0",
        attempt=1
    )

    print(f"\nOriginal Query: calculate average of list")
    print(f"Targeted Query: {feedback['targeted_query']}")
    print(f"Feedback Type: {feedback['feedback_type']}")
    print(f"Error Context:\n{feedback['error_context']}")

    print("\n" + "=" * 70)
