#!/usr/bin/env python3
# evaluation/rag_metrics.py
"""
RAG-Specific Evaluation Metrics

Computes:
1. Faithfulness - Does generated code stay faithful to retrieved examples?
2. Context Precision - Are retrieved examples relevant?
3. Context Recall - Did retrieval find all necessary information?
"""

import ast
from typing import List, Dict, Set, Optional


def extract_api_calls(code: str) -> Set[str]:
    """
    Extract all API calls (module.function, class.method) from code.

    Args:
        code: Python code string

    Returns:
        Set of API call strings (e.g., {'pandas.DataFrame.sort_values', 'numpy.array'})
    """
    api_calls = set()

    try:
        tree = ast.parse(code)

        # Track imports
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{module}.{alias.name}"

        # Extract attribute access (module.function, object.method)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Try to resolve the full path
                parts = []
                current = node

                while isinstance(current, ast.Attribute):
                    parts.insert(0, current.attr)
                    current = current.value

                if isinstance(current, ast.Name):
                    base = current.id
                    if base in imports:
                        # Imported module
                        full_path = f"{imports[base]}.{'.'.join(parts)}"
                    else:
                        # Variable/object method
                        full_path = f"{base}.{'.'.join(parts)}"

                    api_calls.add(full_path)

            # Also track function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in imports:
                        api_calls.add(imports[func_name])
                    else:
                        api_calls.add(func_name)

    except SyntaxError:
        # If code has syntax errors, return empty set
        pass

    return api_calls


def compute_faithfulness(
    generated_code: str,
    retrieved_examples: List[Dict]
) -> float:
    """
    Compute faithfulness score.

    Faithfulness = (APIs in generated that exist in retrieved) / (Total APIs in generated)

    High faithfulness = Low hallucination (LLM stays grounded in context)

    Args:
        generated_code: Generated code string
        retrieved_examples: List of retrieved example dicts with 'code' field

    Returns:
        Faithfulness score (0.0-1.0)
    """
    # Extract APIs from generated code
    generated_apis = extract_api_calls(generated_code)

    if not generated_apis:
        return 1.0  # No APIs used = no hallucination possible

    # Extract APIs from all retrieved examples
    retrieved_apis = set()
    for ex in retrieved_examples:
        ex_apis = extract_api_calls(ex.get("code", ""))
        retrieved_apis.update(ex_apis)

    if not retrieved_apis:
        # No APIs in retrieved examples to compare against
        return 0.0

    # Count faithful APIs (present in both generated and retrieved)
    faithful_apis = generated_apis.intersection(retrieved_apis)

    faithfulness = len(faithful_apis) / len(generated_apis)

    return round(faithfulness, 4)


def compute_context_precision(
    retrieved_examples: List[Dict],
    generated_code: str
) -> float:
    """
    Compute context precision.

    Context Precision = (Relevant retrieved examples) / (Total retrieved)

    An example is "relevant" if any of its APIs appear in the generated code.

    Args:
        retrieved_examples: List of retrieved example dicts
        generated_code: Generated code string

    Returns:
        Context precision score (0.0-1.0)
    """
    if not retrieved_examples:
        return 0.0

    # Extract APIs from generated code
    generated_apis = extract_api_calls(generated_code)

    if not generated_apis:
        # No APIs to check relevance against
        return 1.0

    # Count how many retrieved examples are relevant
    relevant_count = 0
    for ex in retrieved_examples:
        ex_apis = extract_api_calls(ex.get("code", ""))

        # Example is relevant if it shares at least one API with generated code
        if ex_apis.intersection(generated_apis):
            relevant_count += 1

    precision = relevant_count / len(retrieved_examples)

    return round(precision, 4)


def compute_context_recall(
    retrieved_examples: List[Dict],
    ground_truth: Optional[str] = None
) -> Optional[float]:
    """
    Compute context recall.

    Context Recall = (Ground truth APIs found in retrieved) / (Total ground truth APIs)

    Requires ground truth solution to compare against.

    Args:
        retrieved_examples: List of retrieved example dicts
        ground_truth: Ground truth code string (optional)

    Returns:
        Context recall score (0.0-1.0) or None if no ground truth
    """
    if not ground_truth:
        return None  # Cannot compute without ground truth

    # Extract APIs from ground truth
    ground_truth_apis = extract_api_calls(ground_truth)

    if not ground_truth_apis:
        return 1.0  # No APIs to recall

    # Extract APIs from all retrieved examples
    retrieved_apis = set()
    for ex in retrieved_examples:
        ex_apis = extract_api_calls(ex.get("code", ""))
        retrieved_apis.update(ex_apis)

    if not retrieved_apis:
        return 0.0

    # Count how many ground truth APIs were found in retrieval
    found_apis = ground_truth_apis.intersection(retrieved_apis)

    recall = len(found_apis) / len(ground_truth_apis)

    return round(recall, 4)


def compute_all_rag_metrics(
    generated_code: str,
    retrieved_examples: List[Dict],
    ground_truth: Optional[str] = None
) -> Dict:
    """
    Compute all RAG metrics.

    Args:
        generated_code: Generated code
        retrieved_examples: Retrieved examples
        ground_truth: Ground truth code (optional, for recall)

    Returns:
        Dict with faithfulness, context_precision, context_recall
    """
    return {
        "faithfulness": compute_faithfulness(generated_code, retrieved_examples),
        "context_precision": compute_context_precision(retrieved_examples, generated_code),
        "context_recall": compute_context_recall(retrieved_examples, ground_truth)
    }


if __name__ == "__main__":
    # Test RAG metrics
    print("=" * 70)
    print("RAG METRICS TEST")
    print("=" * 70)

    # Test case: Hallucinated API
    print("\n" + "=" * 70)
    print("Test 1: Hallucinated API (Low Faithfulness)")
    print("=" * 70)

    retrieved = [
        {
            "code": """import pandas as pd
def sort_df(df):
    return df.sort_values('column')"""
        },
        {
            "code": """import pandas as pd
df = pd.read_csv('file.csv')"""
        }
    ]

    # Generated code uses hallucinated API
    generated = """import pandas as pd
def process(df):
    return df.transform_apply(lambda x: x * 2)  # HALLUCINATED!
"""

    faithfulness = compute_faithfulness(generated, retrieved)
    precision = compute_context_precision(retrieved, generated)

    print(f"\nFaithfulness: {faithfulness:.4f}")
    print(f"Context Precision: {precision:.4f}")
    print("\nAnalysis:")
    print("  Generated APIs: df.transform_apply")
    print("  Retrieved APIs: df.sort_values, pd.read_csv")
    print("  → transform_apply not in retrieved → Low faithfulness!")

    # Test case: Faithful generation
    print("\n" + "=" * 70)
    print("Test 2: Faithful Generation (High Faithfulness)")
    print("=" * 70)

    generated_faithful = """import pandas as pd
def process(df):
    return df.sort_values('column')  # Uses API from retrieved!
"""

    faithfulness = compute_faithfulness(generated_faithful, retrieved)
    precision = compute_context_precision(retrieved, generated_faithful)

    print(f"\nFaithfulness: {faithfulness:.4f}")
    print(f"Context Precision: {precision:.4f}")
    print("\nAnalysis:")
    print("  Generated APIs: df.sort_values")
    print("  Retrieved APIs: df.sort_values, pd.read_csv")
    print("  → All APIs from retrieved → High faithfulness!")

    print("\n" + "=" * 70)
