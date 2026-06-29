#!/usr/bin/env python3
# validation/layer2_api.py
"""
Layer 2: API/Library Correctness Check

Validates that all imported modules, functions, and attributes actually exist.
Detects hallucinated APIs like pandas.DataFrame.transform_apply() (doesn't exist).
"""

import ast
import importlib
import sys
from typing import Dict, List, Tuple


def extract_imports(code: str) -> Dict[str, str]:
    """
    Extract all imports from code.

    Returns:
        Dict mapping alias -> full module path
        Example: {'pd': 'pandas', 'np': 'numpy', 'DataFrame': 'pandas.DataFrame'}
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    imports = {}

    for node in ast.walk(tree):
        # Handle: import module
        # Handle: import module as alias
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias_name = alias.asname if alias.asname else alias.name
                imports[alias_name] = alias.name

        # Handle: from module import name
        # Handle: from module import name as alias
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                alias_name = alias.asname if alias.asname else alias.name
                full_path = f"{module}.{alias.name}" if module else alias.name
                imports[alias_name] = full_path

    return imports


def extract_api_calls(code: str, imports: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Extract all API calls (module.attribute access).

    Returns:
        List of (module_path, attribute) tuples
        Example: [('pandas.DataFrame', 'sort_values'), ('numpy', 'array')]
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    api_calls = []

    for node in ast.walk(tree):
        # Handle: module.attribute or alias.attribute
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                # Direct attribute access: pd.DataFrame
                name = node.value.id
                if name in imports:
                    module_path = imports[name]
                    api_calls.append((module_path, node.attr))

            elif isinstance(node.value, ast.Attribute):
                # Chained attribute access: pd.DataFrame.sort_values
                # Build the full chain
                chain = [node.attr]
                current = node.value

                while isinstance(current, ast.Attribute):
                    chain.insert(0, current.attr)
                    current = current.value

                if isinstance(current, ast.Name) and current.id in imports:
                    module_path = imports[current.id]
                    # Check intermediate attributes
                    for attr in chain:
                        api_calls.append((module_path, attr))
                        module_path = f"{module_path}.{attr}"

    return api_calls


def check_api_exists(module_path: str, attribute: str) -> Tuple[bool, str]:
    """
    Check if a specific API (module.attribute) actually exists.

    Args:
        module_path: Full module path (e.g., 'pandas.DataFrame')
        attribute: Attribute name (e.g., 'sort_values')

    Returns:
        (exists: bool, error_message: str)
    """
    try:
        # Split module path
        parts = module_path.split('.')

        # Import base module
        try:
            module = importlib.import_module(parts[0])
        except ImportError as e:
            return False, f"Cannot import module '{parts[0]}': {e}"

        # Navigate through submodules/attributes
        obj = module
        for i, part in enumerate(parts[1:], 1):
            if not hasattr(obj, part):
                parent_path = '.'.join(parts[:i])
                return False, f"'{part}' does not exist in '{parent_path}'"
            obj = getattr(obj, part)

        # Check final attribute
        if not hasattr(obj, attribute):
            return False, f"'{attribute}' does not exist on '{module_path}'"

        return True, ""

    except Exception as e:
        return False, f"Error checking '{module_path}.{attribute}': {e}"


def check_api_correctness(code: str) -> Dict:
    """
    Check if all APIs used in code actually exist.

    Args:
        code: Python code string

    Returns:
        Dict with:
            - passed: bool (True if all APIs exist)
            - error: str or None (error message if failed)
            - errors: List[str] (all API errors found)
            - hallucinated_apis: List[str] (list of non-existent APIs)
    """
    # Extract imports
    imports = extract_imports(code)

    if not imports:
        # No imports = nothing to check
        return {
            "passed": True,
            "error": None,
            "errors": [],
            "hallucinated_apis": []
        }

    # Extract API calls
    api_calls = extract_api_calls(code, imports)

    if not api_calls:
        # No API calls = nothing to check
        return {
            "passed": True,
            "error": None,
            "errors": [],
            "hallucinated_apis": []
        }

    # Check each API call
    errors = []
    hallucinated_apis = []

    for module_path, attribute in api_calls:
        exists, error_msg = check_api_exists(module_path, attribute)

        if not exists:
            errors.append(error_msg)
            hallucinated_api = f"{module_path}.{attribute}"
            if hallucinated_api not in hallucinated_apis:
                hallucinated_apis.append(hallucinated_api)

    # Determine result
    passed = len(errors) == 0

    return {
        "passed": passed,
        "error": "; ".join(errors) if errors else None,
        "errors": errors,
        "hallucinated_apis": hallucinated_apis
    }


if __name__ == "__main__":
    # Test Layer 2
    print("=" * 70)
    print("LAYER 2: API/LIBRARY CORRECTNESS TEST")
    print("=" * 70)

    test_cases = [
        # Valid APIs
        {
            "name": "Valid pandas usage",
            "code": """
import pandas as pd

def process_data(data):
    df = pd.DataFrame(data)
    return df.sort_values('column')
"""
        },
        # Hallucinated API - pandas.DataFrame.transform_apply doesn't exist
        {
            "name": "Hallucinated pandas method",
            "code": """
import pandas as pd

def process_data(data):
    df = pd.DataFrame(data)
    return df.transform_apply(lambda x: x * 2)
"""
        },
        # Valid numpy usage
        {
            "name": "Valid numpy usage",
            "code": """
import numpy as np

def calculate_mean(data):
    arr = np.array(data)
    return np.mean(arr)
"""
        },
        # Hallucinated numpy function
        {
            "name": "Hallucinated numpy function",
            "code": """
import numpy as np

def process_array(data):
    return np.super_median(data)  # Doesn't exist
"""
        },
        # Non-existent module
        {
            "name": "Non-existent module",
            "code": """
import fake_module_xyz

def do_something():
    return fake_module_xyz.process()
"""
        },
        # No imports
        {
            "name": "No imports (pure Python)",
            "code": """
def merge_lists(list1, list2):
    return sorted(list1 + list2)
"""
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {test['name']}")
        print(f"{'=' * 70}")

        result = check_api_correctness(test["code"])

        if result["passed"]:
            print("✓ PASSED - All APIs are correct")
        else:
            print(f"✗ FAILED - Found API errors:")
            for error in result["errors"]:
                print(f"   - {error}")

            if result["hallucinated_apis"]:
                print(f"\nHallucinated APIs detected:")
                for api in result["hallucinated_apis"]:
                    print(f"   - {api}")

        print(f"\nCode:")
        print(test["code"])

    print("\n" + "=" * 70)
