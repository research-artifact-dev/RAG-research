#!/usr/bin/env python3
# validation/layer1_syntax.py
"""
Layer 1: Syntax Validation

Checks if generated code is syntactically valid Python using AST parsing.
"""

import ast
from typing import Dict


def check_syntax(code: str) -> Dict:
    """
    Check if code is syntactically valid Python.

    Args:
        code: Python code string to validate

    Returns:
        Dict with:
            - passed: bool (True if syntax is valid)
            - error: str or None (error message if failed)
            - error_line: int or None (line number where error occurred)
    """
    try:
        # Try to parse the code into an AST
        ast.parse(code)

        return {
            "passed": True,
            "error": None,
            "error_line": None
        }

    except SyntaxError as e:
        return {
            "passed": False,
            "error": f"SyntaxError: {e.msg}",
            "error_line": e.lineno
        }

    except Exception as e:
        # Catch any other parsing errors
        return {
            "passed": False,
            "error": f"Parse error: {str(e)}",
            "error_line": None
        }


if __name__ == "__main__":
    # Test Layer 1
    print("=" * 70)
    print("LAYER 1: SYNTAX VALIDATION TEST")
    print("=" * 70)

    # Test cases
    test_cases = [
        # Valid code
        {
            "name": "Valid function",
            "code": """
def merge_lists(list1, list2):
    return sorted(list1 + list2)
"""
        },
        # Syntax error - missing colon
        {
            "name": "Missing colon",
            "code": """
def merge_lists(list1, list2)
    return sorted(list1 + list2)
"""
        },
        # Syntax error - invalid indentation
        {
            "name": "Invalid indentation",
            "code": """
def merge_lists(list1, list2):
return sorted(list1 + list2)
"""
        },
        # Syntax error - unclosed parenthesis
        {
            "name": "Unclosed parenthesis",
            "code": """
def merge_lists(list1, list2):
    return sorted(list1 + list2
"""
        },
        # Valid with imports
        {
            "name": "Valid with imports",
            "code": """
import numpy as np

def calculate_mean(data):
    return np.mean(data)
"""
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {test['name']}")
        print(f"{'=' * 70}")

        result = check_syntax(test["code"])

        if result["passed"]:
            print("✓ PASSED - Code is syntactically valid")
        else:
            print(f"✗ FAILED - {result['error']}")
            if result["error_line"]:
                print(f"   Error on line {result['error_line']}")

        print(f"\nCode:")
        print(test["code"])

    print("\n" + "=" * 70)
