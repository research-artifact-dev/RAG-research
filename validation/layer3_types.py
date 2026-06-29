#!/usr/bin/env python3
# validation/layer3_types.py
"""
Layer 3: Type Checking with mypy

Performs static type checking to detect type mismatches and errors.
"""

import subprocess
import tempfile
import os
from typing import Dict


def check_types(code: str, timeout: int = 15) -> Dict:
    """
    Check code for type errors using mypy.

    Args:
        code: Python code string
        timeout: Maximum seconds for mypy to run

    Returns:
        Dict with:
            - passed: bool (True if no type errors)
            - error: str or None (mypy output if errors found)
            - error_count: int (number of type errors)
    """
    # Write code to temporary file
    with tempfile.NamedTemporaryFile(
        suffix=".py",
        mode="w",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # Run mypy on the temporary file
        result = subprocess.run(
            [
                "mypy",
                tmp_path,
                "--ignore-missing-imports",  # Don't fail on uninstalled packages
                "--no-error-summary",         # Cleaner output
                "--show-error-codes"          # Show error codes for debugging
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # mypy returns 0 if no errors, 1 if errors found
        passed = result.returncode == 0

        # Parse output
        output = result.stdout.strip()

        # Count errors
        error_count = 0
        if output:
            error_count = output.count(tmp_path)  # Each error line contains the file path

        return {
            "passed": passed,
            "error": output if not passed else None,
            "error_count": error_count
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error": "mypy timed out (code may be too complex)",
            "error_count": 1
        }

    except FileNotFoundError:
        return {
            "passed": False,
            "error": "mypy not installed (run: pip install mypy)",
            "error_count": 1
        }

    except Exception as e:
        return {
            "passed": False,
            "error": f"Error running mypy: {str(e)}",
            "error_count": 1
        }

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    # Test Layer 3
    print("=" * 70)
    print("LAYER 3: TYPE CHECKING TEST")
    print("=" * 70)

    test_cases = [
        # Valid typed code
        {
            "name": "Valid typed function",
            "code": """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, 10)
"""
        },
        # Type mismatch
        {
            "name": "Type mismatch",
            "code": """
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers("hello", 10)  # Wrong type!
"""
        },
        # Return type mismatch
        {
            "name": "Return type mismatch",
            "code": """
def get_name() -> str:
    return 42  # Should return str, not int!
"""
        },
        # No type hints (should pass)
        {
            "name": "No type hints",
            "code": """
def merge_lists(list1, list2):
    return list1 + list2
"""
        },
        # Complex types
        {
            "name": "Valid complex types",
            "code": """
from typing import List, Dict

def process_data(items: List[int]) -> Dict[str, int]:
    return {"count": len(items), "sum": sum(items)}
"""
        },
        # Invalid list operation
        {
            "name": "Invalid list operation",
            "code": """
from typing import List

def process_items(items: List[int]) -> int:
    return items + "hello"  # Can't add str to List[int]!
"""
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {test['name']}")
        print(f"{'=' * 70}")

        result = check_types(test["code"])

        if result["passed"]:
            print("✓ PASSED - No type errors")
        else:
            print(f"✗ FAILED - Found {result['error_count']} type error(s)")
            print(f"\nMypy output:")
            print(result["error"])

        print(f"\nCode:")
        print(test["code"])

    print("\n" + "=" * 70)
