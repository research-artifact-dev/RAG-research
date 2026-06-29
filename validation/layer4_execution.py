#!/usr/bin/env python3
# validation/layer4_execution.py
"""
Layer 4: Execution Test

Runs generated code in a sandboxed subprocess to verify it executes without errors.
"""

import subprocess
import tempfile
import os
import ast
import sys
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EXECUTION_TIMEOUT


def extract_function_name(code: str) -> Optional[str]:
    """
    Extract the first function name from code.

    Returns:
        Function name or None if no function found
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None
    except:
        return None


def build_test_wrapper(code: str, test_input: Optional[str] = None) -> str:
    """
    Wrap code with a test call to verify it runs.

    Args:
        code: Generated function code
        test_input: Optional test input to pass to function

    Returns:
        Complete Python script that runs the function
    """
    func_name = extract_function_name(code)

    if not func_name:
        # No function found - just return code as-is (might be a script)
        return code

    # Build a simple test call
    if test_input:
        test_call = f"\nresult = {func_name}({test_input})\nprint(repr(result))\n"
    else:
        # Try to call with empty/default args
        test_call = f"\n# Function defined but not called (no test input provided)\n"
        test_call += f"print('Function {func_name} defined successfully')\n"

    return code + test_call


def check_execution_with_test_cases(
    code: str,
    test_cases: list,
    timeout: int = EXECUTION_TIMEOUT
) -> Dict:
    """
    Execute code with actual test cases (for MBPP evaluation).

    Args:
        code: Python code to execute
        test_cases: List of test assertion strings (e.g., ["assert func(1) == 2", ...])
        timeout: Maximum seconds for execution

    Returns:
        Dict with:
            - passed: bool (True if ALL test cases passed)
            - error: str or None (error message if failed)
            - tests_passed: int (number of test cases that passed)
            - tests_total: int (total number of test cases)
            - failed_test: str or None (first failed test case)
    """
    if not test_cases:
        # No test cases - fall back to basic execution check
        result = check_execution(code, timeout=timeout)
        return {
            "passed": result["passed"],
            "error": result["error"],
            "tests_passed": 1 if result["passed"] else 0,
            "tests_total": 1,
            "failed_test": None
        }

    # Build test script with all assertions
    test_script = code + "\n\n# Test cases\n"
    for i, test_case in enumerate(test_cases):
        test_script += f"{test_case}\n"

    # Write to temporary file
    with tempfile.NamedTemporaryFile(
        suffix=".py",
        mode="w",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(test_script)
        tmp_path = f.name

    try:
        # Execute in subprocess
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Check if execution succeeded
        if result.returncode != 0:
            # Find which test failed
            error_msg = result.stderr.strip()
            failed_test = None

            # Try to extract failing assertion from error
            for test_case in test_cases:
                if test_case in error_msg or any(part in error_msg for part in test_case.split()):
                    failed_test = test_case
                    break

            return {
                "passed": False,
                "error": f"Test case failed:\n{error_msg}",
                "tests_passed": 0,  # We don't know how many passed before failure
                "tests_total": len(test_cases),
                "failed_test": failed_test or test_cases[0]
            }

        # All tests passed!
        return {
            "passed": True,
            "error": None,
            "tests_passed": len(test_cases),
            "tests_total": len(test_cases),
            "failed_test": None
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error": f"Execution timed out after {timeout} seconds (infinite loop or hanging)",
            "tests_passed": 0,
            "tests_total": len(test_cases),
            "failed_test": test_cases[0] if test_cases else None
        }

    except Exception as e:
        return {
            "passed": False,
            "error": f"Error during execution: {str(e)}",
            "tests_passed": 0,
            "tests_total": len(test_cases),
            "failed_test": None
        }

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def check_execution(
    code: str,
    test_input: Optional[str] = None,
    expected_output: Optional[str] = None,
    timeout: int = EXECUTION_TIMEOUT
) -> Dict:
    """
    Execute code in sandboxed subprocess and verify it runs successfully.

    Args:
        code: Python code to execute
        test_input: Optional input to pass to function (as string)
        expected_output: Optional expected output for comparison
        timeout: Maximum seconds for execution

    Returns:
        Dict with:
            - passed: bool (True if execution succeeded)
            - error: str or None (error message if failed)
            - actual_output: str or None (actual program output)
            - output_matches: bool or None (if expected_output provided)
    """
    # Wrap code with test call
    wrapped_code = build_test_wrapper(code, test_input)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(
        suffix=".py",
        mode="w",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(wrapped_code)
        tmp_path = f.name

    try:
        # Execute in subprocess
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Check if execution succeeded
        if result.returncode != 0:
            return {
                "passed": False,
                "error": f"Execution failed with exit code {result.returncode}:\n{result.stderr.strip()}",
                "actual_output": None,
                "output_matches": None
            }

        # Execution succeeded
        actual_output = result.stdout.strip()

        # Check output match if expected output provided
        output_matches = None
        if expected_output is not None:
            output_matches = actual_output == expected_output.strip()

        return {
            "passed": True,
            "error": None,
            "actual_output": actual_output,
            "output_matches": output_matches
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error": f"Execution timed out after {timeout} seconds (infinite loop or hanging)",
            "actual_output": None,
            "output_matches": None
        }

    except Exception as e:
        return {
            "passed": False,
            "error": f"Error during execution: {str(e)}",
            "actual_output": None,
            "output_matches": None
        }

    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    # Test Layer 4
    print("=" * 70)
    print("LAYER 4: EXECUTION TEST")
    print("=" * 70)

    test_cases = [
        # Valid execution
        {
            "name": "Valid function execution",
            "code": """
def add_numbers(a, b):
    return a + b
""",
            "test_input": "5, 10",
            "expected_output": "15"
        },
        # Runtime error - division by zero
        {
            "name": "Runtime error (division by zero)",
            "code": """
def divide(a, b):
    return a / b
""",
            "test_input": "10, 0",
            "expected_output": None
        },
        # Valid list operations
        {
            "name": "Valid list merge",
            "code": """
def merge_lists(list1, list2):
    return sorted(list1 + list2)
""",
            "test_input": "[1, 3, 5], [2, 4, 6]",
            "expected_output": "[1, 2, 3, 4, 5, 6]"
        },
        # Infinite loop (should timeout)
        {
            "name": "Infinite loop",
            "code": """
def infinite_loop():
    while True:
        pass
""",
            "test_input": "",
            "expected_output": None
        },
        # Name error - undefined variable
        {
            "name": "NameError (undefined variable)",
            "code": """
def use_undefined():
    return undefined_variable + 10
""",
            "test_input": "",
            "expected_output": None
        },
        # No function (just script)
        {
            "name": "Script (no function)",
            "code": """
print("Hello, World!")
result = 42
print(result)
""",
            "test_input": None,
            "expected_output": None
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {test['name']}")
        print(f"{'=' * 70}")

        result = check_execution(
            test["code"],
            test["test_input"],
            test["expected_output"],
            timeout=5  # 5 second timeout for testing
        )

        if result["passed"]:
            print("✓ PASSED - Code executed successfully")
            if result["actual_output"]:
                print(f"   Output: {result['actual_output']}")

            if result["output_matches"] is not None:
                if result["output_matches"]:
                    print("   ✓ Output matches expected")
                else:
                    print(f"   ✗ Output mismatch")
                    print(f"      Expected: {test['expected_output']}")
                    print(f"      Actual: {result['actual_output']}")
        else:
            print(f"✗ FAILED - {result['error']}")

        print(f"\nCode:")
        print(test["code"])

    print("\n" + "=" * 70)
