#!/usr/bin/env python3
# generation/prompt_builder.py
"""
Prompt Builder for RAG Generation.

Combines retrieved code examples with user query to create
structured prompts for the LLM.
"""

import sys
import os
import re
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CODE_GENERATION_SYSTEM_PROMPT


def _extract_function_name(test_cases: List[str]) -> Optional[str]:
    """
    Extract function name from test cases.

    Args:
        test_cases: List of test case strings (e.g., "assert prime_num(13)==True")

    Returns:
        Function name or None
    """
    if not test_cases:
        return None

    # Try to extract function name from first test case
    # Pattern: assert function_name(...) or function_name(...)
    for test in test_cases:
        # Remove "assert " prefix
        test = test.replace("assert ", "").strip()
        # Extract function name (word before opening parenthesis)
        match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', test)
        if match:
            return match.group(1)

    return None


def build_prompt(
    user_query: str,
    retrieved_examples: List[Dict],
    max_examples: int = 5,
    attempt: int = 1,
    error_feedback: Optional[Dict] = None,
    test_cases: Optional[List[str]] = None
) -> tuple[str, List[Dict]]:
    """
    Build LLM prompt from user query and retrieved examples.

    Args:
        user_query: User's natural language query
        retrieved_examples: List of retrieved code examples
        max_examples: Maximum number of examples to include
        attempt: Current attempt number (for retry logic)
        error_feedback: Feedback from failed validation attempts
        test_cases: Test cases to extract function name from

    Returns:
        Tuple of (system_prompt, messages) for chat completion
    """

    # Extract function name from test cases
    function_name = _extract_function_name(test_cases) if test_cases else None

    # Limit examples to avoid token overflow
    examples_to_use = retrieved_examples[:max_examples]

    # Build examples section
    examples_text = _format_examples(examples_to_use)

    # Build instruction based on attempt number
    if attempt == 1:
        # First attempt - emphasize following examples closely
        fn_instruction = f"\nIMPORTANT: Your function MUST be named '{function_name}'" if function_name else ""

        instruction = f"""Write a Python function to solve this task:

{user_query}

CRITICAL INSTRUCTIONS:
1. Study the retrieved examples below - they show the CORRECT approach
2. Copy the EXACT same logic, algorithm, and return types from the most similar example
3. DO NOT improve, refactor, or optimize the code
4. DO NOT change return types (e.g., if example returns "None" as string, keep it as string)
5. Keep the same code structure and patterns{fn_instruction}

Use the examples below as your primary reference:"""

    elif attempt == 2 and error_feedback:
        # Second attempt - include error feedback
        failed_layers = error_feedback.get("failed_layers", [])
        errors = error_feedback.get("error_details", "")
        fn_instruction = f"\nREQUIRED function name: '{function_name}'" if function_name else ""

        instruction = f"""Your previous code failed these validation checks: {', '.join(failed_layers)}

Error details:
{errors}

Please write corrected Python code for this task:
{user_query}{fn_instruction}

Copy the logic from the examples - do not improvise."""

    else:
        # Third attempt - self-critique with previous code
        prev_code = error_feedback.get("previous_code", "") if error_feedback else ""
        fn_instruction = f"\nREQUIRED function name: '{function_name}'" if function_name else ""

        instruction = f"""Previous attempt had issues:
```python
{prev_code}
```

Write a corrected version for this task:
{user_query}{fn_instruction}

Use the EXACT logic from the retrieved examples."""

    # Combine into user message
    user_message = f"""{instruction}

{examples_text}

Now write the Python code:"""

    # Return as chat messages format
    messages = [
        {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    return CODE_GENERATION_SYSTEM_PROMPT, messages


def _format_examples(examples: List[Dict]) -> str:
    """
    Format retrieved examples into readable text.

    Args:
        examples: List of code examples

    Returns:
        Formatted string of examples
    """
    examples_text = "Here are relevant code examples:\n\n"

    for i, example in enumerate(examples, 1):
        query = example.get("query", "")
        code = example.get("code", "")
        score = example.get("similarity_score", 0.0)

        examples_text += f"""### Example {i} (similarity: {score:.2f})
# Task: {query}
```python
{code}
```

"""

    return examples_text


def extract_code_from_response(response_text: str) -> str:
    """
    Extract Python code from LLM response.

    Handles cases where LLM returns code with markdown formatting
    or explanatory text.

    Args:
        response_text: Raw LLM response

    Returns:
        Extracted Python code
    """
    # Remove markdown code fences if present
    if "```python" in response_text:
        # Extract code between ```python and ```
        start = response_text.find("```python") + len("```python")
        end = response_text.find("```", start)
        if end != -1:
            code = response_text[start:end].strip()
        else:
            code = response_text[start:].strip()
    elif "```" in response_text:
        # Extract code between ``` and ```
        start = response_text.find("```") + len("```")
        end = response_text.find("```", start)
        if end != -1:
            code = response_text[start:end].strip()
        else:
            code = response_text[start:].strip()
    else:
        # No markdown, use as is
        code = response_text.strip()

    return code


if __name__ == "__main__":
    # Test prompt builder
    print("=" * 70)
    print("PROMPT BUILDER TEST")
    print("=" * 70)

    # Mock retrieved examples
    examples = [
        {
            "query": "sort dictionary by value",
            "code": "sorted(dict.items(), key=lambda x: x[1])",
            "similarity_score": 0.85
        },
        {
            "query": "sort dict items",
            "code": "{k: v for k, v in sorted(d.items(), key=lambda x: x[1])}",
            "similarity_score": 0.82
        }
    ]

    query = "sort a dictionary by its values"

    system_prompt, messages = build_prompt(query, examples, max_examples=2)

    print(f"\nSystem Prompt:\n{system_prompt}\n")
    print(f"\nUser Message:\n{messages[1]['content']}\n")
    print("=" * 70)
