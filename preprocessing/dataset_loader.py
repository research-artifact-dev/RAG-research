# preprocessing/dataset_loader.py
"""
Dataset loading and standardization module.
Loads APPS, CodeSearchNet, HumanEval, and GitHub Code datasets
and converts them to a common schema for the RAG pipeline.
"""

import json
import ast
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAX_APPS, MAX_CODESEARCHNET, MAX_HUMANEVAL, MAX_GITHUB_CODE,
    MIN_CODE_LENGTH, MAX_CODE_LENGTH, LANGUAGE
)


def load_apps_dataset() -> List[Dict]:
    """
    Load APPS dataset (competitive programming problems).

    Returns:
        List of standardized documents
    """
    print("Loading APPS dataset...")
    try:
        dataset = load_dataset("codeparrot/apps", split="train")
        documents = []

        for idx, sample in enumerate(tqdm(dataset, desc="Processing APPS")):
            if idx >= MAX_APPS:
                break

            # Parse solutions (stored as JSON string)
            solutions_raw = sample.get("solutions", "")
            if not solutions_raw:
                continue

            try:
                # Solutions is a JSON-encoded string
                solutions = json.loads(solutions_raw)
                if not solutions or not isinstance(solutions, list) or len(solutions) == 0:
                    continue
                code = solutions[0]
            except:
                continue

            if not code or len(code) < MIN_CODE_LENGTH or len(code) > MAX_CODE_LENGTH:
                continue

            # Use question as query
            query = sample.get("question", "").strip()
            if not query:
                continue

            # Parse input_output (also JSON string)
            input_output_raw = sample.get("input_output", "")
            expected_output = None
            if input_output_raw:
                try:
                    input_output = json.loads(input_output_raw)
                    if "outputs" in input_output and len(input_output["outputs"]) > 0:
                        expected_output = str(input_output["outputs"][0])
                except:
                    pass

            document = {
                "id": f"apps_{idx}",
                "query": query,
                "code": code,
                "expected_output": expected_output,
                "language": "python",
                "source": "apps",
                "metadata": {
                    "difficulty": sample.get("difficulty", "unknown"),
                    "problem_id": sample.get("problem_id", idx)
                }
            }
            documents.append(document)

        print(f"✓ Loaded {len(documents)} documents from APPS")
        return documents

    except Exception as e:
        print(f"✗ Error loading APPS dataset: {e}")
        return []


def load_codesearchnet_dataset() -> List[Dict]:
    """
    Load CodeSearchNet dataset (function-docstring pairs).
    Using sentence-transformers version which is more stable.

    Returns:
        List of standardized documents
    """
    print("Loading CodeSearchNet dataset...")
    try:
        # Use sentence-transformers version (more stable)
        dataset = load_dataset("sentence-transformers/codesearchnet", split="train")
        documents = []

        for idx, sample in enumerate(tqdm(dataset, desc="Processing CodeSearchNet")):
            if idx >= MAX_CODESEARCHNET:
                break

            # sentence-transformers/codesearchnet uses different field names
            code = sample.get("code", "").strip()
            docstring = sample.get("comment", "").strip()

            # Filter invalid samples
            if not code or not docstring:
                continue
            if len(code) < MIN_CODE_LENGTH or len(code) > MAX_CODE_LENGTH:
                continue
            if len(docstring) < 10:  # Skip very short docstrings
                continue

            document = {
                "id": f"codesearchnet_python_{idx}",
                "query": docstring,
                "code": code,
                "expected_output": None,
                "language": "python",
                "source": "codesearchnet",
                "metadata": {
                    "idx": idx
                }
            }
            documents.append(document)

        print(f"✓ Loaded {len(documents)} documents from CodeSearchNet")
        return documents

    except Exception as e:
        print(f"✗ Error loading CodeSearchNet dataset: {e}")
        return []


def load_humaneval_dataset() -> List[Dict]:
    """
    Load HumanEval dataset (hand-crafted Python problems with unit tests).

    Returns:
        List of standardized documents
    """
    print("Loading HumanEval dataset...")
    try:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        documents = []

        for idx, sample in enumerate(tqdm(dataset, desc="Processing HumanEval")):
            task_id = sample.get("task_id", f"HumanEval/{idx}")
            prompt = sample.get("prompt", "").strip()
            canonical_solution = sample.get("canonical_solution", "").strip()

            if not prompt or not canonical_solution:
                continue

            # Extract docstring from prompt as query
            # Prompt typically contains function signature + docstring
            query = prompt

            # Full code = prompt + solution
            code = prompt + canonical_solution

            if len(code) < MIN_CODE_LENGTH or len(code) > MAX_CODE_LENGTH:
                continue

            document = {
                "id": f"humaneval_{idx}",
                "query": query,
                "code": code,
                "expected_output": None,  # Tests are in separate field
                "language": "python",
                "source": "humaneval",
                "metadata": {
                    "task_id": task_id,
                    "entry_point": sample.get("entry_point", "unknown")
                }
            }
            documents.append(document)

        print(f"✓ Loaded {len(documents)} documents from HumanEval")
        return documents

    except Exception as e:
        print(f"✗ Error loading HumanEval dataset: {e}")
        return []


def extract_functions_from_code(code: str) -> List[tuple]:
    """
    Extract individual functions from Python code using AST.

    Args:
        code: Python source code string

    Returns:
        List of tuples (function_name, function_code, docstring)
    """
    functions = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                docstring = ast.get_docstring(node) or func_name.replace("_", " ")

                # Extract function source code
                func_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                func_code = '\n'.join(func_lines)

                if len(func_code) >= MIN_CODE_LENGTH and len(func_code) <= MAX_CODE_LENGTH:
                    functions.append((func_name, func_code, docstring))
    except:
        pass

    return functions


def load_github_code_dataset() -> List[Dict]:
    """
    Load GitHub Code dataset (real-world code samples).
    Note: This dataset uses deprecated scripts. Skipping for now.
    Can be replaced with alternative datasets later.

    Returns:
        List of standardized documents
    """
    print("Loading GitHub Code dataset...")
    print("⚠️  GitHub Code dataset uses deprecated loading scripts.")
    print("Skipping for now. You can add alternative code datasets later.")
    return []


def load_all_datasets(save_path: Optional[str] = "data/processed/all_documents.json") -> List[Dict]:
    """
    Load all datasets and combine into single list.

    Args:
        save_path: Path to save processed documents (optional)

    Returns:
        List of all standardized documents
    """
    print("=" * 60)
    print("DATASET LOADING PIPELINE")
    print("=" * 60)

    all_documents = []

    # Load each dataset
    apps_docs = load_apps_dataset()
    all_documents.extend(apps_docs)

    codesearchnet_docs = load_codesearchnet_dataset()
    all_documents.extend(codesearchnet_docs)

    humaneval_docs = load_humaneval_dataset()
    all_documents.extend(humaneval_docs)

    github_docs = load_github_code_dataset()
    all_documents.extend(github_docs)

    print("=" * 60)
    print(f"TOTAL DOCUMENTS LOADED: {len(all_documents)}")
    print("=" * 60)
    print(f"  - APPS: {len(apps_docs)}")
    print(f"  - CodeSearchNet: {len(codesearchnet_docs)}")
    print(f"  - HumanEval: {len(humaneval_docs)}")
    print(f"  - GitHub Code: {len(github_docs)}")
    print("=" * 60)

    # Save to disk if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_documents, f, indent=2)
        print(f"✓ Saved all documents to {save_path}")

    return all_documents


if __name__ == "__main__":
    # Test the loader
    documents = load_all_datasets()

    # Print sample document
    if documents:
        print("\n" + "=" * 60)
        print("SAMPLE DOCUMENT:")
        print("=" * 60)
        print(json.dumps(documents[0], indent=2))
