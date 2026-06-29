#!/usr/bin/env python3
# retrieval/query_reformulator.py
"""
Query Reformulator for improving low-confidence retrieval.

When initial retrieval has low confidence, uses LLM to reformulate
the query into a more specific, code-search-optimized version.
"""

import sys
import os
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.llm_generator import LLMGenerator
from config import LLM_MODEL_NAME


class QueryReformulator:
    """
    Reformulates queries to improve retrieval confidence.
    Always uses gpt-4o for consistent query reformulation.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize query reformulator with LLM.

        Args:
            model_name: LLM model name in SAP AI Core (default: gpt-4o)
        """
        # Always use gpt-4o for reformulation (consistent across experiments)
        self.generator = LLMGenerator("gpt-4o")

    def reformulate(self, original_query: str, attempt: int = 1) -> str:
        """
        Reformulate a query to be more specific and code-search-friendly.

        Args:
            original_query: Original user query
            attempt: Attempt number (for variation)

        Returns:
            Reformulated query string
        """
        system_prompt = """You are a query optimization expert for code search.
Your task is to reformulate user queries to improve code retrieval accuracy.

Guidelines:
- Make queries more specific and technical
- Include relevant Python keywords and function names
- Focus on the core algorithmic problem
- Keep queries concise (1-2 sentences)
- Only output the reformulated query, nothing else"""

        # Build reformulation prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Original query: "{original_query}"

Attempt: {attempt}

Reformulate this query to improve code retrieval. Make it more specific and search-friendly.
Output only the reformulated query."""
            }
        ]

        # Generate reformulated query
        try:
            reformulated = self.generator.generate_single(
                messages=messages,
                temperature=0.5,  # Slightly higher for variation
                max_tokens=100
            )

            # Clean up the response
            reformulated = reformulated.strip().strip('"').strip("'")

            print(f"\n🔄 Query reformulation (Attempt {attempt}):")
            print(f"   Original: {original_query}")
            print(f"   Reformulated: {reformulated}")

            return reformulated

        except Exception as e:
            print(f"✗ Error reformulating query: {e}")
            # Return original on error
            return original_query

    def generate_variations(
        self,
        original_query: str,
        num_variations: int = 2
    ) -> List[str]:
        """
        Generate multiple query variations.

        Args:
            original_query: Original user query
            num_variations: Number of variations to generate

        Returns:
            List of query variations (including original)
        """
        variations = [original_query]

        for i in range(num_variations):
            variation = self.reformulate(original_query, attempt=i+1)
            if variation != original_query and variation not in variations:
                variations.append(variation)

        return variations


if __name__ == "__main__":
    # Test query reformulator
    print("=" * 70)
    print("QUERY REFORMULATOR TEST")
    print("=" * 70)

    # Initialize reformulator
    print("\n🔄 Initializing query reformulator...")
    reformulator = QueryReformulator()

    # Test queries (intentionally vague)
    test_queries = [
        "sort a list",
        "find something in a string",
        "work with files",
        "make a class for data"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'=' * 70}")

        # Generate 2 variations
        variations = reformulator.generate_variations(query, num_variations=2)

        print(f"\n📝 Generated {len(variations)} query variations:")
        for j, var in enumerate(variations, 1):
            print(f"   {j}. {var}")

        print()

    print("=" * 70)
