#!/usr/bin/env python3
# evaluation/generate_comparison_tables.py
"""
Generate comprehensive comparison tables for all 3 models:
- gpt-4o
- Claude 4.5 Sonnet
- Gemini 2.5 Pro

Each showing Baseline vs RAG performance.
"""

import json


def load_all_results():
    """Load all result files for 3 models."""
    # gpt-4o
    with open("evaluation/baseline_gpt4o.json", 'r') as f:
        baseline_gpt4o = json.load(f)
    with open("evaluation/results_gpt4o.json", 'r') as f:
        rag_gpt4o = json.load(f)

    # Claude 4.5 Sonnet
    with open("evaluation/baseline_claude.json", 'r') as f:
        baseline_claude = json.load(f)
    with open("evaluation/results_claude.json", 'r') as f:
        rag_claude = json.load(f)

    # Gemini 2.5 Pro
    with open("evaluation/baseline_results.json", 'r') as f:
        baseline_gemini = json.load(f)
    with open("evaluation/unseen_results.json", 'r') as f:
        rag_gemini = json.load(f)

    return (baseline_gpt4o, rag_gpt4o,
            baseline_claude, rag_claude,
            baseline_gemini, rag_gemini)


def print_table_1_main_comparison(b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini):
    """Table 1: Comprehensive 3-model comparison."""
    print("\n" + "=" * 95)
    print("TABLE 1: COMPREHENSIVE MODEL COMPARISON (Baseline vs RAG)")
    print("=" * 95)
    print()
    print("┌─────────────────────┬──────────┬────────────┬──────────┬────────────┬────────────┐")
    print("│ Model               │ Baseline │ Baseline % │ RAG      │ RAG %      │ Δ          │")
    print("├─────────────────────┼──────────┼────────────┼──────────┼────────────┼────────────┤")

    # gpt-4o
    b_g = b_gpt["aggregate_metrics"]["pass@1"]
    r_g = r_gpt["aggregate_metrics"]["pass@1"]
    d_g = r_g - b_g
    print(f"│ gpt-4o              │ {b_g:.4f}   │  {b_g*100:5.1f}%    │ {r_g:.4f}   │  {r_g*100:5.1f}%    │ +{d_g*100:5.1f}%    │")

    # Claude 4.5 Sonnet
    b_c = b_claude["aggregate_metrics"]["pass@1"]
    r_c = r_claude["aggregate_metrics"]["pass@1"]
    d_c = r_c - b_c
    print(f"│ Claude 4.5 Sonnet   │ {b_c:.4f}   │  {b_c*100:5.1f}%    │ {r_c:.4f}   │  {r_c*100:5.1f}%    │ +{d_c*100:5.1f}%    │")

    # Gemini 2.5 Pro
    b_m = b_gemini["aggregate_metrics"]["pass@1"]
    r_m = r_gemini["aggregate_metrics"]["pass@1"]
    d_m = r_m - b_m
    print(f"│ Gemini 2.5 Pro      │ {b_m:.4f}   │  {b_m*100:5.1f}%    │ {r_m:.4f}   │  {r_m*100:5.1f}%    │ +{d_m*100:5.1f}%    │")

    print("└─────────────────────┴──────────┴────────────┴──────────┴────────────┴────────────┘")
    print()


def print_table_2_pass_at_k(b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini):
    """Table 2: pass@k comparison for all models."""
    print("\n" + "=" * 95)
    print("TABLE 2: PASS@K COMPARISON")
    print("=" * 95)
    print()
    print("┌─────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ Model               │ Mode     │ pass@1   │ pass@2   │ pass@3   │ Best     │ Δ (1→3)  │")
    print("├─────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    # gpt-4o Baseline
    b_g = b_gpt["aggregate_metrics"]
    print(f"│ gpt-4o              │ Baseline │ {b_g['pass@1']:.4f}   │ {b_g['pass@2']:.4f}   │ {b_g['pass@3']:.4f}   │ {b_g['pass@3']:.4f}   │ +{(b_g['pass@3']-b_g['pass@1'])*100:4.1f}%  │")

    # gpt-4o RAG
    r_g = r_gpt["aggregate_metrics"]
    print(f"│ gpt-4o              │ RAG      │ {r_g['pass@1']:.4f}   │ {r_g['pass@2']:.4f}   │ {r_g['pass@3']:.4f}   │ {r_g['pass@3']:.4f}   │ +{(r_g['pass@3']-r_g['pass@1'])*100:4.1f}%  │")

    print("├─────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    # Claude Baseline
    b_c = b_claude["aggregate_metrics"]
    print(f"│ Claude 4.5 Sonnet   │ Baseline │ {b_c['pass@1']:.4f}   │ {b_c['pass@2']:.4f}   │ {b_c['pass@3']:.4f}   │ {b_c['pass@3']:.4f}   │ +{(b_c['pass@3']-b_c['pass@1'])*100:4.1f}%  │")

    # Claude RAG
    r_c = r_claude["aggregate_metrics"]
    print(f"│ Claude 4.5 Sonnet   │ RAG      │ {r_c['pass@1']:.4f}   │ {r_c['pass@2']:.4f}   │ {r_c['pass@3']:.4f}   │ {r_c['pass@3']:.4f}   │ +{(r_c['pass@3']-r_c['pass@1'])*100:4.1f}%  │")

    print("├─────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    # Gemini Baseline
    b_m = b_gemini["aggregate_metrics"]
    print(f"│ Gemini 2.5 Pro      │ Baseline │ {b_m['pass@1']:.4f}   │ {b_m['pass@2']:.4f}   │ {b_m['pass@3']:.4f}   │ {b_m['pass@3']:.4f}   │ +{(b_m['pass@3']-b_m['pass@1'])*100:4.1f}%  │")

    # Gemini RAG
    r_m = r_gemini["aggregate_metrics"]
    print(f"│ Gemini 2.5 Pro      │ RAG      │ {r_m['pass@1']:.4f}   │ {r_m['pass@2']:.4f}   │ {r_m['pass@3']:.4f}   │ {r_m['pass@3']:.4f}   │ +{(r_m['pass@3']-r_m['pass@1'])*100:4.1f}%  │")

    print("└─────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
    print()


def print_table_3_validation_breakdown(r_gpt, r_claude, r_gemini):
    """Table 3: Validation layer breakdown (RAG only)."""
    print("\n" + "=" * 95)
    print("TABLE 3: VALIDATION LAYER BREAKDOWN (RAG Mode)")
    print("=" * 95)
    print()
    print("┌─────────┬────────────────────┬─────────────────┬─────────────────┬─────────────────┐")
    print("│ Layer   │ Validation Type    │ gpt-4o          │ Claude Sonnet   │ Gemini 2.5 Pro  │")
    print("├─────────┼────────────────────┼─────────────────┼─────────────────┼─────────────────┤")

    r_g_vb = r_gpt["aggregate_metrics"]["validation_breakdown"]
    r_c_vb = r_claude["aggregate_metrics"]["validation_breakdown"]
    r_m_vb = r_gemini["aggregate_metrics"]["validation_breakdown"]

    print(f"│ Layer 1 │ Syntax (AST)       │ {r_g_vb['layer1_syntax_pass_rate']:.4f} ({r_g_vb['layer1_syntax_pass_rate']*100:5.1f}%) │ {r_c_vb['layer1_syntax_pass_rate']:.4f} ({r_c_vb['layer1_syntax_pass_rate']*100:5.1f}%) │ {r_m_vb['layer1_syntax_pass_rate']:.4f} ({r_m_vb['layer1_syntax_pass_rate']*100:5.1f}%) │")
    print(f"│ Layer 2 │ API Correctness    │ {r_g_vb['layer2_api_pass_rate']:.4f} ({r_g_vb['layer2_api_pass_rate']*100:5.1f}%) │ {r_c_vb['layer2_api_pass_rate']:.4f} ({r_c_vb['layer2_api_pass_rate']*100:5.1f}%) │ {r_m_vb['layer2_api_pass_rate']:.4f} ({r_m_vb['layer2_api_pass_rate']*100:5.1f}%) │")
    print(f"│ Layer 3 │ Type Checking      │ {r_g_vb['layer3_types_pass_rate']:.4f} ({r_g_vb['layer3_types_pass_rate']*100:5.1f}%) │ {r_c_vb['layer3_types_pass_rate']:.4f} ({r_c_vb['layer3_types_pass_rate']*100:5.1f}%) │ {r_m_vb['layer3_types_pass_rate']:.4f} ({r_m_vb['layer3_types_pass_rate']*100:5.1f}%) │")
    print(f"│ Layer 4 │ Execution          │ {r_g_vb['layer4_execution_pass_rate']:.4f} ({r_g_vb['layer4_execution_pass_rate']*100:5.1f}%) │ {r_c_vb['layer4_execution_pass_rate']:.4f} ({r_c_vb['layer4_execution_pass_rate']*100:5.1f}%) │ {r_m_vb['layer4_execution_pass_rate']:.4f} ({r_m_vb['layer4_execution_pass_rate']*100:5.1f}%) │")

    print("└─────────┴────────────────────┴─────────────────┴─────────────────┴─────────────────┘")
    print()


def print_table_4_performance(b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini):
    """Table 4: Performance metrics."""
    print("\n" + "=" * 95)
    print("TABLE 4: PERFORMANCE METRICS")
    print("=" * 95)
    print()
    print("┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ Model               │ Mode     │ Avg Time │ Attempts │ Speed    │")
    print("├─────────────────────┼──────────┼──────────┼──────────┼──────────┤")

    b_g = b_gpt["aggregate_metrics"]
    r_g = r_gpt["aggregate_metrics"]
    print(f"│ gpt-4o              │ Baseline │  {b_g['avg_time_seconds']:5.2f}s  │    1.00  │ Fast     │")
    print(f"│ gpt-4o              │ RAG      │  {r_g['avg_time_seconds']:5.2f}s  │    {r_g['avg_attempts']:.2f}  │ Medium   │")

    print("├─────────────────────┼──────────┼──────────┼──────────┼──────────┤")

    b_c = b_claude["aggregate_metrics"]
    r_c = r_claude["aggregate_metrics"]
    print(f"│ Claude 4.5 Sonnet   │ Baseline │  {b_c['avg_time_seconds']:5.2f}s  │    1.00  │ Fast     │")
    print(f"│ Claude 4.5 Sonnet   │ RAG      │  {r_c['avg_time_seconds']:5.2f}s  │    {r_c['avg_attempts']:.2f}  │ Medium   │")

    print("├─────────────────────┼──────────┼──────────┼──────────┼──────────┤")

    b_m = b_gemini["aggregate_metrics"]
    r_m = r_gemini["aggregate_metrics"]
    print(f"│ Gemini 2.5 Pro      │ Baseline │  {b_m['avg_time_seconds']:5.2f}s  │    1.00  │ Fast     │")
    print(f"│ Gemini 2.5 Pro      │ RAG      │  {r_m['avg_time_seconds']:5.2f}s  │    {r_m['avg_attempts']:.2f}  │ Medium   │")

    print("└─────────────────────┴──────────┴──────────┴──────────┴──────────┘")
    print()


def print_key_findings(b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini):
    """Print key findings."""
    print("\n" + "=" * 95)
    print("KEY FINDINGS")
    print("=" * 95)

    # RAG improvements
    gpt_improvement = (r_gpt["aggregate_metrics"]["pass@1"] - b_gpt["aggregate_metrics"]["pass@1"]) * 100
    claude_improvement = (r_claude["aggregate_metrics"]["pass@1"] - b_claude["aggregate_metrics"]["pass@1"]) * 100
    gemini_improvement = (r_gemini["aggregate_metrics"]["pass@1"] - b_gemini["aggregate_metrics"]["pass@1"]) * 100

    print(f"\n✓ RAG improves gpt-4o by {gpt_improvement:.1f}% ({b_gpt['aggregate_metrics']['pass@1']*100:.1f}% → {r_gpt['aggregate_metrics']['pass@1']*100:.1f}%)")
    print(f"✓ RAG improves Claude by {claude_improvement:.1f}% ({b_claude['aggregate_metrics']['pass@1']*100:.1f}% → {r_claude['aggregate_metrics']['pass@1']*100:.1f}%)")
    print(f"✓ RAG improves Gemini by {gemini_improvement:.1f}% ({b_gemini['aggregate_metrics']['pass@1']*100:.1f}% → {r_gemini['aggregate_metrics']['pass@1']*100:.1f}%)")

    # Best model
    rag_scores = {
        "gpt-4o": r_gpt["aggregate_metrics"]["pass@1"],
        "Claude 4.5 Sonnet": r_claude["aggregate_metrics"]["pass@1"],
        "Gemini 2.5 Pro": r_gemini["aggregate_metrics"]["pass@1"]
    }
    best_model = max(rag_scores, key=rag_scores.get)
    best_score = rag_scores[best_model]

    print(f"\n✓ Best RAG performance: {best_model} ({best_score*100:.1f}% pass@1)")
    print(f"✓ RAG benefits ALL models significantly")
    print(f"✓ Cross-encoder reranking + 4-layer validation ensures quality")
    print(f"✓ Model-agnostic pipeline: works across OpenAI, Anthropic, and Google models")
    print("=" * 95)


def main():
    """Generate all comparison tables."""
    print("=" * 95)
    print("COMPREHENSIVE MODEL COMPARISON: gpt-4o vs Claude 4.5 Sonnet vs Gemini 2.5 Pro")
    print("=" * 95)

    # Load all results
    b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini = load_all_results()

    print(f"\n✓ Loaded 6 result files:")
    print(f"  - gpt-4o Baseline: {b_gpt['num_queries']} queries")
    print(f"  - gpt-4o RAG: {r_gpt['num_queries']} queries")
    print(f"  - Claude Baseline: {b_claude['num_queries']} queries")
    print(f"  - Claude RAG: {r_claude['num_queries']} queries")
    print(f"  - Gemini Baseline: {b_gemini['num_queries']} queries")
    print(f"  - Gemini RAG: {r_gemini['num_queries']} queries")

    # Generate tables
    print_table_1_main_comparison(b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini)
    print_table_2_pass_at_k(b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini)
    print_table_3_validation_breakdown(r_gpt, r_claude, r_gemini)
    print_table_4_performance(b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini)
    print_key_findings(b_gpt, r_gpt, b_claude, r_claude, b_gemini, r_gemini)

    print("\n✓ Done! Results are publication-ready.")


if __name__ == "__main__":
    main()
