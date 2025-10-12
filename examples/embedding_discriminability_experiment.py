#!/usr/bin/env python3
"""
Embedding Discriminability Experiment
====================================

Tests different embedding dimensions to measure character separability
using the ΨQRH audit analyzer's analyze_embedding_space function.

Based on the task requirements:
- Test embed_dim values: 64, 128, 256
- Measure "Distância Média Mínima" between character probes
- Determine optimal embed_dim for character discriminability
"""

import sys
import os
from pathlib import Path

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from tools.audit_analyzer import ΨQRHAuditAnalyzer

def run_embedding_discriminability_experiment():
    """
    Run experiment testing different embedding dimensions for character separability.
    """
    print("=" * 80)
    print("🔬 EMBEDDING DISCRIMINABILITY EXPERIMENT")
    print("=" * 80)
    print("\nTesting different embed_dim values to measure character separability...")
    print("Using ΨQRHAuditAnalyzer.analyze_embedding_space()")
    print("\n" + "=" * 80)

    # Initialize analyzer
    analyzer = ΨQRHAuditAnalyzer()

    # Test different embedding dimensions
    embed_dims = [64, 128, 256]
    results = {}

    for embed_dim in embed_dims:
        print(f"\n🧪 Testing embed_dim = {embed_dim}")
        print("-" * 40)

        # Run embedding space analysis
        analysis_result = analyzer.analyze_embedding_space(
            embed_dim=embed_dim,
            save_heatmap=True
        )

        results[embed_dim] = analysis_result

        # Display results
        print("📊 Results:")
        print(f"   - Average Minimum Distance: {analysis_result['avg_min_distance']:.6f}")
        print(f"   - Std Dev of Min Distances: {analysis_result['std_min_distance']:.6f}")
        print(f"   - Most Problematic Pairs (top 5):")

        for char1, char2, similarity in analysis_result['most_problematic_pairs']:
            print(f"     ('{char1}', '{char2}'): similarity = {similarity:.6f}")

        # Interpret results
        avg_min_dist = analysis_result['avg_min_distance']
        if avg_min_dist > 1.0:
            quality = "EXCELLENT"
            desc = "Very good separability between characters"
        elif avg_min_dist > 0.5:
            quality = "GOOD"
            desc = "Adequate separability, some confusion possible"
        elif avg_min_dist > 0.2:
            quality = "MODERATE"
            desc = "Moderate separability, significant confusion likely"
        else:
            quality = "POOR"
            desc = "Poor separability, high confusion expected"

        print(f"   - Quality Assessment: {quality} ({desc})")

    # Comparative Analysis
    print(f"\n" + "=" * 80)
    print("📈 COMPARATIVE ANALYSIS")
    print("=" * 80)

    print("Embed_dim | Avg Min Distance | Std Dev | Quality Assessment")
    print("-----------|------------------|---------|-------------------")

    for embed_dim in embed_dims:
        result = results[embed_dim]
        avg_dist = result['avg_min_distance']
        std_dev = result['std_min_distance']

        if avg_dist > 1.0:
            quality = "EXCELLENT"
        elif avg_dist > 0.5:
            quality = "GOOD"
        elif avg_dist > 0.2:
            quality = "MODERATE"
        else:
            quality = "POOR"

        print("8")

    # Determine optimal embed_dim
    best_embed_dim = max(embed_dims, key=lambda x: results[x]['avg_min_distance'])
    best_distance = results[best_embed_dim]['avg_min_distance']

    print(f"\n🎯 OPTIMAL EMBEDDING DIMENSION: {best_embed_dim}")
    print(f"   - Best Average Minimum Distance: {best_distance:.6f}")
    print(f"   - Improvement over embed_dim=64: {best_distance - results[64]['avg_min_distance']:.6f}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   - Use embed_dim = {best_embed_dim} for better character discriminability")
    print(f"   - Expected reduction in character substitution errors")
    print(f"   - Trade-off: Higher embed_dim increases computational cost")

    if best_distance > results[64]['avg_min_distance'] * 1.5:
        print(f"   - Significant improvement detected - strongly recommended")
    elif best_distance > results[64]['avg_min_distance'] * 1.2:
        print(f"   - Moderate improvement - recommended for accuracy-critical applications")
    else:
        print(f"   - Marginal improvement - consider computational cost vs benefit")

    return results, best_embed_dim

if __name__ == "__main__":
    results, optimal_embed_dim = run_embedding_discriminability_experiment()

    print(f"\n✅ Experiment completed. Optimal embed_dim: {optimal_embed_dim}")
    print("   Results saved as heatmap images (if matplotlib available)")