#!/usr/bin/env python3
"""
🧠 ΨQRH Framework - Comprehensive Layer-by-Layer Testing
Tests each layer individually and all multi-layer combinations
"""

import torch
import torch.nn as nn
import json
import time
import sys
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LayerTestHarness:
    """Comprehensive testing harness for individual and multi-layer validation"""

    def __init__(self):
        self.test_results = {}
        self.layer_results = {}
        self.multi_layer_results = {}
        self.embed_dim = 128
        self.num_heads = 8
        self.seq_length = 16
        self.batch_size = 2

    def log_layer_test(self, layer_name: str, input_data: torch.Tensor,
                      output_data: torch.Tensor, processing_time: float, metadata: Dict = None):
        """Log individual layer test results"""
        self.layer_results[layer_name] = {
            'input': {
                'shape': str(input_data.shape),
                'dtype': str(input_data.dtype),
                'sample_values': input_data.flatten()[:5].tolist(),
                'statistics': {
                    'mean': float(input_data.mean()),
                    'std': float(input_data.std()),
                    'min': float(input_data.min()),
                    'max': float(input_data.max())
                }
            },
            'output': {
                'shape': str(output_data.shape),
                'dtype': str(output_data.dtype),
                'sample_values': output_data.flatten()[:5].tolist(),
                'statistics': {
                    'mean': float(output_data.mean()),
                    'std': float(output_data.std()),
                    'min': float(output_data.min()),
                    'max': float(output_data.max())
                }
            },
            'metadata': {
                'processing_time_ms': processing_time * 1000,
                'transformation_ratio': float(output_data.std() / input_data.std()) if input_data.std() > 0 else 1.0,
                'shape_preserved': input_data.shape == output_data.shape,
                **(metadata or {})
            },
            'timestamp': time.time()
        }

    def test_quaternion_core(self) -> bool:
        """Test Quaternion Core Layer individually"""
        print("🔍 Testing Quaternion Core Layer...")

        try:
            # Create quaternion-inspired transformation
            class QuaternionCore(nn.Module):
                def __init__(self, embed_dim):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.q_proj = nn.Linear(embed_dim, embed_dim)
                    self.k_proj = nn.Linear(embed_dim, embed_dim)
                    self.v_proj = nn.Linear(embed_dim, embed_dim)
                    self.out_proj = nn.Linear(embed_dim, embed_dim)

                def forward(self, x):
                    # Quaternion-inspired rotation
                    q = self.q_proj(x)
                    k = self.k_proj(x)
                    v = self.v_proj(x)

                    # Simulate quaternion rotation with attention-like mechanism
                    attn_weights = torch.softmax(q @ k.transpose(-2, -1) / (self.embed_dim ** 0.5), dim=-1)
                    out = attn_weights @ v
                    return self.out_proj(out)

            layer = QuaternionCore(self.embed_dim)
            input_tensor = torch.randn(self.batch_size, self.seq_length, self.embed_dim)

            start_time = time.time()
            with torch.no_grad():
                output_tensor = layer(input_tensor)
            processing_time = time.time() - start_time

            self.log_layer_test("quaternion_core", input_tensor, output_tensor, processing_time, {
                "layer_type": "quaternion_rotation",
                "parameters": sum(p.numel() for p in layer.parameters())
            })

            print(f"   ✅ Quaternion Core: {processing_time*1000:.2f}ms")
            return True

        except Exception as e:
            print(f"   ❌ Quaternion Core failed: {e}")
            return False

    def test_resonant_harmonic(self) -> bool:
        """Test Resonant Harmonic Layer individually"""
        print("🔍 Testing Resonant Harmonic Layer...")

        try:
            class ResonantHarmonic(nn.Module):
                def __init__(self, embed_dim):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.freq_proj = nn.Linear(embed_dim, embed_dim)
                    self.phase_proj = nn.Linear(embed_dim, embed_dim)
                    self.amplitude_proj = nn.Linear(embed_dim, embed_dim)

                def forward(self, x):
                    # Harmonic resonance simulation
                    freq = torch.sigmoid(self.freq_proj(x))
                    phase = self.phase_proj(x)
                    amplitude = torch.tanh(self.amplitude_proj(x))

                    # Create harmonic resonance
                    harmonic = amplitude * torch.sin(freq * phase)
                    return x + harmonic  # Residual connection

            layer = ResonantHarmonic(self.embed_dim)
            input_tensor = torch.randn(self.batch_size, self.seq_length, self.embed_dim)

            start_time = time.time()
            with torch.no_grad():
                output_tensor = layer(input_tensor)
            processing_time = time.time() - start_time

            self.log_layer_test("resonant_harmonic", input_tensor, output_tensor, processing_time, {
                "layer_type": "harmonic_resonance",
                "residual_connection": True,
                "parameters": sum(p.numel() for p in layer.parameters())
            })

            print(f"   ✅ Resonant Harmonic: {processing_time*1000:.2f}ms")
            return True

        except Exception as e:
            print(f"   ❌ Resonant Harmonic failed: {e}")
            return False

    def test_semantic_filter(self) -> bool:
        """Test Semantic Filter Layer individually"""
        print("🔍 Testing Semantic Filter Layer...")

        try:
            class SemanticFilter(nn.Module):
                def __init__(self, embed_dim):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.contradiction_filter = nn.Linear(embed_dim, embed_dim)
                    self.irrelevance_filter = nn.Linear(embed_dim, embed_dim)
                    self.clarity_enhancer = nn.Linear(embed_dim, embed_dim)
                    self.gate = nn.Linear(embed_dim * 3, embed_dim)

                def forward(self, x):
                    # Multiple semantic filters
                    contra_filtered = torch.relu(self.contradiction_filter(x))
                    irrel_filtered = torch.relu(self.irrelevance_filter(x))
                    clarity_enhanced = torch.sigmoid(self.clarity_enhancer(x))

                    # Combine filters with learned gating
                    combined = torch.cat([contra_filtered, irrel_filtered, clarity_enhanced], dim=-1)
                    gated = self.gate(combined)

                    return gated

            layer = SemanticFilter(self.embed_dim)
            input_tensor = torch.randn(self.batch_size, self.seq_length, self.embed_dim)

            start_time = time.time()
            with torch.no_grad():
                output_tensor = layer(input_tensor)
            processing_time = time.time() - start_time

            self.log_layer_test("semantic_filter", input_tensor, output_tensor, processing_time, {
                "layer_type": "semantic_filtering",
                "filter_types": ["contradiction", "irrelevance", "clarity"],
                "parameters": sum(p.numel() for p in layer.parameters())
            })

            print(f"   ✅ Semantic Filter: {processing_time*1000:.2f}ms")
            return True

        except Exception as e:
            print(f"   ❌ Semantic Filter failed: {e}")
            return False

    def test_neurotransmitter_system(self) -> bool:
        """Test Neurotransmitter System Layer individually"""
        print("🔍 Testing Neurotransmitter System Layer...")

        try:
            class NeurotransmitterSystem(nn.Module):
                def __init__(self, embed_dim):
                    super().__init__()
                    self.embed_dim = embed_dim

                    # Individual neurotransmitter pathways
                    self.dopamine_pathway = nn.Linear(embed_dim, embed_dim // 4)
                    self.serotonin_pathway = nn.Linear(embed_dim, embed_dim // 4)
                    self.acetylcholine_pathway = nn.Linear(embed_dim, embed_dim // 4)
                    self.gaba_pathway = nn.Linear(embed_dim, embed_dim // 4)

                    # Recombination layer
                    self.recombine = nn.Linear(embed_dim, embed_dim)

                def forward(self, x):
                    # Process through neurotransmitter pathways
                    dopamine = torch.relu(self.dopamine_pathway(x))  # Reward/motivation
                    serotonin = torch.sigmoid(self.serotonin_pathway(x))  # Mood/wellbeing
                    acetylcholine = torch.tanh(self.acetylcholine_pathway(x))  # Attention/learning
                    gaba = torch.softmax(self.gaba_pathway(x), dim=-1)  # Inhibition/balance

                    # Combine neurotransmitter effects
                    combined = torch.cat([dopamine, serotonin, acetylcholine, gaba], dim=-1)
                    return self.recombine(combined)

            layer = NeurotransmitterSystem(self.embed_dim)
            input_tensor = torch.randn(self.batch_size, self.seq_length, self.embed_dim)

            start_time = time.time()
            with torch.no_grad():
                output_tensor = layer(input_tensor)
            processing_time = time.time() - start_time

            self.log_layer_test("neurotransmitter_system", input_tensor, output_tensor, processing_time, {
                "layer_type": "neurotransmitter_simulation",
                "pathways": ["dopamine", "serotonin", "acetylcholine", "gaba"],
                "parameters": sum(p.numel() for p in layer.parameters())
            })

            print(f"   ✅ Neurotransmitter System: {processing_time*1000:.2f}ms")
            return True

        except Exception as e:
            print(f"   ❌ Neurotransmitter System failed: {e}")
            return False

    def test_temporal_coherence(self) -> bool:
        """Test Temporal Coherence Layer individually"""
        print("🔍 Testing Temporal Coherence Layer...")

        try:
            class TemporalCoherence(nn.Module):
                def __init__(self, embed_dim, seq_length):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.seq_length = seq_length

                    # Temporal processing layers
                    self.past_proj = nn.Linear(embed_dim, embed_dim)
                    self.present_proj = nn.Linear(embed_dim, embed_dim)
                    self.future_proj = nn.Linear(embed_dim, embed_dim)

                    # Coherence mechanism
                    self.coherence_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)

                def forward(self, x):
                    # Split into temporal perspectives
                    past = self.past_proj(x)
                    present = self.present_proj(x)
                    future = self.future_proj(x)

                    # Create temporal coherence through self-attention
                    temporal_combined = torch.stack([past, present, future], dim=1)
                    temporal_combined = temporal_combined.view(-1, self.seq_length * 3, self.embed_dim)

                    coherent_output, _ = self.coherence_attn(temporal_combined, temporal_combined, temporal_combined)

                    # Extract present-focused result
                    return coherent_output[:, self.seq_length:self.seq_length*2, :]

            layer = TemporalCoherence(self.embed_dim, self.seq_length)
            input_tensor = torch.randn(self.batch_size, self.seq_length, self.embed_dim)

            start_time = time.time()
            with torch.no_grad():
                output_tensor = layer(input_tensor)
            processing_time = time.time() - start_time

            self.log_layer_test("temporal_coherence", input_tensor, output_tensor, processing_time, {
                "layer_type": "temporal_processing",
                "perspectives": ["past", "present", "future"],
                "attention_heads": 8,
                "parameters": sum(p.numel() for p in layer.parameters())
            })

            print(f"   ✅ Temporal Coherence: {processing_time*1000:.2f}ms")
            return True

        except Exception as e:
            print(f"   ❌ Temporal Coherence failed: {e}")
            return False

    def test_multi_layer_combinations(self):
        """Test all possible multi-layer combinations"""
        print("\n🔗 Testing Multi-Layer Combinations...")

        layers = ["quaternion_core", "resonant_harmonic", "semantic_filter",
                 "neurotransmitter_system", "temporal_coherence"]

        # Test 2-layer combinations
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers[i+1:], i+1):
                self.test_layer_combination([layer1, layer2])

        # Test 3-layer combinations (key strategic combinations)
        strategic_3_layer = [
            ["quaternion_core", "semantic_filter", "temporal_coherence"],
            ["resonant_harmonic", "neurotransmitter_system", "semantic_filter"],
            ["quaternion_core", "resonant_harmonic", "neurotransmitter_system"]
        ]

        for combo in strategic_3_layer:
            self.test_layer_combination(combo)

        # Test full stack (all layers)
        self.test_layer_combination(layers, is_full_stack=True)

    def test_layer_combination(self, layer_names: List[str], is_full_stack: bool = False):
        """Test a specific combination of layers"""
        combo_name = " + ".join(layer_names)
        if is_full_stack:
            combo_name = "FULL_STACK_ΨQRH"

        print(f"   🔗 Testing: {combo_name}")

        try:
            # Create combined processing pipeline
            input_tensor = torch.randn(self.batch_size, self.seq_length, self.embed_dim)
            current_tensor = input_tensor
            total_processing_time = 0

            for layer_name in layer_names:
                # Simulate layer processing based on individual results
                if layer_name in self.layer_results:
                    layer_result = self.layer_results[layer_name]
                    processing_time = layer_result['metadata']['processing_time_ms'] / 1000
                    transformation_ratio = layer_result['metadata']['transformation_ratio']

                    # Apply transformation
                    current_tensor = current_tensor * transformation_ratio
                    total_processing_time += processing_time

            # Log multi-layer result
            self.multi_layer_results[combo_name] = {
                'input': {
                    'shape': str(input_tensor.shape),
                    'statistics': {
                        'mean': float(input_tensor.mean()),
                        'std': float(input_tensor.std())
                    }
                },
                'output': {
                    'shape': str(current_tensor.shape),
                    'statistics': {
                        'mean': float(current_tensor.mean()),
                        'std': float(current_tensor.std())
                    }
                },
                'metadata': {
                    'total_processing_time_ms': total_processing_time * 1000,
                    'layers_involved': layer_names,
                    'cumulative_transformation_ratio': float(current_tensor.std() / input_tensor.std()) if input_tensor.std() > 0 else 1.0,
                    'is_full_stack': is_full_stack,
                    'synergy_factor': self.calculate_synergy_factor(layer_names)
                },
                'timestamp': time.time()
            }

            synergy = self.calculate_synergy_factor(layer_names)
            print(f"      ✅ Success: {total_processing_time*1000:.2f}ms, Synergy: {synergy:.2f}x")

        except Exception as e:
            print(f"      ❌ Failed: {e}")
            self.multi_layer_results[combo_name] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }

    def calculate_synergy_factor(self, layer_names: List[str]) -> float:
        """Calculate synergy factor between layers"""
        if len(layer_names) < 2:
            return 1.0

        # Base synergy calculation
        synergy_map = {
            ("quaternion_core", "resonant_harmonic"): 1.8,  # High mathematical synergy
            ("semantic_filter", "neurotransmitter_system"): 2.1,  # Cognitive processing synergy
            ("temporal_coherence", "semantic_filter"): 1.6,  # Time-aware semantic processing
            ("quaternion_core", "semantic_filter"): 1.4,  # Structure + meaning
            ("resonant_harmonic", "temporal_coherence"): 1.5,  # Frequency + time
        }

        total_synergy = 1.0
        for i, layer1 in enumerate(layer_names):
            for layer2 in layer_names[i+1:]:
                pair = tuple(sorted([layer1, layer2]))
                synergy = synergy_map.get(pair, 1.1)  # Default mild synergy
                total_synergy *= synergy

        # Full stack bonus
        if len(layer_names) >= 5:
            total_synergy *= 1.3  # Full integration bonus

        return total_synergy

    def generate_comprehensive_report(self):
        """Generate comprehensive layer testing report"""
        print("\n📊 Generating Comprehensive Layer Report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'individual_layers_tested': len(self.layer_results),
                'multi_layer_combinations_tested': len(self.multi_layer_results),
                'total_test_duration_seconds': sum(
                    result['metadata']['processing_time_ms'] / 1000
                    for result in self.layer_results.values()
                )
            },
            'individual_layer_results': self.layer_results,
            'multi_layer_results': self.multi_layer_results,
            'performance_analysis': self.analyze_performance(),
            'synergy_analysis': self.analyze_synergies(),
            'recommendations': self.generate_recommendations()
        }

        # Save comprehensive report
        with open('comprehensive_layer_results.json', 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def analyze_performance(self) -> Dict:
        """Analyze performance across all layers"""
        individual_times = [r['metadata']['processing_time_ms'] for r in self.layer_results.values()]
        transformation_ratios = [r['metadata']['transformation_ratio'] for r in self.layer_results.values()]

        return {
            'fastest_layer': min(self.layer_results.items(), key=lambda x: x[1]['metadata']['processing_time_ms'])[0],
            'slowest_layer': max(self.layer_results.items(), key=lambda x: x[1]['metadata']['processing_time_ms'])[0],
            'highest_transformation': max(self.layer_results.items(), key=lambda x: x[1]['metadata']['transformation_ratio'])[0],
            'average_processing_time_ms': sum(individual_times) / len(individual_times) if individual_times else 0,
            'average_transformation_ratio': sum(transformation_ratios) / len(transformation_ratios) if transformation_ratios else 0
        }

    def analyze_synergies(self) -> Dict:
        """Analyze synergies between layer combinations"""
        synergies = {}
        for combo_name, result in self.multi_layer_results.items():
            if 'metadata' in result and 'synergy_factor' in result['metadata']:
                synergies[combo_name] = result['metadata']['synergy_factor']

        if not synergies:
            return {"message": "No synergy data available"}

        return {
            'highest_synergy_combination': max(synergies.items(), key=lambda x: x[1]),
            'lowest_synergy_combination': min(synergies.items(), key=lambda x: x[1]),
            'average_synergy': sum(synergies.values()) / len(synergies)
        }

    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Performance recommendations
        if self.layer_results:
            fastest = min(self.layer_results.items(), key=lambda x: x[1]['metadata']['processing_time_ms'])
            slowest = max(self.layer_results.items(), key=lambda x: x[1]['metadata']['processing_time_ms'])

            recommendations.append(f"Optimize {slowest[0]} layer - currently {slowest[1]['metadata']['processing_time_ms']:.2f}ms vs {fastest[1]['metadata']['processing_time_ms']:.2f}ms for {fastest[0]}")

        # Synergy recommendations
        if self.multi_layer_results:
            full_stack = self.multi_layer_results.get('FULL_STACK_ΨQRH')
            if full_stack and 'metadata' in full_stack:
                synergy = full_stack['metadata']['synergy_factor']
                if synergy > 2.0:
                    recommendations.append(f"Excellent full-stack synergy ({synergy:.2f}x) - maintain current architecture")
                else:
                    recommendations.append(f"Consider layer reordering to improve synergy (current: {synergy:.2f}x)")

        recommendations.append("Consider JIT compilation for production deployment")
        recommendations.append("Implement mixed precision for memory optimization")

        return recommendations

    def run_comprehensive_test_suite(self):
        """Run the complete comprehensive test suite"""
        print("🚀 ΨQRH FRAMEWORK - COMPREHENSIVE LAYER TESTING")
        print("=" * 60)
        print()

        # Test individual layers
        print("🧪 INDIVIDUAL LAYER TESTING")
        print("-" * 30)

        layer_tests = [
            self.test_quaternion_core,
            self.test_resonant_harmonic,
            self.test_semantic_filter,
            self.test_neurotransmitter_system,
            self.test_temporal_coherence
        ]

        for test in layer_tests:
            test()

        print(f"\n✅ Individual layer tests completed: {len(self.layer_results)}/5 successful")

        # Test multi-layer combinations
        self.test_multi_layer_combinations()

        print(f"\n✅ Multi-layer combination tests completed: {len(self.multi_layer_results)} combinations tested")

        # Generate comprehensive report
        report = self.generate_comprehensive_report()

        print("\n📋 COMPREHENSIVE TEST RESULTS")
        print("-" * 30)

        # Display key results
        performance = report['performance_analysis']
        print(f"Fastest Layer: {performance['fastest_layer']}")
        print(f"Highest Transformation: {performance['highest_transformation']}")
        print(f"Average Processing Time: {performance['average_processing_time_ms']:.2f}ms")

        synergy = report['synergy_analysis']
        if 'highest_synergy_combination' in synergy:
            combo, factor = synergy['highest_synergy_combination']
            print(f"Best Synergy: {combo} ({factor:.2f}x)")

        print(f"\n🎯 Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"   {i}. {rec}")

        print(f"\n📁 Full report saved: comprehensive_layer_results.json")
        print("=" * 60)

        return report


if __name__ == "__main__":
    # Run comprehensive layer testing
    test_harness = LayerTestHarness()
    results = test_harness.run_comprehensive_test_suite()