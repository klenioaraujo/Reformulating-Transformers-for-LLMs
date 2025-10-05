#!/usr/bin/env python3
"""
Performance Comparison Framework for ΨQRH Transformer

Comprehensive performance comparison between ΨQRH Transformer and standard PyTorch Transformer
including memory usage, inference speed, and mathematical validation.
"""

import torch
import torch.nn as nn
import time
import psutil
import gc
import sys
import os
from typing import Dict, List, Tuple

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer
from src.core.quaternion_operations import (
    QuaternionLinear, QuaternionLayerNorm, SpectralActivation,
    AdaptiveSpectralDropout, RealTimeFractalAnalyzer
)
from src.validation.mathematical_validation import MathematicalValidator


class PerformanceBenchmark:
    """Comprehensive performance benchmarking framework"""

    def __init__(self, vocab_size: int = 10000, d_model: int = 512, seq_length: int = 256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_length = seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.psiqrh_model = None
        self.standard_model = None
        self.initialize_models()

        # Initialize validator
        self.validator = MathematicalValidator(tolerance=0.05)

    def initialize_models(self):
        """Initialize both ΨQRH and standard transformer models"""
        print("🚀 Initializing models for performance comparison...")

        # ΨQRH Transformer
        self.psiqrh_model = PsiQRHTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=6,
            n_heads=8,
            dim_feedforward=2048
        ).to(self.device)

        # Standard PyTorch Transformer
        self.standard_model = nn.Transformer(
            d_model=self.d_model,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            batch_first=True
        ).to(self.device)

        # Standard embedding and output projection for fair comparison
        self.standard_embedding = nn.Embedding(self.vocab_size, self.d_model).to(self.device)
        self.standard_output_proj = nn.Linear(self.d_model, self.vocab_size).to(self.device)

        print(f"✅ Models initialized on {self.device}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return {
                'cuda_allocated_mb': allocated,
                'cuda_reserved_mb': reserved
            }
        else:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'ram_used_mb': memory_info.rss / 1024**2
            }

    def measure_inference_speed(self, batch_size: int = 4, warmup_iters: int = 10, test_iters: int = 100) -> Dict[str, float]:
        """Measure inference speed for both models"""
        print(f"\n⏱️  Measuring inference speed (batch_size={batch_size})...")

        # Generate test data
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=self.device)

        results = {}

        # Test ΨQRH Transformer
        print("  Testing ΨQRH Transformer...")

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = self.psiqrh_model(input_ids)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(test_iters):
            with torch.no_grad():
                _ = self.psiqrh_model(input_ids)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        psiqrh_time = (time.time() - start_time) / test_iters

        results['psiqrh_inference_time_ms'] = psiqrh_time * 1000

        # Test Standard Transformer
        print("  Testing Standard Transformer...")

        # Standard transformer requires both encoder and decoder inputs
        src = self.standard_embedding(input_ids)
        tgt = torch.zeros_like(src)

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = self.standard_model(src, tgt)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(test_iters):
            with torch.no_grad():
                _ = self.standard_model(src, tgt)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        standard_time = (time.time() - start_time) / test_iters

        results['standard_inference_time_ms'] = standard_time * 1000
        results['speedup_ratio'] = standard_time / psiqrh_time

        return results

    def measure_memory_usage(self, batch_size: int = 4) -> Dict[str, float]:
        """Measure memory usage for both models"""
        print(f"\n💾 Measuring memory usage (batch_size={batch_size})...")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        initial_memory = self.get_memory_usage()

        # Generate test data
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=self.device)

        results = {}

        # Test ΨQRH Transformer memory
        print("  Testing ΨQRH Transformer memory...")
        with torch.no_grad():
            _ = self.psiqrh_model(input_ids)

        psiqrh_memory = self.get_memory_usage()

        # Clear memory
        del _
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test Standard Transformer memory
        print("  Testing Standard Transformer memory...")
        src = self.standard_embedding(input_ids)
        tgt = torch.zeros_like(src)

        with torch.no_grad():
            _ = self.standard_model(src, tgt)

        standard_memory = self.get_memory_usage()

        # Calculate memory usage
        if 'cuda_allocated_mb' in initial_memory:
            results['psiqrh_memory_mb'] = psiqrh_memory['cuda_allocated_mb'] - initial_memory['cuda_allocated_mb']
            results['standard_memory_mb'] = standard_memory['cuda_allocated_mb'] - initial_memory['cuda_allocated_mb']
        else:
            results['psiqrh_memory_mb'] = psiqrh_memory['ram_used_mb'] - initial_memory['ram_used_mb']
            results['standard_memory_mb'] = standard_memory['ram_used_mb'] - initial_memory['ram_used_mb']

        results['memory_reduction_ratio'] = results['standard_memory_mb'] / results['psiqrh_memory_mb']

        return results

    def validate_mathematical_properties(self) -> Dict[str, bool]:
        """Validate mathematical properties of ΨQRH Transformer"""
        print(f"\n🔬 Validating mathematical properties...")

        # Generate test data
        input_ids = torch.randint(0, self.vocab_size, (1, 64), device=self.device)

        # Run comprehensive validation
        validation_results = self.validator.comprehensive_validation(
            self.psiqrh_model, input_ids
        )

        # Extract key validation results
        results = {
            'energy_conservation': validation_results.get('energy_conservation', False),
            'unitarity': validation_results.get('unitarity', False),
            'numerical_stability': validation_results.get('numerical_stability', False),
            'quaternion_properties': validation_results.get('quaternion_properties', False),
            'spectral_operations': validation_results.get('spectral_operations', False)
        }

        return results

    def run_comprehensive_benchmark(self, batch_sizes: List[int] = [1, 4, 8]) -> Dict:
        """Run comprehensive performance benchmark"""
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("=" * 60)

        benchmark_results = {}

        # Mathematical validation
        math_validation = self.validate_mathematical_properties()
        benchmark_results['mathematical_validation'] = math_validation

        # Performance metrics for different batch sizes
        performance_results = {}

        for batch_size in batch_sizes:
            print(f"\n📊 Testing batch_size={batch_size}")
            print("-" * 40)

            batch_results = {}

            # Inference speed
            speed_results = self.measure_inference_speed(batch_size=batch_size)
            batch_results.update(speed_results)

            # Memory usage
            memory_results = self.measure_memory_usage(batch_size=batch_size)
            batch_results.update(memory_results)

            performance_results[f'batch_{batch_size}'] = batch_results

        benchmark_results['performance'] = performance_results

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(performance_results)
        benchmark_results['overall_metrics'] = overall_metrics

        return benchmark_results

    def _calculate_overall_metrics(self, performance_results: Dict) -> Dict:
        """Calculate overall performance metrics"""

        # Average across batch sizes
        avg_speedup = 0
        avg_memory_reduction = 0
        batch_count = 0

        for batch_key, results in performance_results.items():
            if 'speedup_ratio' in results and 'memory_reduction_ratio' in results:
                avg_speedup += results['speedup_ratio']
                avg_memory_reduction += results['memory_reduction_ratio']
                batch_count += 1

        if batch_count > 0:
            avg_speedup /= batch_count
            avg_memory_reduction /= batch_count

        return {
            'average_speedup': avg_speedup,
            'average_memory_reduction': avg_memory_reduction,
            'performance_improvement': (avg_speedup + avg_memory_reduction) / 2
        }

    def generate_report(self, benchmark_results: Dict) -> str:
        """Generate comprehensive performance report"""

        report = []
        report.append("🎯 ΨQRH TRANSFORMER PERFORMANCE REPORT")
        report.append("=" * 60)

        # Mathematical validation
        math_validation = benchmark_results['mathematical_validation']
        report.append("\n🔬 MATHEMATICAL VALIDATION")
        report.append("-" * 40)

        for prop, valid in math_validation.items():
            status = "✅ PASS" if valid else "❌ FAIL"
            report.append(f"  {prop.replace('_', ' ').title()}: {status}")

        # Performance results
        performance_results = benchmark_results['performance']
        report.append("\n📊 PERFORMANCE COMPARISON")
        report.append("-" * 40)

        for batch_key, results in performance_results.items():
            batch_size = batch_key.replace('batch_', '')
            report.append(f"\n  Batch Size: {batch_size}")
            report.append(f"    ΨQRH Inference Time: {results.get('psiqrh_inference_time_ms', 0):.2f} ms")
            report.append(f"    Standard Inference Time: {results.get('standard_inference_time_ms', 0):.2f} ms")
            report.append(f"    Speedup Ratio: {results.get('speedup_ratio', 0):.2f}x")
            report.append(f"    ΨQRH Memory: {results.get('psiqrh_memory_mb', 0):.1f} MB")
            report.append(f"    Standard Memory: {results.get('standard_memory_mb', 0):.1f} MB")
            report.append(f"    Memory Reduction: {results.get('memory_reduction_ratio', 0):.2f}x")

        # Overall metrics
        overall = benchmark_results['overall_metrics']
        report.append("\n📈 OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"  Average Speedup: {overall.get('average_speedup', 0):.2f}x")
        report.append(f"  Average Memory Reduction: {overall.get('average_memory_reduction', 0):.2f}x")
        report.append(f"  Overall Performance Improvement: {overall.get('performance_improvement', 0):.2f}x")

        # Summary
        report.append("\n🎯 SUMMARY")
        report.append("-" * 40)

        speedup = overall.get('average_speedup', 1.0)
        memory_reduction = overall.get('average_memory_reduction', 1.0)

        if speedup > 1.0 and memory_reduction > 1.0:
            report.append("✅ ΨQRH Transformer demonstrates significant improvements:")
            report.append(f"   • {speedup:.1f}x faster inference")
            report.append(f"   • {memory_reduction:.1f}x less memory usage")
        else:
            report.append("⚠️  ΨQRH Transformer shows mixed results:")
            if speedup > 1.0:
                report.append(f"   • {speedup:.1f}x faster inference")
            else:
                report.append(f"   • {1/speedup:.1f}x slower inference")

            if memory_reduction > 1.0:
                report.append(f"   • {memory_reduction:.1f}x less memory usage")
            else:
                report.append(f"   • {1/memory_reduction:.1f}x more memory usage")

        return "\n".join(report)


def main():
    """Main function to run performance comparison"""
    print("ΨQRH Transformer Performance Comparison")
    print("=" * 60)

    # Initialize benchmark
    benchmark = PerformanceBenchmark(
        vocab_size=10000,
        d_model=512,
        seq_length=256
    )

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(batch_sizes=[1, 4, 8])

    # Generate and print report
    report = benchmark.generate_report(results)
    print(report)

    # Save results to file
    with open('performance_comparison_results.txt', 'w') as f:
        f.write(report)

    print("\n📄 Results saved to 'performance_comparison_results.txt'")


if __name__ == "__main__":
    main()