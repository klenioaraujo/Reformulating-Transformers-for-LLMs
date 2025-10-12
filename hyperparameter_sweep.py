#!/usr/bin/env python3
"""
ΨQRH Hyperparameter Sweep Script
================================

Systematic optimization of hyperparameters for ΨQRH pipeline training.
Tests combinations of learning rates, batch sizes, and optimizers to find
the optimal configuration for semantic alignment training.
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import argparse
import itertools

# Import ΨQRH components for validation
from psiqrh import ΨQRHPipeline
from evaluate_model import ΨQRHEvaluator


class HyperparameterSweep:
    """Manages systematic hyperparameter optimization for ΨQRH training."""

    def __init__(self, train_data_path: str = "data/training_pairs.json",
                 test_data_path: str = "data/test_cases.json",
                 results_dir: str = "results/hyperparameter_sweep"):
        """
        Initialize the hyperparameter sweep.

        Args:
            train_data_path: Path to training data
            test_data_path: Path to test data
            results_dir: Directory to save sweep results
        """
        self.train_data_path = Path(train_data_path)
        self.test_data_path = Path(test_data_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Define hyperparameter search space
        self.hyperparameter_space = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
            'batch_size': [4, 8, 16],
            'optimizer': ['AdamW', 'SGD']
        }

        # Results storage
        self.sweep_results = []
        self.best_config = None
        self.best_score = float('inf')

        print("🎯 ΨQRH Hyperparameter Sweep Initialized")
        print(f"📊 Search Space: {len(list(itertools.product(*self.hyperparameter_space.values())))} combinations")
        print(f"💾 Results will be saved to: {self.results_dir}")

    def generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate all hyperparameter configurations to test.

        Returns:
            List of hyperparameter configurations
        """
        keys = self.hyperparameter_space.keys()
        values = self.hyperparameter_space.values()
        configurations = []

        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            config['config_id'] = len(configurations)
            configurations.append(config)

        return configurations

    def run_single_training(self, config: Dict[str, Any], epochs: int = 2) -> Dict[str, Any]:
        """
        Run a single training session with given hyperparameters.

        Args:
            config: Hyperparameter configuration
            epochs: Number of epochs to train

        Returns:
            Training results dictionary
        """
        config_id = config['config_id']
        print(f"\n🔬 Testing Configuration {config_id}: {config}")

        # Create unique checkpoint directory for this config
        config_dir = self.results_dir / f"config_{config_id}"
        config_dir.mkdir(exist_ok=True)

        # Prepare training command
        cmd = [
            "python3", "train_pipeline.py",
            "--data-path", str(self.train_data_path),
            "--epochs", str(epochs),
            "--batch-size", str(config['batch_size']),
            "--device", "cpu"
        ]

        # Set environment variables for hyperparameters
        env = os.environ.copy()
        env['PSIQRH_LR'] = str(config['learning_rate'])
        env['PSIQRH_OPTIMIZER'] = config['optimizer']

        print(f"🚀 Executing: {' '.join(cmd)}")
        print(f"   📊 LR: {config['learning_rate']}, Batch: {config['batch_size']}, Optimizer: {config['optimizer']}")

        # Run training
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per config
            )

            training_time = time.time() - start_time

            # Extract final loss from output (simple parsing)
            final_loss = self._extract_final_loss(result.stdout)

            # Check if model was saved
            model_saved = (Path("models/checkpoints") / "best_model.pt").exists()

            config_result = {
                'config_id': config_id,
                'config': config.copy(),
                'success': result.returncode == 0,
                'final_loss': final_loss,
                'training_time': training_time,
                'model_saved': model_saved,
                'stdout': result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
                'stderr': result.stderr[-1000:] if result.stderr else "",
                'timestamp': datetime.now().isoformat()
            }

            if result.returncode == 0:
                print(".6f"            else:
                print(f"❌ Failed (exit code: {result.returncode})")

        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after {300}s")
            config_result = {
                'config_id': config_id,
                'config': config.copy(),
                'success': False,
                'error': 'timeout',
                'training_time': 300,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"💥 Error: {e}")
            config_result = {
                'config_id': config_id,
                'config': config.copy(),
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

        # Save individual config results
        result_file = config_dir / "result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(config_result, f, indent=2, ensure_ascii=False)

        return config_result

    def _extract_final_loss(self, stdout: str) -> float:
        """
        Extract the final loss value from training output.

        Args:
            stdout: Training stdout output

        Returns:
            Final loss value or inf if not found
        """
        try:
            # Look for loss patterns in the output
            lines = stdout.split('\n')
            for line in reversed(lines):
                if 'Loss =' in line or 'loss:' in line.lower():
                    # Extract numeric value
                    import re
                    match = re.search(r'(\d+\.\d+)', line)
                    if match:
                        return float(match.group(1))
                elif 'final_epoch_loss' in line:
                    # Try to extract from variable assignments
                    match = re.search(r'(\d+\.\d+)', line)
                    if match:
                        return float(match.group(1))
        except Exception:
            pass

        return float('inf')

    def evaluate_configuration(self, config: Dict[str, Any]) -> float:
        """
        Evaluate a trained configuration using validation metrics.

        Args:
            config: Hyperparameter configuration

        Returns:
            Validation score (lower is better)
        """
        config_id = config['config_id']
        model_path = Path("models/checkpoints") / "best_model.pt"

        if not model_path.exists():
            print(f"⚠️  No model found for config {config_id}")
            return float('inf')

        try:
            # Quick evaluation with limited test cases
            evaluator = ΨQRHEvaluator(model_path=str(model_path), device='cpu')

            # Load a subset of test data for quick evaluation
            test_cases = evaluator.load_test_data(str(self.test_data_path))
            test_cases = test_cases[:3]  # Only evaluate first 3 cases for speed

            results = evaluator.evaluate_all_cases(test_cases)

            # Use average BLEU score as validation metric (higher is better, so invert)
            avg_bleu = results.get('summary', {}).get('average_bleu_score', 0.0)
            validation_score = 1.0 - avg_bleu  # Convert to loss (lower is better)

            print(".4f"
            return validation_score

        except Exception as e:
            print(f"⚠️  Evaluation failed for config {config_id}: {e}")
            return float('inf')

    def run_sweep(self, epochs_per_config: int = 2, evaluate_each: bool = True) -> Dict[str, Any]:
        """
        Run the complete hyperparameter sweep.

        Args:
            epochs_per_config: Number of epochs to train each configuration
            evaluate_each: Whether to evaluate each configuration after training

        Returns:
            Sweep results summary
        """
        print("🎯 Starting Hyperparameter Sweep")
        print("=" * 50)

        # Generate all configurations
        configurations = self.generate_configurations()
        print(f"📊 Testing {len(configurations)} hyperparameter combinations")

        # Run sweep
        for i, config in enumerate(configurations, 1):
            print(f"\n🔄 Configuration {i}/{len(configurations)}")

            # Train configuration
            result = self.run_single_training(config, epochs_per_config)
            self.sweep_results.append(result)

            # Evaluate if requested and training was successful
            if evaluate_each and result.get('success', False):
                validation_score = self.evaluate_configuration(config)
                result['validation_score'] = validation_score

                # Update best configuration
                if validation_score < self.best_score:
                    self.best_score = validation_score
                    self.best_config = config.copy()
                    print(".4f"
            # Save intermediate results
            self._save_intermediate_results()

            # Small delay between configurations
            time.sleep(1)

        # Generate final report
        summary = self._generate_summary_report()
        self._save_final_report(summary)

        print("\n🎉 Hyperparameter Sweep Completed!")
        print("=" * 50)
        if self.best_config:
            print("🏆 Best Configuration Found:")
            print(f"   📊 Config: {self.best_config}")
            print(".4f"
        return summary

    def _save_intermediate_results(self):
        """Save intermediate sweep results."""
        intermediate_file = self.results_dir / "intermediate_results.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(self.sweep_results, f, indent=2, ensure_ascii=False)

    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        successful_configs = [r for r in self.sweep_results if r.get('success', False)]
        failed_configs = [r for r in self.sweep_results if not r.get('success', False)]

        summary = {
            'sweep_timestamp': datetime.now().isoformat(),
            'total_configurations': len(self.sweep_results),
            'successful_configurations': len(successful_configs),
            'failed_configurations': len(failed_configs),
            'best_configuration': self.best_config,
            'best_validation_score': self.best_score,
            'hyperparameter_space': self.hyperparameter_space,
            'all_results': self.sweep_results
        }

        # Add statistics
        if successful_configs:
            losses = [r.get('final_loss', float('inf')) for r in successful_configs if r.get('final_loss') != float('inf')]
            if losses:
                summary['loss_statistics'] = {
                    'mean_loss': sum(losses) / len(losses),
                    'min_loss': min(losses),
                    'max_loss': max(losses)
                }

        return summary

    def _save_final_report(self, summary: Dict[str, Any]):
        """Save the final sweep report."""
        report_file = self.results_dir / "sweep_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Generate human-readable report
        readable_report = self.results_dir / "sweep_report.md"
        with open(readable_report, 'w', encoding='utf-8') as f:
            f.write("# ΨQRH Hyperparameter Sweep Report\n\n")
            f.write(f"**Timestamp:** {summary['sweep_timestamp']}\n\n")
            f.write(f"**Total Configurations:** {summary['total_configurations']}\n")
            f.write(f"**Successful:** {summary['successful_configurations']}\n")
            f.write(f"**Failed:** {summary['failed_configurations']}\n\n")

            if summary.get('best_configuration'):
                f.write("## Best Configuration\n\n")
                f.write(f"**Config:** {summary['best_configuration']}\n")
                f.write(".4f"                f.write("\n## Hyperparameter Space\n\n")
                for param, values in summary['hyperparameter_space'].items():
                    f.write(f"- **{param}:** {values}\n")

            if 'loss_statistics' in summary:
                f.write("\n## Loss Statistics\n\n")
                stats = summary['loss_statistics']
                f.write(".6f"                f.write(".6f"                f.write(".6f"
        print(f"💾 Final report saved to: {readable_report}")


def main():
    """Main sweep execution function."""
    parser = argparse.ArgumentParser(description="ΨQRH Hyperparameter Sweep")
    parser.add_argument('--train-data', type=str, default='data/training_pairs.json',
                       help='Path to training data')
    parser.add_argument('--test-data', type=str, default='data/test_cases.json',
                       help='Path to test data')
    parser.add_argument('--results-dir', type=str, default='results/hyperparameter_sweep',
                       help='Directory to save results')
    parser.add_argument('--epochs-per-config', type=int, default=2,
                       help='Epochs to train each configuration')
    parser.add_argument('--no-evaluation', action='store_true',
                       help='Skip evaluation phase for faster sweep')

    args = parser.parse_args()

    print("🎯 ΨQRH Hyperparameter Optimization Campaign")
    print("=" * 60)

    # Initialize sweep
    sweep = HyperparameterSweep(
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        results_dir=args.results_dir
    )

    # Run sweep
    summary = sweep.run_sweep(
        epochs_per_config=args.epochs_per_config,
        evaluate_each=not args.no_evaluation
    )

    # Print final results
    print("\n🏆 SWEEP RESULTS SUMMARY")
    print("=" * 40)

    if sweep.best_config:
        print("✅ Best Configuration Found:")
        print(f"   📊 {sweep.best_config}")
        print(".4f"
        print("
💡 Use this configuration for full training:"
        print(f"   make train EPOCHS=50 BATCH_SIZE={sweep.best_config['batch_size']} LR={sweep.best_config['learning_rate']}")
    else:
        print("❌ No successful configurations found")
        print("   Check the logs for issues")

    print(f"\n📊 Full results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()