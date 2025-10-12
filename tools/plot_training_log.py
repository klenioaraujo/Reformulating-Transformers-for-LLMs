#!/usr/bin/env python3
"""
ΨQRH Training Log Plotter
=========================

Visualizes training curves from ΨQRH pipeline logs to analyze learning dynamics,
detect overfitting/underfitting, and monitor training progress.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class TrainingLogPlotter:
    """Plots training curves and analyzes learning dynamics."""

    def __init__(self, log_file: str, output_dir: str = "results/plots"):
        """
        Initialize the plotter.

        Args:
            log_file: Path to the training log file
            output_dir: Directory to save plots
        """
        self.log_file = Path(log_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extracted data
        self.training_losses = []
        self.validation_losses = []
        self.learning_rates = []
        self.epochs = []

        print(f"📊 Loading training log: {log_file}")

    def parse_log_file(self) -> bool:
        """
        Parse the training log file to extract loss values.

        Returns:
            True if parsing was successful
        """
        if not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            return False

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into lines
            lines = content.split('\n')

            current_epoch = 0
            epoch_training_losses = []
            epoch_validation_losses = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for epoch markers
                epoch_match = re.search(r'Epoch (\d+)', line, re.IGNORECASE)
                if epoch_match:
                    # Save previous epoch data if exists
                    if epoch_training_losses:
                        self.training_losses.append(np.mean(epoch_training_losses))
                        self.validation_losses.append(np.mean(epoch_validation_losses) if epoch_validation_losses else 0.0)
                        self.epochs.append(current_epoch)

                    # Start new epoch
                    current_epoch = int(epoch_match.group(1))
                    epoch_training_losses = []
                    epoch_validation_losses = []
                    continue

                # Look for loss values
                loss_match = re.search(r'Loss = (\d+\.\d+)', line, re.IGNORECASE)
                if loss_match:
                    loss_value = float(loss_match.group(1))
                    # Determine if it's training or validation loss based on context
                    if 'validation' in line.lower() or 'val' in line.lower():
                        epoch_validation_losses.append(loss_value)
                    else:
                        epoch_training_losses.append(loss_value)
                    continue

                # Look for learning rate
                lr_match = re.search(r'lr: (\d+\.\d+)', line, re.IGNORECASE)
                if lr_match:
                    self.learning_rates.append(float(lr_match.group(1)))

            # Don't forget the last epoch
            if epoch_training_losses:
                self.training_losses.append(np.mean(epoch_training_losses))
                self.validation_losses.append(np.mean(epoch_validation_losses) if epoch_validation_losses else 0.0)
                self.epochs.append(current_epoch)

            # If no epochs found, try alternative parsing
            if not self.epochs:
                self._parse_alternative_format(content)

            print(f"✅ Parsed {len(self.epochs)} epochs")
            print(f"   📈 Training losses: {len(self.training_losses)}")
            print(f"   📉 Validation losses: {len(self.validation_losses)}")

            return len(self.epochs) > 0

        except Exception as e:
            print(f"❌ Error parsing log file: {e}")
            return False

    def _parse_alternative_format(self, content: str):
        """Alternative parsing for different log formats."""
        # Look for any numeric patterns that might be losses
        loss_pattern = r'(\d+\.\d+)'
        losses = re.findall(loss_pattern, content)

        if losses:
            # Assume every other loss is training/validation
            for i, loss in enumerate(losses):
                if i % 2 == 0:
                    self.training_losses.append(float(loss))
                    self.epochs.append(i // 2 + 1)
                else:
                    self.validation_losses.append(float(loss))

    def plot_learning_curves(self, save_path: Optional[str] = None) -> str:
        """
        Plot training and validation learning curves.

        Args:
            save_path: Path to save the plot (optional)

        Returns:
            Path to the saved plot
        """
        if not self.training_losses:
            print("❌ No training data to plot")
            return ""

        plt.figure(figsize=(12, 8))

        # Plot training loss
        if self.training_losses:
            plt.plot(self.epochs[:len(self.training_losses)],
                    self.training_losses, 'b-', label='Training Loss', linewidth=2, marker='o')

        # Plot validation loss
        if self.validation_losses and any(v > 0 for v in self.validation_losses):
            plt.plot(self.epochs[:len(self.validation_losses)],
                    self.validation_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('ΨQRH Training Learning Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add statistics as text
        if self.training_losses:
            final_train_loss = self.training_losses[-1]
            min_train_loss = min(self.training_losses)

            stats_text = f"Final Training Loss: {final_train_loss:.6f}\n"
            stats_text += f"Min Training Loss: {min_train_loss:.6f}"

            if self.validation_losses and any(v > 0 for v in self.validation_losses):
                final_val_loss = self.validation_losses[-1]
                min_val_loss = min(self.validation_losses)
                stats_text += f"\nFinal Validation Loss: {final_val_loss:.6f}"
                stats_text += f"\nMin Validation Loss: {min_val_loss:.6f}"

            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Determine default save path
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"learning_curves_{timestamp}.png"

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 Learning curves plot saved to: {save_path}")
        return str(save_path)

    def analyze_learning_dynamics(self) -> Dict[str, Any]:
        """
        Analyze the learning dynamics to detect overfitting, underfitting, etc.

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_epochs': len(self.epochs),
            'converged': False,
            'overfitting_detected': False,
            'underfitting_detected': False,
            'learning_stability': 'unknown',
            'recommendations': []
        }

        if not self.training_losses:
            analysis['recommendations'].append("No training data available for analysis")
            return analysis

        # Check for convergence (loss decreasing and stabilizing)
        if len(self.training_losses) >= 3:
            recent_losses = self.training_losses[-3:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

            if abs(loss_trend) < 0.001:  # Very small slope indicates convergence
                analysis['converged'] = True
                analysis['learning_stability'] = 'converged'
            elif loss_trend < -0.01:  # Still decreasing
                analysis['learning_stability'] = 'improving'
            else:  # Plateau or slight increase
                analysis['learning_stability'] = 'plateau'

        # Check for overfitting (training loss << validation loss and gap increasing)
        if (self.validation_losses and len(self.validation_losses) >= 3 and
            any(v > 0 for v in self.validation_losses)):

            # Calculate gap between training and validation loss
            gaps = []
            for t_loss, v_loss in zip(self.training_losses, self.validation_losses):
                if v_loss > 0:
                    gaps.append(v_loss - t_loss)

            if gaps and len(gaps) >= 3:
                # Check if gap is increasing (sign of overfitting)
                gap_trend = np.polyfit(range(len(gaps)), gaps, 1)[0]
                avg_gap = np.mean(gaps)

                if gap_trend > 0.01 and avg_gap > 0.1:  # Gap increasing and significant
                    analysis['overfitting_detected'] = True
                    analysis['recommendations'].append("Overfitting detected: Consider adding regularization (dropout, weight decay)")
                    analysis['recommendations'].append("Early stopping may be beneficial")
                elif avg_gap < 0.01:  # Very small gap
                    analysis['recommendations'].append("Good fit: Training and validation losses are well-aligned")

        # Check for underfitting (high loss values that don't decrease much)
        final_loss = self.training_losses[-1]
        if final_loss > 1.0 and len(self.training_losses) >= 5:
            loss_improvement = self.training_losses[0] - final_loss
            if loss_improvement < 0.1:  # Less than 10% improvement
                analysis['underfitting_detected'] = True
                analysis['recommendations'].append("Underfitting detected: Model may be too simple")
                analysis['recommendations'].append("Consider increasing model capacity or training longer")

        # Overall assessment
        if analysis['overfitting_detected']:
            analysis['overall_assessment'] = 'overfitting'
        elif analysis['underfitting_detected']:
            analysis['overall_assessment'] = 'underfitting'
        elif analysis['converged']:
            analysis['overall_assessment'] = 'well_fitted'
        else:
            analysis['overall_assessment'] = 'still_learning'

        return analysis

    def generate_analysis_report(self, plot_path: str) -> str:
        """
        Generate a comprehensive analysis report.

        Args:
            plot_path: Path to the learning curves plot

        Returns:
            Formatted analysis report
        """
        analysis = self.analyze_learning_dynamics()

        report_lines = []
        report_lines.append("# ΨQRH Training Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Basic statistics
        report_lines.append("## Training Statistics")
        report_lines.append(f"- **Total Epochs:** {analysis['total_epochs']}")
        report_lines.append(f"- **Final Training Loss:** {self.training_losses[-1]:.6f}" if self.training_losses else "- **Final Training Loss:** N/A")

        if self.validation_losses and any(v > 0 for v in self.validation_losses):
            report_lines.append(f"- **Final Validation Loss:** {self.validation_losses[-1]:.6f}")

        report_lines.append("")

        # Learning dynamics analysis
        report_lines.append("## Learning Dynamics Analysis")
        report_lines.append(f"- **Overall Assessment:** {analysis['overall_assessment'].replace('_', ' ').title()}")
        report_lines.append(f"- **Learning Stability:** {analysis['learning_stability'].title()}")
        report_lines.append(f"- **Converged:** {'Yes' if analysis['converged'] else 'No'}")
        report_lines.append(f"- **Overfitting Detected:** {'Yes' if analysis['overfitting_detected'] else 'No'}")
        report_lines.append(f"- **Underfitting Detected:** {'Yes' if analysis['underfitting_detected'] else 'No'}")
        report_lines.append("")

        # Recommendations
        if analysis['recommendations']:
            report_lines.append("## Recommendations")
            for rec in analysis['recommendations']:
                report_lines.append(f"- {rec}")
            report_lines.append("")

        # Plot reference
        report_lines.append("## Learning Curves")
        report_lines.append(f"![Learning Curves]({plot_path})")
        report_lines.append("")

        # Detailed loss progression
        if self.training_losses:
            report_lines.append("## Loss Progression")
            report_lines.append("| Epoch | Training Loss | Validation Loss |")
            report_lines.append("|-------|---------------|-----------------|")

            for i, epoch in enumerate(self.epochs):
                train_loss = self.training_losses[i] if i < len(self.training_losses) else "N/A"
                val_loss = self.validation_losses[i] if i < len(self.validation_losses) and self.validation_losses[i] > 0 else "N/A"
                report_lines.append(f"| {epoch} | {train_loss} | {val_loss} |")

        return "\n".join(report_lines)


def find_latest_log(log_dir: str = "logs/training") -> Optional[str]:
    """
    Find the most recent training log file.

    Args:
        log_dir: Directory containing log files

    Returns:
        Path to the latest log file or None
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    # Find all log files
    log_files = list(log_path.glob("*.log"))
    if not log_files:
        return None

    # Return the most recent one
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    return str(latest_log)


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description="ΨQRH Training Log Plotter")
    parser.add_argument('--log-file', type=str,
                       help='Path to training log file (auto-detects latest if not specified)')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                       help='Directory to save plots and reports')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip learning dynamics analysis')

    args = parser.parse_args()

    print("📊 ΨQRH Training Log Analysis")
    print("=" * 40)

    # Find log file
    log_file = args.log_file
    if not log_file:
        log_file = find_latest_log()
        if log_file:
            print(f"📄 Using latest log file: {log_file}")
        else:
            print("❌ No log files found in logs/training/")
            print("   Run training first or specify --log-file")
            return 1

    # Initialize plotter
    plotter = TrainingLogPlotter(log_file, args.output_dir)

    # Parse log file
    if not plotter.parse_log_file():
        print("❌ Failed to parse log file")
        return 1

    # Generate plot
    plot_path = plotter.plot_learning_curves()

    # Generate analysis report
    if not args.no_analysis:
        analysis = plotter.analyze_learning_dynamics()
        report = plotter.generate_analysis_report(plot_path)

        # Save report
        report_path = Path(args.output_dir) / "training_analysis.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📋 Analysis report saved to: {report_path}")

        # Print key findings
        print("\n🔍 KEY FINDINGS:")
        print(f"   📊 Overall Assessment: {analysis['overall_assessment'].replace('_', ' ').title()}")
        print(f"   🎯 Converged: {'Yes' if analysis['converged'] else 'No'}")
        print(f"   ⚠️  Overfitting: {'Detected' if analysis['overfitting_detected'] else 'Not detected'}")
        print(f"   ⚠️  Underfitting: {'Detected' if analysis['underfitting_detected'] else 'Not detected'}")

        if analysis['recommendations']:
            print("   💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"      • {rec}")

    print("
✅ Analysis completed!"    print(f"   📊 Plot: {plot_path}")

    return 0


if __name__ == "__main__":
    exit(main())