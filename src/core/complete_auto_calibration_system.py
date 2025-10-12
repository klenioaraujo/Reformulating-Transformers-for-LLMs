#!/usr/bin/env python3
"""
Complete Auto-Calibration System for ΨQRH Pipeline
==================================================

Orchestrates all parameter calibrators to provide complete auto-calibration
of the ΨQRH system based on input characteristics.
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path

# Import individual calibrators
from .physical_parameter_calibrator import PhysicalParameterCalibrator
from .architecture_parameter_calibrator import ArchitectureParameterCalibrator
from .processing_parameter_calibrator import ProcessingParameterCalibrator
from .control_parameter_calibrator import ControlParameterCalibrator


class CompleteAutoCalibrationSystem:
    """
    Complete auto-calibration system that orchestrates all parameter calibrators
    """

    def __init__(self):
        """Initialize the complete auto-calibration system"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize individual calibrators
        self.physical_calibrator = PhysicalParameterCalibrator()
        self.architecture_calibrator = ArchitectureParameterCalibrator()
        self.processing_calibrator = ProcessingParameterCalibrator()
        self.control_calibrator = ControlParameterCalibrator()

        print("🔧 Complete Auto-Calibration System initialized")

    def calibrate_all_parameters(self, text: str, fractal_signal: Optional[torch.Tensor] = None,
                               D_fractal: Optional[float] = None) -> Dict[str, Any]:
        """
        Calibrate all system parameters based on input analysis

        Args:
            text: Input text for analysis
            fractal_signal: Fractal signal tensor (optional, will be computed if not provided)
            D_fractal: Fractal dimension (optional, will be computed if not provided)

        Returns:
            Dict containing all calibrated parameters organized by category
        """
        print(f"🔧 Starting complete auto-calibration for text: '{text[:50]}...'")

        # Compute fractal signal if not provided
        if fractal_signal is None:
            fractal_signal = self._compute_fractal_signal(text)

        # Compute fractal dimension if not provided
        if D_fractal is None:
            D_fractal = self._compute_fractal_dimension(fractal_signal)

        print(f"   📐 Computed fractal dimension: D = {D_fractal:.3f}")

        # Calibrate all parameter categories
        physical_params = self.physical_calibrator.calibrate_all(D_fractal, text, fractal_signal)
        architecture_params = self.architecture_calibrator.calibrate_all(text)
        processing_params = self.processing_calibrator.calibrate_all(text)
        control_params = self.control_calibrator.calibrate_all(text)

        # Combine all parameters
        calibrated_params = {
            'physical_params': physical_params,
            'architecture_params': architecture_params,
            'processing_params': processing_params,
            'control_params': control_params,
            # Metadata
            'input_analysis': {
                'text_length': len(text),
                'fractal_dimension': D_fractal,
                'signal_shape': fractal_signal.shape if fractal_signal is not None else None
            }
        }

        # Validate all parameters
        validation_results = self.validate_all_parameters(calibrated_params)

        calibrated_params['validation'] = validation_results

        print(f"✅ Auto-calibration completed successfully")
        print(f"   🔬 Physical: {len(physical_params)} parameters")
        print(f"   🏗️  Architecture: {len(architecture_params)} parameters")
        print(f"   ⚙️  Processing: {len(processing_params)} parameters")
        print(f"   🎛️  Control: {len(control_params)} parameters")

        if not validation_results['all_checks_pass']:
            print(f"⚠️  Some validation checks failed - review parameters")

        return calibrated_params

    def _compute_fractal_signal(self, text: str) -> torch.Tensor:
        """
        Compute fractal signal from text (simplified version)

        Args:
            text: Input text

        Returns:
            Fractal signal tensor
        """
        # Simple text-to-signal conversion (same as in pipeline)
        char_values = torch.tensor([ord(c) / 127.0 for c in text], dtype=torch.float32)

        # Basic spectral processing
        spectrum = torch.fft.fft(char_values)

        # Expand to reasonable dimension
        magnitude = torch.abs(spectrum)
        if len(magnitude) < 64:
            # Upsampling
            magnitude = torch.nn.functional.interpolate(
                magnitude.unsqueeze(0).unsqueeze(0),
                size=64,
                mode='linear',
                align_corners=False
            ).squeeze()

        return magnitude.to(self.device)

    def _compute_fractal_dimension(self, signal: torch.Tensor) -> float:
        """
        Compute fractal dimension from signal (simplified version)

        Args:
            signal: Signal tensor

        Returns:
            Fractal dimension
        """
        # Simplified power-law fitting
        spectrum = torch.fft.fft(signal)
        power_spectrum = torch.abs(spectrum) ** 2

        k = torch.arange(1, len(power_spectrum) + 1, dtype=torch.float32)

        log_k = torch.log(k + 1e-10)
        log_P = torch.log(power_spectrum + 1e-10)

        # Simple linear regression
        n = len(log_k)
        sum_x = log_k.sum()
        sum_y = log_P.sum()
        sum_xy = (log_k * log_P).sum()
        sum_x2 = (log_k ** 2).sum()

        beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        D = (3.0 - beta.item()) / 2.0

        # Clamp to physical range
        D = max(1.0, min(D, 2.0))

        return D

    def validate_all_parameters(self, calibrated_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all calibrated parameters

        Args:
            calibrated_params: All calibrated parameters

        Returns:
            Validation results
        """
        # Validate each category
        physical_validation = self.physical_calibrator.validate_physical_consistency(
            calibrated_params['physical_params']
        )
        architecture_validation = self.architecture_calibrator.validate_architecture_consistency(
            calibrated_params['architecture_params']
        )
        processing_validation = self.processing_calibrator.validate_processing_consistency(
            calibrated_params['processing_params']
        )
        control_validation = self.control_calibrator.validate_control_consistency(
            calibrated_params['control_params']
        )

        # Overall validation
        all_checks_pass = (
            physical_validation['all_checks_pass'] and
            architecture_validation['all_checks_pass'] and
            processing_validation['all_checks_pass'] and
            control_validation['all_checks_pass']
        )

        return {
            'physical_validation': physical_validation,
            'architecture_validation': architecture_validation,
            'processing_validation': processing_validation,
            'control_validation': control_validation,
            'all_checks_pass': all_checks_pass
        }

    def get_calibration_report(self, calibrated_params: Dict[str, Any]) -> str:
        """
        Generate a comprehensive calibration report

        Args:
            calibrated_params: Calibrated parameters

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ΨQRH COMPLETE AUTO-CALIBRATION REPORT")
        report_lines.append("=" * 80)

        # Input analysis
        input_analysis = calibrated_params['input_analysis']
        report_lines.append(f"📊 INPUT ANALYSIS:")
        report_lines.append(f"   Text Length: {input_analysis['text_length']} characters")
        report_lines.append(f"   Fractal Dimension: {input_analysis['fractal_dimension']:.3f}")
        if input_analysis['signal_shape']:
            report_lines.append(f"   Signal Shape: {input_analysis['signal_shape']}")
        report_lines.append("")

        # Physical parameters
        phys = calibrated_params['physical_params']
        report_lines.append(f"🔬 PHYSICAL PARAMETERS:")
        report_lines.append(f"   I₀ (Amplitude): {phys['I0']:.3f}")
        report_lines.append(f"   ω (Angular Frequency): {phys['omega']:.3f}")
        report_lines.append(f"   k (Wave Number): {phys['k']:.3f}")
        report_lines.append(f"   α (Spectral): {phys['alpha']:.3f}")
        report_lines.append(f"   β (Nonlinear): {phys['beta']:.3f}")
        report_lines.append("")

        # Architecture parameters
        arch = calibrated_params['architecture_params']
        report_lines.append(f"🏗️  ARCHITECTURE PARAMETERS:")
        report_lines.append(f"   embed_dim: {arch['embed_dim']}")
        report_lines.append(f"   num_heads: {arch['num_heads']}")
        report_lines.append(f"   hidden_dim: {arch['hidden_dim']}")
        report_lines.append(f"   num_layers: {arch['num_layers']}")
        report_lines.append("")

        # Processing parameters
        proc = calibrated_params['processing_params']
        report_lines.append(f"⚙️  PROCESSING PARAMETERS:")
        report_lines.append(f"   dropout: {proc['dropout']:.3f}")
        report_lines.append(f"   max_history: {proc['max_history']}")
        report_lines.append(f"   vocab_size: {proc['vocab_size']}")
        report_lines.append(f"   epsilon: {proc['epsilon']:.2e}")
        report_lines.append("")

        # Control parameters
        ctrl = calibrated_params['control_params']
        report_lines.append(f"🎛️  CONTROL PARAMETERS:")
        report_lines.append(f"   temperature: {ctrl['temperature']:.3f}")
        report_lines.append(f"   top_k: {ctrl['top_k']}")
        report_lines.append(f"   learning_rate: {ctrl['learning_rate']:.2e}")
        report_lines.append("")

        # Validation summary
        validation = calibrated_params['validation']
        report_lines.append(f"✅ VALIDATION SUMMARY:")
        report_lines.append(f"   Physical: {'✅' if validation['physical_validation']['all_checks_pass'] else '❌'}")
        report_lines.append(f"   Architecture: {'✅' if validation['architecture_validation']['all_checks_pass'] else '❌'}")
        report_lines.append(f"   Processing: {'✅' if validation['processing_validation']['all_checks_pass'] else '❌'}")
        report_lines.append(f"   Control: {'✅' if validation['control_validation']['all_checks_pass'] else '❌'}")
        report_lines.append(f"   Overall: {'✅ ALL PASSED' if validation['all_checks_pass'] else '❌ ISSUES FOUND'}")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def save_calibration_config(self, calibrated_params: Dict[str, Any], filepath: str):
        """
        Save calibrated parameters to a configuration file

        Args:
            calibrated_params: Calibrated parameters
            filepath: Path to save configuration
        """
        import json

        # Flatten parameters for config file
        config = {}

        # Add physical parameters
        config.update(calibrated_params['physical_params'])

        # Add architecture parameters
        config.update(calibrated_params['architecture_params'])

        # Add processing parameters
        config.update(calibrated_params['processing_params'])

        # Add control parameters
        config.update(calibrated_params['control_params'])

        # Add metadata
        config['_metadata'] = {
            'calibration_system': 'CompleteAutoCalibrationSystem',
            'input_analysis': calibrated_params['input_analysis'],
            'validation': calibrated_params['validation']['all_checks_pass']
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"💾 Calibration config saved to: {filepath}")

    def load_calibration_config(self, filepath: str) -> Dict[str, Any]:
        """
        Load calibrated parameters from a configuration file

        Args:
            filepath: Path to configuration file

        Returns:
            Loaded parameters
        """
        import json

        with open(filepath, 'r') as f:
            config = json.load(f)

        # Reconstruct the calibrated_params structure
        calibrated_params = {
            'physical_params': {
                'alpha': config.get('alpha', 1.0),
                'beta': config.get('beta', 0.5),
                'I0': config.get('I0', 1.0),
                'omega': config.get('omega', 1.0),
                'k': config.get('k', 2.0)
            },
            'architecture_params': {
                'embed_dim': config.get('embed_dim', 64),
                'num_heads': config.get('num_heads', 8),
                'hidden_dim': config.get('hidden_dim', 512),
                'num_layers': config.get('num_layers', 3)
            },
            'processing_params': {
                'dropout': config.get('dropout', 0.1),
                'max_history': config.get('max_history', 10),
                'vocab_size': config.get('vocab_size', 256),
                'epsilon': config.get('epsilon', 1e-10)
            },
            'control_params': {
                'temperature': config.get('temperature', 1.0),
                'top_k': config.get('top_k', 10),
                'learning_rate': config.get('learning_rate', 1e-4)
            },
            'input_analysis': config.get('_metadata', {}).get('input_analysis', {}),
            'validation': {'all_checks_pass': config.get('_metadata', {}).get('validation', False)}
        }

        return calibrated_params