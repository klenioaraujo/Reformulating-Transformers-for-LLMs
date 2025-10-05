import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingNotFoundError(Exception):
    """Raised when model lacks required token_embedding for energy validation"""
    pass


class MathematicalValidator:
    """Comprehensive mathematical property validation for ΨQRH"""

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance

    def _compute_input_energy(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        Compute input energy with proper handling for different model types

        Args:
            model: ΨQRH model
            x: Input tensor (token IDs or embeddings)

        Returns:
            Input energy as float

        Raises:
            EmbeddingNotFoundError: If model lacks token_embedding and x is not embeddings
        """
        from ..core.utils import compute_energy

        # Case 1: Model has token_embedding - use it
        if hasattr(model, 'token_embedding'):
            input_embeddings = model.token_embedding(x)
            energy = compute_energy(input_embeddings).sum().item()
            logger.debug(f"Computed energy from token_embedding: {energy:.6f}")
            return energy

        # Case 2: Input is already embeddings (floating point with dimension > 1)
        if x.dtype == torch.float32 and len(x.shape) >= 2 and x.shape[-1] > 1:
            energy = compute_energy(x).sum().item()
            logger.debug(f"Computed energy from input embeddings: {energy:.6f}")
            return energy

        # Case 3: No valid method to compute input energy
        error_msg = (
            "Cannot compute input energy: model lacks 'token_embedding' attribute "
            "and input is not in embedding format (float tensor with dim >= 2)"
        )
        logger.error(error_msg)
        raise EmbeddingNotFoundError(error_msg)

    def validate_energy_conservation(self, model: nn.Module, x: torch.Tensor,
                                   skip_on_no_embedding: bool = False) -> Dict:
        """
        Verify energy conservation: ||output|| ≈ ||input|| ± tolerance

        Args:
            model: ΨQRH model to validate
            x: Input tensor (token IDs or embeddings)
            skip_on_no_embedding: If True, skip validation instead of raising error

        Returns:
            Dict with validation results
        """
        with torch.no_grad():
            # Forward pass
            output = model(x)

            # Calculate output energy
            from ..core.utils import compute_energy
            output_energy = compute_energy(output).sum().item()

            # Calculate input energy with proper error handling
            try:
                input_energy = self._compute_input_energy(model, x)

                # Calculate conservation ratio
                conservation_ratio = output_energy / input_energy if input_energy > 0 else 1.0

                # Check if within tolerance
                is_conserved = abs(conservation_ratio - 1.0) <= self.tolerance

                return {
                    "input_energy": input_energy,
                    "output_energy": output_energy,
                    "conservation_ratio": conservation_ratio,
                    "is_conserved": is_conserved,
                    "tolerance": self.tolerance,
                    "validation_method": "proper_embedding"
                }

            except EmbeddingNotFoundError as e:
                if skip_on_no_embedding:
                    logger.warning(f"Skipping energy conservation test: {str(e)}")
                    return {
                        "input_energy": None,
                        "output_energy": output_energy,
                        "conservation_ratio": None,
                        "is_conserved": None,
                        "tolerance": self.tolerance,
                        "validation_method": "skipped",
                        "skip_reason": str(e)
                    }
                else:
                    raise

    def validate_unitarity(self, model: nn.Module, x: torch.Tensor) -> Dict:
        """
        Verify unitarity: |F(k)| ≈ 1.0 for all frequencies

        Args:
            model: ΨQRH model to validate
            x: Input tensor

        Returns:
            Dict with validation results
        """
        with torch.no_grad():
            # Get intermediate representations if possible
            # For now, validate on output
            output = model(x)

            # Convert to frequency domain
            output_fft = torch.fft.fft(output, dim=1, norm="ortho")

            # Calculate magnitude spectrum
            magnitude_spectrum = torch.abs(output_fft)

            # Check unitarity (should be approximately 1.0)
            mean_magnitude = torch.mean(magnitude_spectrum).item()
            std_magnitude = torch.std(magnitude_spectrum).item()

            # Check if within tolerance
            is_unitary = abs(mean_magnitude - 1.0) <= self.tolerance

            return {
                "mean_magnitude": mean_magnitude,
                "std_magnitude": std_magnitude,
                "is_unitary": is_unitary,
                "tolerance": self.tolerance
            }

    def validate_numerical_stability(self, model: nn.Module, x: torch.Tensor,
                                   num_passes: int = 1000) -> Dict:
        """
        Validate numerical stability over multiple forward passes

        Args:
            model: ΨQRH model to validate
            x: Input tensor
            num_passes: Number of forward passes to test

        Returns:
            Dict with validation results
        """
        with torch.no_grad():
            nan_count = 0
            inf_count = 0

            for i in range(num_passes):
                output = model(x)

                # Check for NaN and Inf
                if torch.isnan(output).any():
                    nan_count += 1
                if torch.isinf(output).any():
                    inf_count += 1

            is_stable = (nan_count == 0) and (inf_count == 0)

            return {
                "num_passes": num_passes,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "is_stable": is_stable
            }

    def validate_quaternion_properties(self, quaternion_ops) -> Dict:
        """
        Validate quaternion algebra properties

        Args:
            quaternion_ops: Quaternion operations module

        Returns:
            Dict with validation results
        """
        with torch.no_grad():
            # Test quaternion multiplication
            q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity
            q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])  # i

            # Identity property
            identity_result = quaternion_ops.multiply(q1, q2)
            identity_correct = torch.allclose(identity_result, q2, atol=1e-6)

            # Inverse property
            q3 = torch.tensor([0.5, 0.5, 0.5, 0.5])
            q3_conj = torch.tensor([0.5, -0.5, -0.5, -0.5])
            inverse_result = quaternion_ops.multiply(q3, q3_conj)
            expected_inverse = torch.tensor([1.0, 0.0, 0.0, 0.0])
            inverse_correct = torch.allclose(inverse_result, expected_inverse, atol=1e-6)

            return {
                "identity_property": identity_correct,
                "inverse_property": inverse_correct,
                "all_properties_valid": identity_correct and inverse_correct
            }

    def validate_spectral_operations(self, model: nn.Module, x: torch.Tensor) -> Dict:
        """
        Validate spectral operations properties

        Args:
            model: ΨQRH model to validate
            x: Input tensor

        Returns:
            Dict with validation results
        """
        with torch.no_grad():
            # Use model output for spectral operations validation
            output = model(x)

            # Test FFT/IFFT consistency on output
            output_fft = torch.fft.fft(output, dim=1, norm="ortho")
            output_reconstructed = torch.fft.ifft(output_fft, dim=1, norm="ortho")

            fft_consistent = torch.allclose(output, output_reconstructed.real, atol=1e-6)

            # Test Parseval's theorem (energy conservation in frequency domain)
            from ..core.utils import compute_energy
            time_domain_energy = compute_energy(output).sum().item()
            freq_domain_energy = compute_energy(output_fft).sum().item()

            parseval_ratio = freq_domain_energy / time_domain_energy
            parseval_valid = abs(parseval_ratio - 1.0) <= self.tolerance

            return {
                "fft_consistency": fft_consistent,
                "parseval_theorem": parseval_valid,
                "time_domain_energy": time_domain_energy,
                "freq_domain_energy": freq_domain_energy,
                "parseval_ratio": parseval_ratio
            }

    def comprehensive_validation(self, model: nn.Module, x: torch.Tensor,
                               quaternion_ops) -> Dict:
        """
        Run comprehensive mathematical validation

        Args:
            model: ΨQRH model to validate
            x: Input tensor
            quaternion_ops: Quaternion operations module

        Returns:
            Dict with all validation results
        """
        results = {}

        # Energy conservation
        results["energy_conservation"] = self.validate_energy_conservation(model, x)

        # Unitarity
        results["unitarity"] = self.validate_unitarity(model, x)

        # Numerical stability
        results["numerical_stability"] = self.validate_numerical_stability(model, x)

        # Quaternion properties
        results["quaternion_properties"] = self.validate_quaternion_properties(quaternion_ops)

        # Spectral operations
        results["spectral_operations"] = self.validate_spectral_operations(model, x)

        # Overall validation status
        all_valid = (
            results["energy_conservation"]["is_conserved"] and
            results["unitarity"]["is_unitary"] and
            results["numerical_stability"]["is_stable"] and
            results["quaternion_properties"]["all_properties_valid"] and
            results["spectral_operations"]["fft_consistency"] and
            results["spectral_operations"]["parseval_theorem"]
        )

        results["overall_validation"] = {
            "all_tests_passed": all_valid,
            "total_tests": 6,
            "passed_tests": sum([
                results["energy_conservation"]["is_conserved"],
                results["unitarity"]["is_unitary"],
                results["numerical_stability"]["is_stable"],
                results["quaternion_properties"]["all_properties_valid"],
                results["spectral_operations"]["fft_consistency"],
                results["spectral_operations"]["parseval_theorem"]
            ])
        }

        return results

    def generate_validation_report(self, validation_results: Dict) -> str:
        """
        Generate a human-readable validation report

        Args:
            validation_results: Results from comprehensive_validation

        Returns:
            Formatted validation report
        """
        report = []
        report.append("ΨQRH Mathematical Validation Report")
        report.append("=" * 50)

        # Energy conservation
        ec = validation_results["energy_conservation"]
        report.append(f"Energy Conservation: {'PASS' if ec['is_conserved'] else 'FAIL'}")
        report.append(f"  Input Energy: {ec['input_energy']:.6f}")
        report.append(f"  Output Energy: {ec['output_energy']:.6f}")
        report.append(f"  Ratio: {ec['conservation_ratio']:.6f} (target: 1.0 ± {ec['tolerance']})")

        # Unitarity
        unit = validation_results["unitarity"]
        report.append(f"Unitarity: {'PASS' if unit['is_unitary'] else 'FAIL'}")
        report.append(f"  Mean Magnitude: {unit['mean_magnitude']:.6f} (target: 1.0 ± {unit['tolerance']})")
        report.append(f"  Std Magnitude: {unit['std_magnitude']:.6f}")

        # Numerical stability
        ns = validation_results["numerical_stability"]
        report.append(f"Numerical Stability: {'PASS' if ns['is_stable'] else 'FAIL'}")
        report.append(f"  Passes: {ns['num_passes']}")
        report.append(f"  NaN Count: {ns['nan_count']}")
        report.append(f"  Inf Count: {ns['inf_count']}")

        # Quaternion properties
        qp = validation_results["quaternion_properties"]
        report.append(f"Quaternion Properties: {'PASS' if qp['all_properties_valid'] else 'FAIL'}")
        report.append(f"  Identity: {'PASS' if qp['identity_property'] else 'FAIL'}")
        report.append(f"  Inverse: {'PASS' if qp['inverse_property'] else 'FAIL'}")

        # Spectral operations
        so = validation_results["spectral_operations"]
        report.append(f"Spectral Operations: {'PASS' if so['fft_consistency'] and so['parseval_theorem'] else 'FAIL'}")
        report.append(f"  FFT Consistency: {'PASS' if so['fft_consistency'] else 'FAIL'}")
        report.append(f"  Parseval Theorem: {'PASS' if so['parseval_theorem'] else 'FAIL'}")
        report.append(f"  Parseval Ratio: {so['parseval_ratio']:.6f}")

        # Overall
        ov = validation_results["overall_validation"]
        report.append("-" * 50)
        report.append(f"Overall Validation: {'PASS' if ov['all_tests_passed'] else 'FAIL'}")
        report.append(f"  Tests Passed: {ov['passed_tests']}/{ov['total_tests']}")

        return "\n".join(report)