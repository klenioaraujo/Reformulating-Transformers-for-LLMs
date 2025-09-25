import torch
from typing import Dict

class GateController:
    """
    Gate controller based on numerical "receipts" for flow control.

    Implements ABSTAIN/DELIVER/CLARIFY mechanism based on:
    - Orthogonality error
    - Removed energy ratio
    - Drift angle
    """

    def __init__(self,
                 orthogonal_threshold: float = 1e-6,
                 energy_threshold: float = 0.1,
                 drift_threshold: float = 0.1):
        self.orthogonal_threshold = orthogonal_threshold
        self.energy_threshold = energy_threshold
        self.drift_threshold = drift_threshold

    def calculate_receipts(self,
                          input_tensor: torch.Tensor,
                          output_tensor: torch.Tensor,
                          rotation_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Calculates numerical receipts for gate decision-making.

        Args:
            input_tensor: Input tensor
            output_tensor: Output tensor
            rotation_params: Current rotation parameters

        Returns:
            Dictionary with receipt values
        """
        receipts = {}

        # 1. Orthogonality error (norm preservation)
        input_norm = torch.norm(input_tensor, dim=-1)
        output_norm = torch.norm(output_tensor, dim=-1)
        receipts['orthogonal_error'] = torch.mean(torch.abs(input_norm - output_norm)).item()

        # 2. Energy ratio removed by the spectral filter
        input_energy = torch.mean(input_tensor ** 2)
        output_energy = torch.mean(output_tensor ** 2)
        receipts['energy_ratio'] = (input_energy - output_energy) / (input_energy + 1e-10)
        receipts['energy_ratio'] = receipts['energy_ratio'].item()

        # 3. Drift angle (change in rotation parameters)
        if all(key in rotation_params for key in ['theta_left', 'omega_left', 'phi_left', 'theta_right', 'omega_right', 'phi_right']):
            drift_angle = torch.sqrt(
                rotation_params['theta_left'].detach()**2 +
                rotation_params['omega_left'].detach()**2 +
                rotation_params['phi_left'].detach()**2 +
                rotation_params['theta_right'].detach()**2 +
                rotation_params['omega_right'].detach()**2 +
                rotation_params['phi_right'].detach()**2
            ).item()
        else:
            drift_angle = 0.0
        receipts['drift_angle'] = drift_angle

        return receipts

    def decide_gate(self, receipts: Dict[str, float]) -> str:
        """
        Makes a gate decision based on the receipts.

        Returns:
            'ABSTAIN': Refuse processing (error too high)
            'DELIVER': Deliver the result (successful processing)
            'CLARIFY': Request clarification (uncertain results)
        """
        orthogonal_error = receipts['orthogonal_error']
        energy_ratio = abs(receipts['energy_ratio'])
        drift_angle = receipts['drift_angle']

        # ABSTAIN: If orthogonality error is too high
        if orthogonal_error > self.orthogonal_threshold:
            return 'ABSTAIN'

        # DELIVER: If all criteria are within limits
        if (orthogonal_error <= self.orthogonal_threshold * 0.1 and
            energy_ratio <= self.energy_threshold and
            drift_angle <= self.drift_threshold):
            return 'DELIVER'

        # CLARIFY: Intermediate case
        return 'CLARIFY'

    def apply_gate_policy(self,
                         gate_decision: str,
                         input_tensor: torch.Tensor,
                         output_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies gate policy based on the decision.

        Args:
            gate_decision: Gate decision ('ABSTAIN', 'DELIVER', 'CLARIFY')
            input_tensor: Original input tensor
            output_tensor: Output tensor from the 4D layer

        Returns:
            Final tensor after applying the policy
        """
        if gate_decision == 'ABSTAIN':
            # Return original input (no modification)
            return input_tensor
        elif gate_decision == 'DELIVER':
            # Deliver processed output
            return output_tensor
        elif gate_decision == 'CLARIFY':
            # Mix input and output with weights
            confidence_weight = 0.5  # Could be adaptive
            return confidence_weight * output_tensor + (1 - confidence_weight) * input_tensor
        else:
            # Fallback to deliver
            return output_tensor
