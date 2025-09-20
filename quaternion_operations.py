import torch

class QuaternionOperations:
    """Utility class for quaternion operations"""

    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiplies two quaternions: q1 * q2.

        Args:
            q1, q2: Tensors of shape [..., 4] (w, x, y, z)
        Returns:
            Quaternion product of shape [..., 4]
        """
        # More efficient version using direct operations
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def create_unit_quaternion(theta: torch.Tensor,
                              omega: torch.Tensor,
                              phi: torch.Tensor) -> torch.Tensor:
        """Creates a unit quaternion from angles"""
        cos_theta_2 = torch.cos(theta / 2)
        sin_theta_2 = torch.sin(theta / 2)
        cos_omega = torch.cos(omega)
        sin_omega = torch.sin(omega)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        return torch.stack([
            cos_theta_2,
            sin_theta_2 * cos_omega,
            sin_theta_2 * sin_omega * cos_phi,
            sin_theta_2 * sin_omega * sin_phi
        ], dim=-1)

    @staticmethod
    def create_unit_quaternion_batch(theta: torch.Tensor,
                                     omega: torch.Tensor,
                                     phi: torch.Tensor) -> torch.Tensor:
        """Creates a batch of unit quaternions from angle tensors."""
        cos_theta_2 = torch.cos(theta / 2)
        sin_theta_2 = torch.sin(theta / 2)
        cos_omega = torch.cos(omega)
        sin_omega = torch.sin(omega)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        return torch.stack([
            cos_theta_2,
            sin_theta_2 * cos_omega,
            sin_theta_2 * sin_omega * cos_phi,
            sin_theta_2 * sin_omega * sin_phi
        ], dim=-1)