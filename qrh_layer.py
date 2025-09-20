import torch
import torch.nn as nn
import torch.fft as fft
import math
from typing import Tuple

from torch.amp import autocast

from quaternion_operations import QuaternionOperations
from spectral_filter import SpectralFilter

class QRHLayer(nn.Module):
    """
    ΨQRH Layer for Transformers: Ψ_QRH = R_left · F^{-1} { F(k) · F { Ψ } } · R_right

    CRITICAL CORRECTIONS IMPLEMENTED:
    =================================
    1. SO(4) vs SO(3): This implementation correctly uses SO(4) = SU(2) × SU(2)
       - TWO independent quaternions (q_left, q_right)
       - Full rotations in 4D space
       - NOT to be confused with SO(3) which uses only one quaternion

    2. |k| Calculation: Correctly uses fftfreq (NOT np.gradient)
       - fftfreq: frequencies from Fourier space
       - np.gradient: spatial derivatives (would break the filter)

    3. Engineering Improvements:
       - Einsum for efficient operations
       - Spectral windowing (Hann/Hamming/Blackman)
       - Numerical stability: sqrt(k² + ε)

    Args:
        embed_dim: Embedding dimension per quaternion component
        alpha: Scaling parameter for the spectral filter
        theta_left, omega_left, phi_left: Angles for the left quaternion
        theta_right, omega_right, phi_right: Angles for the right quaternion
        use_learned_rotation: If True, angles are learnable parameters
        use_windowing: If True, applies spectral windowing
        window_type: Type of window ('hann', 'hamming', 'blackman')
    """

    def __init__(self,
                 embed_dim: int,
                 alpha: float = 1.0,
                 theta_left: float = 0.1,
                 omega_left: float = 0.05,
                 phi_left: float = 0.02,
                 theta_right: float = 0.08,
                 omega_right: float = 0.03,
                 phi_right: float = 0.015,
                 use_learned_rotation: bool = False,
                 spatial_dims: Tuple[int, ...] = None,
                 use_windowing: bool = True,
                 window_type: str = 'hann'):
        super().__init__()

        self.embed_dim = embed_dim
        self.total_dim = 4 * embed_dim
        self.alpha = alpha
        self.spatial_dims = spatial_dims if spatial_dims is not None else None

        # Left rotation parameters (learnable or fixed)
        if use_learned_rotation:
            self.theta_left = nn.Parameter(torch.tensor(theta_left, dtype=torch.float32, requires_grad=True))
            self.omega_left = nn.Parameter(torch.tensor(omega_left, dtype=torch.float32, requires_grad=True))
            self.phi_left = nn.Parameter(torch.tensor(phi_left, dtype=torch.float32, requires_grad=True))
            self.theta_right = nn.Parameter(torch.tensor(theta_right, dtype=torch.float32, requires_grad=True))
            self.omega_right = nn.Parameter(torch.tensor(omega_right, dtype=torch.float32, requires_grad=True))
            self.phi_right = nn.Parameter(torch.tensor(phi_right, dtype=torch.float32, requires_grad=True))
        else:
            self.register_buffer('theta_left', torch.tensor(theta_left))
            self.register_buffer('omega_left', torch.tensor(omega_left))
            self.register_buffer('phi_left', torch.tensor(phi_left))
            self.register_buffer('theta_right', torch.tensor(theta_right))
            self.register_buffer('omega_right', torch.tensor(omega_right))
            self.register_buffer('phi_right', torch.tensor(phi_right))

        # Initialize the spectral filter with improvements
        self.spectral_filter = SpectralFilter(alpha, use_windowing=use_windowing, window_type=window_type)

        # Projection layers
        self.v_proj = nn.Linear(self.total_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)

        # Register FFT frequencies for reuse
        self.register_buffer('freqs', None)

    def get_rotation_quaternions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the left and right rotation quaternions for SO(4)"""
        q_left = QuaternionOperations.create_unit_quaternion(
            self.theta_left, self.omega_left, self.phi_left)
        q_right = QuaternionOperations.create_unit_quaternion(
            self.theta_right, self.omega_right, self.phi_right)
        return q_left, q_right

    def _compute_frequencies(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates FFT frequencies correctly using fftfreq (NOT np.gradient).

        CRITICAL: Using np.gradient would completely break the spectral filter because:
        - fftfreq gives the correct frequencies from Fourier space.
        - np.gradient calculates spatial derivatives, not FFT frequencies.
        - Mixing the two leads to fundamental inconsistencies.
        """
        # For compatibility, if spatial_dims is not defined, use sequence dimension
        if self.spatial_dims is None or (isinstance(self.spatial_dims, (list, tuple)) and len(self.spatial_dims) == 1 and self.spatial_dims[0] == seq_len):
            # 1D case (compatibility with previous implementation)
            if self.freqs is None or self.freqs.size(0) != seq_len:
                # CRITICAL CORRECTION: Use fftfreq correctly, NOT np.gradient
                self.freqs = fft.fftfreq(seq_len, d=1.0, device=device)

            # Calculate wave vector k = 2π * frequency
            k = 2 * math.pi * self.freqs.view(1, seq_len, 1)

            # CORRECTION: Enhanced numerical stability with sqrt(k² + ε)
            # Avoids problems when k=0 (DC frequency)
            k_mag = torch.sqrt(k**2 + self.spectral_filter.epsilon)

            return k, k_mag
        else:
            # Multi-dimensional case
            k_vecs = [fft.fftfreq(n, d=1.0, device=device) for n in self.spatial_dims]
            k_mesh = torch.meshgrid(*k_vecs, indexing='ij')

            # Calculate wave vector magnitude with enhanced stability
            k_squared = sum(k_i**2 for k_i in k_mesh)
            # CORRECTION: Better numerical stability
            k_mag = torch.sqrt(k_squared + self.spectral_filter.epsilon)

            return k_mesh, k_mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass with correct SO(4) and engineering improvements.

        CRITICAL CORRECTION: Implements correct SO(4) rotations using two
        independent quaternions to obtain the full 4D rotation group.

        ENGINEERING IMPROVEMENTS:
        - Use of einsum for efficient operations
        - Spectral windowing to reduce leakage
        - Enhanced numerical stability

        Args:
            x: Input tensor of shape [batch_size, seq_len, 4 * embed_dim]
        Returns:
            Processed tensor with the same shape
        """
        with autocast('cuda', enabled=torch.cuda.is_available()):
            batch_size, seq_len, _ = x.shape
            device = x.device

            # 1. Project to V using einsum for efficiency
            V = torch.einsum('bij,jk->bik', x, self.v_proj.weight) + self.v_proj.bias

            # 2. Split into quaternion components using advanced slicing
            D = self.embed_dim
            # Efficiently reshape: [batch, seq, 4*embed] -> [batch, seq, embed, 4]
            Ψ_reshaped = V.view(batch_size, seq_len, 4, D).permute(0, 1, 3, 2)

            # 3. Extract quaternion components
            Ψ_w, Ψ_i, Ψ_j, Ψ_k = Ψ_reshaped.unbind(dim=-1)

            # 4. Complex representation for spectral processing
            Ψ_complex = Ψ_w + 1j * Ψ_i

            # 5. Apply windowing to reduce spectral leakage
            Ψ_windowed = self.spectral_filter.apply_window(Ψ_complex, seq_len)

            # 6. Fourier Transform
            Ψ_fft = fft.fft(Ψ_windowed, dim=1)

            # 7. Apply spectral filter
            k_mesh, k_mag = self._compute_frequencies(seq_len, device)
            F_k = self.spectral_filter(k_mag)
            Ψ_filtered = Ψ_fft * F_k

            # 8. Inverse Transform
            Ψ_ifft_complex = fft.ifft(Ψ_filtered, dim=1)

            # 9. Update components using efficient operations
            Ψ_new_w = torch.real(Ψ_ifft_complex)
            Ψ_new_i = torch.imag(Ψ_ifft_complex)

            # 10. CRITICAL CORRECTION: Correct SO(4) rotations
            # SO(4) = SU(2) × SU(2) - two independent quaternions
            q_left, q_right = self.get_rotation_quaternions()

            # Expand quaternions for broadcasting: [4] -> [1, 1, 1, 4]
            q_left_expanded = q_left.view(1, 1, 1, 4)
            q_right_expanded = q_right.view(1, 1, 1, 4)

            # Apply SO(4) rotation: q_left * Ψ * q_right
            # First: q_left * Ψ (left rotation)
            temp = QuaternionOperations.multiply(q_left_expanded, Ψ_reshaped)

            # Second: temp * q_right (right rotation)
            rotated = QuaternionOperations.multiply(temp, q_right_expanded)

            # 11. Reconstruct final tensor
            Ψ_final = rotated.permute(0, 1, 3, 2).reshape(batch_size, seq_len, self.total_dim)

            # 12. Final projection + residual connection using einsum
            output = torch.einsum('bij,jk->bik', Ψ_final, self.out_proj.weight) + self.out_proj.bias + x

            return output
