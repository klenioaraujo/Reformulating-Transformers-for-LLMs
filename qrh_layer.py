import torch
import torch.nn as nn
import torch.fft as fft
import math
import warnings
from typing import Tuple, Dict, Optional, Literal, Callable
from dataclasses import dataclass

from torch.amp import autocast

from quaternion_operations import QuaternionOperations
from spectral_filter import SpectralFilter

# Custom Exception Hierarchy
class QRHError(Exception):
    """Base exception for all errors raised by the QRHLayer."""
    pass

class QRHDimensionError(QRHError):
    """Raised for dimension-related errors."""
    pass

class QRHSpectralError(QRHError):
    """Raised for errors during spectral (FFT, filtering) operations."""
    pass

# Caching Utilities
class FFTCache:
    """A simple FIFO cache for storing FFT results."""
    def __init__(self, max_size: int = 10):
        self.cache: Dict[Tuple, torch.Tensor] = {}
        self.max_size = max_size

    def get(self, key: Tuple, compute_func: Callable[[], torch.Tensor]) -> torch.Tensor:
        if key in self.cache:
            return self.cache[key]
        
        if len(self.cache) >= self.max_size:
            # Evict the first item added (FIFO)
            self.cache.pop(next(iter(self.cache)))

        result = compute_func()
        self.cache[key] = result
        return result

@dataclass
class QRHConfig:
    embed_dim: int = 64
    alpha: float = 1.0
    theta_left: float = 0.1
    omega_left: float = 0.05
    phi_left: float = 0.02
    theta_right: float = 0.08
    omega_right: float = 0.03
    phi_right: float = 0.015
    use_learned_rotation: bool = False
    spatial_dims: Optional[Tuple[int, ...]] = None
    use_windowing: bool = True
    window_type: str = 'hann'
    fft_cache_size: int = 10
    device: str = 'cpu'
    enable_warnings: bool = True

class QRHLayer(nn.Module):
    """
    ΨQRH Layer for Transformers: Ψ_QRH = R_left · F^{-1} { F(k) · F { Ψ } } · R_right
    """

    def __init__(self, config: QRHConfig):
        super().__init__()
        self.config = config
        self.total_dim = 4 * config.embed_dim
        self._freq_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self.fft_cache = FFTCache(max_size=config.fft_cache_size)

        # Left rotation parameters (learnable or fixed)
        if config.use_learned_rotation:
            self.theta_left = nn.Parameter(torch.tensor(config.theta_left, dtype=torch.float32, requires_grad=True))
            self.omega_left = nn.Parameter(torch.tensor(config.omega_left, dtype=torch.float32, requires_grad=True))
            self.phi_left = nn.Parameter(torch.tensor(config.phi_left, dtype=torch.float32, requires_grad=True))
            self.theta_right = nn.Parameter(torch.tensor(config.theta_right, dtype=torch.float32, requires_grad=True))
            self.omega_right = nn.Parameter(torch.tensor(config.omega_right, dtype=torch.float32, requires_grad=True))
            self.phi_right = nn.Parameter(torch.tensor(config.phi_right, dtype=torch.float32, requires_grad=True))
        else:
            self.register_buffer('theta_left', torch.tensor(config.theta_left))
            self.register_buffer('omega_left', torch.tensor(config.omega_left))
            self.register_buffer('phi_left', torch.tensor(config.phi_left))
            self.register_buffer('theta_right', torch.tensor(config.theta_right))
            self.register_buffer('omega_right', torch.tensor(config.omega_right))
            self.register_buffer('phi_right', torch.tensor(config.phi_right))

        # Initialize the spectral filter with improvements
        self.spectral_filter = SpectralFilter(config.alpha, use_windowing=config.use_windowing, window_type=config.window_type)

        # Projection layers
        self.v_proj = nn.Linear(self.total_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)

    def get_rotation_quaternions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the left and right rotation quaternions for SO(4) using a vectorized operation."""
        # Stack angles for batch processing
        thetas = torch.stack([self.theta_left, self.theta_right])
        omegas = torch.stack([self.omega_left, self.omega_right])
        phis = torch.stack([self.phi_left, self.phi_right])

        # Create quaternions in a single batch call
        quaternions = QuaternionOperations.create_unit_quaternion_batch(thetas, omegas, phis)

        # Unpack
        q_left, q_right = quaternions[0], quaternions[1]
        return q_left, q_right

    def _get_freq_cache(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache_key = (seq_len, device.type)
        if cache_key not in self._freq_cache:
            freq = fft.fftfreq(seq_len, d=1.0, device=device)
            self._freq_cache[cache_key] = freq
        return self._freq_cache[cache_key].to(device)

    def _compute_frequencies(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates FFT frequencies correctly using fftfreq (NOT np.gradient).
        """
        # For compatibility, if spatial_dims is not defined, use sequence dimension
        if self.config.spatial_dims is None or (isinstance(self.config.spatial_dims, (list, tuple)) and len(self.config.spatial_dims) == 1 and self.config.spatial_dims[0] == seq_len):
            # 1D case (compatibility with previous implementation)
            freqs = self._get_freq_cache(seq_len, device)

            # Calculate wave vector k = 2π * frequency
            k = 2 * math.pi * freqs.view(1, seq_len, 1)

            # CORRECTION: Enhanced numerical stability with sqrt(k² + ε)
            k_mag = torch.sqrt(k**2 + self.spectral_filter.epsilon)

            return k, k_mag
        else:
            # Multi-dimensional case
            k_vecs = [self._get_freq_cache(n, device) for n in self.config.spatial_dims]
            k_mesh = torch.meshgrid(*k_vecs, indexing='ij')

            # Calculate wave vector magnitude with enhanced stability
            k_squared = sum(k_i**2 for k_i in k_mesh)
            # CORRECTION: Better numerical stability
            k_mag = torch.sqrt(k_squared + self.spectral_filter.epsilon)

            return k_mesh, k_mag

    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        """Comprehensive input validation to ensure layer robustness with NaN handling."""
        # Check for tensor dimensions and feature size
        if x.dim() != 3:
            raise QRHDimensionError(f"Expected 3D tensor, but got {x.dim()}D")
        expected_features = 4 * self.config.embed_dim
        if x.size(-1) != expected_features:
            raise QRHDimensionError(f"Expected {expected_features} features for the last dimension, but got {x.size(-1)}")

        # Check for numerical stability with graceful NaN handling
        finite_mask = torch.isfinite(x)
        if not finite_mask.all():
            if not finite_mask.any():
                # All values are NaN/Inf - return zeros
                warnings.warn("Input tensor contains only NaN/Inf values. Returning zero tensor.")
                return torch.zeros_like(x)
            else:
                # Some values are NaN/Inf - replace with zeros
                x_clean = x.clone()
                x_clean[~finite_mask] = 0.0
                if self.config.enable_warnings:
                    warnings.warn(f"Input tensor contains {(~finite_mask).sum().item()} NaN/Inf values. Replacing with zeros.")
                return x_clean

        # Warn about untested devices
        if x.device.type not in ['cpu', 'cuda', 'mps', 'xla']:
            warnings.warn(f"Unsupported device type: '{x.device.type}'. The layer may work but has not been tested on this device.")

        return x

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # Tudo aqui usa device de x
        V = torch.einsum('bij,jk->bik', x, self.v_proj.weight)
        if self.v_proj.bias is not None:
            V = V + self.v_proj.bias
        return V.view(*x.shape[:-1], 4, self.config.embed_dim).permute(0, 1, 3, 2)

    def _apply_spectral_filtering(self, Ψ: torch.Tensor) -> torch.Tensor:
        """
        Applies spectral filtering in frequency domain:
        Ψ_filtered = F^{-1} { F(k) · F { Ψ } }

        Args:
            Ψ: Input tensor [B, T, D, 4] (batch, seq_len, embed_dim, quaternion_components)
        Returns:
            Filtered tensor with same shape
        """
        batch_size, seq_len, embed_dim, quat_dim = Ψ.shape

        # Apply windowing if enabled
        Ψ_windowed = self.spectral_filter.apply_window(Ψ, seq_len)

        # Forward FFT along sequence dimension - MUST be complex
        Ψ_fft = fft.fft(Ψ_windowed, dim=1)
        assert Ψ_fft.dtype in [torch.complex64, torch.complex128], f"FFT must be complex, got {Ψ_fft.dtype}"

        # Compute wave vector magnitudes
        k, k_mag = self._compute_frequencies(seq_len, Ψ.device)

        # Apply spectral filter F(k) - MUST return complex tensor
        filter_response = self.spectral_filter(k_mag)
        assert filter_response.dtype in [torch.complex64, torch.complex128], f"Filter must be complex, got {filter_response.dtype}"

        # Expand filter to match tensor dimensions [1, T, 1, 1]
        filter_expanded = filter_response.view(1, seq_len, 1, 1)

        # Apply filter in frequency domain - complex × complex multiplication
        Ψ_filtered_fft = Ψ_fft * filter_expanded

        # CRITICAL: Verify that phase rotation is actually applied
        assert Ψ_filtered_fft.dtype in [torch.complex64, torch.complex128], "Filtered FFT must be complex!"
        assert not torch.allclose(Ψ_filtered_fft.imag, torch.zeros_like(Ψ_filtered_fft.imag), atol=1e-10), \
            "Imaginary part is zero — filter may not be applying phase rotation!"

        # Inverse FFT to return to time domain - take real part only after IFFT
        Ψ_filtered = fft.ifft(Ψ_filtered_fft, dim=1).real

        return Ψ_filtered

    def _apply_quaternion_rotations(self, Ψ_filtered: torch.Tensor) -> torch.Tensor:
        """
        Applies quaternion rotations: R_left · Ψ_filtered · R_right

        Args:
            Ψ_filtered: Filtered tensor [B, T, D, 4] (quaternion components in last dim)
        Returns:
            Rotated tensor with same shape
        """
        # Get rotation quaternions
        q_left, q_right = self.get_rotation_quaternions()

        batch_size, seq_len, embed_dim, quat_dim = Ψ_filtered.shape

        # Reshape for quaternion operations: [B*T*D, 4]
        Ψ_flat = Ψ_filtered.reshape(-1, 4)

        # Apply left rotation: q_left * Ψ
        # Expand q_left to match batch dimensions
        q_left_expanded = q_left.unsqueeze(0).expand(Ψ_flat.size(0), -1)
        Ψ_left_rotated = QuaternionOperations.multiply(q_left_expanded, Ψ_flat)

        # Apply right rotation: (q_left * Ψ) * q_right
        # Expand q_right to match batch dimensions
        q_right_expanded = q_right.unsqueeze(0).expand(Ψ_flat.size(0), -1)
        Ψ_rotated_flat = QuaternionOperations.multiply(Ψ_left_rotated, q_right_expanded)

        # Reshape back to original dimensions
        Ψ_rotated = Ψ_rotated_flat.view(batch_size, seq_len, embed_dim, quat_dim)

        return Ψ_rotated

    def _postprocess_output(self, Ψ_rotated: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Postprocesses the rotated quaternion output and applies residual connection.

        Args:
            Ψ_rotated: Rotated tensor [B, T, D, 4]
            x: Original input tensor [B, T, 4*D] for residual connection
        Returns:
            Final output tensor [B, T, 4*D]
        """
        batch_size, seq_len, embed_dim, quat_dim = Ψ_rotated.shape

        # Reshape from [B, T, D, 4] to [B, T, 4*D]
        Ψ_reshaped = Ψ_rotated.permute(0, 1, 3, 2).reshape(batch_size, seq_len, self.total_dim)

        # Apply output projection
        Ψ_projected = torch.einsum('bij,jk->bik', Ψ_reshaped, self.out_proj.weight)
        if self.out_proj.bias is not None:
            Ψ_projected = Ψ_projected + self.out_proj.bias

        # Apply residual connection
        output = Ψ_projected + x

        return output

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core implementation of the forward pass, executed within an autocast context.

        Args:
            x: Input tensor [B, T, 4*D].
        Returns:
            Output tensor with the same shape and device as input.
        """
        x_validated = self._validate_input(x)
        Ψ = self._preprocess_input(x_validated)
        Ψ_filtered = self._apply_spectral_filtering(Ψ)
        Ψ_rotated = self._apply_quaternion_rotations(Ψ_filtered)
        # Pass validated 'x' for residual connection
        return self._postprocess_output(Ψ_rotated, x_validated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Top-level forward pass with device-aware Automatic Mixed Precision (AMP).
        Uses bfloat16 on compatible devices (CUDA, MPS) and float32 on CPU.
        """
        # Only use autocast on GPU devices that support it
        if x.device.type != 'cpu' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            with torch.autocast(device_type=x.device.type, dtype=dtype):
                return self._forward_impl(x)
        else:
            # Run directly without autocast on CPU to avoid warnings
            return self._forward_impl(x)

    def check_health(self, x: torch.Tensor) -> dict:
        """Performs validation checks inspired by the test suite."""
        health_report = {}
        with torch.no_grad():
            output = self.forward(x)
            
            # 1. Energy Conservation Test
            input_energy = torch.norm(x).item()
            output_energy = torch.norm(output).item()
            # Avoid division by zero
            if input_energy > 1e-6:
                energy_ratio = output_energy / input_energy
                # Healthy DNA should keep energy ratio within a reasonable bound (e.g., 0.5 to 2.0)
                health_report['energy_ratio'] = energy_ratio
                health_report['is_stable'] = 0.5 < energy_ratio < 2.0
            else:
                health_report['energy_ratio'] = 0
                health_report['is_stable'] = False

        return health_report
