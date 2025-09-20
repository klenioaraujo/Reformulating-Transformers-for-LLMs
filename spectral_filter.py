import torch
import torch.nn as nn

class SpectralFilter(nn.Module):
    """Implements the logarithmic phase filter F(k) for negentropy filtering with enhanced numerical stability."""

    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-10, use_stable_activation: bool = True,
                 use_windowing: bool = True, window_type: str = 'hann'):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.use_stable_activation = use_stable_activation
        self.use_windowing = use_windowing
        self.window_type = window_type

        # Parameters for enhanced stabilization
        self.register_buffer('k_min', torch.tensor(1e-12))  # More stable
        self.register_buffer('k_max', torch.tensor(1e6))    # Larger range

    def forward(self, k_mag: torch.Tensor) -> torch.Tensor:
        """
        Applies the spectral filter with enhanced numerical stabilization.

        Args:
            k_mag: Magnitude of the wave vector, shape [..., dims]
        Returns:
            Applied filter with the same shape as k_mag
        """
        # Clamp to avoid extreme values with better precision
        k_mag_clamped = torch.clamp(k_mag, self.k_min, self.k_max)

        if self.use_stable_activation:
            # Stabilized version using GELU instead of arctan
            # F(k) = exp(i * alpha * GELU(log(k_mag + epsilon)))
            log_k = torch.log(k_mag_clamped + self.epsilon)

            # More robust normalization for multi-dimensional tensors
            log_k_mean = log_k.mean(dim=-1, keepdim=True)
            log_k_std = log_k.std(dim=-1, keepdim=True) + self.epsilon
            log_k_normalized = (log_k - log_k_mean) / log_k_std

            # Use GELU for stable smoothing
            phase = self.alpha * torch.nn.functional.gelu(log_k_normalized)
        else:
            # Original version with improvements
            log_k = torch.log(k_mag_clamped + self.epsilon)
            phase = self.alpha * torch.arctan(log_k)

        # Apply filter with enhanced NaN/Inf check
        filter_response = torch.exp(1j * phase)

        # Replace invalid values with identity for better precision
        invalid_mask = torch.isnan(filter_response) | torch.isinf(filter_response)
        filter_response = torch.where(invalid_mask, torch.ones_like(filter_response), filter_response)

        return filter_response

    def apply_window(self, signal: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Applies windowing to reduce spectral leakage.

        Args:
            signal: Input signal [batch, seq_len, ...]
            seq_len: Sequence length
        Returns:
            Signal with windowing applied
        """
        if not self.use_windowing:
            return signal

        if self.window_type == 'hann':
            # Hann window for leakage reduction
            window = torch.hann_window(seq_len, device=signal.device)
        elif self.window_type == 'hamming':
            # Hamming window
            window = torch.hamming_window(seq_len, device=signal.device)
        elif self.window_type == 'blackman':
            # Blackman window
            window = torch.blackman_window(seq_len, device=signal.device)
        else:
            # Rectangular window (no windowing)
            window = torch.ones(seq_len, device=signal.device)

        # Apply windowing using einsum for efficiency
        # signal: [batch, seq_len, ...], window: [seq_len]
        window_expanded = window.view(1, seq_len, *(1,) * (signal.dim() - 2))
        return torch.einsum('b... , b... -> b...', signal, window_expanded)