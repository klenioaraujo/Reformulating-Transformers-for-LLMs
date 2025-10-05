import torch
import torch.nn as nn
import torch.fft as fft
import math


class HarmonicEvolutionLayer(nn.Module):
    """
    Harmonic Evolution Layer implementing the formula: R ⋅ F⁻¹{ F(k) ⋅ F{Ψ} }

    This layer replaces the traditional feed-forward network with a mathematically
    efficient harmonic evolution operation that preserves the rotational structure
    of quaternion representations.

    Based on Section 2.9.3 of the ΨQRH paper - "Evolução Harmônica"
    """

    def __init__(self, d_model: int, rotation_dim: int = 4, use_learnable_kernel: bool = True, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.rotation_dim = rotation_dim
        self.use_learnable_kernel = use_learnable_kernel
        self.max_seq_len = max_seq_len

        # Learnable rotation parameters R
        # R is a rotation matrix that operates on the quaternion space
        self.rotation_matrix = nn.Parameter(
            torch.eye(rotation_dim, dtype=torch.float32)
        )

        # Learnable frequency kernel k
        # This represents F(k) in the frequency domain
        if use_learnable_kernel:
            self.frequency_kernel = nn.Parameter(
                torch.ones(max_seq_len, dtype=torch.float32)  # Apply along sequence dimension
            )
        else:
            # Fixed frequency kernel (low-pass filter)
            self.register_buffer('frequency_kernel',
                torch.ones(max_seq_len, dtype=torch.float32)
            )

        # Normalization - use full quaternion dimension
        self.layer_norm = nn.LayerNorm(d_model * rotation_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing: R ⋅ F⁻¹{ F(k) ⋅ F{Ψ} }

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, d_model = x.shape

        # Reshape input to quaternion representation
        # Assuming d_model is divisible by 4 for quaternion representation
        quat_dim = d_model // self.rotation_dim
        x_quat = x.view(batch_size, seq_len, quat_dim, self.rotation_dim)

        # Apply Fourier transform along the sequence dimension
        # F{Ψ} - Fourier transform of the quaternion state
        x_freq = fft.fft(x_quat, dim=1)

        # Apply frequency kernel F(k) ⋅ F{Ψ}
        # The frequency kernel operates along the sequence dimension
        kernel_slice = self.frequency_kernel[:seq_len].view(1, seq_len, 1, 1)
        filtered_freq = x_freq * kernel_slice

        # Apply inverse Fourier transform
        # F⁻¹{ F(k) ⋅ F{Ψ} }
        filtered_time = fft.ifft(filtered_freq, dim=1).real

        # Apply rotation R ⋅ [result]
        # R operates on the quaternion dimension
        rotated = torch.einsum('bstq,oq->bsto', filtered_time, self.rotation_matrix)

        # Reshape back to original dimensions
        output = rotated.reshape(batch_size, seq_len, d_model)

        # Apply layer normalization
        output = self.layer_norm(output)

        return output

    def get_parameter_count(self) -> int:
        """
        Get the total number of parameters in this layer

        Returns:
            Total parameter count
        """
        total_params = 0

        # Rotation matrix parameters
        total_params += self.rotation_matrix.numel()

        # Frequency kernel parameters
        if self.use_learnable_kernel:
            total_params += self.frequency_kernel.numel()

        # Layer normalization parameters
        total_params += sum(p.numel() for p in self.layer_norm.parameters())

        return total_params