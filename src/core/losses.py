#!/usr/bin/env python3
"""
ΨQRH Spectral Entropy Loss Module
==================================

Implements spectral entropy loss for self-supervised training of quantum states.
This loss function measures the spectral structure of quantum states and encourages
the model to generate states with low entropy (high structure), characteristic of
coherent linguistic signals.
"""

import torch
import torch.nn as nn
import torch.fft as fft


class SpectralEntropyLoss(nn.Module):
    """
    Spectral Entropy Loss for quantum state regularization.

    This loss computes the spectral entropy of quantum states, encouraging
    the model to generate states with structured spectral signatures similar
    to natural language signals.
    """

    def __init__(self, embed_dim: int, epsilon: float = 1e-12):
        """
        Initialize the spectral entropy loss.

        Args:
            embed_dim: Embedding dimension of the quantum states
            epsilon: Small value to avoid log(0)
        """
        super(SpectralEntropyLoss, self).__init__()
        self.embed_dim = embed_dim
        self.epsilon = epsilon

    def forward(self, psi_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Spectral Entropy of the quantum state.

        Loss = -Σ(P(k) * log(P(k))) where P(k) is the normalized power spectrum

        Args:
            psi_tensor: Quantum state tensor of shape [batch, seq_len, embed_dim, 4]
                       where the last dimension represents quaternion components

        Returns:
            Normalized spectral entropy loss (0-1 range)
        """
        # 1. Get the power spectrum
        # Average the norm of quaternions to get a 1D signal per sequence element
        norm_psi = torch.norm(psi_tensor, p=2, dim=-1)  # [batch, seq_len, embed_dim]
        signal = torch.mean(norm_psi, dim=-1)  # [batch, seq_len]

        # FFT of the signal
        psi_fft = fft.fft(signal, dim=-1)
        power_spectrum = torch.abs(psi_fft)**2

        # 2. Normalize to get a probability distribution P(k)
        total_power = torch.sum(power_spectrum, dim=-1, keepdim=True)
        power_dist = power_spectrum / (total_power + self.epsilon)

        # 3. Calculate the entropy
        log_power_dist = torch.log2(power_dist + self.epsilon)
        entropy = -torch.sum(power_dist * log_power_dist, dim=-1)

        # Normalize by log of number of bins to have a value between 0 and 1
        seq_len = signal.shape[-1]
        max_entropy = torch.log2(torch.tensor(seq_len, dtype=torch.float32, device=signal.device))
        normalized_entropy = entropy / max_entropy

        # Return the mean entropy for the batch
        return torch.mean(normalized_entropy)


class QuantumContrastiveLoss(nn.Module):
    """
    Contrastive loss for quantum embeddings in the ΨQRH domain.

    This loss teaches the model to distinguish between correct and incorrect
    quantum representations of characters. For a given context, the correct
    target character (Ψ_positivo) should be close while incorrect characters
    (Ψ_negativo) should be far.

    Loss = Distancia(Contexto, Ψ_positivo) - Distancia(Contexto, Ψ_negativo)
    """

    def __init__(self, margin: float = 1.0, epsilon: float = 1e-12):
        """
        Initialize the quantum contrastive loss.

        Args:
            margin: Margin for the contrastive loss
            epsilon: Small epsilon for numerical stability
        """
        super(QuantumContrastiveLoss, self).__init__()
        self.margin = margin
        self.epsilon = epsilon

    def forward(self, psi_context: torch.Tensor, psi_positive: torch.Tensor,
                psi_negative: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss between quantum states.

        Loss = Distancia(Contexto, Ψ_positivo) - Distancia(Contexto, Ψ_negativo)

        Args:
            psi_context: Context quantum state [batch, embed_dim] or [batch, embed_dim, 4]
            psi_positive: Positive target quantum state [batch, embed_dim] or [batch, embed_dim, 4]
            psi_negative: Negative target quantum state [batch, embed_dim] or [batch, embed_dim, 4]

        Returns:
            Contrastive loss value
        """
        # Compute distances using quaternion distance metric
        dist_positive = self._quaternion_distance(psi_context, psi_positive)  # [batch]
        dist_negative = self._quaternion_distance(psi_context, psi_negative)  # [batch]

        # Contrastive loss: maximize distance to negative, minimize distance to positive
        # Loss = max(0, margin + dist_pos - dist_neg)
        loss = torch.clamp(self.margin + dist_positive - dist_negative, min=0.0)

        return torch.mean(loss)

    def _quaternion_distance(self, psi1: torch.Tensor, psi2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between two quaternion tensors.

        Uses the quaternion angular distance metric.

        Args:
            psi1: First quaternion tensor [batch, embed_dim] or [batch, embed_dim, 4]
            psi2: Second quaternion tensor [batch, embed_dim] or [batch, embed_dim, 4]

        Returns:
            Distance scores [batch]
        """
        # Handle different input formats
        if psi1.dim() == 2 and psi2.dim() == 2:
            # Both are flattened embeddings [batch, embed_dim]
            # Convert to quaternion format by reshaping
            batch_size = psi1.shape[0]
            embed_dim = psi1.shape[1]
            quat_dim = embed_dim // 4

            psi1_quat = psi1.view(batch_size, quat_dim, 4)
            psi2_quat = psi2.view(batch_size, quat_dim, 4)
        elif psi1.dim() == 3 and psi2.dim() == 3:
            # Both are quaternion format [batch, embed_dim, 4]
            psi1_quat = psi1
            psi2_quat = psi2
        else:
            raise ValueError(f"Incompatible tensor dimensions: psi1 {psi1.shape}, psi2 {psi2.shape}")

        # Quaternion distance: angular distance between unit quaternions
        # Distance = arccos(|<q1, q2>|) where <q1, q2> is the quaternion dot product
        dot_product = torch.sum(psi1_quat * psi2_quat, dim=[1, 2])  # [batch]

        # Normalize dot product (should be in [-1, 1] for unit quaternions)
        norm1 = torch.norm(psi1_quat, dim=[1, 2])  # [batch]
        norm2 = torch.norm(psi2_quat, dim=[1, 2])  # [batch]
        cos_similarity = dot_product / (norm1 * norm2 + self.epsilon)
        cos_similarity = torch.clamp(cos_similarity, -1.0, 1.0)

        # Angular distance in radians
        distance = torch.acos(cos_similarity)

        return distance


class QuantumEmbedding(nn.Module):
    """
    Learnable quantum embedding layer.

    Maps character IDs to quantum states (quaternions) through a learnable
    embedding followed by a transformation to quaternion space.
    """

    def __init__(self, vocab_size: int, embed_dim: int, init_scale: float = 0.1):
        """
        Initialize the quantum embedding layer.

        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension (will be mapped to embed_dim//4 quaternions)
            init_scale: Scale for random initialization
        """
        super(QuantumEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Standard embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformation to quaternion space
        # We use embed_dim//4 quaternions per character
        quaternion_dim = embed_dim // 4
        self.to_quaternion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, quaternion_dim * 4)  # Output 4 components per quaternion
        )

        # Initialize weights
        self._init_weights(init_scale)

    def _init_weights(self, init_scale: float):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum embedding.

        Args:
            input_ids: Character IDs [batch, seq_len]

        Returns:
            Quantum states [batch, seq_len, embed_dim, 4]
        """
        # Get standard embeddings
        embeddings = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        # Transform to quaternion space
        batch_size, seq_len, embed_dim = embeddings.shape
        quaternion_output = self.to_quaternion(embeddings.view(-1, embed_dim))  # [batch*seq_len, embed_dim]

        # Reshape to quaternion format
        quaternion_dim = embed_dim // 4
        psi = quaternion_output.view(batch_size, seq_len, quaternion_dim, 4)  # [batch, seq_len, quaternion_dim, 4]

        # Normalize to unit quaternions (optional, but helps with stability)
        psi_norm = torch.norm(psi, dim=-1, keepdim=True)
        psi_normalized = torch.div(psi, psi_norm + 1e-8)

        return psi_normalized


class QuantumStateRegularizationLoss(nn.Module):
    """
    Combined loss for quantum state regularization including spectral entropy
    and additional quantum coherence constraints.
    """

    def __init__(self, embed_dim: int, spectral_weight: float = 1.0,
                 coherence_weight: float = 0.1, epsilon: float = 1e-12):
        """
        Initialize the combined quantum regularization loss.

        Args:
            embed_dim: Embedding dimension
            spectral_weight: Weight for spectral entropy component
            coherence_weight: Weight for quantum coherence component
            epsilon: Small epsilon for numerical stability
        """
        super(QuantumStateRegularizationLoss, self).__init__()
        self.spectral_entropy = SpectralEntropyLoss(embed_dim, epsilon)
        self.spectral_weight = spectral_weight
        self.coherence_weight = coherence_weight
        self.epsilon = epsilon

    def forward(self, psi_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined quantum regularization loss.

        Args:
            psi_tensor: Quantum state tensor [batch, seq_len, embed_dim, 4]

        Returns:
            Combined loss value
        """
        # Spectral entropy component
        spectral_loss = self.spectral_entropy(psi_tensor)

        # Quantum coherence component (encourage unit quaternion norms)
        quaternion_norms = torch.norm(psi_tensor, p=2, dim=-1)  # [batch, seq_len, embed_dim]
        target_norm = torch.ones_like(quaternion_norms)
        coherence_loss = torch.mean((quaternion_norms - target_norm)**2)

        # Combined loss
        total_loss = (self.spectral_weight * spectral_loss +
                     self.coherence_weight * coherence_loss)

        return total_loss


class SpectralCoherenceLoss(nn.Module):
    """
    Spectral Coherence Loss for training quantum states to match linguistic spectral signatures.

    This loss compares the power spectrum of generated quantum states with reference
    linguistic signals, encouraging the model to produce states with spectral properties
    characteristic of natural language (following power-law distributions like Zipf's law).
    """

    def __init__(self, epsilon: float = 1e-12):
        """
        Initialize the spectral coherence loss.

        Args:
            epsilon: Small value to avoid division by zero and log(0)
        """
        super(SpectralCoherenceLoss, self).__init__()
        self.epsilon = epsilon
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, psi_output: torch.Tensor, psi_reference: torch.Tensor) -> torch.Tensor:
        """
        Calculate spectral coherence loss between output and reference quantum states.

        Loss = KL_Divergence(PowerSpectrum(Ψ_output) || PowerSpectrum(Ψ_reference))

        Args:
            psi_output: Generated quantum states [batch, seq_len, embed_dim, 4]
            psi_reference: Reference quantum states [batch, seq_len, embed_dim, 4]

        Returns:
            KL divergence loss between power spectra
        """
        # Convert quantum states to power spectra
        power_spectrum_output = self._compute_power_spectrum(psi_output)
        power_spectrum_reference = self._compute_power_spectrum(psi_reference)

        # Convert to log-probability distributions for KL divergence
        # KLDivLoss expects log probabilities as input and probabilities as target
        log_power_dist_output = torch.log(power_spectrum_output + self.epsilon)
        power_dist_reference = power_spectrum_reference

        # Calculate KL divergence
        loss = self.kl_div(log_power_dist_output, power_dist_reference)

        return loss

    def _compute_power_spectrum(self, psi_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized power spectrum from quantum states.

        Args:
            psi_tensor: Quantum states [batch, seq_len, embed_dim, 4]

        Returns:
            Normalized power spectrum [batch, freq_bins]
        """
        # Convert quaternion tensor to 1D signal
        # Use the norm of quaternions as the signal amplitude
        signal = torch.norm(psi_tensor, p=2, dim=-1)  # [batch, seq_len, embed_dim]

        # Average across embedding dimension to get sequence signal
        signal = torch.mean(signal, dim=-1)  # [batch, seq_len]

        # Compute FFT
        psi_fft = fft.fft(signal, dim=-1)
        power_spectrum = torch.abs(psi_fft) ** 2  # Power spectrum

        # Normalize to probability distribution
        total_power = torch.sum(power_spectrum, dim=-1, keepdim=True)
        normalized_spectrum = power_spectrum / (total_power + self.epsilon)

        return normalized_spectrum


class QuantumLinguisticAlignmentLoss(nn.Module):
    """
    Quantum-Linguistic Alignment Loss for connecting quantum and linguistic spaces.

    This loss function bridges the gap between the quantum space (where Kuramoto operates)
    and the linguistic space (where characters live) by:

    1. Converting target text to quantum space using embeddings
    2. Calculating alignment loss between generated and target quantum states
    3. Ensuring optical decoding produces the correct text

    This addresses the fundamental disconnect where quantum states don't correspond
    to meaningful linguistic representations.
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, device: str = 'cpu'):
        """
        Initialize the quantum-linguistic alignment loss.

        Args:
            vocab_size: Size of the character vocabulary
            embed_dim: Embedding dimension
            device: Device for tensor operations
        """
        super(QuantumLinguisticAlignmentLoss, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device = device

        # Components for alignment
        self.quantum_embedding = QuantumEmbedding(vocab_size, embed_dim)
        self.optical_probe = None  # Will be set externally

        # Loss weights
        self.space_alignment_weight = 1.0
        self.decoding_weight = 1.0

    def set_optical_probe(self, optical_probe):
        """Set the optical probe for decoding validation."""
        self.optical_probe = optical_probe

    def forward(self, quantum_state: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Calculate quantum-linguistic alignment loss.

        Args:
            quantum_state: Generated quantum state [batch, seq_len, embed_dim, 4]
            target_text: Target text string

        Returns:
            Combined alignment loss
        """
        # 1. Convert target text to quantum space
        target_quantum = self._text_to_quantum(target_text, quantum_state.shape)

        # 2. Calculate space alignment loss (MSE between quantum states)
        space_alignment_loss = self._calculate_space_alignment_loss(quantum_state, target_quantum)

        # 3. Calculate decoding consistency loss
        decoding_loss = self._calculate_decoding_loss(quantum_state, target_text)

        # 4. Combine losses
        total_loss = (self.space_alignment_weight * space_alignment_loss +
                     self.decoding_weight * decoding_loss)

        return total_loss

    def _text_to_quantum(self, text: str, target_shape: torch.Size) -> torch.Tensor:
        """
        Convert text to quantum space using embeddings.

        Args:
            text: Input text string
            target_shape: Target shape for quantum state [batch, seq_len, embed_dim, 4]

        Returns:
            Quantum representation of the text
        """
        # Convert text to character IDs
        char_ids = torch.tensor([ord(c) for c in text], dtype=torch.long, device=self.device)
        char_ids = char_ids.unsqueeze(0)  # [1, seq_len]

        # Get quantum embedding
        quantum_state = self.quantum_embedding(char_ids)  # [1, seq_len, embed_dim, 4]

        # Pad or truncate to match target shape
        batch_size, seq_len, embed_dim, _ = target_shape

        if quantum_state.shape[1] < seq_len:
            # Pad with zeros
            padding = torch.zeros(batch_size, seq_len - quantum_state.shape[1], embed_dim, 4, device=self.device)
            quantum_state = torch.cat([quantum_state, padding], dim=1)
        elif quantum_state.shape[1] > seq_len:
            # Truncate
            quantum_state = quantum_state[:, :seq_len]

        return quantum_state

    def _calculate_space_alignment_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate alignment loss between quantum spaces.

        Args:
            generated: Generated quantum state
            target: Target quantum state

        Returns:
            MSE loss between quantum states
        """
        # Ensure same shape
        min_seq_len = min(generated.shape[1], target.shape[1])
        generated = generated[:, :min_seq_len]
        target = target[:, :min_seq_len]

        # MSE loss between quantum states
        loss = nn.functional.mse_loss(generated, target)
        return loss

    def _calculate_decoding_loss(self, quantum_state: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Calculate decoding consistency loss.

        Args:
            quantum_state: Quantum state to decode
            target_text: Expected decoded text

        Returns:
            Cross-entropy loss between decoded and target text
        """
        if self.optical_probe is None:
            return torch.tensor(0.0, device=self.device)

        # Decode using optical probe
        decoded_text = self.optical_probe(quantum_state)

        # Convert target text to character IDs for loss calculation
        target_ids = torch.tensor([ord(c) for c in target_text], dtype=torch.long, device=self.device)

        # Convert decoded text to IDs
        decoded_ids = torch.tensor([ord(c) for c in decoded_text], dtype=torch.long, device=self.device)

        # Pad or truncate to same length
        min_len = min(len(target_ids), len(decoded_ids))
        target_ids = target_ids[:min_len]
        decoded_ids = decoded_ids[:min_len]

        # Cross-entropy loss
        loss = nn.functional.cross_entropy(
            decoded_ids.unsqueeze(0).float(),
            target_ids.unsqueeze(0),
            reduction='mean'
        )

        return loss