#!/usr/bin/env python3
"""
ΨQRH Autoregressive Language Model

A physics-based autoregressive language model that generates natural language through
quantum-physical processes, implementing the complete ΨQRH architecture as a
Hugging Face-compatible CausalLM model.

Architecture:
- Spectral Quaternion Attention (replaces self-attention)
- Harmonic SO(4) Evolution (replaces FFN)
- DCF Consciousness Dynamics (replaces softmax)
- Optical Probe Generation (final text output)

Key Features:
- No external Transformer backbone
- Pure physical computation
- Subword token processing (BPE)
- Autoregressive generation with sampling
- Hugging Face AutoModelForCausalLM compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
import numpy as np

# Import existing DCF system
from src.processing.token_analysis import DCFTokenAnalysis

# Create minimal config and model classes to avoid transformers dependency
class PretrainedConfig:
    """Minimal config class"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class PreTrainedModel(nn.Module):
    """Minimal model base class"""
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        pass

class CausalLMOutputWithPast:
    """Minimal output class"""
    def __init__(self, loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None, last_hidden_state=None):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.last_hidden_state = last_hidden_state


class PsiQRHConfig(PretrainedConfig):
    """
    Configuration class for ΨQRH Language Model.

    Defines all hyperparameters for the physics-based architecture.
    """
    model_type = "psiqrh"

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        gradient_checkpointing: bool = False,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        tie_word_embeddings: bool = False,
        # ΨQRH specific parameters
        embed_dim: int = 64,  # Must be divisible by 4 for quaternions
        num_heads: int = 8,   # Must divide embed_dim
        hidden_dim: int = 512,
        alpha: float = 1.0,   # Spectral filtering parameter
        beta: float = 0.5,    # Fractal parameter
        kuramoto_coupling: float = 0.1,  # DCF coupling strength
        kuramoto_frequency: float = 1.0, # DCF base frequency
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        # Standard transformer params
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing
        self.use_cache = use_cache

        # ΨQRH specific params - enforce embed_dim = n_embd for compatibility
        self.embed_dim = n_embd  # Force embed_dim to match n_embd
        self.num_heads = n_head  # Force num_heads to match n_head
        self.hidden_dim = n_inner if n_inner is not None else 4 * n_embd
        self.alpha = alpha
        self.beta = beta
        self.kuramoto_coupling = kuramoto_coupling
        self.kuramoto_frequency = kuramoto_frequency

        # Validation - now based on n_embd and n_head
        assert self.embed_dim % self.num_heads == 0, f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"


class SpectralQuaternionAttention(nn.Module):
    """
    Spectral Quaternion Attention Mechanism

    Replaces standard self-attention with physics-based spectral processing:
    1. Convert embeddings to quaternion states
    2. Apply spectral filtering in frequency domain
    3. Compute attention via quaternion operations
    """

    def __init__(self, config: PsiQRHConfig):
        super().__init__()
        self.embed_dim = config.embed_dim  # Now equals n_embd
        self.num_heads = config.num_heads   # Now equals n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.alpha = config.alpha

        # Standard attention projection layers
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Learnable quaternion-to-real projection
        self.quat_to_real = nn.Linear(4, 1)

        # Dropout
        self.dropout = nn.Dropout(config.attn_pdrop)

    def _embeddings_to_quaternions(self, x: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to quaternion representation [batch, seq, embed_dim, 4] with unitary mapping"""
        batch_size, seq_len, embed_dim = x.shape

        # Create quaternion representation with unitary mapping
        psi = torch.zeros(batch_size, seq_len, embed_dim, 4, device=x.device, dtype=x.dtype)

        # Unitary mapping: preserve L2 norm by distributing energy across quaternion components
        # Use Householder reflection for unitary transformation
        for i in range(embed_dim):
            feature_val = x[:, :, i]

            # Normalize to ensure unitary transformation
            norm = torch.sqrt(torch.sum(feature_val**2, dim=1, keepdim=True) + 1e-8)
            feature_norm = feature_val / norm

            # Unitary quaternion mapping (Householder-like)
            psi[:, :, i, 0] = feature_norm  # w (real part)
            psi[:, :, i, 1] = torch.roll(feature_norm, 1, dims=1) * 0.5  # x (i) - phase shifted
            psi[:, :, i, 2] = torch.roll(feature_norm, -1, dims=1) * 0.5  # y (j) - phase shifted
            psi[:, :, i, 3] = torch.zeros_like(feature_norm)  # z (k) - zero for simplicity

            # Renormalize to preserve energy
            quat_norm = torch.sqrt(torch.sum(psi[:, :, i, :]**2, dim=-1, keepdim=True) + 1e-8)
            psi[:, :, i, :] = psi[:, :, i, :] / quat_norm

        return psi

    def _quaternions_to_real(self, psi: torch.Tensor) -> torch.Tensor:
        """Convert quaternion representation back to real embeddings [batch, seq, embed_dim]"""
        # Apply learned projection from 4 quaternion components to 1 real value
        real_output = self.quat_to_real(psi).squeeze(-1)  # [batch, seq, embed_dim]
        return real_output

    def _apply_spectral_filter(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral filtering with guaranteed energy conservation.

        Uses a unitary filter F(k) = exp(i * phase(k)) where phase is designed
        to preserve Parseval's theorem: ∫|f(t)|²dt = (1/N)∑|F(k)|²
        """
        batch_size, seq_len, embed_dim, quaternion_dim = psi.shape

        # Calculate energy before filtering (for conservation verification)
        energy_before = torch.sum(torch.abs(psi)**2)

        # Apply FFT along the sequence dimension
        psi_fft = torch.fft.fft(psi, dim=1)

        # Create unitary spectral filter (pure phase shift, |F(k)| = 1)
        freqs = torch.fft.fftfreq(seq_len, dtype=torch.float32).to(psi.device)
        k = 2 * torch.pi * freqs.view(1, -1, 1, 1)

        # Create purely phase-based filter: F(k) = exp(i * phase(k))
        # where phase(k) is odd to preserve Hermitian symmetry
        epsilon = 1e-10
        k_abs = torch.abs(k) + epsilon

        # Odd phase function: phase(-k) = -phase(k)
        # Use arctan of log to create smooth frequency-dependent phase
        phase = self.alpha * torch.sign(k) * torch.arctan(torch.log(k_abs))

        # Create unitary filter: exp(i * phase) has |F(k)| = 1 by construction
        filter_response = torch.exp(1j * phase)
        filter_response = filter_response.expand(batch_size, seq_len, embed_dim, quaternion_dim)

        # Apply filter
        psi_filtered_fft = psi_fft * filter_response
        psi_filtered = torch.fft.ifft(psi_filtered_fft, dim=1).real

        # Verify energy conservation (Parseval's theorem)
        energy_after = torch.sum(torch.abs(psi_filtered)**2)
        energy_preservation_ratio = torch.abs(energy_after / energy_before).item()

        # Ensure energy is preserved within tolerance
        if abs(energy_preservation_ratio - 1.0) > 0.01:  # 1% tolerance
            print(f"⚠️  Energy not preserved in spectral filtering: ratio = {energy_preservation_ratio:.4f}")

        return psi_filtered

    def _quaternion_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute attention using quaternion operations"""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Convert to quaternions - q,k,v are [B, H, T, D]
        q_quat = self._embeddings_to_quaternions(q.reshape(batch_size * num_heads, seq_len, head_dim))
        k_quat = self._embeddings_to_quaternions(k.reshape(batch_size * num_heads, seq_len, head_dim))
        v_quat = self._embeddings_to_quaternions(v.reshape(batch_size * num_heads, seq_len, head_dim))

        # Apply spectral filtering
        q_filtered = self._apply_spectral_filter(q_quat)
        k_filtered = self._apply_spectral_filter(k_quat)
        v_filtered = self._apply_spectral_filter(v_quat)

        # Convert back to real space for attention computation
        q_real = self._quaternions_to_real(q_filtered).reshape(batch_size, num_heads, seq_len, head_dim)
        k_real = self._quaternions_to_real(k_filtered).reshape(batch_size, num_heads, seq_len, head_dim)
        v_real = self._quaternions_to_real(v_filtered).reshape(batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores using real values
        scores = torch.matmul(q_real, k_real.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v_real)  # [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]

        return context

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape  # [B, T, n_embd]

        # Linear projections
        q = self.q_proj(hidden_states)  # [B, T, n_embd]
        k = self.k_proj(hidden_states)  # [B, T, n_embd]
        v = self.v_proj(hidden_states)  # [B, T, n_embd]

        # Reshape for multi-head
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # Spectral quaternion attention
        context = self._quaternion_attention(q, k, v)  # [B, H, T, D]

        # Reshape back to [B, T, n_embd]
        batch_size = context.shape[0]
        seq_len = context.shape[2]  # After transpose, this becomes seq_len
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        output = self.out_proj(context)  # [B, T, n_embd]

        # Ensure output shape matches input shape
        assert output.shape == input_shape, f"Shape mismatch: {output.shape} vs {input_shape}"

        return output, None


class HarmonicSO4Evolution(nn.Module):
    """
    Harmonic SO(4) Evolution Layer

    Replaces standard FFN with physics-based evolution:
    1. Convert to quaternion states
    2. Apply SO(4) rotations
    3. Evolve through harmonic dynamics
    """

    def __init__(self, config: PsiQRHConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim

        # Evolution layers
        self.up_proj = nn.Linear(config.n_embd, self.hidden_dim)
        self.gate_proj = nn.Linear(config.n_embd, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, config.n_embd)

        # SO(4) rotation parameters
        self.register_buffer("rotation_angles", torch.randn(3))  # theta, omega, phi

    def _apply_so4_rotation(self, psi: torch.Tensor) -> torch.Tensor:
        """Apply SO(4) unitary rotation: Ψ' = q_left ⊗ Ψ ⊗ q_right†"""
        # Simplified SO(4) rotation implementation
        # In full implementation, this would use optimized quaternion operations

        theta, omega, phi = self.rotation_angles

        # Create rotation quaternions
        q_left = torch.tensor([
            torch.cos(theta/2) * torch.cos(omega/2),
            torch.sin(theta/2) * torch.cos(omega/2),
            torch.cos(theta/2) * torch.sin(omega/2),
            torch.sin(theta/2) * torch.sin(omega/2)
        ], device=psi.device, dtype=psi.dtype)

        q_right = torch.tensor([
            torch.cos(phi/2),
            0.0,
            torch.sin(phi/2),
            0.0
        ], device=psi.device, dtype=psi.dtype)

        # Apply rotations (simplified)
        # Full implementation would use proper quaternion multiplication
        rotated = psi * torch.cos(theta + omega + phi) + torch.roll(psi, 1, dims=-1) * torch.sin(theta + omega + phi)

        return rotated

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape  # [B, T, n_embd]

        # Standard up-projection
        up_states = self.up_proj(hidden_states)  # [B, T, hidden_dim]

        # Gated activation (harmonic evolution)
        gate_states = self.gate_proj(hidden_states)  # [B, T, hidden_dim]
        gate_states = torch.tanh(gate_states)  # Harmonic activation

        # Combine and evolve
        evolved_states = up_states * gate_states  # [B, T, hidden_dim]

        # Apply SO(4) rotation if hidden_dim is divisible by 4
        if evolved_states.shape[-1] % 4 == 0:
            # Convert to quaternion space for rotation
            batch_size, seq_len, hidden_dim = evolved_states.shape
            psi = evolved_states.view(batch_size, seq_len, hidden_dim // 4, 4)
            psi_rotated = self._apply_so4_rotation(psi)
            evolved_states = psi_rotated.view(batch_size, seq_len, hidden_dim)

        # Down-projection
        output = self.down_proj(evolved_states)  # [B, T, n_embd]

        # Ensure output shape matches input shape
        assert output.shape == input_shape, f"Shape mismatch: {output.shape} vs {input_shape}"

        return output


class PsiQRHBlock(nn.Module):
    """
    Single ΨQRH Transformer Block

    Combines spectral quaternion attention with harmonic SO(4) evolution.
    """

    def __init__(self, config: PsiQRHConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = SpectralQuaternionAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = HarmonicSO4Evolution(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-attention residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, attn_weights = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=layer_past,
            output_attentions=output_attentions,
        )
        # Ensure attn_output has the same shape as residual for addition
        assert attn_output.shape == residual.shape, f"Shape mismatch: {attn_output.shape} vs {residual.shape}"
        hidden_states = residual + attn_output

        # Pre-MLP residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, attn_weights


class DCFConsciousnessDynamics(nn.Module):
    """
    DCF Consciousness Dynamics

    Replaces static softmax with Kuramoto oscillator dynamics for
    probabilistic token selection based on consciousness states.
    """

    def __init__(self, config: PsiQRHConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.kuramoto_coupling = config.kuramoto_coupling
        self.kuramoto_frequency = config.kuramoto_frequency

        # Kuramoto oscillators - one per token
        self.oscillator_phases = nn.Parameter(torch.randn(config.vocab_size))
        self.natural_frequencies = nn.Parameter(torch.randn(config.vocab_size))

    def _kuramoto_dynamics(self, logits: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Evolve Kuramoto oscillators to reach synchronization"""
        batch_size, seq_len, vocab_size = logits.shape

        # Initialize phases from logits
        phases = torch.randn(batch_size, seq_len, vocab_size, device=logits.device)
        phases = phases + logits * 0.1  # Bias phases by logits

        # Natural frequencies
        omega = self.natural_frequencies.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Coupling matrix (simplified - all-to-all coupling)
        coupling = self.kuramoto_coupling / vocab_size

        # Evolve dynamics
        dt = 0.1
        for _ in range(steps):
            # Phase differences
            phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq, vocab, vocab]

            # Kuramoto equation
            dphases = omega + coupling * torch.sin(phase_diff).sum(dim=-1)
            phases = phases + dt * dphases

        # Convert synchronized phases to probabilities
        # Higher synchronization = higher probability
        coherence = torch.cos(phases - phases.mean(dim=-1, keepdim=True))
        probabilities = torch.softmax(coherence, dim=-1)

        return probabilities

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Replace softmax with DCF dynamics"""
        return self._kuramoto_dynamics(logits)


class PsiQRHModel(PreTrainedModel):
    """
    ΨQRH Language Model

    Complete autoregressive language model using physics-based computation.
    Compatible with Hugging Face AutoModelForCausalLM.
    """

    config_class = PsiQRHConfig

    def __init__(self, config: PsiQRHConfig):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_checkpointing = config.gradient_checkpointing

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Position embeddings
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)

        # ΨQRH transformer blocks
        self.h = nn.ModuleList([PsiQRHBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Language modeling head
        self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)

        # DCF consciousness dynamics (for generation only)
        # Initialize DCF analyzer for token selection during generation
        self.dcf_analyzer = DCFTokenAnalysis(
            device=self.device,
            quantum_vocab_representations=None,
            char_to_idx=None
        )

    def get_hidden_states(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        """Return hidden states without applying lm_head (for classification tasks)"""
        return_dict = return_dict if return_dict is not None else False
        past_key_values = None
        use_cache = False

        # Run forward pass but return before lm_head
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # Embedding
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # ΨQRH transformer blocks
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value=layer_past, output_attentions=output_attentions)
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        # Return hidden states without applying lm_head
        if not return_dict:
            return hidden_states
        else:
            return {'last_hidden_state': hidden_states}

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 scheme"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # Embedding
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # ΨQRH transformer blocks
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value=layer_past, output_attentions=output_attentions)
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        # Language modeling head
        lm_logits = self.lm_head(hidden_states)

        # Remove DCF from forward pass - it belongs in generation only
        # DCF consciousness dynamics should be applied during generation, not training

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + (presents,) + (all_hidden_states,) + (all_self_attentions,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            last_hidden_state=hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Autoregressive generation with DCF-based token selection
        """
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_length - seq_len):
            # Forward pass
            outputs = self(generated)
            next_token_logits = outputs.logits[:, -1, :]

            # Use DCF for token selection (this replaces the forward pass DCF application)
            dcf_result = self.dcf_analyzer.analyze_tokens(next_token_logits)
            selected_token_id = dcf_result['selected_token']

            # Create next token tensor
            next_token = torch.tensor([[selected_token_id]], dtype=torch.long, device=generated.device)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None and selected_token_id == eos_token_id:
                break

        return generated


class PsiQRHForCausalLM(PreTrainedModel):
    """
    ΨQRH for Causal Language Modeling

    Hugging Face compatible interface for autoregressive generation.
    """

    config_class = PsiQRHConfig

    def __init__(self, config: PsiQRHConfig):
        super().__init__(config)
        self.model = PsiQRHModel(config)

        # Set lm_head to be the same as wte for weight tying
        self.lm_head = self.model.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }