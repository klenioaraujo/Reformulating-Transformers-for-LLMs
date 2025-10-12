"""
ΨQRH Transformer Architecture

Main transformer implementation integrating quaternionic operations,
spectral analysis, and fractal consciousness metrics.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
Project: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs
"""

import torch
import torch.nn as nn
import math
import yaml
from pathlib import Path
from typing import Optional, Dict, Tuple, Union

from ..core.quaternion_operations import (
    QuaternionLinear,
    QuaternionLayerNorm,
    SpectralActivation,
    AdaptiveSpectralDropout,
    RealTimeFractalAnalyzer
)
from ..core.spectral_harmonic_processor import QuaternionMLP


def load_transformer_config(config_path: Optional[Union[str, Path]] = None, preset: str = 'standard') -> Dict:
    """
    Carrega configuração do transformer de arquivo YAML.

    Args:
        config_path: Caminho para arquivo de configuração. Se None, usa padrão.
        preset: Preset a usar ('minimal', 'standard', 'full', 'consciousness')

    Returns:
        Dicionário com configurações
    """
    if config_path is None:
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "configs" / "psiqrh_transformer_config.yaml"

    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    # Config base
    config = full_config['psiqrh_transformer']

    # Aplicar preset se especificado
    if preset and preset in full_config['presets']:
        preset_config = full_config['presets'][preset]
        # Merge recursivo
        config = _merge_configs(config, preset_config)

    return config


def _merge_configs(base: Dict, override: Dict) -> Dict:
    """Merge recursivo de configurações"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


class QuaternionTokenEmbedding(nn.Module):
    """RIGOROUS token embedding with quaternion representation (doe.md 2.9.1)

    Implements EXACTLY:
    - ψ₀ = Re(MLP(x))
    - ψ₁ = Im(MLP(x))
    - ψ₂, ψ₃ via SO(4) rotational transformations
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Standard embedding (converts token IDs to vectors)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # RIGOROUS: QuaternionMLP for ψ₀ = Re(MLP(x)), ψ₁ = Im(MLP(x))
        self.quaternion_mlp = QuaternionMLP(embed_dim=d_model)

        # Lightweight rotational parameters for ψ₂, ψ₃
        self.rotation_angles = nn.Parameter(torch.randn(d_model, 2) * 0.01)
        self.rotation_scales = nn.Parameter(torch.ones(d_model, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RIGOROUS forward following doe.md 2.9.1

        Args:
            x: Token IDs [B, T]

        Returns:
            Quaternion embedding [B, T, d_model * 4]
        """
        # Standard embedding: token IDs → vectors
        embedded = self.embedding(x)  # [B, T, d_model]

        # RIGOROUS: MLP → complex (ψ₀ + iψ₁)
        complex_out = self.quaternion_mlp(embedded)  # [B, T, d_model] complex

        # Extract ψ₀ and ψ₁
        psi_0 = complex_out.real  # [B, T, d_model]
        psi_1 = complex_out.imag  # [B, T, d_model]

        # Generate ψ₂ and ψ₃ via SO(4) rotational transformations
        # Using lightweight rotation parameters
        psi_2 = psi_0 * self.rotation_scales[:, 0] + psi_1 * self.rotation_scales[:, 1]
        psi_3 = psi_1 * self.rotation_scales[:, 0] - psi_0 * self.rotation_scales[:, 1]

        # Apply rotation angles
        psi_2 = psi_2 * torch.cos(self.rotation_angles[:, 0])
        psi_3 = psi_3 * torch.sin(self.rotation_angles[:, 1])

        # Stack all four quaternion components: Ψ = ψ₀ + ψ₁i + ψ₂j + ψ₃k
        quaternion_embedded = torch.stack([psi_0, psi_1, psi_2, psi_3], dim=-1)
        quaternion_embedded = quaternion_embedded.view(*quaternion_embedded.shape[:-2], self.d_model * 4)

        return quaternion_embedded


class SpectralPositionalEncoding(nn.Module):
    """Positional encoding using spectral decomposition"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Learnable frequency components
        self.frequencies = nn.Parameter(
            torch.randn(d_model // 4) * 2 * math.pi
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Generate spectral positional encoding
        positions = torch.arange(seq_len, device=x.device).float()

        # Apply frequency modulation
        spectral_encoding = torch.zeros_like(x)
        for i, freq in enumerate(self.frequencies):
            phase = positions * freq
            spectral_encoding[:, :, i*4:(i+1)*4] = torch.stack([
                torch.cos(phase), torch.sin(phase),
                torch.cos(phase * 1.5), torch.sin(phase * 1.5)
            ], dim=-1)

        return x + spectral_encoding


class AdaptiveSpectralFilter(nn.Module):
    """Adaptive spectral filter for ΨQRH attention"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Learnable filter parameters - match the head dimension
        # head_dim = (d_model * 4) // n_heads, but we need to handle variable n_heads
        # Use a larger dimension that can broadcast to common head dimensions
        self.alpha = nn.Parameter(torch.ones(256))  # head_dim
        self.beta = nn.Parameter(torch.zeros(256))  # head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply logarithmic phase filter
        magnitude = torch.abs(x)
        phase = torch.angle(x)

        # Adaptive filtering - ensure proper broadcasting
        # Get the actual size of the last dimension
        last_dim = x.size(-1)

        # Slice or repeat parameters to match the input size
        if last_dim <= self.alpha.size(0):
            alpha_slice = self.alpha[:last_dim]
            beta_slice = self.beta[:last_dim]
        else:
            # Repeat parameters if input is larger
            repeat_factor = (last_dim + self.alpha.size(0) - 1) // self.alpha.size(0)
            alpha_slice = self.alpha.repeat(repeat_factor)[:last_dim]
            beta_slice = self.beta.repeat(repeat_factor)[:last_dim]

        # Expand to match input dimensions
        alpha_expanded = alpha_slice.view(1, 1, 1, -1)
        beta_expanded = beta_slice.view(1, 1, 1, -1)

        filtered_magnitude = magnitude * torch.sigmoid(alpha_expanded)
        filtered_phase = phase + beta_expanded

        # Reconstruct complex tensor
        filtered_x = filtered_magnitude * torch.exp(1j * filtered_phase)

        # Preserve Parseval by normalizing energy
        input_energy = torch.sum(torch.abs(x)**2)
        output_energy = torch.sum(torch.abs(filtered_x)**2)

        # Avoid division by zero
        if output_energy > 1e-8:
            scale = torch.sqrt(input_energy / output_energy)
            filtered_x = filtered_x * scale

        return filtered_x

    def update_alpha(self, new_alpha: torch.Tensor):
        """Update alpha parameter based on fractal analysis"""
        with torch.no_grad():
            self.alpha.data = 0.9 * self.alpha.data + 0.1 * new_alpha


class SpectralStateDecomposer(nn.Module):
    """
    Spectral State Decomposer - Derives Q, K, V from spectral decomposition
    of a single quaternion state, eliminating the need for separate projections.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = (d_model * 4) // n_heads  # Quaternion expands by 4

        # Ultra-lightweight spectral filters with extreme bottleneck
        # Use much smaller bottleneck dimensions for maximum parameter efficiency
        bottleneck_dim = max(d_model // 8, 16)  # Extreme bottleneck

        # Use depthwise separable convolutions for maximum efficiency
        self.spectral_filter_q = nn.Sequential(
            nn.Conv1d(d_model * 4, bottleneck_dim, kernel_size=3, padding=1, groups=4),  # Grouped convolution
            nn.ReLU(),
            nn.Conv1d(bottleneck_dim, d_model * 4, kernel_size=1)  # Pointwise convolution
        )

        self.spectral_filter_k = nn.Sequential(
            nn.Conv1d(d_model * 4, bottleneck_dim, kernel_size=3, padding=1, groups=4),  # Grouped convolution
            nn.ReLU(),
            nn.Conv1d(bottleneck_dim, d_model * 4, kernel_size=1)  # Pointwise convolution
        )

        self.spectral_filter_v = nn.Sequential(
            nn.Conv1d(d_model * 4, bottleneck_dim, kernel_size=3, padding=1, groups=4),  # Grouped convolution
            nn.ReLU(),
            nn.Conv1d(bottleneck_dim, d_model * 4, kernel_size=1)  # Pointwise convolution
        )

        # Spectral normalization
        self.spectral_norm = nn.LayerNorm(d_model * 4)

    def forward(self, psi_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Derive Q, K, V from spectral decomposition of input state Ψ(x)

        Args:
            psi_x: Input quaternion state [batch_size, seq_len, d_model * 4]

        Returns:
            Q, K, V tensors for spectral attention
        """
        batch_size, seq_len, _ = psi_x.shape

        # Transform to frequency domain (use magnitude for filtering)
        psi_x_fft = torch.fft.fft(psi_x, dim=1)  # [batch_size, seq_len, d_model * 4]
        psi_x_fft_magnitude = torch.abs(psi_x_fft)  # Use magnitude for spectral filtering

        # Apply spectral filters to derive Q, K, V components
        # Permute for Conv1d: [batch_size, channels, seq_len]
        psi_x_fft_permuted = psi_x_fft_magnitude.permute(0, 2, 1)

        # Apply ultra-lightweight spectral filters
        F_psi_q = self.spectral_filter_q(psi_x_fft_permuted)  # [batch_size, d_model*4, seq_len]
        F_psi_k = self.spectral_filter_k(psi_x_fft_permuted)  # [batch_size, d_model*4, seq_len]
        F_psi_v = self.spectral_filter_v(psi_x_fft_permuted)  # [batch_size, d_model*4, seq_len]

        # Transform back to time domain
        Q = torch.fft.ifft(F_psi_q.permute(0, 2, 1), dim=1).real  # [batch_size, seq_len, d_model*4]
        K = torch.fft.ifft(F_psi_k.permute(0, 2, 1), dim=1).real  # [batch_size, seq_len, d_model*4]
        V = torch.fft.ifft(F_psi_v.permute(0, 2, 1), dim=1).real  # [batch_size, seq_len, d_model*4]

        # Apply spectral normalization
        Q = self.spectral_norm(Q)
        K = self.spectral_norm(K)
        V = self.spectral_norm(V)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

        return Q, K, V


class PsiQRHAttention(nn.Module):
    """Attention mechanism using ΨQRH spectral operations with latent coupling"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model  # This is already d_model * 4 from the transformer
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # d_model is already expanded
        self.d_latent = 4 * (d_model // 4)  # Latent dimension based on original d_model

        # Projeção Latente Compartilhada
        self.z_proj = nn.Linear(d_model, self.d_latent)

        # Normalização após projeção latente
        self.z_norm = nn.LayerNorm(self.d_latent)

        # Projeções Derivadas - output n_heads * head_dim for multi-head
        self.q_proj = nn.Linear(self.d_latent, self.n_heads * self.head_dim)
        self.r_proj = nn.Linear(self.d_latent, self.n_heads * self.head_dim)
        self.h_proj = nn.Linear(self.d_latent, self.n_heads * self.head_dim)

        # Ativação de Fase Ψ
        self.phi_proj = nn.Linear(self.head_dim, self.head_dim)

        # Single output projection to combine heads
        self.out_proj = nn.Linear(d_model, d_model)

    def _phasor_activation(self, v: torch.Tensor) -> torch.Tensor:
        """
        Ativação de Fase Ψ(v) = v ⊙ exp(i ⋅ vW_φ)

        Args:
            v: Input tensor [batch_size, seq_len, n_heads, head_dim]

        Returns:
            Phase-activated tensor with same shape
        """
        # Aplicar projeção linear para obter ângulos de fase
        phi = self.phi_proj(v)  # [batch_size, seq_len, n_heads, head_dim]

        # Criar números complexos: exp(i ⋅ phi)
        phase_factors = torch.exp(1j * phi)

        # Aplicar multiplicação elemento-a-elemento: v ⊙ exp(i ⋅ phi)
        activated = v * phase_factors

        return activated

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape

        # Projeção Latente Compartilhada
        Z = self.z_proj(query)  # [batch_size, seq_len, d_latent]
        Z = self.z_norm(Z)      # LayerNorm após projeção

        # Projeções Derivadas
        Q = self.q_proj(Z)  # [batch_size, seq_len, n_heads * head_dim]
        R = self.r_proj(Z)  # [batch_size, seq_len, n_heads * head_dim]
        H = self.h_proj(Z)  # [batch_size, seq_len, n_heads * head_dim]

        # Reshape para multi-head: [batch_size, seq_len, n_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        R = R.view(batch_size, seq_len, self.n_heads, self.head_dim)
        H = H.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Ativação de Fase Ψ
        Q_prime = self._phasor_activation(Q)  # [batch_size, seq_len, n_heads, head_dim]
        R_prime = self._phasor_activation(R)  # [batch_size, seq_len, n_heads, head_dim]

        # Calcular scores: Re(Q' @ R'*) - atenção T×T entre tokens
        # Q_prime, R_prime: [batch_size, seq_len, n_heads, head_dim]
        scores = torch.einsum('bqhd,bkhd->bhqk', Q_prime, R_prime.conj())  # [batch_size, n_heads, seq_len, seq_len]
        scores = torch.real(scores)  # Tomar parte real

        # Remover softmax - usar normalização física ou identidade
        # attention_weights = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)  # REMOVIDO
        # Usar normalização L2 ou identidade para manter funcionamento básico
        attention_weights = torch.nn.functional.normalize(scores, p=2, dim=-1)  # Normalização L2

        # Aplicar atenção: attention_weights @ H
        attention_output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, H)  # [batch_size, seq_len, n_heads, head_dim]

        # Combinar cabeças: [batch_size, seq_len, n_heads * head_dim]
        attention_output = attention_output.reshape(batch_size, seq_len, -1)

        # Ensure output has correct dimensions (d_model)
        if attention_output.size(-1) != self.d_model:
            # If dimensions don't match, use a temporary linear layer to adjust
            if not hasattr(self, '_dim_adjuster'):
                self._dim_adjuster = nn.Linear(attention_output.size(-1), self.d_model).to(attention_output.device)
            attention_output = self._dim_adjuster(attention_output)

        return self.out_proj(attention_output)



class PsiQRHFeedForward(nn.Module):
    """Feed-forward network with ΨQRH spectral processing"""

    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()

        # Quaternion-based linear layers
        self.linear1 = QuaternionLinear(d_model, dim_feedforward)
        self.linear2 = QuaternionLinear(dim_feedforward, d_model)

        # Spectral activation
        self.activation = SpectralActivation()

        # Adaptive dropout
        self.dropout = AdaptiveSpectralDropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First linear transformation
        x = self.linear1(x)

        # Spectral activation
        x = self.activation(x)

        # Adaptive dropout
        x = self.dropout(x)

        # Second linear transformation
        x = self.linear2(x)

        return x


class PsiQRHTransformerBlock(nn.Module):
    """Complete ΨQRH transformer block with configurable components"""

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None, preset: str = 'standard'):
        super().__init__()

        # Load configuration
        if config is None:
            config = load_transformer_config(config_path, preset)

        self.config = config
        self.d_model = config['model']['d_model']
        self.n_heads = config['model']['n_heads']
        self.dim_feedforward = config['model']['dim_feedforward']
        self.device = config['performance']['device']

        # Component flags
        self.use_kuramoto = config['components']['use_kuramoto']
        self.use_working_memory = config['components']['use_working_memory']
        self.use_phase_sync = config['components']['use_phase_sync']
        self.use_fractal_analysis = config['components']['use_fractal_analysis']
        self.use_harmonic_coupling = config['harmonic_coupling']['enabled']

        # ΨQRH attention - input is d_model * quaternion_multiplier
        input_dim = self.d_model * config['model']['quaternion_multiplier']
        self.self_attention = PsiQRHAttention(input_dim, self.n_heads)
        self.attention_norm = nn.LayerNorm(input_dim)

        # Kuramoto Spectral Layer (optional)
        if self.use_kuramoto:
            from ..core.kuramoto_spectral_neurons import create_kuramoto_layer
            kuramoto_cfg = config['kuramoto']
            self.kuramoto_layer = create_kuramoto_layer(
                device=kuramoto_cfg.get('device', self.device)
            )

            # Phase Synchronization Module (optional)
            if self.use_phase_sync:
                from ..core.phase_synchronization import create_phase_sync_module
                phase_cfg = config['phase_synchronization']
                self.phase_sync = create_phase_sync_module(
                    grid_size=phase_cfg['grid_size'],
                    embed_dim=self.d_model * config['model']['quaternion_multiplier'],
                    device=self.device
                )

        # Conscious Working Memory (optional)
        if self.use_working_memory:
            from ..core.conscious_working_memory import create_conscious_working_memory
            self.working_memory = create_conscious_working_memory(
                config_path=config['working_memory'].get('config_path'),
                device=self.device
            )

        # Harmonic Evolution Layer - replaces traditional feed-forward
        from ..core.harmonic_evolution_layer import HarmonicEvolutionLayer
        self.feed_forward = HarmonicEvolutionLayer(
            self.d_model,
            rotation_dim=4,
            use_learnable_kernel=True,
            max_seq_len=config['model'].get('max_seq_length', 1024)
        )
        self.ffn_norm = QuaternionLayerNorm(self.d_model)

        # Fractal analysis (optional)
        if self.use_fractal_analysis:
            self.fractal_analyzer = RealTimeFractalAnalyzer()

        # Layer scaling from config
        full_dim = self.d_model * config['model']['quaternion_multiplier']
        layer_scales = config['layer_scaling']['init_values']
        self.layer_scale_attention = nn.Parameter(torch.ones(full_dim) * layer_scales['attention'])
        self.layer_scale_kuramoto = nn.Parameter(torch.ones(full_dim) * layer_scales['kuramoto']) if self.use_kuramoto else None
        self.layer_scale_memory = nn.Parameter(torch.ones(full_dim) * layer_scales['memory']) if self.use_working_memory else None
        self.layer_scale_ffn = nn.Parameter(torch.ones(full_dim) * layer_scales['feedforward'])

        # Energy conservation validator
        if config['energy_conservation']['enabled']:
            from ..core.phase_synchronization import EnergyConservationValidator
            self.energy_validator = EnergyConservationValidator(
                tolerance=config['energy_conservation']['tolerance']
            )

        # Harmonic Layer Coupling (optional)
        if self.use_harmonic_coupling:
            from ..core.harmonic_layer_coupling import create_harmonic_coupling
            coupling_cfg = config['harmonic_coupling']
            # Determinar número de camadas ativas
            n_active_layers = 2  # attention + ffn sempre presentes
            if self.use_kuramoto:
                n_active_layers += 1
            if self.use_working_memory:
                n_active_layers += 1

            self.harmonic_coupling = create_harmonic_coupling(
                embed_dim=self.d_model * config['model']['quaternion_multiplier'],
                n_layers=n_active_layers,
                config_path=coupling_cfg.get('config_path'),
                preset=coupling_cfg.get('preset', 'standard'),
                device=self.device
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Import energy normalization from BKP implementation
        from ..core.utils import energy_normalize

        metrics = {}
        layer_outputs_for_coupling = {}

        # Self-attention with residual and energy conservation
        residual = x
        x = self.attention_norm(x)
        attention_out = self.self_attention(x, x, x)

        # Apply energy normalization to preserve energy conservation
        attention_out = energy_normalize(x, attention_out)
        x = residual + self.layer_scale_attention * attention_out

        # Apply energy conservation after layer scale
        x = energy_normalize(residual, x)

        # Salvar output da atenção para acoplamento
        if self.use_harmonic_coupling:
            layer_outputs_for_coupling['attention'] = x.clone()

        # Real-time fractal analysis (optional)
        if self.use_fractal_analysis:
            fractal_metrics = self.fractal_analyzer.analyze(x)
            self._adapt_parameters(fractal_metrics)
        else:
            fractal_metrics = {'entropy': 0.5, 'dimension': 2.0, 'complexity': 0.5}

        # Prepare consciousness state for downstream components
        consciousness_state = {
            'entropy': fractal_metrics.get('entropy', 0.5),
            'fractal_dimension': fractal_metrics.get('dimension', 2.0).item() if torch.is_tensor(fractal_metrics.get('dimension')) else fractal_metrics.get('dimension', 2.0),
            'fci': fractal_metrics.get('complexity', 0.5)
        }

        # Kuramoto Spectral Neurons (after attention)
        if self.use_kuramoto:
            residual = x
            kuramoto_out, kuramoto_metrics = self.kuramoto_layer(x, return_metrics=True)

            # Phase Synchronization (if enabled)
            if self.use_phase_sync:
                batch_size, seq_len, embed_dim_full = kuramoto_out.shape
                # Extrair fases dos neurônios
                phases = torch.atan2(
                    torch.sin(kuramoto_out.mean(dim=1)),
                    torch.cos(kuramoto_out.mean(dim=1))
                )  # [batch, embed_dim]

                # Expandir para n_neurons
                n_neurons = self.phase_sync.n_neurons
                if phases.shape[1] < n_neurons:
                    phases = phases.repeat(1, (n_neurons // phases.shape[1]) + 1)[:, :n_neurons]

                # Aplicar sincronização de fase
                phase_cfg = self.config['phase_synchronization']
                kuramoto_out, phase_metrics = self.phase_sync(
                    kuramoto_out,
                    phases,
                    extract_rhythms=phase_cfg.get('extract_rhythms', False)
                )
                metrics['phase_sync'] = phase_metrics

            # Normalize kuramoto output to match residual energy
            kuramoto_out = energy_normalize(residual, kuramoto_out)

            # Apply layer scale
            x = residual + self.layer_scale_kuramoto * kuramoto_out

            # Apply energy conservation after layer scale
            x = energy_normalize(residual, x)

            # Validate energy conservation
            if hasattr(self, 'energy_validator'):
                is_valid, ratio, energy_report = self.energy_validator.validate(residual, x)
                if self.config['energy_conservation']['log_violations'] and not is_valid:
                    print(f"⚠️  Energia não conservada após Kuramoto: {ratio:.4f}")
                metrics['kuramoto_energy'] = energy_report

            if kuramoto_metrics:
                metrics['kuramoto'] = kuramoto_metrics

            # Salvar output do Kuramoto para acoplamento
            if self.use_harmonic_coupling:
                layer_outputs_for_coupling['kuramoto'] = x.clone()

        # Conscious Working Memory (after Kuramoto)
        if self.use_working_memory:
            residual = x
            memory_out, memory_state = self.working_memory(
                x,
                consciousness_state,
                return_memory_state=True
            )
            # Normalize memory output to match residual energy
            memory_out = energy_normalize(residual, memory_out)

            # Apply layer scale
            x = residual + self.layer_scale_memory * memory_out

            # Apply energy conservation after layer scale
            x = energy_normalize(residual, x)

            # Validate energy conservation
            if hasattr(self, 'energy_validator'):
                is_valid, ratio, energy_report = self.energy_validator.validate(residual, x)
                if self.config['energy_conservation']['log_violations'] and not is_valid:
                    print(f"⚠️  Energia não conservada após Memory: {ratio:.4f}")
                metrics['memory_energy'] = energy_report

            if memory_state is not None:
                metrics['memory'] = {
                    'memory_norm': torch.norm(memory_state).item(),
                    'memory_mean': memory_state.mean().item()
                }

            # Salvar output da memória para acoplamento
            if self.use_harmonic_coupling:
                layer_outputs_for_coupling['memory'] = x.clone()

        # Feed-forward with residual and energy conservation
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.feed_forward(x)

        # Apply energy normalization to preserve energy conservation
        ffn_out = energy_normalize(x, ffn_out)
        x = residual + self.layer_scale_ffn * ffn_out

        # Apply energy conservation after layer scale
        x = energy_normalize(residual, x)

        # Salvar output final para acoplamento
        if self.use_harmonic_coupling:
            layer_outputs_for_coupling['feedforward'] = x.clone()

        # Aplicar Acoplamento Harmônico entre todas as camadas
        if self.use_harmonic_coupling and len(layer_outputs_for_coupling) > 1:
            layer_names = list(layer_outputs_for_coupling.keys())
            harmonized_output, coupling_metrics = self.harmonic_coupling(
                layer_outputs_for_coupling,
                layer_names
            )

            # Substituir output final pelo harmonizado
            x = harmonized_output

            # Adicionar métricas de acoplamento
            metrics['harmonic_coupling'] = coupling_metrics

            # Log de sincronização
            if coupling_metrics['is_synchronized']:
                print(f"✨ Camadas harmonizadas: r={coupling_metrics['global_synchronization']:.4f}")
            else:
                print(f"⚠️  Camadas dessincronizadas: r={coupling_metrics['global_synchronization']:.4f}")

        metrics['fractal'] = fractal_metrics

        return x, metrics

    def _adapt_parameters(self, fractal_metrics: Dict):
        """Adapt parameters based on fractal analysis"""
        # Update spectral filter parameters
        new_alpha = self._map_fractal_to_alpha(fractal_metrics['dimension'])
        self.self_attention.spectral_filter.update_alpha(new_alpha)

    def _map_fractal_to_alpha(self, dimension: torch.Tensor) -> torch.Tensor:
        """Map fractal dimension to spectral filter alpha parameter"""
        # Higher dimension = more complex signal = stronger filtering
        return torch.sigmoid(dimension - 1.5)  # Map to [0, 1] range


class PsiQRHTransformer(nn.Module):
    """Complete ΨQRH-based transformer architecture"""

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dim_feedforward: int = 1024,  # Reduced from 2048 for memory optimization
                 max_seq_length: int = 1024,
                 fractal_analysis_freq: int = 1000,
                 quaternion_multiplier: int = 4):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.quaternion_multiplier = quaternion_multiplier

        # ΨQRH-based components
        self.token_embedding = QuaternionTokenEmbedding(vocab_size, d_model)
        self.positional_encoding = SpectralPositionalEncoding(d_model * quaternion_multiplier, max_seq_length)

        # ΨQRH transformer blocks - optimized for parameter efficiency
        self.layers = nn.ModuleList([
            PsiQRHTransformerBlock(
                config={'model': {'d_model': d_model, 'n_heads': n_heads, 'dim_feedforward': dim_feedforward // 2, 'quaternion_multiplier': quaternion_multiplier, 'max_seq_length': max_seq_length},
                        'performance': {'device': 'cpu'},
                        'components': {'use_kuramoto': False, 'use_working_memory': False, 'use_phase_sync': False, 'use_fractal_analysis': False},
                        'harmonic_coupling': {'enabled': False},
                        'layer_scaling': {'init_values': {'attention': 1.0, 'kuramoto': 1.0, 'memory': 1.0, 'feedforward': 1.0}},
                        'energy_conservation': {'enabled': False, 'tolerance': 0.01, 'log_violations': False}}
            ) for _ in range(n_layers)
        ])

        # Adaptive fractal controller (placeholder - would be implemented in optimization module)
        # self.fractal_controller = AdaptiveFractalController(
        #     window_size=fractal_analysis_freq
        # )

        # Tied embeddings with projection back to real space
        # Instead of output_projection, we use the embedding matrix transposed
        self.to_real_space = nn.Linear(d_model * quaternion_multiplier, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Import energy normalization from BKP implementation
        from ..core.utils import energy_normalize

        # Embed tokens as quaternions
        x = self.token_embedding(x)
        x_embedded = x  # Save embedded representation for energy reference

        # Apply spectral positional encoding
        x = self.positional_encoding(x)

        # Process through ΨQRH layers with energy conservation
        for i, layer in enumerate(self.layers):
            x_before_layer = x
            x, metrics = layer(x)

            # Apply energy conservation after each layer (from BKP)
            x = energy_normalize(x_before_layer, x)

            # Adaptive fractal analysis and parameter adjustment (placeholder)
            # if i % self.fractal_analysis_freq == 0:
            #     self.fractal_controller.update_parameters(x, layer)

        # Project back from quaternion space to real space
        final_real_state = self.to_real_space(x)

        # Use tied embeddings: calculate logits using embedding matrix transposed
        logits = final_real_state @ self.token_embedding.embedding.weight.T

        # Final energy normalization to maintain overall energy conservation
        logits = energy_normalize(x_embedded, logits)

        return logits

    def get_quaternion_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        RIGOROUS: Get quaternion embedding following doe.md 2.9.1

        Uses internal MLP to generate ψ₀ = Re(MLP(x)), ψ₁ = Im(MLP(x))

        Args:
            token_ids: Token IDs [B, T]

        Returns:
            Quaternion state [B, T, d_model, 4]
        """
        # Get embedding via rigorous QuaternionMLP
        embedded = self.token_embedding.embedding(token_ids)  # [B, T, d_model]

        # Pass through MLP to get complex output
        complex_out = self.token_embedding.quaternion_mlp(embedded)  # [B, T, d_model] complex

        # Extract quaternion components
        psi_0 = complex_out.real
        psi_1 = complex_out.imag

        # Generate ψ₂, ψ₃ via rotational transformation
        rotation_scales = self.token_embedding.rotation_scales
        rotation_angles = self.token_embedding.rotation_angles

        psi_2 = psi_0 * rotation_scales[:, 0] + psi_1 * rotation_scales[:, 1]
        psi_3 = psi_1 * rotation_scales[:, 0] - psi_0 * rotation_scales[:, 1]

        psi_2 = psi_2 * torch.cos(rotation_angles[:, 0])
        psi_3 = psi_3 * torch.sin(rotation_angles[:, 1])

        # Stack as quaternion [B, T, d_model, 4]
        quaternion = torch.stack([psi_0, psi_1, psi_2, psi_3], dim=-1)

        return quaternion

    def get_model_info(self) -> Dict:
        """Get information about the model architecture"""
        # Get dim_feedforward safely
        dim_feedforward = 0
        if self.layers:
            first_layer = self.layers[0]
            # Check if it's HarmonicEvolutionLayer (from complete implementation)
            if hasattr(first_layer, 'evolution'):
                dim_feedforward = self.d_model * 4  # Default expansion
            # Check if it's standard feed_forward
            elif hasattr(first_layer, 'feed_forward'):
                if hasattr(first_layer.feed_forward, 'linear1'):
                    dim_feedforward = first_layer.feed_forward.linear1.out_features
                else:
                    dim_feedforward = self.d_model * 4  # Default expansion
            else:
                dim_feedforward = self.d_model * 4  # Default expansion

        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.layers[0].self_attention.n_heads if self.layers and hasattr(self.layers[0], 'self_attention') else 8,
            "dim_feedforward": dim_feedforward,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "architecture": "ΨQRH Transformer"
        }