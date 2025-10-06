#!/usr/bin/env python3
"""
Memória de Trabalho Consciente - Conscious Working Memory
==========================================================

Implementa memória de trabalho baseada em:
1. Equação de Padilha: f(λ,t) = I₀·sin(ωt+αλ)·e^(i(ωt-kλ+βλ²))
2. Mapeamentos fractais: α(D) = α₀(1+λ·(D-D_e)/D_e), β = (2n+1)-2D
3. Modulação caótica (mapa logístico)
4. Neurotransmissores sintéticos
5. Integração com Kuramoto

Mathematical Framework:
-----------------------
Padilha Wave Equation:
    f(λ,t) = I₀·sin(ωt + α(D)·λ)·exp(i(ωt - kλ + β(D)·λ²))

Fractal Mappings:
    α(D) = α₀·(1 + λ·(D - D_euclidean)/D_euclidean)
    β(D) = (2n + 1) - 2D

Logistic Map:
    x_{n+1} = r·x_n·(1 - x_n)

Memory Evolution:
    M(t+dt) = decay·M(t) + wave_component + chaotic_modulation + neurotransmitter_influence

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

from .quaternion_operations import (
    quaternion_multiply,
    quaternion_normalize,
    QuaternionLinear
)


def load_working_memory_config(config_path: Optional[str] = None) -> Dict:
    """Carrega configuração da memória de trabalho"""
    if config_path is None:
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "configs" / "working_memory_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config['working_memory']


class PadilhaWaveEquation(nn.Module):
    """
    Implementa a equação de onda de Padilha com mapeamentos fractais:
    f(λ,t) = I₀·sin(ωt + α(D)·λ)·e^(i(ωt - kλ + β(D)·λ²))
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Parâmetros da equação
        wave_cfg = config['padilha_wave']
        self.I0 = wave_cfg['I0']
        self.omega = wave_cfg['omega']
        self.k = wave_cfg['k']
        self.alpha_base = wave_cfg['alpha_base']
        self.lambda_coupling = wave_cfg['lambda_coupling']
        self.D_euclidean = wave_cfg['euclidean_dim']
        self.chirp_order = wave_cfg['chirp_order']

        # Limites
        self.alpha_min = wave_cfg['alpha_min']
        self.alpha_max = wave_cfg['alpha_max']

    def compute_alpha(self, fractal_dimension: float) -> float:
        """
        Computa α(D) = α₀(1 + λ·(D - D_euclid)/D_euclid)
        Bounded to [alpha_min, alpha_max]
        """
        D = fractal_dimension
        complexity_ratio = (D - self.D_euclidean) / self.D_euclidean
        alpha = self.alpha_base * (1.0 + self.lambda_coupling * complexity_ratio)

        # Clip to bounds
        alpha = np.clip(alpha, self.alpha_min, self.alpha_max)
        return alpha

    def compute_beta(self, fractal_dimension: float) -> float:
        """
        Computa β(D) = (2n + 1) - 2D
        Coeficiente de chirp quadrático
        """
        n = self.chirp_order
        beta = (2 * n + 1) - 2 * fractal_dimension
        return beta

    def apply_wave_component(
        self,
        x: torch.Tensor,
        entropy: float,
        fractal_dimension: float,
        t: float = 0.0
    ) -> torch.Tensor:
        """
        Aplica componente de onda harmônica ao tensor x.

        Args:
            x: Tensor de entrada [batch, seq_len, embed_dim * 4]
            entropy: Entropia do estado de consciência
            fractal_dimension: Dimensão fractal D
            t: Tempo atual

        Returns:
            Tensor modulado pela equação de Padilha
        """
        batch_size, seq_len, embed_dim = x.shape

        # Computar α(D) e β(D)
        alpha = self.compute_alpha(fractal_dimension)
        beta = self.compute_beta(fractal_dimension)

        # Posições espaciais λ
        lambda_positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        lambda_positions = lambda_positions / seq_len  # Normalizar [0, 1]

        # Componente de amplitude (parte real)
        amplitude_term = self.I0 * torch.sin(
            self.omega * t + alpha * lambda_positions
        )  # [seq_len]

        # Componente de fase complexa
        phase_term = self.omega * t - self.k * lambda_positions + beta * lambda_positions ** 2
        complex_modulation = torch.exp(1j * phase_term)

        # Combinar amplitude e fase
        wave_function = amplitude_term * complex_modulation.real  # Usar parte real

        # Expandir para broadcast
        wave_function = wave_function.view(1, seq_len, 1)  # [1, seq_len, 1]

        # Modular entrada
        modulated = x * (1.0 + 0.1 * wave_function)  # Modulação suave

        return modulated


class LogisticMapChaoticUpdater(nn.Module):
    """
    Modulação caótica usando mapa logístico:
    x_{n+1} = r·x_n·(1 - x_n)
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        chaos_cfg = config['chaotic_modulation']
        self.r = chaos_cfg['r_parameter']
        self.strength = chaos_cfg['modulation_strength']
        self.num_iterations = chaos_cfg['num_iterations']

    def modulate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica modulação caótica via mapa logístico.

        Args:
            x: Tensor [batch, seq_len, embed_dim * 4]

        Returns:
            Tensor modulado caoticamente
        """
        batch_size, seq_len, embed_dim = x.shape

        # Inicializar estados caóticos
        # Usar norma do tensor para seed
        x_norm = torch.norm(x, dim=-1)  # [batch, seq_len]
        chaotic_state = torch.sigmoid(x_norm)  # Map to [0, 1]

        # Iterar mapa logístico
        for _ in range(self.num_iterations):
            chaotic_state = self.r * chaotic_state * (1 - chaotic_state)

        # Expandir para modulação
        chaotic_mod = chaotic_state.unsqueeze(-1)  # [batch, seq_len, 1]

        # Aplicar modulação
        modulated = x * (1.0 + self.strength * (chaotic_mod - 0.5))

        return modulated


class SyntheticNeurotransmitterSystem(nn.Module):
    """Sistema de neurotransmissores sintéticos para regulação de memória"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        nt_cfg = config['neurotransmitters']

        # Dopamina (recompensa/motivação)
        self.dopamine_baseline = nt_cfg['dopamine']['baseline']
        self.dopamine_sensitivity = nt_cfg['dopamine']['sensitivity']
        self.dopamine_decay = nt_cfg['dopamine']['decay_rate']

        # Serotonina (estabilidade/humor)
        self.serotonin_baseline = nt_cfg['serotonin']['baseline']
        self.serotonin_sensitivity = nt_cfg['serotonin']['sensitivity']
        self.serotonin_decay = nt_cfg['serotonin']['decay_rate']

        # Norepinefrina (atenção/alerta)
        self.norepi_baseline = nt_cfg['norepinephrine']['baseline']
        self.norepi_sensitivity = nt_cfg['norepinephrine']['sensitivity']
        self.norepi_decay = nt_cfg['norepinephrine']['decay_rate']

        # Estados internos
        self.register_buffer('dopamine_level', torch.tensor(self.dopamine_baseline))
        self.register_buffer('serotonin_level', torch.tensor(self.serotonin_baseline))
        self.register_buffer('norepi_level', torch.tensor(self.norepi_baseline))

    def compute_dopamine(self, fci: float) -> torch.Tensor:
        """Computa sinal de dopamina baseado em FCI (recompensa por coerência)"""
        reward = torch.sigmoid(torch.tensor(fci * self.dopamine_sensitivity))
        self.dopamine_level = (1 - self.dopamine_decay) * self.dopamine_level + self.dopamine_decay * reward
        return self.dopamine_level

    def compute_serotonin(self, stability: float) -> torch.Tensor:
        """Computa sinal de serotonina baseado em estabilidade"""
        stab = torch.sigmoid(torch.tensor(stability * self.serotonin_sensitivity))
        self.serotonin_level = (1 - self.serotonin_decay) * self.serotonin_level + self.serotonin_decay * stab
        return self.serotonin_level

    def compute_norepinephrine(self, attention_demand: float) -> torch.Tensor:
        """Computa sinal de norepinefrina baseado em demanda de atenção"""
        alert = torch.sigmoid(torch.tensor(attention_demand * self.norepi_sensitivity))
        self.norepi_level = (1 - self.norepi_decay) * self.norepi_level + self.norepi_decay * alert
        return self.norepi_level

    def modulate_memory(
        self,
        memory: torch.Tensor,
        fci: float,
        stability: float,
        attention_demand: float
    ) -> torch.Tensor:
        """
        Modula memória usando neurotransmissores.

        Args:
            memory: Estado de memória [batch, memory_size, embed_dim * 4]
            fci: FCI atual
            stability: Estabilidade do sistema
            attention_demand: Demanda de atenção

        Returns:
            Memória modulada
        """
        # Computar níveis
        dopamine = self.compute_dopamine(fci)
        serotonin = self.compute_serotonin(stability)
        norepi = self.compute_norepinephrine(attention_demand)

        # Combinação weighted
        modulation = (
            0.4 * dopamine +      # Reforço de memórias relevantes
            0.3 * serotonin +     # Estabilização geral
            0.3 * norepi          # Aumento de saliência
        )

        # Aplicar modulação
        modulated_memory = memory * modulation

        return modulated_memory


class QuaternionicAttentionRetriever(nn.Module):
    """Sistema de atenção quaterniônica para recuperação seletiva de memórias"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        embed_dim = config['memory']['embed_dim']
        attn_cfg = config['attention_retrieval']

        self.num_heads = attn_cfg['num_heads']
        self.head_dim = (embed_dim * 4) // self.num_heads

        # Parâmetros para filtro espectral
        self.alpha = attn_cfg.get('alpha', 1.5)
        self.epsilon = attn_cfg.get('epsilon', 1e-10)

        # Projeções quaterniônicas otimizadas
        # Usar projeções lineares regulares com reshape para eficiência
        self.q_proj = nn.Linear(embed_dim * 4, embed_dim * 4)
        self.k_proj = nn.Linear(embed_dim * 4, embed_dim * 4)
        self.v_proj = nn.Linear(embed_dim * 4, embed_dim * 4)

        self.dropout = nn.Dropout(attn_cfg['dropout'])

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Recupera informações relevantes da memória via atenção espectral ΨQRH.

        Implementa atenção espectral baseada em doe.md 2.9.2:
        - Mapeamento para quaternions Ψ(Q), Ψ(K), Ψ(V)
        - Transformada de Fourier
        - Filtragem espectral F(k) = exp(i α · arctan(ln(|k| + ε)))
        - Interação via produto de Hamilton
        - Transformada inversa

        Args:
            query: Query tensor [batch, seq_len, embed_dim * 4]
            memory: Memory tensor [batch, memory_size, embed_dim * 4]

        Returns:
            Retrieved information [batch, seq_len, embed_dim * 4]
        """
        batch_size = query.shape[0]

        # Projetar para estados quaterniônicos Ψ(Q), Ψ(K), Ψ(V)
        Q = self.q_proj(query)  # Ψ(Q) [batch, seq_len, embed_dim * 4]
        K = self.k_proj(memory)  # Ψ(K) [batch, memory_size, embed_dim * 4]
        V = self.v_proj(memory)  # Ψ(V) [batch, memory_size, embed_dim * 4]

        # Aplicar FFT aos estados quaterniônicos
        Q_fft = torch.fft.fft(Q, dim=-2)  # FFT sobre dimensão de sequência
        K_fft = torch.fft.fft(K, dim=-2)
        V_fft = torch.fft.fft(V, dim=-2)

        # Criar filtro espectral F(k) = exp(i α · arctan(ln(|k| + ε)))
        seq_len = Q.shape[-2]
        k = torch.arange(seq_len, device=Q.device, dtype=torch.float32)
        filter_kernel = torch.exp(1j * self.alpha * torch.arctan(torch.log(k + self.epsilon)))

        # Aplicar filtro espectral
        Q_filtered = Q_fft * filter_kernel.unsqueeze(-1).unsqueeze(0)
        K_filtered = K_fft * filter_kernel.unsqueeze(-1).unsqueeze(0)
        V_filtered = V_fft * filter_kernel.unsqueeze(-1).unsqueeze(0)

        # Interação via produto de Hamilton: F[Ψ(Q)] ⊗ F[Ψ(K)] ⊗ F[Ψ(V)]
        # Primeiro Q ⊗ K
        QK_interaction = quaternion_multiply(Q_filtered, K_filtered)
        # Depois ⊗ V
        QKV_interaction = quaternion_multiply(QK_interaction, V_filtered)

        # Aplicar IFFT para retornar ao domínio tempo/espaço
        attended_spectral = torch.fft.ifft(QKV_interaction, dim=-2)

        # Retornar parte real e garantir formato correto
        attended = attended_spectral.real

        # Aplicar dropout se necessário
        attended = self.dropout(attended)

        return attended


class ConsciousWorkingMemory(nn.Module):
    """
    Memória de Trabalho Consciente completa integrand:
    - Equação de Padilha
    - Mapa logístico
    - Neurotransmissores
    - Atenção quaterniônica
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()

        # Carregar configuração
        self.config = load_working_memory_config(config_path)
        self.device = self.config['performance']['device']

        # Dimensões
        self.memory_size = self.config['memory']['size']
        self.embed_dim = self.config['memory']['embed_dim']
        self.full_dim = self.embed_dim * 4  # quaternion

        # Campo de memória persistente
        self.register_buffer(
            'memory_state',
            torch.zeros(1, self.memory_size, self.full_dim, device=self.device)
        )

        # Tempo interno
        self.register_buffer('internal_time', torch.tensor(0.0))

        # Componentes
        self.wave_equation = PadilhaWaveEquation(self.config)
        self.chaotic_updater = LogisticMapChaoticUpdater(self.config)
        self.neurotransmitters = SyntheticNeurotransmitterSystem(self.config)
        self.attention_retriever = QuaternionicAttentionRetriever(self.config)

        # Projeções
        self.input_projection = nn.Linear(self.full_dim, self.full_dim)
        self.output_projection = nn.Linear(self.full_dim, self.full_dim)

        # Decay
        self.persistence_decay = self.config['memory']['persistence_decay']

    def update(
        self,
        current_input: torch.Tensor,
        consciousness_state: Dict[str, float]
    ) -> torch.Tensor:
        """
        Atualiza o estado de memória com base na entrada atual e estado consciente.

        Args:
            current_input: Tensor de entrada [batch, seq_len, embed_dim * 4]
            consciousness_state: Dict com 'entropy', 'fractal_dimension', 'fci'

        Returns:
            Estado de memória atualizado
        """
        batch_size, seq_len, _ = current_input.shape

        # 1. Projetar entrada
        projected_input = self.input_projection(current_input)

        # 2. Modulação caótica (Mapa Logístico)
        if self.config['chaotic_modulation']['enable']:
            chaotic_input = self.chaotic_updater.modulate(projected_input)
        else:
            chaotic_input = projected_input

        # 3. Componente de onda harmônica (Equação de Padilha)
        if self.config['wave_dynamics']['use_wave_equation']:
            wave_modulated = self.wave_equation.apply_wave_component(
                chaotic_input,
                consciousness_state.get('entropy', 0.5),
                consciousness_state.get('fractal_dimension', 2.0),
                t=self.internal_time.item()
            )
        else:
            wave_modulated = chaotic_input

        # 4. Integrar com memória existente
        # Expandir wave_modulated para memory_size se necessário
        if wave_modulated.shape[1] < self.memory_size:
            # Pad ou repeat
            repeat_factor = (self.memory_size + seq_len - 1) // seq_len
            wave_expanded = wave_modulated.repeat(1, repeat_factor, 1)[:, :self.memory_size, :]
        else:
            wave_expanded = wave_modulated[:, :self.memory_size, :]

        # Broadcast batch dimension
        if self.memory_state.shape[0] != batch_size:
            self.memory_state = self.memory_state.repeat(batch_size, 1, 1)

        # 5. Evolução temporal com decay
        updated_memory = (
            self.persistence_decay * self.memory_state +
            (1 - self.persistence_decay) * wave_expanded
        )

        # 6. Modulação por neurotransmissores
        if self.config['neurotransmitters']['enable']:
            updated_memory = self.neurotransmitters.modulate_memory(
                updated_memory,
                fci=consciousness_state.get('fci', 0.5),
                stability=1.0 - consciousness_state.get('entropy', 0.5),
                attention_demand=0.5  # Placeholder
            )

        # Atualizar estado
        self.memory_state = updated_memory

        # Incrementar tempo
        self.internal_time += 1.0

        return self.memory_state

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Recupera informações relevantes do campo de memória.

        Args:
            query: Query tensor [batch, seq_len, embed_dim * 4]

        Returns:
            Informação recuperada [batch, seq_len, embed_dim * 4]
        """
        batch_size = query.shape[0]

        # Ensure memory has correct batch size
        if self.memory_state.shape[0] != batch_size:
            memory = self.memory_state.repeat(batch_size, 1, 1)
        else:
            memory = self.memory_state

        # Recuperação via atenção quaterniônica
        retrieved = self.attention_retriever(query, memory)

        # Projetar saída
        output = self.output_projection(retrieved)

        return output

    def forward(
        self,
        x: torch.Tensor,
        consciousness_state: Dict[str, float],
        return_memory_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass completo: atualizar + recuperar.

        Args:
            x: Input tensor [batch, seq_len, embed_dim * 4]
            consciousness_state: Estado de consciência
            return_memory_state: Se True, retorna estado de memória

        Returns:
            output: Informação recuperada
            memory_state: Estado de memória (opcional)
        """
        # Atualizar memória
        updated_memory = self.update(x, consciousness_state)

        # Recuperar informação
        retrieved = self.retrieve(x)

        # Combinar com input (conexão residual weighted)
        weight = self.config.get('transformer_integration', {}).get('memory_influence_weight', 0.3)
        output = (1 - weight) * x + weight * retrieved

        if return_memory_state:
            return output, updated_memory
        else:
            return output, None


# ============================================================================
# FUNÇÃO DE CRIAÇÃO
# ============================================================================

def create_conscious_working_memory(
    config_path: Optional[str] = None,
    device: str = "cpu"
) -> ConsciousWorkingMemory:
    """Factory function para criar memória de trabalho consciente"""
    memory = ConsciousWorkingMemory(config_path)
    memory.to(device)
    return memory
