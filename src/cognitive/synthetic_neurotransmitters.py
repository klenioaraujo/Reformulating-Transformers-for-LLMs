#!/usr/bin/env python3
"""
SYNTHETIC NEUROTRANSMITTER ALIGNMENT LAYER

Camada inspirada em neurotransmissores para alinhamento de componentes QRH:
1. Dopamina Sintética - Recompensa e otimização de performance
2. Serotonina Sintética - Estabilização e harmonia entre módulos
3. Acetilcolina Sintética - Atenção e foco computacional
4. GABA Sintético - Inibição de ruído e estabilização
5. Glutamato Sintético - Excitação e amplificação de sinal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from src.core.quaternion_operations import quaternion_multiply


def smooth_dimension_handling(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Smooth dimensional handling for neurotransmitter inputs
    Returns: (normalized_tensor, (batch_size, seq_len, embed_dim))
    """
    if len(x.shape) == 2:
        batch_size, embed_dim = x.shape
        seq_len = 1
        x = x.unsqueeze(1)  # Add sequence dimension
    elif len(x.shape) == 3:
        batch_size, seq_len, embed_dim = x.shape
    else:
        # Smooth fallback for unexpected dimensions
        original_shape = x.shape
        x = x.view(-1, 1, x.shape[-1])
        batch_size, seq_len, embed_dim = x.shape

    return x, (batch_size, seq_len, embed_dim)


@dataclass
class NeurotransmitterConfig:
    """Configuração dos neurotransmissores sintéticos"""
    embed_dim: int = 32

    # Dopamina - Recompensa e Performance
    dopamine_strength: float = 0.8
    dopamine_learning_rate: float = 0.01

    # Serotonina - Estabilização
    serotonin_stability: float = 0.6
    serotonin_harmony_factor: float = 1.2

    # Acetilcolina - Atenção
    acetylcholine_focus: float = 0.9
    acetylcholine_selectivity: float = 2.0

    # GABA - Inibição
    gaba_inhibition: float = 0.4
    gaba_noise_threshold: float = 0.3

    # Glutamato - Excitação
    glutamate_excitation: float = 1.1
    glutamate_amplification: float = 1.5


class SyntheticDopamine(nn.Module):
    """
    Dopamina Sintética - Sistema de recompensa para otimização
    Reforça padrões que melhoram performance do QRH
    """

    def __init__(self, config: NeurotransmitterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4  # QRH dimension

        # Receptor de dopamina
        self.dopamine_receptor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Sistema de recompensa adaptativo
        self.reward_memory = nn.Parameter(torch.zeros(1))
        self.performance_tracker = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, performance_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply dopaminergic modulation based on performance with smooth dimensional handling
        """
        # Smooth dimensional handling
        if len(x.shape) == 2:
            batch_size, embed_dim = x.shape
            seq_len = 1
            x = x.unsqueeze(1)  # Add sequence dimension
        elif len(x.shape) == 3:
            batch_size, seq_len, embed_dim = x.shape
        else:
            # Fallback for unexpected dimensions
            x = x.view(-1, 1, x.shape[-1])
            batch_size, seq_len, embed_dim = x.shape

        # Calculate reward based on signal quality
        signal_quality = torch.norm(x, dim=-1, keepdim=True)
        signal_stability = 1.0 / (1.0 + torch.var(signal_quality, dim=1, keepdim=True))

        # Receptor de dopamina avalia o sinal
        dopamine_response = self.dopamine_receptor(x)  # [B, T, 1]

        # Sistema de recompensa
        if performance_signal is not None:
            reward = performance_signal
        else:
            reward = dopamine_response * signal_stability

        # Modulação dopaminérgica
        dopamine_modulation = 1.0 + self.config.dopamine_strength * reward

        # Aplica modulação com memória adaptativa
        modulated_output = x * dopamine_modulation

        # Atualiza memória de recompensa
        with torch.no_grad():
            avg_reward = reward.mean()
            self.reward_memory.data = (0.9 * self.reward_memory.data +
                                     0.1 * avg_reward)

        return modulated_output


class SyntheticSerotonin(nn.Module):
    """
    Serotonina Sintética - Estabilização e harmonia entre módulos
    Sincroniza diferentes componentes do sistema QRH
    """

    def __init__(self, config: NeurotransmitterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4

        # Receptores de serotonina para harmonia
        self.serotonin_5ht1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.serotonin_5ht2 = nn.Linear(self.embed_dim, self.embed_dim)

        # Sistema de harmonização
        self.harmony_layer = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh()
        )

        # Buffer de estabilização
        self.register_buffer('stability_state', torch.zeros(1, 1, self.embed_dim))

    def forward(self, x: torch.Tensor, harmony_targets: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Aplica estabilização serotoninérgica
        """
        # Ativação dos receptores
        serotonin_5ht1_response = torch.sigmoid(self.serotonin_5ht1(x))
        serotonin_5ht2_response = torch.tanh(self.serotonin_5ht2(x))

        # Harmonização entre sinais
        harmony_signal = self.harmony_layer(x)

        # Se há múltiplos alvos, harmoniza entre eles
        if harmony_targets:
            harmony_sum = sum(target for target in harmony_targets)
            harmony_avg = harmony_sum / len(harmony_targets)

            # Alinhamento harmônico
            alignment_factor = torch.cosine_similarity(
                x.flatten(1), harmony_avg.flatten(1), dim=1
            ).unsqueeze(-1).unsqueeze(-1)

            harmony_signal = harmony_signal * (1.0 + alignment_factor * self.config.serotonin_harmony_factor)

        # Estabilização temporal
        stabilized_signal = (
            self.config.serotonin_stability * x +
            (1.0 - self.config.serotonin_stability) * harmony_signal
        )

        # Modulação serotoninérgica final
        serotonin_modulation = serotonin_5ht1_response * serotonin_5ht2_response
        stabilized_output = stabilized_signal * serotonin_modulation

        # Atualiza estado de estabilização
        with torch.no_grad():
            self.stability_state = 0.95 * self.stability_state + 0.05 * x.mean(dim=(0, 1), keepdim=True)

        return stabilized_output


class SyntheticAcetylcholine(nn.Module):
    """
    Acetilcolina Sintética - Sistema atencional e foco computacional
    Direciona atenção para componentes relevantes do QRH
    """

    def __init__(self, config: NeurotransmitterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4

        # Parâmetros para atenção espectral
        self.alpha = 1.5
        self.epsilon = 1e-10

        # Receptores nicotínicos e muscarínicos
        self.nicotinic_receptor = nn.MultiheadAttention(
            self.embed_dim, num_heads=4, dropout=0.0, batch_first=True
        )

        self.muscarinic_receptor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Foco atencional
        self.attention_focus = nn.Parameter(torch.ones(1))

    def _spectral_attention(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Atenção espectral ΨQRH baseada em doe.md 2.9.2
        """
        batch_size, seq_len, embed_dim = signal.shape

        # Mapeamento para quaternions (reshape para [batch, seq_len, embed_dim//4, 4])
        d_model = embed_dim // 4
        signal_quat = signal.view(batch_size, seq_len, d_model, 4)

        # FFT
        signal_fft = torch.fft.fft(signal_quat, dim=-3)

        # Filtro espectral
        k = torch.arange(seq_len, device=signal.device, dtype=torch.float32)
        filter_kernel = torch.exp(1j * self.alpha * torch.arctan(torch.log(k + self.epsilon)))
        filter_kernel = filter_kernel.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # Aplicar filtro
        signal_filtered = signal_fft * filter_kernel

        # Interação Hamilton (Q ⊗ K ⊗ V, aqui auto-atenção)
        interaction = quaternion_multiply(quaternion_multiply(signal_filtered, signal_filtered), signal_filtered)

        # IFFT
        attended = torch.fft.ifft(interaction, dim=-3).real

        # Reshape de volta
        attended = attended.view(batch_size, seq_len, embed_dim)

        return attended

    def forward(self, x: torch.Tensor, attention_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aplica modulação colinérgica para foco atencional
        """
        # Ativação nicotínica (atenção rápida)
        if attention_targets is not None:
            nicotinic_output, _ = self.nicotinic_receptor(x, attention_targets, attention_targets)
        else:
            nicotinic_output, _ = self.nicotinic_receptor(x, x, x)

        # Ativação muscarínica (modulação lenta)
        muscarinic_output = self.muscarinic_receptor(x)

        # Integração dos sistemas colinérgicos
        cholinergic_signal = (
            self.config.acetylcholine_focus * nicotinic_output +
            (1.0 - self.config.acetylcholine_focus) * muscarinic_output
        )

        # Aplicar seletividade atencional via atenção espectral
        attention_weights = self._spectral_attention(
            cholinergic_signal * self.config.acetylcholine_selectivity
        )

        focused_output = x * attention_weights

        return focused_output


class SyntheticGABA(nn.Module):
    """
    GABA Sintético - Sistema inibitório para controle de ruído
    Remove interferência e estabiliza o sistema QRH
    """

    def __init__(self, config: NeurotransmitterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4

        # Receptores GABA-A e GABA-B
        self.gaba_a_receptor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, 1),
            nn.Sigmoid()
        )

        self.gaba_b_receptor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh()
        )

        # Threshold para ruído
        self.noise_threshold = nn.Parameter(torch.tensor(config.gaba_noise_threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica inibição GABAérgica para redução de ruído
        """
        # Detecção de ruído
        signal_magnitude = torch.norm(x, dim=-1, keepdim=True)
        noise_level = torch.var(x, dim=-1, keepdim=True)

        # Ativação GABA-A (inibição rápida)
        gaba_a_response = self.gaba_a_receptor(x)

        # Ativação GABA-B (modulação lenta)
        gaba_b_response = self.gaba_b_receptor(x)

        # Inibição baseada no nível de ruído
        noise_mask = (noise_level > self.noise_threshold).float()
        inhibition_strength = self.config.gaba_inhibition * noise_mask * gaba_a_response

        # Aplicar inibição
        inhibited_signal = x * (1.0 - inhibition_strength)
        modulated_output = inhibited_signal + gaba_b_response * (1.0 - inhibition_strength)

        return modulated_output


class SyntheticGlutamate(nn.Module):
    """
    Glutamato Sintético - Sistema excitatório para amplificação
    Amplifica sinais importantes no sistema QRH
    """

    def __init__(self, config: NeurotransmitterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4

        # Receptores AMPA, NMDA, Kainate
        self.ampa_receptor = nn.Linear(self.embed_dim, self.embed_dim)
        self.nmda_receptor = nn.Linear(self.embed_dim, self.embed_dim)
        self.kainate_receptor = nn.Linear(self.embed_dim, self.embed_dim)

        # Sistema de amplificação
        self.amplification_gate = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, excitation_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aplica excitação glutamatérgica
        """
        # Ativação dos receptores
        ampa_response = torch.relu(self.ampa_receptor(x))  # Resposta rápida
        nmda_response = torch.sigmoid(self.nmda_receptor(x))  # Plasticidade
        kainate_response = torch.tanh(self.kainate_receptor(x))  # Modulação

        # Integração glutamatérgica
        glutamate_signal = (
            0.5 * ampa_response +
            0.3 * nmda_response +
            0.2 * kainate_response
        )

        # Amplificação adaptativa
        amplification_factor = self.amplification_gate(x)
        excitation_strength = self.config.glutamate_excitation * amplification_factor

        # Aplicar excitação
        excited_output = x + excitation_strength * glutamate_signal

        # Amplificação final
        if excitation_signal is not None:
            excited_output = excited_output * (1.0 + excitation_signal * self.config.glutamate_amplification)

        return excited_output


class SyntheticNeurotransmitterSystem(nn.Module):
    """
    Sistema Completo de Neurotransmissores Sintéticos
    Orquestra todos os neurotransmissores para alinhamento perfeito do QRH
    """

    def __init__(self, config: NeurotransmitterConfig):
        super().__init__()
        self.config = config

        # Inicializar todos os neurotransmissores
        self.dopamine = SyntheticDopamine(config)
        self.serotonin = SyntheticSerotonin(config)
        self.acetylcholine = SyntheticAcetylcholine(config)
        self.gaba = SyntheticGABA(config)
        self.glutamate = SyntheticGlutamate(config)

        # Sistema de coordenação central
        self.neural_coordinator = nn.Sequential(
            nn.Linear(config.embed_dim * 4, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim * 4),
            nn.LayerNorm(config.embed_dim * 4)
        )

        # Pesos adaptativos para cada neurotransmissor
        self.neurotransmitter_weights = nn.Parameter(torch.ones(5) / 5)  # 5 neurotransmissores

    def forward(self, x: torch.Tensor,
                performance_signal: Optional[torch.Tensor] = None,
                harmony_targets: Optional[List[torch.Tensor]] = None,
                attention_targets: Optional[torch.Tensor] = None,
                excitation_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aplicação coordenada de todos os neurotransmissores sintéticos
        """
        # Aplicar cada neurotransmissor
        dopamine_output = self.dopamine(x, performance_signal)
        serotonin_output = self.serotonin(x, harmony_targets)
        acetylcholine_output = self.acetylcholine(x, attention_targets)
        gaba_output = self.gaba(x)
        glutamate_output = self.glutamate(x, excitation_signal)

        # Pesos normalizados
        weights = F.softmax(self.neurotransmitter_weights, dim=0)

        # Combinação ponderada
        combined_output = (
            weights[0] * dopamine_output +
            weights[1] * serotonin_output +
            weights[2] * acetylcholine_output +
            weights[3] * gaba_output +
            weights[4] * glutamate_output
        )

        # Coordenação neural final
        coordinated_output = self.neural_coordinator(combined_output)

        # Residual connection para estabilidade
        final_output = x + 0.3 * coordinated_output

        return final_output

    def get_neurotransmitter_status(self) -> Dict[str, float]:
        """
        Retorna status atual dos neurotransmissores
        """
        weights = F.softmax(self.neurotransmitter_weights, dim=0)

        return {
            'dopamine_activity': weights[0].item(),
            'serotonin_activity': weights[1].item(),
            'acetylcholine_activity': weights[2].item(),
            'gaba_activity': weights[3].item(),
            'glutamate_activity': weights[4].item(),
            'reward_memory': self.dopamine.reward_memory.item(),
            'stability_level': torch.norm(self.serotonin.stability_state).item(),
            'attention_focus': self.acetylcholine.attention_focus.item()
        }


def create_aligned_qrh_component(component: nn.Module, config: NeurotransmitterConfig) -> nn.Module:
    """
    Factory function para criar componentes QRH alinhados com neurotransmissores
    """
    class AlignedQRHComponent(nn.Module):
        def __init__(self, original_component, nt_config):
            super().__init__()
            self.component = original_component
            self.neurotransmitters = SyntheticNeurotransmitterSystem(nt_config)

            # Replace problematic methods with smooth transition
            self._fix_script_methods()

        def _fix_script_methods(self):
            """Smoothly replace problematic JIT methods with gradual transition"""
            if hasattr(self.component, 'fast_quaternion_opposition'):
                # Create smooth transition between original method and neurotransmitter
                original_method = getattr(self.component, 'fast_quaternion_opposition', None)

                def smooth_transition_opposition(x_quat):
                    return self._smooth_opposition_transition(x_quat, original_method)

                self.component.fast_quaternion_opposition = smooth_transition_opposition

        def _smooth_opposition_transition(self, x_quat: torch.Tensor, original_method) -> torch.Tensor:
            """Smooth transition between original method and neurotransmitter-based detection"""
            batch_size, seq_len = x_quat.shape[:2]

            if seq_len <= 1:
                return torch.zeros(batch_size, seq_len, device=x_quat.device)

            try:
                # Try original method first with error handling
                if original_method and callable(original_method) and not isinstance(original_method, type(torch.jit.ScriptMethod)):
                    original_result = original_method(x_quat)

                    # Apply serotonin smoothing to original result
                    smoothed_original = self.neurotransmitters.serotonin(
                        original_result.unsqueeze(-1).expand(-1, -1, self.neurotransmitters.config.embed_dim * 4)
                    ).mean(dim=-1)

                    return smoothed_original
                else:
                    # Fall back to neurotransmitter-based detection
                    return self._neurotransmitter_opposition_detection(x_quat)

            except Exception:
                # Smooth fallback using neurotransmitter system
                return self._neurotransmitter_opposition_detection(x_quat)

        def _neurotransmitter_opposition_detection(self, x_quat: torch.Tensor) -> torch.Tensor:
            """Neurotransmitter-based opposition detection with smooth transitions"""
            batch_size, seq_len = x_quat.shape[:2]

            if seq_len <= 1:
                return torch.zeros(batch_size, seq_len, device=x_quat.device)

            # Reshape for neurotransmitter processing
            x_flat = x_quat.view(batch_size, seq_len, -1)

            # Use GABA for noise suppression (opposition detection)
            gaba_analysis = self.neurotransmitters.gaba(x_flat)

            # Use serotonin for harmonic stabilization
            serotonin_harmony = self.neurotransmitters.serotonin(x_flat)

            # Combine GABA and serotonin for smooth opposition detection
            opposition_raw = torch.norm(x_flat - gaba_analysis, dim=-1)
            harmony_factor = torch.cosine_similarity(x_flat.flatten(1), serotonin_harmony.flatten(1), dim=1).unsqueeze(-1)

            # Smooth transition with harmony modulation
            opposition_smooth = opposition_raw * (1.0 - 0.3 * torch.sigmoid(harmony_factor))

            return torch.sigmoid(opposition_smooth - opposition_smooth.mean(dim=-1, keepdim=True))

        def _safe_opposition_detection(self, x_quat: torch.Tensor) -> torch.Tensor:
            """Safe version of quaternion opposition detection"""
            batch_size, seq_len = x_quat.shape[:2]

            if seq_len <= 1:
                return torch.zeros(batch_size, seq_len, device=x_quat.device)

            # Usar sistema GABA para detecção de ruído/oposição
            gaba_analysis = self.neurotransmitters.gaba(x_quat.view(batch_size, seq_len, -1))

            # Calcular oposição baseada na análise GABA
            opposition_scores = torch.norm(x_quat.view(batch_size, seq_len, -1) - gaba_analysis, dim=-1)
            opposition_normalized = torch.sigmoid(opposition_scores - opposition_scores.mean(dim=-1, keepdim=True))

            return opposition_normalized

        def forward(self, *args, **kwargs):
            """Forward pass com alinhamento neurotransmissor"""
            # Executar componente original
            output = self.component(*args, **kwargs)

            # Aplicar alinhamento neurotransmissor se a saída for tensor
            if isinstance(output, torch.Tensor):
                aligned_output = self.neurotransmitters(output)
                return aligned_output
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                aligned_first = self.neurotransmitters(output[0])
                return (aligned_first,) + output[1:]
            else:
                return output

    return AlignedQRHComponent(component, config)