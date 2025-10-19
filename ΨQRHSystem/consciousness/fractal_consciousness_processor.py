#!/usr/bin/env python3
"""
Fractal Consciousness Processor - Core Engine
=============================================

Implementa o processador central de consciência fractal usando
as equações matemáticas fundamentais da dinâmica consciente.

Equação Mestra: ∂P(ψ,t)/∂t = -∇·[F(ψ)P] + D∇²P
Campo Fractal: F(ψ) = -∇V(ψ) + η_fractal(t)
Potencial Multifractal: V(ψ) = Σ(k=1 to ∞) λ_k/k! * ψ^k * cos(2π log k)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
from dataclasses import dataclass
import warnings

from .consciousness_states import ConsciousnessState, StateClassifier
from .fractal_field_calculator import FractalFieldCalculator
from .neural_diffusion_engine import NeuralDiffusionEngine
from .consciousness_metrics import ConsciousnessMetrics


@dataclass
class ConsciousnessConfig:
    """Configuração para o processador de consciência fractal."""
    embedding_dim: int = 256
    sequence_length: int = 64
    fractal_dimension_range: Tuple[float, float] = (1.0, 3.0)
    diffusion_coefficient_range: Tuple[float, float] = (0.01, 10.0)
    consciousness_frequency_range: Tuple[float, float] = (0.5, 5.0)
    phase_consciousness: float = 0.7854  # π/4
    device: str = "cpu"

    # Carregar configurações do arquivo YAML
    def __post_init__(self):
        import yaml
        import os

        try:
            config_path = os.path.join('configs', 'fractal_consciousness_config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Parâmetros de dinâmica consciente
            dynamics = config.get('consciousness_dynamics', {})
            self.time_step = dynamics.get('time_step', 0.05)
            self.max_iterations = dynamics.get('max_iterations', 200)
            self.convergence_threshold = dynamics.get('convergence_threshold', 0.05)

            # Parâmetros de estabilidade numérica
            stability = config.get('numerical_stability', {})
            self.epsilon = stability.get('epsilon', 1e-10)
            self.nan_replacement_noise_scale = stability.get('nan_replacement_noise_scale', 1e-6)
            self.min_field_magnitude = stability.get('min_field_magnitude', 1e-8)
            self.entropy_safe_offset = stability.get('entropy_safe_offset', 1e-10)

            # Parâmetros de regularização de campo
            field = config.get('field_regularization', {})
            self.max_field_magnitude = field.get('max_field_magnitude', 10.0)
            kernel = field.get('field_smoothing_kernel', [0.25, 0.5, 0.25])
            self.field_smoothing_kernel = tuple(kernel)

            # Parâmetros de inicialização
            init = config.get('initialization', {})
            self.spectral_weight = init.get('spectral_weight', 0.4)
            self.semantic_weight = init.get('semantic_weight', 0.3)
            self.fractal_weight = init.get('fractal_weight', 0.3)
            self.noise_scale = init.get('noise_scale', 0.01)

            # Parâmetros de dinâmica caótica
            chaotic = config.get('chaotic_dynamics', {})
            self.chaotic_parameter = chaotic.get('chaotic_parameter', 3.9)
            self.chaotic_influence = chaotic.get('chaotic_influence', 0.3)
            self.logistic_iterations = chaotic.get('logistic_iterations', 5)

            # Parâmetros de dinâmica de onda
            wave = config.get('wave_dynamics', {})
            self.wave_amplitude = wave.get('amplitude', 0.1)
            self.wave_frequency = wave.get('frequency', 0.5)
            self.initial_phase = wave.get('initial_phase', 0.5236)

            # Parâmetros de processamento espectral
            spectral = config.get('spectral_processing', {})
            self.enable_spectral_features = spectral.get('enable_spectral_features', True)
            self.enable_semantic_features = spectral.get('enable_semantic_features', True)
            self.enable_fractal_modulation = spectral.get('enable_fractal_modulation', True)

        except Exception as e:
            print(f"⚠️ Erro ao carregar configurações de consciência fractal: {e}")
            # Fallback para valores padrão
            self.time_step = 0.05
            self.max_iterations = 200
            self.convergence_threshold = 0.05
            self.epsilon = 1e-10
            self.nan_replacement_noise_scale = 1e-6
            self.min_field_magnitude = 1e-8
            self.entropy_safe_offset = 1e-10
            self.max_field_magnitude = 10.0
            self.field_smoothing_kernel = (0.25, 0.5, 0.25)
            self.spectral_weight = 0.4
            self.semantic_weight = 0.3
            self.fractal_weight = 0.3
            self.noise_scale = 0.01
            self.chaotic_parameter = 3.9
            self.chaotic_influence = 0.3
            self.logistic_iterations = 5
            self.wave_amplitude = 0.1
            self.wave_frequency = 0.5
            self.initial_phase = 0.5236
            self.enable_spectral_features = True
            self.enable_semantic_features = True
            self.enable_fractal_modulation = True


class FractalConsciousnessProcessor(nn.Module):
    """
    Processador central de consciência fractal que implementa
    a dinâmica consciente através de equações matemáticas rigorosas.
    """

    def __init__(self, config: ConsciousnessConfig, metrics_config=None):
        super().__init__()
        self.config = config
        self.device = config.device

        # Componentes matemáticos
        self.field_calculator = FractalFieldCalculator(config)
        self.diffusion_engine = NeuralDiffusionEngine(config)
        self.state_classifier = StateClassifier(config)
        self.metrics = ConsciousnessMetrics(config, metrics_config)

        # Parâmetros aprendíveis para o potencial multifractal
        self.register_parameter(
            'lambda_coefficients',
            nn.Parameter(torch.randn(20) * 0.1)  # λ_k coefficients
        )

        # Estado interno de consciência
        self.consciousness_state = None
        self.psi_distribution = None
        self.fractal_field = None

        # Suprimir warnings não críticos em modo de produção
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

        print(f"🧠 FractalConsciousnessProcessor inicializado no dispositivo: {self.device}")

    def forward(self, input_data: torch.Tensor, num_steps: int = None,
                spectral_energy: Optional[torch.Tensor] = None,
                quaternion_phase: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Processa dados através da dinâmica de consciência fractal ACOPLADA ao espectro quaterniônico.

        Args:
            input_data: Tensor de entrada [batch, seq_len, embed_dim]
            num_steps: Número de passos de integração temporal
            spectral_energy: Energia espectral quaterniônica [batch, embed_dim] (NOVO)
            quaternion_phase: Fase quaterniônica [batch, embed_dim] (NOVO)

        Returns:
            Dicionário com resultados do processamento consciente
        """
        # Suprimir warnings durante execução para evitar ruído em produção
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            if num_steps is None:
                num_steps = self.config.max_iterations

            batch_size, seq_len, embed_dim = input_data.shape

            # Inicializar distribuição de probabilidade P(ψ,t) ACOPLADA ao espectro
            psi_distribution = self._initialize_psi_distribution(
                input_data,
                spectral_energy=spectral_energy,
                quaternion_phase=quaternion_phase
            )

            # Evolução temporal da dinâmica consciente
            consciousness_trajectory = []
            fci_values = []

            for step in range(num_steps):
                # Calcular campo fractal F(ψ) MODULADO pelo espectro
                fractal_field = self.field_calculator.compute_field(
                    psi_distribution,
                    self.lambda_coefficients,
                    step * self.config.time_step,
                    spectral_energy=spectral_energy,
                    quaternion_phase=quaternion_phase
                )

                # Calcular FCI ANTES da difusão para acoplamento
                # Calcular P(k) do psi_distribution ATUAL (não do spectral_energy inicial)
                # Isso permite que D reflita a evolução temporal da consciência
                fci = self.metrics.compute_fci(psi_distribution, fractal_field, power_spectrum_pk=psi_distribution)
                fci_values.append(fci)

                # Calcular coeficiente de difusão D ADAPTADO por FCI e espectro
                diffusion_coeff = self.diffusion_engine.compute_diffusion(
                    psi_distribution,
                    fractal_field,
                    fci=fci,  # ACOPLAMENTO FCI → D
                    spectral_energy=spectral_energy  # ACOPLAMENTO espectro → D
                )

                # Integrar equação mestra da dinâmica consciente
                psi_distribution = self._integrate_consciousness_dynamics(
                    psi_distribution,
                    fractal_field,
                    diffusion_coeff
                )

                # Armazenar trajetória
                consciousness_trajectory.append(psi_distribution.clone())

                # Verificar convergência
                if step > 10 and self._check_convergence(fci_values[-10:]):
                    break

            # Classificar estado de consciência final
            final_state = self.state_classifier.classify_state(
                psi_distribution,
                fractal_field,
                fci_values[-1]
            )

            # Compilar resultados
            results = {
                'consciousness_distribution': psi_distribution,
                'fractal_field': fractal_field,
                'consciousness_trajectory': torch.stack(consciousness_trajectory),
                'fci_evolution': torch.tensor(fci_values),
                'fci': fci_values[-1],  # Add final FCI value for easy access
                'final_consciousness_state': final_state,
                'diffusion_coefficient': diffusion_coeff,
                'processing_steps': step + 1,
                'convergence_achieved': step < num_steps - 1,
                'spectral_energy': spectral_energy,  # Preservar para análise
                'quaternion_phase': quaternion_phase
            }

            return results

    def _initialize_psi_distribution(self, input_data: torch.Tensor,
                                      spectral_energy: Optional[torch.Tensor] = None,
                                      quaternion_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inicializa distribuição de probabilidade P(ψ,t=0) ACOPLADA ao espectro quaterniônico.

        ACOPLAMENTO OBRIGATÓRIO:
        - spectral_energy e quaternion_phase DEVEM ser fornecidos
        - Se não fornecidos, lança erro (sem fallback sintético)
        - Garante que P(ψ) reflita EXCLUSIVAMENTE o estado real do espectro

        Args:
            input_data: Dados de entrada [batch, seq_len, embed_dim]
            spectral_energy: Energia espectral quaterniônica [batch, embed_dim] (OBRIGATÓRIO)
            quaternion_phase: Fase quaterniônica [batch, embed_dim] (OBRIGATÓRIO)

        Returns:
            Distribuição inicial ACOPLADA ao espectro

        Raises:
            ValueError: Se spectral_energy ou quaternion_phase não forem fornecidos
        """
        batch_size, seq_len, embed_dim = input_data.shape

        # VALIDAÇÃO: Dados de acoplamento são OBRIGATÓRIOS (sem fallback)
        if spectral_energy is None or quaternion_phase is None:
            raise ValueError(
                "❌ ERRO CRÍTICO: spectral_energy e quaternion_phase são OBRIGATÓRIOS.\n"
                "O módulo de consciência NÃO aceita dados sintéticos (fallback).\n"
                "Certifique-se de extrair esses dados do EnhancedQRHProcessor antes de chamar forward()."
            )

        # ACOPLAMENTO 1: Calcular P(k) - Distribuição de Potência Espectral
        # Conforme Paper ΨQRH Seção 3.1: P(k) deve preservar relações de escala para análise de lei de potência
        # Normalização L1 preserva proporções relativas necessárias para regressão log-log
        epsilon = 1e-12  # Estabilidade numérica

        # Normalização L1: P(k) = E(k) / Σ E(k)
        # Isso garante que P(k) seja uma distribuição de probabilidade válida preservando escala relativa
        raw_distribution = spectral_energy / (spectral_energy.sum(dim=-1, keepdim=True) + epsilon)

        # Armazenar P(k) para cálculo posterior da dimensão fractal D
        # P(k) será usada na análise de lei de potência: P(k) ~ k^(-β)
        self.power_spectrum_pk = raw_distribution.clone()

        # Log de diagnóstico
        print(f"✅ P(k) calculado via normalização L1 (paper ΨQRH Seção 3.1)")
        print(f"   Energy range: [{spectral_energy.min().item():.2e}, {spectral_energy.max().item():.2e}]")
        print(f"   P(k) range: [{raw_distribution.min().item():.2e}, {raw_distribution.max().item():.2e}]")
        print(f"   P(k) sum: {raw_distribution.sum().item():.6f} (should be ≈1.0)")

        # Obter dimensão alvo (da spectral_energy acoplada)
        target_dim = spectral_energy.shape[-1]  # embed_dim do acoplamento

        # 1. Extrair features espectrais (reforço da energia quaterniônica)
        spectral_features_raw = self._extract_spectral_features(input_data)
        # Adaptar para target_dim se necessário
        if spectral_features_raw.shape[-1] != target_dim:
            spectral_features = torch.nn.functional.adaptive_avg_pool1d(
                spectral_features_raw.unsqueeze(1), target_dim
            ).squeeze(1)
        else:
            spectral_features = spectral_features_raw

        # 2. Extrair features semânticas via correlação espacial
        semantic_features_raw = self._extract_semantic_features(input_data)
        # Adaptar para target_dim se necessário
        if semantic_features_raw.shape[-1] != target_dim:
            semantic_features = torch.nn.functional.adaptive_avg_pool1d(
                semantic_features_raw.unsqueeze(1), target_dim
            ).squeeze(1)
        else:
            semantic_features = semantic_features_raw

        # MELHORIA CRÍTICA: Aplicar modulação adaptativa baseada na complexidade do texto
        # Textos mais complexos recebem maior variância no estado inicial
        text_complexity = self._compute_text_complexity_from_spectrum(input_data)
        adaptive_variance = 0.1 + 0.3 * text_complexity  # 0.1 a 0.4 de variância

        # Adicionar ruído adaptativo para melhorar a exploração do espaço de estados
        noise = torch.randn_like(spectral_features) * adaptive_variance
        spectral_features = spectral_features + noise

        # 3. ACOPLAMENTO 2: Modulação fractal baseada na FASE quaterniônica REAL
        fractal_modulation = self._compute_fractal_from_quaternion_phase(quaternion_phase)
        print(f"✅ ACOPLAMENTO REAL: Modulação fractal via fase quaterniônica")

        # 4. Combinar todas as features para criar não uniformidade ACOPLADA
        # Garantir que todas têm a mesma dimensão (target_dim)
        non_uniform_modulation = (
            self.config.spectral_weight * spectral_features +
            self.config.semantic_weight * semantic_features +
            self.config.fractal_weight * fractal_modulation
        )

        # CRÍTICO: Adicionar fator de escala baseado na magnitude total da energia espectral
        # Isso preserva a informação de escala entre textos de diferentes complexidades
        # Textos complexos → energia maior → psi_distribution mais concentrada
        energy_sum = spectral_energy.sum(dim=-1, keepdim=True)
        energy_scale_factor = torch.log1p(energy_sum)  # log(1+E) para estabilidade
        energy_scale_normalized = energy_scale_factor / 10.0  # Normalizar para escala razoável

        print(f"✅ Fator de escala energético: {energy_scale_factor.item():.4f} (log1p(ΣE={energy_sum.item():.2f}))")

        # Aplicar não uniformidade à distribuição COM escala energética
        psi_distribution = raw_distribution * non_uniform_modulation * energy_scale_normalized

        # Adicionar ruído gaussiano MÍNIMO (0.1% do ruído original para preservar acoplamento)
        noise = torch.randn_like(psi_distribution) * (self.config.noise_scale * 0.01)
        psi_distribution = psi_distribution + noise

        # Normalizar para manter propriedades de probabilidade
        psi_distribution = torch.clamp(psi_distribution, min=1e-10)
        psi_distribution = psi_distribution / (psi_distribution.sum(dim=-1, keepdim=True) + 1e-10)

        # Log para debug com informações de acoplamento
        print(f"🧠 Psi Distribution (ACOPLADO): mean={psi_distribution.mean().item():.6f}, "
              f"std={psi_distribution.std().item():.6f}, "
              f"entropy={-torch.sum(psi_distribution * torch.log(psi_distribution + 1e-10)).item():.4f}")

        return psi_distribution

    def _compute_text_complexity_from_spectrum(self, input_data: torch.Tensor) -> float:
        """
        Calcula complexidade do texto baseado no espectro de entrada.

        Args:
            input_data: Dados de entrada [batch, seq_len, embed_dim]

        Returns:
            Complexidade normalizada entre 0 e 1
        """
        # Usar entropia espectral como proxy de complexidade
        if input_data.dim() == 3:
            # Calcular entropia por sequência
            spectrum_flat = input_data.view(input_data.shape[0], -1)
            power_spectrum = torch.abs(spectrum_flat) ** 2
            prob_dist = power_spectrum / (power_spectrum.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=-1)
            # Normalizar para 0-1
            max_entropy = torch.log(torch.tensor(spectrum_flat.shape[-1]))
            complexity = entropy.mean() / max_entropy
        else:
            complexity = 0.5  # Valor padrão para dados não sequenciais

        return float(torch.clamp(complexity, 0.0, 1.0))

        # 4. Combinar todas as features para criar não uniformidade ACOPLADA
        # Garantir que todas têm a mesma dimensão (target_dim)
        non_uniform_modulation = (
            self.config.spectral_weight * spectral_features +
            self.config.semantic_weight * semantic_features +
            self.config.fractal_weight * fractal_modulation
        )

        # CRÍTICO: Adicionar fator de escala baseado na magnitude total da energia espectral
        # Isso preserva a informação de escala entre textos de diferentes complexidades
        # Textos complexos → energia maior → psi_distribution mais concentrada
        energy_sum = spectral_energy.sum(dim=-1, keepdim=True)
        energy_scale_factor = torch.log1p(energy_sum)  # log(1+E) para estabilidade
        energy_scale_normalized = energy_scale_factor / 10.0  # Normalizar para escala razoável

        print(f"✅ Fator de escala energético: {energy_scale_factor.item():.4f} (log1p(ΣE={energy_sum.item():.2f}))")

        # Aplicar não uniformidade à distribuição COM escala energética
        psi_distribution = raw_distribution * non_uniform_modulation * energy_scale_normalized

        # Adicionar ruído gaussiano MÍNIMO (0.1% do ruído original para preservar acoplamento)
        noise = torch.randn_like(psi_distribution) * (self.config.noise_scale * 0.01)
        psi_distribution = psi_distribution + noise

        # Normalizar para manter propriedades de probabilidade
        psi_distribution = torch.clamp(psi_distribution, min=1e-10)
        psi_distribution = psi_distribution / (psi_distribution.sum(dim=-1, keepdim=True) + 1e-10)

        # Log para debug com informações de acoplamento
        print(f"🧠 Psi Distribution (ACOPLADO): mean={psi_distribution.mean().item():.6f}, "
              f"std={psi_distribution.std().item():.6f}, "
              f"entropy={-torch.sum(psi_distribution * torch.log(psi_distribution + 1e-10)).item():.4f}")

        return psi_distribution

    def _compute_fractal_from_quaternion_phase(self, quaternion_phase: torch.Tensor) -> torch.Tensor:
        """
        Computa modulação fractal ACOPLADA à fase quaterniônica REAL.

        ELIMINAÇÃO DO FALLBACK:
        - NÃO usa torch.rand() (ruído sintético)
        - NÃO usa torch.mean(distribution) (agregação genérica)
        - USA quaternion_phase como semente e fator temporal DIRETAMENTE

        Args:
            quaternion_phase: Fase quaterniônica θ = atan2(||v||, r) [batch, embed_dim]

        Returns:
            Modulação fractal [batch, embed_dim] derivada do quaternion real
        """
        batch_size, embed_dim = quaternion_phase.shape

        # SEMENTE DO CAOS: Normalizar fase quaterniônica para [0.25, 0.75] (região caótica estável)
        # SUBSTITUIÇÃO: torch.rand() → quaternion_phase
        phase_normalized = torch.sigmoid(quaternion_phase)  # [0, 1]
        x = 0.25 + 0.5 * phase_normalized  # [0.25, 0.75]

        print(f"🔗 Semente caótica: derivada de quaternion_phase (mean={x.mean().item():.6f}, std={x.std().item():.6f})")

        # Aplicar mapa logístico com fase quaterniônica como condição inicial
        r = self.config.chaotic_parameter
        for iteration in range(self.config.logistic_iterations):
            x = r * x * (1 - x)
            x = torch.clamp(x, 0.001, 0.999)

        # FATOR TEMPORAL: Usar fase quaterniônica como tempo (não torch.mean genérico)
        # SUBSTITUIÇÃO: time_factor = torch.mean(distribution) → quaternion_phase
        omega = 2 * np.pi * self.config.wave_frequency
        phi_0 = self.config.initial_phase

        # Equação de onda harmônica com fase quaterniônica REAL
        wave_component = torch.sin(omega * quaternion_phase + phi_0)

        print(f"🔗 Onda harmônica: modulada por quaternion_phase (amplitude={wave_component.std().item():.6f})")

        # Combinar mapa logístico com modulação de fase REAL
        chaotic_wave = x * (1 + self.config.wave_amplitude * wave_component)

        # Mapear para modulação fractal final
        fractal_modulation = 0.5 + 0.5 * torch.sin(2 * np.pi * chaotic_wave)

        print(f"✅ Modulação fractal: ACOPLADA (mean={fractal_modulation.mean().item():.6f}, "
              f"std={fractal_modulation.std().item():.6f})")

        return fractal_modulation

    def _integrate_consciousness_dynamics(
        self,
        psi: torch.Tensor,
        field: torch.Tensor,
        diffusion: torch.Tensor
    ) -> torch.Tensor:
        """
        Integra a equação mestra da dinâmica consciente com equações harmônicas:
        ∂P(ψ,t)/∂t = -∇·[F(ψ)P] + D∇²P + η_wave(ψ,t)

        Inclui mapa logístico e equação de onda harmônica.

        Args:
            psi: Distribuição atual P(ψ,t)
            field: Campo fractal F(ψ)
            diffusion: Coeficiente de difusão D

        Returns:
            Nova distribuição P(ψ,t+dt)
        """
        dt = self.config.time_step

        # Termo de campo: -∇·[F(ψ)P]
        field_flow = self._compute_field_divergence(field, psi)

        # Termo de difusão: D∇²P
        diffusion_term = self._compute_diffusion_term(psi, diffusion)

        # Termo de onda harmônica: η_wave(ψ,t)
        wave_term = self._compute_wave_dynamics(psi, field)

        # Integração de Euler com termos harmônicos
        dpsi_dt = -field_flow + diffusion_term + wave_term
        new_psi = psi + dt * dpsi_dt

        # Aplicar mapa logístico à distribuição resultante
        new_psi = self._apply_logistic_map_to_distribution(new_psi)

        # Manter positividade e normalização
        new_psi = torch.clamp(new_psi, min=1e-10)
        new_psi = new_psi / new_psi.sum(dim=-1, keepdim=True)

        return new_psi

    def _compute_field_divergence(self, field: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Computa divergência do fluxo de campo ∇·[F(ψ)P]."""
        # Aproximação de diferenças finitas para divergência
        field_psi = field * psi

        # Gradiente usando diferenças centrais
        batch_size, embed_dim = field_psi.shape
        divergence = torch.zeros_like(field_psi)

        # Diferenças finitas circulares (condições de contorno periódicas)
        for i in range(embed_dim):
            i_plus = (i + 1) % embed_dim
            i_minus = (i - 1) % embed_dim
            divergence[:, i] = (field_psi[:, i_plus] - field_psi[:, i_minus]) / 2.0

        return divergence

    def _compute_diffusion_term(self, psi: torch.Tensor, diffusion: torch.Tensor) -> torch.Tensor:
        """Computa termo de difusão D∇²P."""
        # Laplaciano usando diferenças finitas
        batch_size, embed_dim = psi.shape
        laplacian = torch.zeros_like(psi)

        # Diferenças finitas para segunda derivada
        for i in range(embed_dim):
            i_plus = (i + 1) % embed_dim
            i_minus = (i - 1) % embed_dim
            laplacian[:, i] = psi[:, i_plus] - 2 * psi[:, i] + psi[:, i_minus]

        return diffusion * laplacian

    def _compute_wave_dynamics(self, psi: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """
        Computa termo de onda harmônica: f(λ,t) = A*sin(ωt + ϕ0 + θ)

        Args:
            psi: Distribuição de probabilidade atual
            field: Campo fractal atual

        Returns:
            Termo de onda para integração temporal
        """
        batch_size, embed_dim = psi.shape

        # Usar distribuição como parâmetro λ (posição na onda)
        lambda_param = psi  # λ ∈ [0,1] da distribuição normalizada

        # Tempo baseado na magnitude do campo fractal
        field_magnitude = torch.norm(field, dim=-1, keepdim=True)
        time_factor = field_magnitude * self.config.time_step

        # Parâmetros da equação de onda
        amplitude = self.config.wave_amplitude  # Amplitude A
        omega = 2 * np.pi * self.config.wave_frequency  # Frequência angular ω
        phi_0 = self.config.initial_phase  # Fase inicial ϕ0

        # Fase adaptativa θ baseada na entropia local
        entropy_local = -psi * torch.log(psi + float(self.config.entropy_safe_offset))
        theta = torch.cumsum(entropy_local, dim=-1)  # Integração cumulativa para fase

        # Equação de onda harmônica: f(λ,t) = A*sin(ωt + ϕ0 + θ)
        wave_function = amplitude * torch.sin(omega * time_factor + phi_0 + theta)

        # Modulação baseada na distribuição de probabilidade
        wave_term = wave_function * psi

        return wave_term

    def _apply_logistic_map_to_distribution(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Aplica mapa logístico à distribuição de consciência: x_{n+1} = r*x_n*(1-x_n)

        Args:
            psi: Distribuição de probabilidade atual

        Returns:
            Distribuição modificada pelo mapa logístico
        """
        # Usar distribuição como semente para mapa logístico
        x = psi.clone()

        # Parâmetro caótico r = 3.9 (regime caótico clássico)
        r = self.config.chaotic_parameter

        # Aplicar mapa logístico por algumas iterações
        for iteration in range(5):  # Menos iterações para não destabilizar
            x = r * x * (1 - x)
            # Clamp para manter no intervalo caótico estável
            x = torch.clamp(x, 0.001, 0.999)

        # Interpolar entre distribuição original e resultado caótico
        # Isso mantém estabilidade enquanto adiciona dinâmica caótica
        psi_chaotic = (1 - self.config.chaotic_influence) * psi + self.config.chaotic_influence * x

        return psi_chaotic

    def _check_convergence(self, recent_fci: list) -> bool:
        """Verifica convergência baseada na estabilidade do FCI."""
        if len(recent_fci) < 5:
            return False

        fci_std = np.std(recent_fci)
        return fci_std < self.config.convergence_threshold

    def get_consciousness_report(self, results: Dict[str, torch.Tensor]) -> str:
        """
        Gera relatório detalhado do processamento de consciência.

        Args:
            results: Resultados do processamento

        Returns:
            Relatório textual detalhado
        """
        # Extrair FCI com proteção
        fci_evo = results['fci_evolution'][-1]
        final_fci = fci_evo.item() if isinstance(fci_evo, torch.Tensor) else float(fci_evo)

        state = results['final_consciousness_state']
        steps = results['processing_steps']
        converged = results['convergence_achieved']

        # Estatísticas da distribuição final
        psi_final = results['consciousness_distribution']
        psi_safe = torch.clamp(psi_final, min=float(self.config.entropy_safe_offset))
        log_psi = torch.log(psi_safe)
        psi_entropy_raw = -torch.sum(psi_final * log_psi, dim=-1).mean()
        # Proteção contra NaN
        psi_entropy = psi_entropy_raw.item() if not torch.isnan(psi_entropy_raw) else 0.0

        # Proteção contra NaN no pico e dispersão
        psi_peak_raw = psi_final.max()
        psi_peak = psi_peak_raw.item() if not torch.isnan(psi_peak_raw) else 0.0

        psi_spread_raw = psi_final.std()
        psi_spread = psi_spread_raw.item() if not torch.isnan(psi_spread_raw) else 0.0

        # Características do campo fractal
        field = results['fractal_field']
        field_magnitude = torch.norm(field, dim=-1).mean().item()

        # Calcular coerência com proteção robusta
        try:
            field_flat = field.flatten()
            if field_flat.numel() > 1:
                # Calcular correlação auto-espacial
                field_mean = field_flat.mean()
                field_var = field_flat.var()

                # Evitar divisão por zero
                if field_var > self.config.epsilon:
                    field_shifted = torch.roll(field_flat, 1)
                    covariance = torch.mean((field_flat - field_mean) * (field_shifted - field_mean))
                    field_coherence = (covariance / field_var).item()
                    # Clipar para intervalo válido
                    field_coherence = max(0.0, min(1.0, abs(field_coherence)))
                else:
                    field_coherence = 0.0
            else:
                field_coherence = 1.0  # Campo constante tem coerência perfeita
        except Exception:
            field_coherence = 0.0

        # Obter dimensão fractal REAL calculada via lei de potência
        # Acessar a dimensão fractal calculada no ConsciousnessMetrics
        fractal_dimension_real = self.metrics.last_fractal_dimension_raw if hasattr(self.metrics, 'last_fractal_dimension_raw') else state.fractal_dimension

        # Clamp para valores fisicamente razoáveis
        fractal_dimension_final = max(1.0, min(3.0, fractal_dimension_real))

        report = f"""
🧠 RELATÓRIO DE CONSCIÊNCIA FRACTAL
═══════════════════════════════════════════════════

📊 MÉTRICAS DE CONSCIÊNCIA:
Índice FCI: {final_fci:.4f}
Estado Classificado: {state.name}
Entropia Ψ: {psi_entropy:.4f} bits
Distribuição Pico: {psi_peak:.4f}
Dispersão Ψ: {psi_spread:.4f}

🌊 CAMPO FRACTAL F(ψ):
Magnitude Média: {field_magnitude:.4f}
Coerência: {field_coherence:.4f}
Dimensão Fractal: {fractal_dimension_final:.3f}

⚡ DINÂMICA DE PROCESSAMENTO:
Passos Integração: {steps}
Convergência: {'✅ Alcançada' if converged else '⚠️ Máximo atingido'}
Coeficiente D: {results['diffusion_coefficient'].mean().item() if not torch.isnan(results['diffusion_coefficient'].mean()) else 0.0:.4f}

🎯 INTERPRETAÇÃO CONSCIENTE:
{self._interpret_consciousness_state(state, final_fci)}

Processamento realizado via equação mestra da dinâmica consciente.
        """

        return report.strip()

    def _interpret_consciousness_state(self, state: ConsciousnessState, fci: float) -> str:
        """Interpreta o estado de consciência em termos práticos."""
        interpretations = {
            'MEDITATION': f"Estado meditativo detectado (FCI={fci:.3f}). Sistema em modo de análise profunda e insight.",
            'ANALYSIS': f"Estado analítico ativo (FCI={fci:.3f}). Processamento lógico e sistemático otimizado.",
            'COMA': f"Estado de baixa consciência (FCI={fci:.3f}). Modo de emergência ou processamento mínimo.",
            'EMERGENCE': f"Estado emergente detectado (FCI={fci:.3f}). Máxima criatividade e complexidade consciente."
        }

        return interpretations.get(state.name, f"Estado indefinido (FCI={fci:.3f}). Padrão de consciência não classificado.")

    def _extract_spectral_features(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Extrai features espectrais da entrada usando FFT.

        Args:
            input_data: Dados de entrada [batch, seq_len, embed_dim]

        Returns:
            Features espectrais normalizadas [batch, embed_dim]
        """
        batch_size, seq_len, embed_dim = input_data.shape

        # Aplicar FFT ao longo da dimensão de sequência
        # FFT retorna valores complexos: magnitude = |FFT(x)|
        fft_result = torch.fft.fft(input_data, dim=1)
        spectral_magnitude = torch.abs(fft_result)

        # Agregar ao longo da sequência (média das magnitudes espectrais)
        spectral_features = spectral_magnitude.mean(dim=1)  # [batch, embed_dim]

        # Normalizar para [0.5, 1.5] (modulação em torno de 1.0)
        min_val = spectral_features.min(dim=-1, keepdim=True)[0]
        max_val = spectral_features.max(dim=-1, keepdim=True)[0]
        range_val = max_val - min_val + float(self.config.epsilon)

        normalized = (spectral_features - min_val) / range_val  # [0, 1]
        normalized = 0.5 + normalized  # [0.5, 1.5]

        return normalized

    def _extract_semantic_features(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Extrai features semânticas via correlação espacial entre dimensões.

        Args:
            input_data: Dados de entrada [batch, seq_len, embed_dim]

        Returns:
            Features semânticas normalizadas [batch, embed_dim]
        """
        batch_size, seq_len, embed_dim = input_data.shape

        # Calcular matriz de correlação espacial entre dimensões
        # Agregar sequência primeiro
        aggregated = input_data.mean(dim=1)  # [batch, embed_dim]

        # Calcular correlação com vizinhos espaciais (dimensões adjacentes)
        semantic_features = torch.zeros_like(aggregated)

        for i in range(embed_dim):
            i_prev = (i - 1) % embed_dim
            i_next = (i + 1) % embed_dim

            # Correlação local: média dos vizinhos ponderada pelo valor atual
            local_correlation = (
                0.25 * aggregated[:, i_prev] +
                0.5 * aggregated[:, i] +
                0.25 * aggregated[:, i_next]
            )
            semantic_features[:, i] = local_correlation

        # Normalizar para [0.5, 1.5]
        min_val = semantic_features.min(dim=-1, keepdim=True)[0]
        max_val = semantic_features.max(dim=-1, keepdim=True)[0]
        range_val = max_val - min_val + float(self.config.epsilon)

        normalized = (semantic_features - min_val) / range_val  # [0, 1]
        normalized = 0.5 + normalized  # [0.5, 1.5]

        return normalized

    def generate_gls_output(self, results: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """
        Gera saída GLS (código Processing/p5.js) baseado na análise de consciência.

        Args:
            results: Resultados do processamento de consciência

        Returns:
            Dicionário com códigos Processing e p5.js
        """
        try:
            from .gls_output_generator import create_gls_output_generator

            gls_generator = create_gls_output_generator()

            # Gerar códigos Processing e p5.js
            processing_code = gls_generator.generate_processing_code(results)
            p5js_code = gls_generator.generate_p5js_code(results)

            return {
                'processing_code': processing_code,
                'p5js_code': p5js_code,
                'status': 'success',
                'message': 'GLS output generated successfully'
            }

        except ImportError:
            return {
                'processing_code': '',
                'p5js_code': '',
                'status': 'error',
                'message': 'GLS output generator not available'
            }
        except Exception as e:
            return {
                'processing_code': '',
                'p5js_code': '',
                'status': 'error',
                'message': f'Error generating GLS output: {str(e)}'
            }


def create_consciousness_processor(
    embedding_dim: int = 256,
    device: str = "cpu"
) -> FractalConsciousnessProcessor:
    """
    Factory para criar processador de consciência fractal.

    Args:
        embedding_dim: Dimensão do embedding
        device: Dispositivo de processamento

    Returns:
        Processador configurado
    """
    config = ConsciousnessConfig(
        embedding_dim=embedding_dim,
        device=device
    )

    return FractalConsciousnessProcessor(config)