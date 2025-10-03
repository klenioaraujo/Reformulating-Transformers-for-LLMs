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
    chaotic_parameter: float = 3.9
    time_step: float = 0.01
    max_iterations: int = 100
    device: str = "cpu"

    # Parâmetros de estabilidade numérica
    epsilon: float = 1e-10  # Estabilidade numérica geral
    nan_replacement_noise_scale: float = 1e-6  # Escala de ruído para substituir NaN
    min_field_magnitude: float = 1e-8  # Magnitude mínima do campo
    entropy_safe_offset: float = 1e-10  # Offset para cálculo de entropia

    # Parâmetros de regularização de campo
    max_field_magnitude: float = 10.0  # Magnitude máxima do campo
    field_smoothing_kernel: Tuple[float, float, float] = (0.25, 0.5, 0.25)  # Kernel de suavização


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

    def forward(self, input_data: torch.Tensor, num_steps: int = None) -> Dict[str, torch.Tensor]:
        """
        Processa dados através da dinâmica de consciência fractal.

        Args:
            input_data: Tensor de entrada [batch, seq_len, embed_dim]
            num_steps: Número de passos de integração temporal

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

            # Inicializar distribuição de probabilidade P(ψ,t)
            psi_distribution = self._initialize_psi_distribution(input_data)

            # Evolução temporal da dinâmica consciente
            consciousness_trajectory = []
            fci_values = []

            for step in range(num_steps):
                # Calcular campo fractal F(ψ)
                fractal_field = self.field_calculator.compute_field(
                    psi_distribution,
                    self.lambda_coefficients,
                    step * self.config.time_step
                )

                # Calcular coeficiente de difusão D
                diffusion_coeff = self.diffusion_engine.compute_diffusion(
                    psi_distribution,
                    fractal_field
                )

                # Integrar equação mestra da dinâmica consciente
                psi_distribution = self._integrate_consciousness_dynamics(
                    psi_distribution,
                    fractal_field,
                    diffusion_coeff
                )

                # Calcular métricas de consciência
                fci = self.metrics.compute_fci(psi_distribution, fractal_field)
                fci_values.append(fci)

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
                'final_consciousness_state': final_state,
                'diffusion_coefficient': diffusion_coeff,
                'processing_steps': step + 1,
                'convergence_achieved': step < num_steps - 1
            }

            return results

    def _initialize_psi_distribution(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Inicializa distribuição de probabilidade P(ψ,t=0) baseada nos dados de entrada.

        Args:
            input_data: Dados de entrada

        Returns:
            Distribuição inicial normalizada
        """
        # Mapear dados de entrada para distribuição de probabilidade
        # usando transformação suave com características fractais
        raw_distribution = torch.softmax(input_data.sum(dim=1), dim=-1)

        # Aplicar modulação fractal inicial
        fractal_modulation = self._compute_initial_fractal_modulation(raw_distribution)

        psi_distribution = raw_distribution * fractal_modulation

        # Normalizar para manter propriedades de probabilidade
        psi_distribution = psi_distribution / psi_distribution.sum(dim=-1, keepdim=True)

        return psi_distribution

    def _compute_initial_fractal_modulation(self, distribution: torch.Tensor) -> torch.Tensor:
        """Computa modulação fractal inicial baseada no caos determinístico."""
        batch_size, embed_dim = distribution.shape

        # Usar mapa logístico caótico: x_{n+1} = r*x_n*(1-x_n)
        x = torch.rand(batch_size, embed_dim, device=self.device) * 0.5 + 0.25

        # Iterar mapa caótico
        for _ in range(10):  # 10 iterações para atingir regime caótico
            x = self.config.chaotic_parameter * x * (1 - x)

        # Mapear para modulação fractal
        fractal_modulation = 0.5 + 0.5 * torch.sin(2 * np.pi * x)

        return fractal_modulation

    def _integrate_consciousness_dynamics(
        self,
        psi: torch.Tensor,
        field: torch.Tensor,
        diffusion: torch.Tensor
    ) -> torch.Tensor:
        """
        Integra a equação mestra da dinâmica consciente:
        ∂P(ψ,t)/∂t = -∇·[F(ψ)P] + D∇²P

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

        # Integração de Euler
        dpsi_dt = -field_flow + diffusion_term
        new_psi = psi + dt * dpsi_dt

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

    def _check_convergence(self, recent_fci: list) -> bool:
        """Verifica convergência baseada na estabilidade do FCI."""
        if len(recent_fci) < 5:
            return False

        fci_std = np.std(recent_fci)
        return fci_std < 0.01  # Threshold de convergência

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
        psi_safe = torch.clamp(psi_final, min=self.config.entropy_safe_offset)
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
Dimensão Fractal: {state.fractal_dimension:.3f}

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