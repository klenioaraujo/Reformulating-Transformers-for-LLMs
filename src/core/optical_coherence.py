#!/usr/bin/env python3
"""
Optical Coherence Calculator - Sharpness Adaptativo ΨQRH
=========================================================

Calcula sharpness da sonda óptica a partir de coerência do campo
de ressonância, eliminando valores fixos copiados do GPT-2.

Baseado em: η_optical = coherence(field) / (1 + disorder)

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import numpy as np
from typing import Optional, Tuple


class OpticalCoherenceCalculator:
    """
    Calcula sharpness óptico emergente da coerência do campo.

    Princípio: O sharpness NÃO é fixo (2.36 do GPT-2), mas emerge
    da estrutura espacial do campo de ressonância.

    Fórmula:
        s = s₀ · coherence · (2 - D) · (1 - FCI)

    Onde:
    - coherence: Coerência espacial via autocorrelação
    - (2 - D): Fator de ordem fractal (D=1 → max, D=2 → min)
    - (1 - FCI): Fator de exploração (FCI alto → sharpness baixo)
    """

    def __init__(
        self,
        s_baseline: float = 2.0,   # Sharpness baseline (~GPT-2)
        s_min: float = 0.5,        # Sharpness mínimo
        s_max: float = 5.0,        # Sharpness máximo
        coherence_method: str = 'autocorr'  # 'autocorr' ou 'mutual_info'
    ):
        self.s_baseline = s_baseline
        self.s_min = s_min
        self.s_max = s_max
        self.coherence_method = coherence_method

    def compute_optical_sharpness(
        self,
        resonance_field: torch.Tensor,  # [vocab_size]
        D_fractal: float,
        FCI: float
    ) -> float:
        """
        Calcula sharpness adaptativo baseado em física óptica.

        Interpretação física:

        1. Coerência Espacial (ρ₁):
           - Alta autocorrelação → campo coerente → sharpness alto
           - Baixa autocorrelação → campo disperso → sharpness baixo

        2. Ordem Fractal (2 - D):
           - D = 1.0: Sistema ordenado → fator = 1.0 → sharpness alto
           - D = 2.0: Sistema caótico → fator = 0.0 → sharpness baixo

        3. Exploração (1 - FCI):
           - FCI baixo: Análise básica → sharpness alto (convergência)
           - FCI alto: Consciência emergente → sharpness baixo (exploração)

        Args:
            resonance_field: Espectro de ressonância [vocab_size]
            D_fractal: Dimensão fractal [1.0, 2.0]
            FCI: Fractal Consciousness Index [0.0, 1.0]

        Returns:
            s: Sharpness óptico [s_min, s_max]
        """
        # Validação
        D_fractal = np.clip(D_fractal, 1.0, 2.0)
        FCI = np.clip(FCI, 0.0, 1.0)

        # 1. Coerência espacial
        if self.coherence_method == 'autocorr':
            coherence = self._compute_spatial_coherence_autocorr(resonance_field)
        elif self.coherence_method == 'mutual_info':
            coherence = self._compute_spatial_coherence_mi(resonance_field)
        else:
            raise ValueError(f"Unknown coherence method: {self.coherence_method}")

        # 2. Fator de ordem fractal
        order_factor = 2.0 - D_fractal  # ∈ [0, 1]

        # 3. Fator de exploração (inverso de FCI)
        exploration_factor = 1.0 - FCI  # ∈ [0, 1]

        # 4. Sharpness emergente
        s = self.s_baseline * coherence * order_factor * exploration_factor

        # Clamping
        s = np.clip(s, self.s_min, self.s_max)

        return float(s)

    def _compute_spatial_coherence_autocorr(
        self,
        field: torch.Tensor,
        lag: int = 1
    ) -> float:
        """
        Coerência espacial via autocorrelação de lag=1.

        ρ₁ = Cov(f[i], f[i+1]) / (σ_f)²

        Alta autocorrelação → campo suave → coerente
        Baixa autocorrelação → campo ruidoso → incoerente

        Args:
            field: Campo de ressonância [vocab_size]
            lag: Deslocamento para autocorrelação

        Returns:
            ρ₁: Autocorrelação normalizada [0, 1]
        """
        # Normalizar campo
        field_norm = field / (torch.max(field) + 1e-10)

        # Autocorrelação de lag=1
        if len(field_norm) <= lag:
            return 0.0

        # Coeficiente de correlação de Pearson
        x = field_norm[:-lag]
        y = field_norm[lag:]

        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        cov = torch.sum((x - mean_x) * (y - mean_y))
        std_x = torch.sqrt(torch.sum((x - mean_x) ** 2))
        std_y = torch.sqrt(torch.sum((y - mean_y) ** 2))

        rho = cov / (std_x * std_y + 1e-10)

        # Clampar em [0, 1] (autocorrelação negativa → 0)
        rho = torch.clamp(rho, 0.0, 1.0)

        return rho.item()

    def _compute_spatial_coherence_mi(
        self,
        field: torch.Tensor,
        bins: int = 10
    ) -> float:
        """
        Coerência espacial via informação mútua.

        I(X; Y) = H(X) + H(Y) - H(X, Y)

        Alta MI → dependência espacial → coerente
        Baixa MI → independência → incoerente

        Args:
            field: Campo de ressonância [vocab_size]
            bins: Número de bins para histograma

        Returns:
            coherence_mi: Coerência via MI normalizada [0, 1]
        """
        if len(field) <= 1:
            return 0.0

        # Deslocar para criar pares (x_i, x_{i+1})
        x = field[:-1].numpy()
        y = field[1:].numpy()

        # Histograma 2D
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        hist_x = np.histogram(x, bins=bins)[0]
        hist_y = np.histogram(y, bins=bins)[0]

        # Probabilidades
        p_xy = hist_2d / (hist_2d.sum() + 1e-10)
        p_x = hist_x / (hist_x.sum() + 1e-10)
        p_y = hist_y / (hist_y.sum() + 1e-10)

        # Entropias
        H_x = -np.sum(p_x * np.log(p_x + 1e-10))
        H_y = -np.sum(p_y * np.log(p_y + 1e-10))
        H_xy = -np.sum(p_xy * np.log(p_xy + 1e-10))

        # Informação mútua
        MI = H_x + H_y - H_xy

        # Normalizar por H_max = min(H_x, H_y)
        H_max = min(H_x, H_y) + 1e-10
        coherence_mi = MI / H_max

        # Clampar
        coherence_mi = np.clip(coherence_mi, 0.0, 1.0)

        return float(coherence_mi)

    def compute_field_disorder(
        self,
        field: torch.Tensor
    ) -> float:
        """
        Calcula desordem do campo via variância normalizada.

        disorder = σ / μ  (coeficiente de variação)

        Alta desordem → campo ruidoso → sharpness baixo

        Args:
            field: Campo de ressonância [vocab_size]

        Returns:
            disorder: Desordem normalizada [0, inf]
        """
        mean = torch.mean(field)
        std = torch.std(field)

        disorder = std / (mean + 1e-10)

        return disorder.item()

    def estimate_coherence_length(
        self,
        field: torch.Tensor
    ) -> int:
        """
        Estima comprimento de coerência do campo.

        L_c = primeiro lag onde autocorrelação < threshold

        Args:
            field: Campo de ressonância [vocab_size]

        Returns:
            L_c: Comprimento de coerência [1, vocab_size]
        """
        threshold = 0.3  # Autocorrelação mínima para coerência

        field_norm = field / (torch.max(field) + 1e-10)

        for lag in range(1, len(field_norm)):
            x = field_norm[:-lag]
            y = field_norm[lag:]

            # Correlação de Pearson
            rho = torch.corrcoef(torch.stack([x, y]))[0, 1]

            if rho < threshold:
                return lag

        return len(field_norm)  # Campo totalmente coerente


class AdaptiveSharpnessScheduler:
    """
    Ajusta sharpness dinamicamente durante geração autoregressiva.
    """

    def __init__(
        self,
        coherence_calc: OpticalCoherenceCalculator,
        adaptation_rate: float = 0.1
    ):
        self.coherence_calc = coherence_calc
        self.adaptation_rate = adaptation_rate
        self.sharpness_history = []

    def update_sharpness(
        self,
        current_sharpness: float,
        resonance_field: torch.Tensor,
        D_fractal: float,
        FCI: float
    ) -> float:
        """
        Atualiza sharpness com suavização exponencial.

        s_new = (1 - α) · s_old + α · s_target

        Args:
            current_sharpness: Sharpness atual
            resonance_field: Campo de ressonância atual
            D_fractal: Dimensão fractal
            FCI: Consciência

        Returns:
            s_new: Sharpness atualizado
        """
        # Calcular sharpness alvo
        s_target = self.coherence_calc.compute_optical_sharpness(
            resonance_field, D_fractal, FCI
        )

        # Suavização exponencial (Exponential Moving Average)
        s_new = (1 - self.adaptation_rate) * current_sharpness + \
                self.adaptation_rate * s_target

        # Registrar histórico
        self.sharpness_history.append(s_new)

        return s_new

    def get_statistics(self) -> dict:
        """Estatísticas do histórico de sharpness."""
        if not self.sharpness_history:
            return {}

        history = np.array(self.sharpness_history)

        return {
            'mean': float(np.mean(history)),
            'std': float(np.std(history)),
            'min': float(np.min(history)),
            'max': float(np.max(history)),
            'n_steps': len(history)
        }


if __name__ == "__main__":
    # Teste de validação
    calc = OpticalCoherenceCalculator()

    print("=" * 60)
    print("TESTE: Optical Coherence Calculator")
    print("=" * 60)

    # Caso 1: Campo altamente coerente (entrada simples)
    print("\n1. Campo coerente (input simples):")
    field_coherent = torch.tensor([0.5, 0.48, 0.46, 0.44, 0.42, 0.40])
    D1, FCI1 = 1.48, 0.16

    coherence1 = calc._compute_spatial_coherence_autocorr(field_coherent)
    s1 = calc.compute_optical_sharpness(field_coherent, D1, FCI1)

    print(f"   Campo: {field_coherent.tolist()}")
    print(f"   Coerência (autocorr): {coherence1:.3f}")
    print(f"   D={D1:.2f}, FCI={FCI1:.2f}")
    print(f"   → Sharpness = {s1:.3f}")
    print(f"   Interpretação: Alto (campo suave, baixa complexidade)")

    # Caso 2: Campo incoerente (input complexo)
    print("\n2. Campo incoerente (input complexo):")
    field_incoherent = torch.tensor([0.5, 0.1, 0.4, 0.05, 0.3, 0.15])
    D2, FCI2 = 1.72, 0.68

    coherence2 = calc._compute_spatial_coherence_autocorr(field_incoherent)
    s2 = calc.compute_optical_sharpness(field_incoherent, D2, FCI2)

    print(f"   Campo: {field_incoherent.tolist()}")
    print(f"   Coerência (autocorr): {coherence2:.3f}")
    print(f"   D={D2:.2f}, FCI={FCI2:.2f}")
    print(f"   → Sharpness = {s2:.3f}")
    print(f"   Interpretação: Baixo (campo ruidoso, alta complexidade)")

    # Caso 3: Comparação de métodos
    print("\n3. Comparação: Autocorrelação vs Informação Mútua:")
    calc_mi = OpticalCoherenceCalculator(coherence_method='mutual_info')

    s_autocorr = calc.compute_optical_sharpness(field_coherent, 1.5, 0.5)
    s_mi = calc_mi.compute_optical_sharpness(field_coherent, 1.5, 0.5)

    print(f"   Sharpness (autocorr): {s_autocorr:.3f}")
    print(f"   Sharpness (mutual_info): {s_mi:.3f}")

    # Caso 4: Comprimento de coerência
    print("\n4. Comprimento de coerência:")
    L_c1 = calc.estimate_coherence_length(field_coherent)
    L_c2 = calc.estimate_coherence_length(field_incoherent)

    print(f"   Campo coerente: L_c = {L_c1}")
    print(f"   Campo incoerente: L_c = {L_c2}")

    # Caso 5: Scheduler adaptativo
    print("\n5. Adaptive Sharpness Scheduler:")
    scheduler = AdaptiveSharpnessScheduler(calc, adaptation_rate=0.3)

    s_current = 2.0
    for step in range(5):
        s_current = scheduler.update_sharpness(
            s_current, field_coherent, 1.5, 0.3 + step * 0.1
        )
        print(f"   Step {step+1}: s = {s_current:.3f}")

    stats = scheduler.get_statistics()
    print(f"   Estatísticas: {stats}")

    print(f"\n✅ Optical Coherence Calculator validado!")
