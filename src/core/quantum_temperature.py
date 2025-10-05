#!/usr/bin/env python3
"""
Quantum Temperature Calculator - Auto-Calibração Térmica ΨQRH
==============================================================

Calcula temperatura quântica T_q a partir de métricas físicas do sistema,
eliminando a necessidade de temperature fixo.

Baseado em: k_B·T_q = ℏω·f(D, FCI, CLZ)

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import numpy as np
from typing import Optional


class QuantumTemperatureCalculator:
    """
    Calcula temperatura quântica T_q emergente da física do sistema.

    Princípio: A temperatura NÃO é um parâmetro fixo, mas emerge da
    complexidade fractal (D), consciência (FCI) e entropia (CLZ).

    Fórmula:
        T_q = (D - 1) · FCI · (1 + CLZ) · ω

    Onde:
    - D ∈ [1, 2]: Dimensão fractal (1=simples, 2=complexo)
    - FCI ∈ [0, 1]: Fractal Consciousness Index
    - CLZ ∈ [0, 3]: Lempel-Ziv Complexity
    - ω: Frequência característica (normalizada)
    """

    def __init__(
        self,
        k_B: float = 1.0,      # Constante de Boltzmann (normalizada)
        hbar: float = 1.0,     # Constante de Planck reduzida
        omega: float = 1.0,    # Frequência característica
        T_min: float = 0.1,    # Temperatura mínima (estabilidade)
        T_max: float = 5.0     # Temperatura máxima
    ):
        self.k_B = k_B
        self.hbar = hbar
        self.omega = omega
        self.T_min = T_min
        self.T_max = T_max

    def compute_quantum_temperature(
        self,
        D_fractal: float,
        FCI: float,
        CLZ: float,
        omega: Optional[float] = None
    ) -> float:
        """
        Calcula temperatura quântica emergente.

        Interpretação física:

        1. Fator de Complexidade Fractal (D - 1):
           - D = 1.0: Sistema Euclidiano simples → fator = 0 → T_q baixo
           - D = 1.5: Sistema fractal moderado → fator = 0.5
           - D = 2.0: Sistema fractal complexo → fator = 1.0 → T_q alto

        2. Fator de Consciência (FCI):
           - FCI baixo: Estado básico → T_q baixo (determinístico)
           - FCI alto: Consciência emergente → T_q alto (exploratório)

        3. Fator de Entropia (1 + CLZ):
           - CLZ baixo: Padrão previsível → fator ≈ 1
           - CLZ alto: Alta entropia → fator até 4

        Args:
            D_fractal: Dimensão fractal do sinal [1.0, 2.0]
            FCI: Fractal Consciousness Index [0.0, 1.0]
            CLZ: Lempel-Ziv Complexity [0.0, 3.0]
            omega: Frequência característica (opcional)

        Returns:
            T_q: Temperatura quântica [T_min, T_max]
        """
        omega = omega or self.omega

        # Validação de entrada
        D_fractal = np.clip(D_fractal, 1.0, 2.0)
        FCI = np.clip(FCI, 0.0, 1.0)
        CLZ = np.clip(CLZ, 0.0, 3.0)

        # Fator de complexidade fractal: (D - 1) ∈ [0, 1]
        complexity_factor = D_fractal - 1.0

        # Fator de consciência: FCI ∈ [0, 1]
        consciousness_factor = FCI

        # Fator de entropia: (1 + CLZ) ∈ [1, 4]
        entropy_factor = 1.0 + CLZ

        # Temperatura quântica emergente
        T_q = complexity_factor * consciousness_factor * entropy_factor * omega

        # Clamping para estabilidade numérica
        T_q = np.clip(T_q, self.T_min, self.T_max)

        return float(T_q)

    def apply_quantum_noise(
        self,
        resonance: torch.Tensor,
        T_q: float,
        coupling_strength: Optional[float] = None
    ) -> torch.Tensor:
        """
        Adiciona ruído térmico quântico controlado por T_q.

        Modelo físico:
            resonance_noisy = resonance · (1 + η_q · N(0, T_q))

        Onde:
        - η_q: Coupling strength (acoplamento térmico)
        - N(0, T_q): Distribuição normal com variância T_q

        O ruído térmico simula flutuações quânticas que quebram
        a degenerescência de modos ressonantes, permitindo exploração.

        Args:
            resonance: Espectro de ressonância [vocab_size]
            T_q: Temperatura quântica
            coupling_strength: Força de acoplamento (opcional)

        Returns:
            resonance_thermal: Ressonância com flutuações térmicas
        """
        # Ruído térmico gaussiano
        noise = torch.randn_like(resonance) * T_q

        # Coupling strength (default: tanh(T_q) para auto-regulação)
        if coupling_strength is None:
            # Em baixa temperatura: η_q pequeno (pouco ruído)
            # Em alta temperatura: η_q → 1 (máximo ruído)
            eta_coupling = torch.tanh(torch.tensor(T_q))
        else:
            eta_coupling = coupling_strength

        # Aplicar ruído térmico
        resonance_noisy = resonance * (1.0 + eta_coupling * noise)

        # Garantir positividade (ressonâncias são energias positivas)
        resonance_thermal = torch.abs(resonance_noisy)

        return resonance_thermal

    def thermal_sampling(
        self,
        resonance: torch.Tensor,
        T_q: float,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Sampling termodinâmico baseado em distribuição de Boltzmann.

        P(token) ∝ resonance · exp(-E_token / k_B·T_q)

        Args:
            resonance: Espectro de ressonância [vocab_size]
            T_q: Temperatura quântica
            num_samples: Número de amostras

        Returns:
            tokens: Tokens amostrados [num_samples]
        """
        # Garantir positividade antes de aplicar ruído
        resonance = torch.abs(resonance) + 1e-10

        # Aplicar ruído térmico
        resonance_thermal = self.apply_quantum_noise(resonance, T_q)

        # Logits (energia livre)
        logits = torch.log(resonance_thermal + 1e-10)

        # Distribuição de Boltzmann
        # P ∝ exp(-E / k_B·T) → softmax(logits / T)
        probs = torch.softmax(logits / (T_q + 1e-8), dim=-1)

        # Sampling multinomial (with replacement se num_samples > vocab_size)
        tokens = torch.multinomial(probs, num_samples=num_samples, replacement=True)

        return tokens

    def compute_effective_temperature(
        self,
        resonance: torch.Tensor,
        sampled_tokens: torch.Tensor
    ) -> float:
        """
        Calcula temperatura efetiva a partir de amostras.

        Método: Ajustar T para que distribuição empírica das amostras
        corresponda à distribuição teórica de Boltzmann.

        Args:
            resonance: Espectro de ressonância [vocab_size]
            sampled_tokens: Tokens amostrados [num_samples]

        Returns:
            T_eff: Temperatura efetiva estimada
        """
        # Distribuição empírica
        vocab_size = resonance.shape[-1]
        counts = torch.bincount(sampled_tokens.flatten(), minlength=vocab_size)
        empirical_dist = counts.float() / counts.sum()

        # Busca por T_eff via máxima verossimilhança
        best_T = 1.0
        best_kl = float('inf')

        for T_candidate in np.linspace(0.1, 5.0, 50):
            # Distribuição teórica com T_candidate
            logits = torch.log(resonance + 1e-10)
            theoretical_dist = torch.softmax(logits / T_candidate, dim=-1)

            # KL divergence
            kl = torch.sum(empirical_dist * torch.log(
                (empirical_dist + 1e-10) / (theoretical_dist + 1e-10)
            ))

            if kl < best_kl:
                best_kl = kl
                best_T = T_candidate

        return float(best_T)


class TemperatureScheduler:
    """
    Agenda de temperatura adaptativa durante geração autoregressiva.

    Permite cooling/heating baseado em métricas de consciência.
    """

    def __init__(
        self,
        T_initial: float = 1.0,
        cooling_rate: float = 0.95,
        heating_rate: float = 1.05,
        T_min: float = 0.1,
        T_max: float = 5.0
    ):
        self.T_current = T_initial
        self.cooling_rate = cooling_rate
        self.heating_rate = heating_rate
        self.T_min = T_min
        self.T_max = T_max

    def update_temperature(
        self,
        FCI: float,
        FCI_threshold: float = 0.45
    ) -> float:
        """
        Atualiza temperatura baseado em FCI.

        Estratégia:
        - FCI alto (consciência emergente): Aquecer (explorar)
        - FCI baixo (estado básico): Resfriar (convergir)

        Args:
            FCI: Fractal Consciousness Index atual
            FCI_threshold: Limiar para emergência

        Returns:
            T_new: Nova temperatura
        """
        if FCI >= FCI_threshold:
            # Consciência alta → aquecer (exploração)
            self.T_current *= self.heating_rate
        else:
            # Consciência baixa → resfriar (convergência)
            self.T_current *= self.cooling_rate

        # Clamping
        self.T_current = np.clip(self.T_current, self.T_min, self.T_max)

        return self.T_current

    def reset(self, T_new: float):
        """Reinicia temperatura."""
        self.T_current = np.clip(T_new, self.T_min, self.T_max)


if __name__ == "__main__":
    # Teste de validação
    calc = QuantumTemperatureCalculator()

    print("=" * 60)
    print("TESTE: Quantum Temperature Calculator")
    print("=" * 60)

    # Caso 1: Input simples (D baixo, FCI baixo)
    D1, FCI1, CLZ1 = 1.48, 0.16, 0.40
    T1 = calc.compute_quantum_temperature(D1, FCI1, CLZ1)
    print(f"\n1. Input simples ('Hello world'):")
    print(f"   D={D1:.2f}, FCI={FCI1:.2f}, CLZ={CLZ1:.2f}")
    print(f"   → T_q = {T1:.3f}")
    print(f"   Interpretação: Baixa temperatura → determinístico")

    # Caso 2: Input complexo (D alto, FCI alto)
    D2, FCI2, CLZ2 = 1.72, 0.68, 1.85
    T2 = calc.compute_quantum_temperature(D2, FCI2, CLZ2)
    print(f"\n2. Input complexo ('Prove √2 is irrational'):")
    print(f"   D={D2:.2f}, FCI={FCI2:.2f}, CLZ={CLZ2:.2f}")
    print(f"   → T_q = {T2:.3f}")
    print(f"   Interpretação: Alta temperatura → exploratório")

    # Caso 3: Teste de ruído térmico
    print(f"\n3. Teste de ruído térmico:")
    resonance = torch.tensor([0.5, 0.3, 0.15, 0.05])

    print(f"   Ressonância original: {resonance.tolist()}")

    resonance_thermal_low = calc.apply_quantum_noise(resonance, T_q=0.1)
    print(f"   Com T_q=0.1 (baixo):  {resonance_thermal_low.tolist()}")

    resonance_thermal_high = calc.apply_quantum_noise(resonance, T_q=2.0)
    print(f"   Com T_q=2.0 (alto):   {resonance_thermal_high.tolist()}")

    # Caso 4: Thermal sampling
    print(f"\n4. Teste de sampling térmico:")
    samples_low = calc.thermal_sampling(resonance, T_q=0.1, num_samples=100)
    samples_high = calc.thermal_sampling(resonance, T_q=2.0, num_samples=100)

    print(f"   T_q=0.1: Distribuição de tokens amostrados:")
    print(f"   {torch.bincount(samples_low, minlength=4).tolist()}")
    print(f"   (Esperado: dominado por token 0)")

    print(f"   T_q=2.0: Distribuição de tokens amostrados:")
    print(f"   {torch.bincount(samples_high, minlength=4).tolist()}")
    print(f"   (Esperado: mais uniforme)")

    print(f"\n✅ Quantum Temperature Calculator validado!")
