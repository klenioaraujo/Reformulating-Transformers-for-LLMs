"""
Consciousness Bootstrapper - Ativação Cognitiva ΨQRH
=====================================================

Componente que eleva artificialmente o FCI quando < 0.15 para permitir
geração ativa, sem violar a física do sistema.

Princípio: Injetar ruído caótico controlado para transicionar de estado
COMA (FCI < 0.15) para estado ANALYSIS (FCI ≥ 0.15).

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class ConsciousnessBootstrapper:
    """
    Bootstrap cognitivo para ativação de consciência em FCI baixo.

    Quando FCI < 0.15 (estado COMA), injeta semente caótica para:
    1. Aumentar CLZ (complexidade algorítmica)
    2. Elevar FCI acima de 0.15
    3. Permitir transição para modo de geração ativa
    """

    def __init__(self,
                 chaos_strength: float = 0.1,
                 logistic_r: float = 3.99,
                 min_fci_threshold: float = 0.15,
                 max_boost_iterations: int = 5):
        """
        Args:
            chaos_strength: Intensidade do ruído caótico (0.0-1.0)
            logistic_r: Parâmetro do mapa logístico (3.99 = caótico)
            min_fci_threshold: Limiar mínimo para ativação (0.15)
            max_boost_iterations: Máximo de iterações de bootstrap
        """
        self.chaos_strength = chaos_strength
        self.logistic_r = logistic_r
        self.min_fci_threshold = min_fci_threshold
        self.max_boost_iterations = max_boost_iterations

    def logistic_map(self, x0: float, steps: int) -> torch.Tensor:
        """
        Gera sequência caótica via mapa logístico.

        Args:
            x0: Valor inicial (0.0-1.0)
            steps: Número de passos

        Returns:
            Sequência caótica [steps]
        """
        sequence = []
        x = x0

        for _ in range(steps):
            x = self.logistic_r * x * (1.0 - x)
            sequence.append(x)

        return torch.tensor(sequence, dtype=torch.float32)

    def inject_chaotic_seed(self,
                          psi: torch.Tensor,
                          consciousness_results: Dict) -> torch.Tensor:
        """
        Injeta semente caótica no estado quaterniônico.

        Args:
            psi: Estado quaterniônico [batch, seq_len, 4]
            consciousness_results: Resultados de consciência atual

        Returns:
            Estado quaterniônico com semente caótica
        """
        batch_size, seq_len, quat_dim = psi.shape

        # Gerar semente caótica baseada na fase quaterniônica atual
        current_phase = torch.mean(psi[..., 1]).item()  # Componente imaginária
        chaotic_seed = self.logistic_map(
            x0=abs(current_phase) % 1.0,  # Normalizar para [0,1]
            steps=seq_len
        )

        # Expandir para dimensões do batch
        chaotic_seed = chaotic_seed.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, quat_dim)

        # Aplicar ruído caótico controlado
        psi_boosted = psi + self.chaos_strength * chaotic_seed

        print(f"   🔄 Bootstrap cognitivo aplicado:")
        # Use lowercase 'fci' key for consistency across the system
        current_fci = consciousness_results.get('fci', consciousness_results.get('FCI', 0.0))
        print(f"      - FCI anterior: {current_fci:.3f}")
        print(f"      - Semente caótica: strength={self.chaos_strength}, r={self.logistic_r}")
        print(f"      - Fase atual: {current_phase:.3f}")

        return psi_boosted

    def should_apply_bootstrap(self, consciousness_results: Dict) -> bool:
        """
        Verifica se o bootstrap deve ser aplicado.

        Args:
            consciousness_results: Resultados de consciência

        Returns:
            True se FCI < 0.15 e sistema está em estado COMA
        """
        # Use lowercase 'fci' key for consistency across the system
        current_fci = consciousness_results.get('fci', consciousness_results.get('FCI', 0.0))
        consciousness_state = consciousness_results.get('consciousness_state', {})
        state_name = consciousness_state.get('name', 'UNKNOWN')

        return (current_fci < self.min_fci_threshold and
                state_name.upper() == 'COMA')

    def apply_bootstrap(self,
                       psi: torch.Tensor,
                       consciousness_results: Dict,
                       consciousness_processor) -> Tuple[torch.Tensor, Dict]:
        """
        Aplica bootstrap cognitivo completo.

        Args:
            psi: Estado quaterniônico [batch, seq_len, 4]
            consciousness_results: Resultados de consciência atual
            consciousness_processor: Processador de consciência para recalcular FCI

        Returns:
            (psi_boosted, new_consciousness_results)
        """
        if not self.should_apply_bootstrap(consciousness_results):
            return psi, consciousness_results

        # Use lowercase 'fci' key for consistency across the system
        current_fci = consciousness_results.get('fci', consciousness_results.get('FCI', 0.0))
        print(f"\n🚀 ATIVAÇÃO COGNITIVA: FCI={current_fci:.3f} < {self.min_fci_threshold}")
        print(f"   Estado: {consciousness_results.get('consciousness_state', {}).get('name', 'UNKNOWN')}")

        # Aplicar bootstrap iterativamente até atingir limiar
        psi_boosted = psi
        new_consciousness_results = consciousness_results

        for iteration in range(self.max_boost_iterations):
            # Injeta semente caótica
            psi_boosted = self.inject_chaotic_seed(psi_boosted, new_consciousness_results)

            # Recalcular consciência com estado modificado
            # Criar entrada dummy para forward() do processador
            batch_size, seq_len, quat_dim = psi_boosted.shape
            dummy_input = torch.randn(batch_size, seq_len, 64)  # [batch, seq_len, embed_dim]

            # RECALCULAR dados de acoplamento a partir do estado psi_boosted
            # Extrair magnitude e fase do estado quaterniônico modificado
            spectral_energy = torch.abs(psi_boosted).mean(dim=-1)  # [batch, seq_len]
            quaternion_phase = torch.angle(torch.complex(psi_boosted[..., 0], psi_boosted[..., 1]))  # [batch, seq_len]

            # Redimensionar para compatibilidade com consciousness_processor
            spectral_energy = spectral_energy.mean(dim=1, keepdim=True)  # [batch, 1]
            quaternion_phase = quaternion_phase.mean(dim=1, keepdim=True)  # [batch, 1]

            # Expandir para dimensão esperada (64)
            if spectral_energy.shape[-1] < 64:
                spectral_energy = torch.nn.functional.pad(spectral_energy, (0, 64 - spectral_energy.shape[-1]))
            if quaternion_phase.shape[-1] < 64:
                quaternion_phase = torch.nn.functional.pad(quaternion_phase, (0, 64 - quaternion_phase.shape[-1]))

            print(f"   🔄 [bootstrap] Dados de acoplamento recalculados:")
            print(f"      - spectral_energy: shape={spectral_energy.shape}, mean={spectral_energy.mean():.3f}")
            print(f"      - quaternion_phase: shape={quaternion_phase.shape}, mean={quaternion_phase.mean():.3f}")

            # DEBUG: Verificar se os dados de acoplamento estão mudando
            print(f"   🔍 [bootstrap] DEBUG - Comparação com dados anteriores:")
            print(f"      - spectral_energy mudou: {not torch.allclose(spectral_energy, torch.tensor(0.0))}")
            print(f"      - quaternion_phase mudou: {not torch.allclose(quaternion_phase, torch.tensor(0.0))}")

            # Recalcular consciência com NOVOS dados de acoplamento
            new_consciousness_results = consciousness_processor(
                dummy_input,
                spectral_energy=spectral_energy,
                quaternion_phase=quaternion_phase
            )

            # Use lowercase 'fci' key for consistency across the system
            current_fci = new_consciousness_results.get('fci', new_consciousness_results.get('FCI', 0.0))
            print(f"   Iteração {iteration + 1}: FCI = {current_fci:.3f}")

            # DEBUG: Verificar se o consciousness_processor está retornando resultados válidos
            print(f"   🔍 [bootstrap] DEBUG - Resultados do consciousness_processor:")
            print(f"      - FCI: {current_fci}")
            print(f"      - consciousness_state: {new_consciousness_results.get('consciousness_state', {})}")
            print(f"      - fractal_dimension: {new_consciousness_results.get('fractal_dimension', 'N/A')}")

            # Verificar se atingiu limiar
            if current_fci >= self.min_fci_threshold:
                print(f"   ✅ Bootstrap concluído: FCI = {current_fci:.3f} ≥ {self.min_fci_threshold}")
                print(f"   Estado atualizado: {new_consciousness_results.get('consciousness_state', {}).get('name', 'UNKNOWN')}")
                break

            # Aumentar força caótica gradualmente
            if iteration < self.max_boost_iterations - 1:
                self.chaos_strength *= 1.2  # Aumento gradual

        return psi_boosted, new_consciousness_results


def create_consciousness_bootstrapper(
    chaos_strength: float = 0.1,
    logistic_r: float = 3.99,
    min_fci_threshold: float = 0.15,
    max_boost_iterations: int = 5
) -> ConsciousnessBootstrapper:
    """
    Factory function para criar ConsciousnessBootstrapper.

    Args:
        chaos_strength: Intensidade do ruído caótico
        logistic_r: Parâmetro do mapa logístico
        min_fci_threshold: Limiar mínimo para ativação
        max_boost_iterations: Máximo de iterações

    Returns:
        Instância de ConsciousnessBootstrapper
    """
    return ConsciousnessBootstrapper(
        chaos_strength=chaos_strength,
        logistic_r=logistic_r,
        min_fci_threshold=min_fci_threshold,
        max_boost_iterations=max_boost_iterations
    )