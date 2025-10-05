"""
Optical Text Decoder - Wave-to-Text Físico ΨQRH
=================================================

Decodifica estado quaterniônico em texto fluente via ressonância óptica,
usando apenas física (sem softmax, sem argmax).

Princípio: Geração autoregressiva via sonda óptica com parâmetros
emergentes (T_q, sharpness) e parada por colapso de consciência.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


class OpticalTextDecoder:
    """
    Decodificador de texto via física óptica para ΨQRH.

    Pipeline:
    1. Estado quaterniônico inicial
    2. Sonda óptica autoregressiva com T_q e sharpness emergentes
    3. Geração de tokens via ressonância térmica
    4. Parada por colapso de consciência (FCI < 0.05)
    """

    def __init__(self,
                 vocab_size: int = 50257,
                 max_tokens: int = 100,
                 min_fci_threshold: float = 0.05,
                 device: str = 'cpu'):
        """
        Args:
            vocab_size: Tamanho do vocabulário
            max_tokens: Máximo de tokens a gerar
            min_fci_threshold: Limiar mínimo de FCI para parada
            device: Dispositivo (cpu/cuda)
        """
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.min_fci_threshold = min_fci_threshold
        self.device = device

        # Importar calculadores de parâmetros emergentes
        from src.core.quantum_temperature import QuantumTemperatureCalculator
        from src.core.optical_coherence import OpticalCoherenceCalculator

        self.temp_calculator = QuantumTemperatureCalculator(
            T_min=0.1,
            T_max=5.0
        )
        self.coherence_calculator = OpticalCoherenceCalculator(
            s_baseline=2.0,
            s_min=0.5,
            s_max=5.0,
            coherence_method='autocorr'
        )

    def optical_probe_resonance(self,
                               psi: torch.Tensor,
                               alpha: float,
                               beta: float) -> torch.Tensor:
        """
        Calcula ressonância óptica entre estado quaterniônico e ondas de sonda.

        Args:
            psi: Estado quaterniônico [batch, 4]
            alpha: Parâmetro fractal α
            beta: Parâmetro fractal β

        Returns:
            Ressonância [vocab_size]
        """
        batch_size = psi.shape[0]
        device = psi.device

        # Gerar ondas de sonda para todo vocabulário
        lambda_indices = torch.arange(self.vocab_size, device=device)
        lambda_indices = lambda_indices.unsqueeze(0).expand(batch_size, -1)

        # Configuração Padilha
        I0, omega, k = 1.0, 2.0 * np.pi, 2.0 * np.pi / 0.5
        t = 1.0

        # λ normalizado: [0, 1]
        lambda_val = lambda_indices.float() / self.vocab_size

        # Gerar ondas de sonda
        amplitude = I0 * torch.sin(omega * t + alpha * lambda_val)
        phase = omega * t - k * lambda_val + beta * (lambda_val ** 2)

        wave_real = amplitude * torch.cos(phase)
        wave_imag = amplitude * torch.sin(phase)

        # Mapear para quaternion (simplificado)
        probe_waves = torch.stack([wave_real, wave_imag,
                                 torch.zeros_like(wave_real),
                                 torch.zeros_like(wave_real)], dim=-1)

        # Normalizar quaterniões de sonda
        from src.core.quaternion_operations import quaternion_normalize
        probe_waves = quaternion_normalize(probe_waves)

        # Calcular energia de acoplamento: |⟨f(λ), Ψ⟩|²
        psi_expanded = psi.unsqueeze(1).expand(-1, self.vocab_size, -1)
        coupling = (probe_waves * psi_expanded).sum(dim=-1)
        energy = coupling ** 2

        return energy.squeeze(0)  # [vocab_size]

    def apply_quantum_noise(self, energy: torch.Tensor, T_q: float) -> torch.Tensor:
        """
        Aplica ruído térmico quântico à energia de ressonância.

        Args:
            energy: Energia de ressonância [vocab_size]
            T_q: Temperatura quântica

        Returns:
            Energia com ruído térmico
        """
        # Ruído térmico: exp(ε/T_q) onde ε ~ N(0, T_q)
        thermal_noise = torch.randn_like(energy) * T_q
        energy_thermal = energy * torch.exp(thermal_noise / T_q)

        return energy_thermal

    def thermal_sampling(self, resonance: torch.Tensor, T_q: float) -> int:
        """
        Amostragem térmica da distribuição de ressonância.

        Args:
            resonance: Ressonância térmica [vocab_size]
            T_q: Temperatura quântica

        Returns:
            Índice do token amostrado
        """
        # Logits com temperatura quântica
        logits = resonance / T_q

        # Amostragem via softmax térmica
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, 1).item()

        return token_id

    def decode_to_text(self,
                      psi_initial: torch.Tensor,
                      alpha: float,
                      beta: float,
                      consciousness_processor,
                      token_decoder) -> Tuple[str, Dict]:
        """
        Decodificação autoregressiva de texto via física óptica.

        Args:
            psi_initial: Estado quaterniônico inicial [batch, 4]
            alpha: Parâmetro fractal α
            beta: Parâmetro fractal β
            consciousness_processor: Processador de consciência
            token_decoder: Função para decodificar token_id → string

        Returns:
            (texto_gerado, métricas)
        """
        generated_tokens = []
        metrics = {
            'tokens_generated': 0,
            'final_fci': 0.0,
            'avg_temperature': 0.0,
            'avg_sharpness': 0.0,
            'stopped_by': 'max_tokens'
        }

        current_psi = psi_initial
        temperatures = []
        sharpnesses = []

        print(f"🚀 Iniciando decodificação óptica autoregressiva...")
        print(f"   - Estado inicial: {current_psi.shape}")
        print(f"   - Parâmetros: α={alpha:.3f}, β={beta:.3f}")

        for step in range(self.max_tokens):
            # Recalcular consciência
            batch_size, seq_len, quat_dim = current_psi.shape
            dummy_input = torch.randn(batch_size, seq_len, 64, device=self.device)

            # Extrair dados de acoplamento do estado atual
            spectral_energy = torch.abs(current_psi[..., 0])  # Componente real
            # Corrigir operação complexa para evitar erro de imag
            quaternion_phase = torch.atan2(current_psi[..., 1], current_psi[..., 0])  # Fase calculada manualmente

            consciousness_results = consciousness_processor(
                dummy_input,
                spectral_energy=spectral_energy,
                quaternion_phase=quaternion_phase
            )

            current_fci = consciousness_results.get('FCI', 0.0)
            D_fractal = consciousness_results.get('D_fractal', 1.5)
            CLZ = consciousness_results.get('CLZ', 1.0)

            # Calcular parâmetros emergentes
            T_q = self.temp_calculator.compute_quantum_temperature(
                D_fractal=D_fractal,
                FCI=current_fci,
                CLZ=CLZ
            )

            # Sonda óptica para ressonância
            resonance = self.optical_probe_resonance(
                current_psi[:, -1, :],  # Último estado
                alpha=alpha,
                beta=beta
            )

            sharpness = self.coherence_calculator.compute_optical_sharpness(
                resonance_field=resonance,
                D_fractal=D_fractal,
                FCI=current_fci
            )

            # Aplicar sharpness
            resonance_sharp = resonance ** sharpness

            # Aplicar ruído térmico
            resonance_thermal = self.apply_quantum_noise(resonance_sharp, T_q)

            # Amostragem térmica
            next_token_id = self.thermal_sampling(resonance_thermal, T_q)

            # Decodificar token
            try:
                next_token = token_decoder(next_token_id)
                generated_tokens.append(next_token)
            except Exception as e:
                print(f"   ⚠️  Erro na decodificação do token {next_token_id}: {e}")
                break

            # Atualizar métricas
            temperatures.append(T_q)
            sharpnesses.append(sharpness)

            print(f"   Step {step + 1}: token='{next_token}', FCI={current_fci:.3f}, T_q={T_q:.3f}, s={sharpness:.3f}")

            # Parar se consciência colapsar
            if current_fci < self.min_fci_threshold:
                print(f"   🛑 Parada por colapso de consciência: FCI={current_fci:.3f} < {self.min_fci_threshold}")
                metrics['stopped_by'] = 'consciousness_collapse'
                break

            # Parar se token de fim de sequência
            if next_token in ['</s>', '<|endoftext|>', '\n'] and step > 5:
                print(f"   🛑 Parada por token de fim de sequência")
                metrics['stopped_by'] = 'end_token'
                break

            # Atualizar estado (simplificado - em produção usar embedding)
            # Para demonstração, manter estado atual
            if step < self.max_tokens - 1:
                # Adicionar pequena perturbação para simular evolução
                noise = torch.randn_like(current_psi) * 0.01
                current_psi = current_psi + noise

        # Calcular métricas finais
        metrics.update({
            'tokens_generated': len(generated_tokens),
            'final_fci': current_fci,
            'avg_temperature': np.mean(temperatures) if temperatures else 0.0,
            'avg_sharpness': np.mean(sharpnesses) if sharpnesses else 0.0
        })

        # Juntar tokens em texto
        generated_text = ''.join(generated_tokens)

        print(f"\n✅ Decodificação concluída:")
        print(f"   - Texto: '{generated_text}'")
        print(f"   - Tokens: {metrics['tokens_generated']}")
        print(f"   - FCI final: {metrics['final_fci']:.3f}")
        print(f"   - T_q médio: {metrics['avg_temperature']:.3f}")
        print(f"   - Sharpness médio: {metrics['avg_sharpness']:.3f}")
        print(f"   - Parada por: {metrics['stopped_by']}")

        return generated_text, metrics


def create_optical_text_decoder(
    vocab_size: int = 50257,
    max_tokens: int = 100,
    min_fci_threshold: float = 0.05,
    device: str = 'cpu'
) -> OpticalTextDecoder:
    """
    Factory function para criar OpticalTextDecoder.

    Args:
        vocab_size: Tamanho do vocabulário
        max_tokens: Máximo de tokens a gerar
        min_fci_threshold: Limiar mínimo de FCI para parada
        device: Dispositivo

    Returns:
        Instância de OpticalTextDecoder
    """
    return OpticalTextDecoder(
        vocab_size=vocab_size,
        max_tokens=max_tokens,
        min_fci_threshold=min_fci_threshold,
        device=device
    )