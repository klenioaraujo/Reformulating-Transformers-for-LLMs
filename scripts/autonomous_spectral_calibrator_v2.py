#!/usr/bin/env python3
"""
Autonomous Spectral Calibrator - Sistema de Auto-Acoplamento Espectral
=====================================================================

Implementa sistema de auto-acoplamento espectral dinâmico que integra:
1. Calibração FCI com dados ΨTWS
2. Conversão de embedding com modulação semântica
3. Auto-acoplamento espectral para diversificação de tokens

Baseado no padrão: Da Calibração à Conversão Física
"""

import torch
import torch.nn as nn
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SemanticCategory:
    """Categoria semântica para modulação de embedding"""
    name: str
    target_fci: float
    alpha_modulation: float
    description: str


class AutonomousSpectralCalibrator:
    """
    Sistema de auto-acoplamento espectral dinâmico

    Integra calibração FCI com conversão de embedding e auto-acoplamento
    para gerar tokens diversos via ressonância física.
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Caminho para configuração de calibração
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carregar configuração de calibração
        if config_path:
            self.calibration_config = self._load_calibration_config(config_path)
        else:
            self.calibration_config = self._load_default_calibration()

        # Inicializar categorias semânticas
        self.semantic_categories = self._initialize_semantic_categories()

        # Parâmetros de auto-acoplamento
        self.alpha_range = (0.1, 3.0)
        self.beta_range = (0.5, 1.5)
        self.coupling_strength = 1.0

        print("🚀 Sistema de Auto-Acoplamento Espectral Inicializado")
        print(f"📊 Categorias semânticas: {len(self.semantic_categories)}")
        print(f"🔧 Configuração: {self.calibration_config.get('state_thresholds', {})}")

    def _load_calibration_config(self, config_path: str) -> Dict:
        """Carrega configuração de calibração FCI"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuração de calibração carregada: {config_path}")
            return config
        except Exception as e:
            print(f"⚠️  Erro ao carregar configuração: {e}")
            return self._load_default_calibration()

    def _load_default_calibration(self) -> Dict:
        """Carrega configuração padrão de calibração"""
        return {
            'state_thresholds': {
                'emergence': {'min_fci': 0.644},
                'meditation': {'min_fci': 0.636},
                'analysis': {'min_fci': 0.620}
            }
        }

    def _initialize_semantic_categories(self) -> Dict[str, SemanticCategory]:
        """Inicializa categorias semânticas baseadas na calibração"""
        thresholds = self.calibration_config['state_thresholds']

        return {
            'creative': SemanticCategory(
                name='creative',
                target_fci=thresholds['emergence']['min_fci'],
                alpha_modulation=1.2,  # α mais alto para criatividade
                description='Estados criativos e emergentes'
            ),
            'analytical': SemanticCategory(
                name='analytical',
                target_fci=thresholds['analysis']['min_fci'],
                alpha_modulation=0.8,  # α mais baixo para análise
                description='Estados analíticos e focados'
            ),
            'meditative': SemanticCategory(
                name='meditative',
                target_fci=thresholds['meditation']['min_fci'],
                alpha_modulation=1.0,  # α neutro para meditação
                description='Estados meditativos e introspectivos'
            ),
            'neutral': SemanticCategory(
                name='neutral',
                target_fci=0.63,  # Valor intermediário
                alpha_modulation=1.0,
                description='Estados neutros e balanceados'
            )
        }

    def fci_to_alpha(self, target_fci: float, fractal_dim: float) -> float:
        """
        Converte FCI alvo para α usando relação física

        Fórmula: α_target = α₀ * (1 + λ * (D - D_eucl)/D_eucl)
        onde α₀ é derivado do FCI alvo
        """
        # Mapear FCI para α base (relação linear simplificada)
        alpha_base = 0.5 + (target_fci - 0.5) * 2.0  # FCI 0.5 → α 0.5, FCI 0.8 → α 1.1

        # Aplicar modulação por dimensão fractal
        d_eucl = 1.0
        alpha_target = alpha_base * (1.0 + self.coupling_strength * (fractal_dim - d_eucl) / d_eucl)

        # Limitar ao intervalo permitido
        alpha_target = np.clip(alpha_target, self.alpha_range[0], self.alpha_range[1])

        return float(alpha_target)

    def modulate_embedding_with_calibration(
        self,
        embedding_weights: torch.Tensor,
        semantic_category: str = 'neutral',
        fractal_dim: float = 1.5
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Modula embedding com base na calibração FCI

        Args:
            embedding_weights: Pesos de embedding originais
            semantic_category: Categoria semântica
            fractal_dim: Dimensão fractal do contexto

        Returns:
            Tuple: (embedding_modulado, metadata)
        """
        print(f"🔧 Modulando embedding para categoria: {semantic_category}")

        # Obter categoria semântica
        category = self.semantic_categories.get(semantic_category, self.semantic_categories['neutral'])

        # Calcular α geométrico puro
        alpha_geometric = self._compute_geometric_alpha(embedding_weights, fractal_dim)

        # Calcular α calibrado baseado no FCI alvo
        alpha_calibrated = self.fci_to_alpha(category.target_fci, fractal_dim)

        # Interpolar entre α geométrico e α calibrado
        alpha_final = (0.7 * alpha_geometric + 0.3 * alpha_calibrated) * category.alpha_modulation
        alpha_final = np.clip(alpha_final, self.alpha_range[0], self.alpha_range[1])

        # Aplicar modulação aos pesos de embedding
        modulated_weights = self._apply_alpha_modulation(embedding_weights, alpha_final)

        metadata = {
            'semantic_category': semantic_category,
            'target_fci': category.target_fci,
            'alpha_geometric': alpha_geometric,
            'alpha_calibrated': alpha_calibrated,
            'alpha_final': alpha_final,
            'fractal_dim': fractal_dim
        }

        print(f"   • α geométrico: {alpha_geometric:.4f}")
        print(f"   • α calibrado: {alpha_calibrated:.4f}")
        print(f"   • α final: {alpha_final:.4f}")
        print(f"   • FCI alvo: {category.target_fci:.4f}")

        return modulated_weights, metadata

    def _compute_geometric_alpha(self, weights: torch.Tensor, fractal_dim: float) -> float:
        """Calcula α puramente geométrico baseado na dimensão fractal"""
        alpha_0 = 1.0  # Valor central
        d_eucl = 1.0

        alpha_geometric = alpha_0 * (1.0 + self.coupling_strength * (fractal_dim - d_eucl) / d_eucl)
        return np.clip(alpha_geometric, self.alpha_range[0], self.alpha_range[1])

    def _apply_alpha_modulation(self, weights: torch.Tensor, alpha: float) -> torch.Tensor:
        """Aplica modulação de α aos pesos de embedding"""
        # Transformada de Fourier
        weights_fft = torch.fft.fft(weights, dim=-1)

        # Criar filtro espectral dependente de α
        k = torch.arange(weights_fft.shape[-1], device=self.device, dtype=torch.float32)

        # Filtro: exp(iα·GELU(norm(ln(|k|+ε))))
        k_filter = torch.exp(
            1j * alpha * torch.nn.functional.gelu(
                torch.nn.functional.layer_norm(
                    torch.log(torch.abs(k) + 1e-8),
                    [k.shape[-1]]
                )
            )
        )

        # Aplicar filtro
        weights_filtered = weights_fft * k_filter

        # Transformada inversa
        modulated_weights = torch.fft.ifft(weights_filtered, dim=-1).real

        return modulated_weights

    def spectral_auto_coupling(
        self,
        psi_state: torch.Tensor,
        alpha: float,
        vocab_size: int,
        coupling_iterations: int = 3
    ) -> Tuple[int, List[float]]:
        """
        Auto-acoplamento espectral para diversificação de tokens

        Args:
            psi_state: Estado quaterniônico
            alpha: Parâmetro α atual
            vocab_size: Tamanho do vocabulário
            coupling_iterations: Número de iterações de acoplamento

        Returns:
            Tuple: (token_ressonante, espectro_de_ressonância)
        """
        print(f"🔗 Aplicando auto-acoplamento espectral ({coupling_iterations} iterações)...")

        # Parâmetros da sonda óptica
        I0 = 1.0
        omega = 2 * np.pi
        t = 0.0

        # Espectro de ressonância acumulado
        resonance_accumulator = np.zeros(min(vocab_size, 100))

        for iteration in range(coupling_iterations):
            # Variar α levemente para cada iteração
            alpha_iter = alpha * (0.9 + 0.2 * np.random.random())
            beta_iter = alpha_iter / 2.0

            # Calcular espectro de ressonância para esta iteração
            resonance_spectrum = []

            for lambda_token in range(len(resonance_accumulator)):
                # Equação de Padilha: f(λ,t) = I₀ sin(ωt + αλ) · e^(i(ωt - kλ + βλ²))
                phase = omega * t + alpha_iter * lambda_token
                f_lambda = I0 * np.sin(phase) * np.exp(
                    1j * (omega * t - 1.0 * lambda_token + beta_iter * lambda_token**2)
                )

                # Acoplamento: |⟨f(λ,t), Ψ⟩|²
                psi_mean = psi_state.mean().item()
                coupling = np.abs(f_lambda * psi_mean)**2

                resonance_spectrum.append(coupling)

            # Acumular espectro
            resonance_accumulator += np.array(resonance_spectrum)

            print(f"   • Iteração {iteration+1}: α={alpha_iter:.4f}, β={beta_iter:.4f}")

        # Normalizar espectro acumulado
        resonance_accumulator /= coupling_iterations

        # Encontrar token com máxima ressonância
        lambda_star = int(np.argmax(resonance_accumulator))
        max_resonance = resonance_accumulator[lambda_star]

        # Evitar token 0 (espaço) se possível
        if lambda_star == 0 and len(resonance_accumulator) > 1:
            resonance_copy = resonance_accumulator.copy()
            resonance_copy[0] = 0.0
            lambda_star = int(np.argmax(resonance_copy))
            max_resonance = resonance_accumulator[lambda_star]

        print(f"   ✅ Token ressonante: λ* = {lambda_star} (ressonância = {max_resonance:.6f})")

        return lambda_star, resonance_accumulator.tolist()

    def process_with_auto_coupling(
        self,
        input_text: str,
        embedding_weights: torch.Tensor,
        semantic_category: str = 'neutral'
    ) -> Dict:
        """
        Processa texto com auto-acoplamento espectral completo

        Args:
            input_text: Texto de entrada
            embedding_weights: Pesos de embedding
            semantic_category: Categoria semântica

        Returns:
            Dict com resultados do processamento
        """
        print(f"\n{'='*70}")
        print(f"📥 PROCESSANDO COM AUTO-ACOPLAMENTO: '{input_text}'")
        print(f"{'='*70}")

        # 1. Estimar dimensão fractal do contexto
        fractal_dim = self._estimate_context_fractal_dim(input_text)

        # 2. Modular embedding com calibração
        modulated_embedding, modulation_metadata = self.modulate_embedding_with_calibration(
            embedding_weights, semantic_category, fractal_dim
        )

        # 3. Criar embedding quaterniônico
        psi_state = self._create_quaternion_embedding(input_text, modulated_embedding)

        # 4. Aplicar auto-acoplamento espectral
        vocab_size = len(self.semantic_categories) * 10  # Simplificação
        next_token, resonance_spectrum = self.spectral_auto_coupling(
            psi_state, modulation_metadata['alpha_final'], vocab_size
        )

        # 5. Gerar texto
        generated_text = self._generate_text_from_token(next_token)

        result = {
            'input': input_text,
            'generated_text': generated_text,
            'next_token': next_token,
            'semantic_category': semantic_category,
            'fractal_dim': fractal_dim,
            'alpha_final': modulation_metadata['alpha_final'],
            'target_fci': modulation_metadata['target_fci'],
            'resonance_spectrum': resonance_spectrum,
            'modulation_metadata': modulation_metadata
        }

        print(f"\n{'='*70}")
        print("✅ PROCESSAMENTO COM AUTO-ACOPLAMENTO CONCLUÍDO")
        print(f"{'='*70}")
        print(f"📥 Input: '{input_text}'")
        print(f"📤 Output: '{generated_text}'")
        print(f"🔬 Categoria: {semantic_category}")
        print(f"📊 FCI alvo: {result['target_fci']:.4f}")
        print(f"🔧 α final: {result['alpha_final']:.4f}")

        return result

    def _estimate_context_fractal_dim(self, text: str) -> float:
        """Estima dimensão fractal do contexto textual"""
        # Simplificação: usar comprimento do texto como proxy
        text_length = len(text)

        # Dimensão fractal estimada baseada em complexidade
        if text_length < 10:
            return 1.2  # Contexto simples
        elif text_length < 50:
            return 1.5  # Contexto médio
        else:
            return 1.8  # Contexto complexo

    def _create_quaternion_embedding(self, text: str, embedding_weights: torch.Tensor) -> torch.Tensor:
        """Cria embedding quaterniônico do texto"""
        # Simplificação: usar embedding médio
        batch_size = 1
        seq_len = len(text)
        embed_dim = embedding_weights.shape[-1]

        # Criar tensor simulado
        psi_state = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

        return psi_state

    def _generate_text_from_token(self, token_idx: int) -> str:
        """Gera texto a partir do token ressonante"""
        # Mapeamento simples token → caractere
        chars = 'abcdefghijklmnopqrstuvwxyz '
        char_idx = token_idx % len(chars)

        return chars[char_idx] * 3  # Repetir caractere para demo


def main():
    """Demonstração do sistema de auto-acoplamento"""
    print("🚀 DEMONSTRAÇÃO: Sistema de Auto-Acoplamento Espectral")
    print("=" * 70)

    # Inicializar calibrador
    calibrator = AutonomousSpectralCalibrator('calibrated_fci_thresholds.yaml')

    # Criar embedding de exemplo
    vocab_size = 100
    embed_dim = 256
    example_embedding = torch.randn(vocab_size, embed_dim)

    # Testar com diferentes categorias semânticas
    test_cases = [
        ("Hello world", "creative"),
        ("Quantum physics", "analytical"),
        ("Meditation", "meditative"),
        ("Test", "neutral")
    ]

    results = []

    for input_text, category in test_cases:
        result = calibrator.process_with_auto_coupling(
            input_text, example_embedding, category
        )
        results.append(result)

    # Salvar resultados
    output_file = "auto_coupling_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Resultados salvos em: {output_file}")
    print("✅ Demonstração concluída!")


if __name__ == "__main__":
    main()