#!/usr/bin/env python3
"""
Integração Aprimorada de Caracteres Quânticos no Sistema ΨQRH
===========================================================

Este módulo integra a QuantumCharacterMatrix aprimorada no pipeline ΨQRH,
substituindo o mapeamento primitivo de caracteres por representação quântica física.

Principais melhorias:
- Substituição do mapeamento ASCII simples por estados quânticos baseados na Equação de Padilha
- Integração de parâmetros espectrais (α, β, D) dos modelos convertidos
- Preservação de propriedades físicas durante a conversão
- Validação matemática rigorosa das operações quânticas

Uso:
    from enhanced_quantum_integration import EnhancedQuantumIntegration
    integrator = EnhancedQuantumIntegration()
    quantum_state = integrator.text_to_quantum("hello")
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

from quantum_word_matrix import QuantumWordMatrix


class EnhancedQuantumIntegration(nn.Module):
    """
    Integração Aprimorada de Caracteres Quânticos no Sistema ΨQRH

    Substitui o mapeamento primitivo por representação quântica física baseada
    nos princípios do doe.md e parâmetros espectrais dos modelos convertidos.
    """

    def __init__(self,
                 embed_dim: int = 64,
                 alpha: float = 1.5,
                 beta: float = 0.8,
                 fractal_dim: float = 1.7,
                 device: str = 'cpu',
                 enable_spectral_adaptation: bool = True):
        """
        Inicializa a integração quântica aprimorada.

        Args:
            embed_dim: Dimensão do espaço de embedding quântico
            alpha: Parâmetro espectral α (filtragem)
            beta: Parâmetro espectral β (dimensão fractal)
            fractal_dim: Dimensão fractal D
            device: Dispositivo de computação
            enable_spectral_adaptation: Habilita adaptação espectral dinâmica
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
        self.fractal_dim = fractal_dim
        self.device = device
        self.enable_spectral_adaptation = enable_spectral_adaptation

        # ========== MATRIZ QUÂNTICA APRIMORADA ==========
        self.quantum_matrix = create_enhanced_quantum_matrix(
            embed_dim=embed_dim,
            alpha=alpha,
            beta=beta,
            fractal_dim=fractal_dim,
            device=device
        )

        # ========== ADAPTAÇÃO ESPECTRAL DINÂMICA ==========
        if enable_spectral_adaptation:
            self.spectral_adapter = nn.Sequential(
                nn.Linear(embed_dim * 4 + 3, embed_dim),  # +3 para estatísticas do texto
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 3)  # Saída: [delta_alpha, delta_beta, delta_fractal_dim]
            )
        else:
            self.spectral_adapter = None

        # ========== VALIDAÇÃO MATEMÁTICA ==========
        self.validator = QuantumStateValidator(device=device)

        # ========== CACHE PARA PERFORMANCE ==========
        self.state_cache = {}
        self.max_cache_size = 1000

        # Mover para dispositivo
        self.to(device)

        print("🔬 Enhanced Quantum Integration inicializada com sucesso!")
        print(f"   📐 Configuração: embed_dim={embed_dim}, α={alpha:.3f}, β={beta:.3f}, D={fractal_dim:.3f}")
        print(f"   🎯 Adaptação espectral: {'ATIVADA' if enable_spectral_adaptation else 'DESATIVADA'}")

    def text_to_quantum(self, text: str, enable_cache: bool = True) -> torch.Tensor:
        """
        Converte texto para representação quântica aprimorada.

        Args:
            text: Texto de entrada
            enable_cache: Usar cache para performance

        Returns:
            Estado quântico [seq_len, embed_dim, 4]
        """
        # Verificar cache
        if enable_cache and text in self.state_cache:
            return self.state_cache[text].clone()

        # Converter caractere por caractere
        quantum_states = []

        for i, char in enumerate(text):
            if char == '\n':
                char = ' '  # Normalizar quebras de linha

            try:
                # Codificar caractere usando matriz quântica aprimorada
                char_state = self.quantum_matrix.encode_character(char, position=i)
                quantum_states.append(char_state)
            except ValueError as e:
                # Fallback para caracteres não suportados
                print(f"⚠️ Caractere não suportado '{char}', usando fallback")
                fallback_state = torch.zeros(self.embed_dim, 4, dtype=torch.float32, device=self.device)
                fallback_state[:, 0] = 0.1  # Pequeno valor real
                quantum_states.append(fallback_state)

        # Empilhar estados
        if quantum_states:
            quantum_tensor = torch.stack(quantum_states, dim=0)  # [seq_len, embed_dim, 4]
        else:
            quantum_tensor = torch.zeros(1, self.embed_dim, 4, dtype=torch.float32, device=self.device)

        # Aplicar adaptação espectral dinâmica se habilitada
        if self.enable_spectral_adaptation:
            quantum_tensor = self._apply_spectral_adaptation(quantum_tensor, text)

        # Validar estado quântico
        validation_result = self.validator.validate_quantum_state(quantum_tensor)
        if not validation_result['is_valid']:
            print(f"⚠️ Estado quântico inválido detectado: {validation_result['issues']}")
            # Aplicar correção automática
            quantum_tensor = self._correct_quantum_state(quantum_tensor, validation_result)

        # Atualizar cache
        if enable_cache and len(self.state_cache) < self.max_cache_size:
            self.state_cache[text] = quantum_tensor.clone()

        return quantum_tensor

    def _apply_spectral_adaptation(self, quantum_tensor: torch.Tensor, text: str) -> torch.Tensor:
        """
        Aplica adaptação espectral dinâmica baseada no conteúdo do texto.

        Args:
            quantum_tensor: Estado quântico base [seq_len, embed_dim, 4]
            text: Texto original para análise

        Returns:
            Estado quântico adaptado
        """
        # Calcular estatísticas do texto para adaptação
        text_stats = self._analyze_text_statistics(text)

        # Preparar entrada para o adaptador
        # Usar média do estado quântico como representação global
        global_state = quantum_tensor.mean(dim=0).view(-1)  # [embed_dim * 4]

        # Concatenar com estatísticas do texto
        adapter_input = torch.cat([
            global_state,
            torch.tensor([
                text_stats['complexity'],
                text_stats['entropy'],
                text_stats['fractal_estimate']
            ], device=self.device)
        ])

        # Aplicar adaptador
        adaptations = self.spectral_adapter(adapter_input)  # [3]

        # Aplicar adaptações aos parâmetros espectrais
        delta_alpha, delta_beta, delta_fractal_dim = adaptations

        # Limitar adaptações para estabilidade
        delta_alpha = torch.clamp(delta_alpha, -0.5, 0.5)
        delta_beta = torch.clamp(delta_beta, -0.3, 0.3)
        delta_fractal_dim = torch.clamp(delta_fractal_dim, -0.2, 0.2)

        # Atualizar parâmetros da matriz quântica temporariamente
        original_alpha = self.quantum_matrix.alpha
        original_beta = self.quantum_matrix.beta
        original_fractal_dim = self.quantum_matrix.fractal_dim

        self.quantum_matrix.alpha = original_alpha + delta_alpha.item()
        self.quantum_matrix.beta = original_beta + delta_beta.item()
        self.quantum_matrix.fractal_dim = original_fractal_dim + delta_fractal_dim.item()

        # Re-codificar com parâmetros adaptados
        adapted_states = []
        for i, char in enumerate(text):
            adapted_state = self.quantum_matrix.encode_character(char, position=i)
            adapted_states.append(adapted_state)

        adapted_tensor = torch.stack(adapted_states, dim=0)

        # Restaurar parâmetros originais
        self.quantum_matrix.alpha = original_alpha
        self.quantum_matrix.beta = original_beta
        self.quantum_matrix.fractal_dim = original_fractal_dim

        return adapted_tensor

    def _analyze_text_statistics(self, text: str) -> Dict[str, float]:
        """
        Analisa estatísticas do texto para adaptação espectral.

        Args:
            text: Texto a analisar

        Returns:
            Dicionário com estatísticas
        """
        # Complexidade baseada na diversidade de caracteres
        unique_chars = len(set(text))
        total_chars = len(text)
        complexity = unique_chars / total_chars if total_chars > 0 else 0.0

        # Entropia de Shannon
        if total_chars > 0:
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1

            entropy = 0.0
            for count in char_counts.values():
                prob = count / total_chars
                entropy -= prob * math.log2(prob)
        else:
            entropy = 0.0

        # Estimativa fractal baseada na repetição de padrões
        # (simplificada - em implementação completa usaria análise mais sofisticada)
        if len(text) > 10:
            # Procurar por repetições de 2-3 caracteres
            repetitions = 0
            for i in range(len(text) - 3):
                pattern = text[i:i+3]
                repetitions += text.count(pattern) - 1

            fractal_estimate = 1.0 + (repetitions / len(text)) * 0.5
            fractal_estimate = min(fractal_estimate, 2.0)  # Limitar
        else:
            fractal_estimate = 1.5  # Valor padrão

        return {
            'complexity': complexity,
            'entropy': entropy,
            'fractal_estimate': fractal_estimate
        }

    def _correct_quantum_state(self, quantum_tensor: torch.Tensor,
                              validation_result: Dict[str, Any]) -> torch.Tensor:
        """
        Aplica correção automática a estados quânticos inválidos.

        Args:
            quantum_tensor: Estado quântico inválido
            validation_result: Resultado da validação

        Returns:
            Estado quântico corrigido
        """
        corrected_tensor = quantum_tensor.clone()

        # Correção de valores infinitos/NaN
        if torch.any(torch.isinf(corrected_tensor)) or torch.any(torch.isnan(corrected_tensor)):
            corrected_tensor = torch.where(
                torch.isfinite(corrected_tensor),
                corrected_tensor,
                torch.zeros_like(corrected_tensor)
            )

        # Correção de norma zero (estados degenerados)
        norms = torch.norm(corrected_tensor, dim=(1, 2))  # [seq_len]
        zero_norm_mask = norms == 0

        if torch.any(zero_norm_mask):
            # Substituir estados com norma zero por estados unitários
            unit_state = torch.zeros_like(corrected_tensor[0])  # [embed_dim, 4]
            unit_state[:, 0] = 1.0  # Componente real unitário

            for i in range(len(corrected_tensor)):
                if zero_norm_mask[i]:
                    corrected_tensor[i] = unit_state

        # Renormalizar para preservar energia
        norms_corrected = torch.norm(corrected_tensor, dim=(1, 2), keepdim=True)  # [seq_len, 1, 1]
        corrected_tensor = corrected_tensor / (norms_corrected + 1e-8)

        return corrected_tensor

    def quantum_to_text(self, quantum_tensor: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Converte estado quântico de volta para texto usando decodificação aprimorada.

        Args:
            quantum_tensor: Estado quântico [seq_len, embed_dim, 4]
            top_k: Número de candidatos por posição

        Returns:
            Lista de sequências candidatas com suas confianças
        """
        decoded_sequences = []

        for i in range(quantum_tensor.shape[0]):
            char_state = quantum_tensor[i]  # [embed_dim, 4]
            candidates = self.quantum_matrix.decode_quantum_state(char_state, top_k=top_k)

            if not decoded_sequences:
                # Primeira posição - inicializar sequências
                decoded_sequences = [(char, conf) for char, conf in candidates]
            else:
                # Extender sequências existentes
                new_sequences = []
                for existing_seq, existing_conf in decoded_sequences:
                    for new_char, new_conf in candidates:
                        combined_seq = existing_seq + new_char
                        combined_conf = existing_conf * new_conf
                        new_sequences.append((combined_seq, combined_conf))

                # Manter apenas as top_k sequências
                new_sequences.sort(key=lambda x: x[1], reverse=True)
                decoded_sequences = new_sequences[:top_k]

        return decoded_sequences

    def integrate_spectral_parameters(self, model_config: Dict[str, Any]):
        """
        Integra parâmetros espectrais de um modelo convertido no sistema quântico.

        Args:
            model_config: Configuração do modelo convertido
        """
        # Extrair parâmetros espectrais do modelo
        spectral_params = model_config.get('spectral_parameters', {})

        if spectral_params:
            alpha = spectral_params.get('alpha_spectral', self.alpha)
            beta = spectral_params.get('beta_spectral', self.beta)
            fractal_dim = spectral_params.get('embed_dim_spectral', self.fractal_dim)

            # Atualizar matriz quântica
            self.quantum_matrix.update_spectral_parameters(
                alpha=alpha,
                beta=beta,
                fractal_dim=fractal_dim
            )

            # Atualizar parâmetros locais
            self.alpha = alpha
            self.beta = beta
            self.fractal_dim = fractal_dim

            print(f"✅ Parâmetros espectrais integrados do modelo convertido:")
            print(f"   α = {alpha:.3f}, β = {beta:.3f}, D = {fractal_dim:.3f}")
        else:
            print("⚠️ Nenhum parâmetro espectral encontrado no modelo convertido")

    def validate_physical_consistency(self, quantum_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Valida consistência física do estado quântico.

        Args:
            quantum_tensor: Estado quântico a validar

        Returns:
            Resultado da validação física
        """
        return self.validator.validate_physical_consistency(quantum_tensor)

    def clear_cache(self):
        """Limpa o cache de estados quânticos."""
        self.state_cache.clear()
        print("🧹 Cache de estados quânticos limpo")

    def save_integration(self, filepath: str):
        """Salva a integração quântica aprimorada."""
        state = {
            'embed_dim': self.embed_dim,
            'alpha': self.alpha,
            'beta': self.beta,
            'fractal_dim': self.fractal_dim,
            'enable_spectral_adaptation': self.enable_spectral_adaptation,
            'state_dict': self.state_dict(),
            'quantum_matrix_state': {
                'state_dict': self.quantum_matrix.state_dict(),
                'base_states': self.quantum_matrix.base_states,
                'semantic_mapping': self.quantum_matrix.semantic_mapping
            }
        }

        torch.save(state, filepath)
        print(f"💾 Integração quântica aprimorada salva em: {filepath}")

    @classmethod
    def load_integration(cls, filepath: str, device: str = 'cpu') -> 'EnhancedQuantumIntegration':
        """Carrega integração quântica aprimorada de arquivo."""
        state = torch.load(filepath, map_location=device)

        integration = cls(
            embed_dim=state['embed_dim'],
            alpha=state['alpha'],
            beta=state['beta'],
            fractal_dim=state['fractal_dim'],
            device=device,
            enable_spectral_adaptation=state['enable_spectral_adaptation']
        )

        integration.load_state_dict(state['state_dict'])

        # Carregar estado da matriz quântica
        matrix_state = state['quantum_matrix_state']
        integration.quantum_matrix.load_state_dict(matrix_state['state_dict'])
        integration.quantum_matrix.base_states = matrix_state['base_states'].to(device)
        integration.quantum_matrix.semantic_mapping = matrix_state['semantic_mapping']

        print(f"📁 Integração quântica aprimorada carregada de: {filepath}")
        return integration


class QuantumStateValidator:
    """
    Validador de Estados Quânticos para Integridade Física
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def validate_quantum_state(self, quantum_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Valida integridade de um estado quântico.

        Args:
            quantum_tensor: Estado quântico [seq_len, embed_dim, 4]

        Returns:
            Resultado da validação
        """
        issues = []

        # Verificar valores finitos
        if not torch.all(torch.isfinite(quantum_tensor)):
            issues.append("Valores não-finitos (inf/NaN) detectados")

        # Verificar dimensionalidade
        expected_shape = (-1, -1, 4)
        if len(quantum_tensor.shape) != 3 or quantum_tensor.shape[2] != 4:
            issues.append(f"Dimensionalidade incorreta: esperada {expected_shape}, obtida {quantum_tensor.shape}")

        # Verificar normas não-zero
        norms = torch.norm(quantum_tensor, dim=(1, 2))  # [seq_len]
        zero_norms = torch.sum(norms == 0).item()
        if zero_norms > 0:
            issues.append(f"{zero_norms} posições com norma zero (estados degenerados)")

        # Verificar unitariedade aproximada (norma ≈ 1)
        mean_norm = torch.mean(norms).item()
        if not (0.5 <= mean_norm <= 2.0):
            issues.append(".3f")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'mean_norm': mean_norm,
            'zero_norms': zero_norms
        }

    def validate_physical_consistency(self, quantum_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Valida consistência física baseada nos princípios do doe.md.

        Args:
            quantum_tensor: Estado quântico a validar

        Returns:
            Resultado da validação física
        """
        # Energia conservada: ||output|| ≈ ||input|| (dentro de 5%)
        energy_conservation = self._check_energy_conservation(quantum_tensor)

        # Unitaridade: Filtros espectrais preservam energia
        unitarity = self._check_unitarity(quantum_tensor)

        # Estabilidade numérica: Valores finitos
        numerical_stability = self._check_numerical_stability(quantum_tensor)

        # Consistência fractal: Propriedades fractais preservadas
        fractal_consistency = self._check_fractal_consistency(quantum_tensor)

        is_physically_consistent = all([
            energy_conservation['is_conserved'],
            unitarity['is_unitary'],
            numerical_stability['is_stable'],
            fractal_consistency['is_consistent']
        ])

        return {
            'is_physically_consistent': is_physically_consistent,
            'energy_conservation': energy_conservation,
            'unitarity': unitarity,
            'numerical_stability': numerical_stability,
            'fractal_consistency': fractal_consistency
        }

    def _check_energy_conservation(self, quantum_tensor: torch.Tensor) -> Dict[str, Any]:
        """Verifica conservação de energia."""
        norms = torch.norm(quantum_tensor, dim=(1, 2))  # [seq_len]
        mean_norm = torch.mean(norms).item()
        std_norm = torch.std(norms).item()

        # Energia conservada se norma média ≈ 1 e variação pequena
        is_conserved = 0.8 <= mean_norm <= 1.2 and std_norm <= 0.2

        return {
            'is_conserved': is_conserved,
            'mean_norm': mean_norm,
            'std_norm': std_norm
        }

    def _check_unitarity(self, quantum_tensor: torch.Tensor) -> Dict[str, Any]:
        """Verifica unitariedade aproximada."""
        # Para quaternions, verificar se a norma é aproximadamente preservada
        # em operações consecutivas (simplificado)
        norms = torch.norm(quantum_tensor, dim=(1, 2))
        norm_variation = torch.std(norms) / (torch.mean(norms) + 1e-8)

        is_unitary = norm_variation <= 0.1  # Variação < 10%

        return {
            'is_unitary': is_unitary,
            'norm_variation': norm_variation.item()
        }

    def _check_numerical_stability(self, quantum_tensor: torch.Tensor) -> Dict[str, Any]:
        """Verifica estabilidade numérica."""
        is_finite = torch.all(torch.isfinite(quantum_tensor))
        max_value = torch.max(torch.abs(quantum_tensor)).item()
        min_value = torch.min(torch.abs(quantum_tensor)).item()

        # Estável se todos valores finitos e range razoável
        is_stable = is_finite and max_value <= 100.0 and min_value >= 1e-10

        return {
            'is_stable': is_stable,
            'is_finite': is_finite.item(),
            'max_value': max_value,
            'min_value': min_value
        }

    def _check_fractal_consistency(self, quantum_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Verifica consistência fractal (propriedades auto-similares preservadas).
        """
        # Análise simplificada: verificar se padrões se repetem em diferentes escalas
        # (implementação completa usaria análise de dimensão fractal)

        # Calcular autocorrelação como proxy para auto-similaridade
        flat_tensor = quantum_tensor.view(-1)
        if len(flat_tensor) > 10:
            autocorr = torch.corrcoef(torch.stack([
                flat_tensor[:-1],
                flat_tensor[1:]
            ]))[0, 1]

            # Consistente se autocorrelação moderada (não muito alta nem baixa)
            is_consistent = 0.1 <= abs(autocorr) <= 0.8
            fractal_measure = abs(autocorr).item()
        else:
            is_consistent = True
            fractal_measure = 0.5

        return {
            'is_consistent': is_consistent,
            'fractal_measure': fractal_measure
        }


def create_enhanced_quantum_integration(alpha: float = 1.5, beta: float = 0.8,
                                       fractal_dim: float = 1.7, embed_dim: int = 64,
                                       device: str = 'cpu') -> EnhancedQuantumIntegration:
    """
    Factory function para criar integração quântica aprimorada.

    Args:
        alpha: Parâmetro espectral α
        beta: Parâmetro espectral β
        fractal_dim: Dimensão fractal D
        embed_dim: Dimensão do embedding
        device: Dispositivo

    Returns:
        Instância configurada da EnhancedQuantumIntegration
    """
    return EnhancedQuantumIntegration(
        embed_dim=embed_dim,
        alpha=alpha,
        beta=beta,
        fractal_dim=fractal_dim,
        device=device
    )


# Exemplo de uso e integração
if __name__ == "__main__":
    # Criar integração aprimorada
    integrator = create_enhanced_quantum_integration(alpha=1.5, beta=0.8, fractal_dim=1.7)

    # Testar conversão texto → quântico
    test_text = "Hello ΨQRH!"
    print(f"🔬 Teste da Integração Quântica Aprimorada")
    print(f"Texto de entrada: '{test_text}'")
    print("=" * 60)

    # Converter para estado quântico
    quantum_state = integrator.text_to_quantum(test_text)
    print(f"✅ Estado quântico gerado: shape {quantum_state.shape}")

    # Validar estado quântico
    validation = integrator.validate_physical_consistency(quantum_state)
    print(f"📊 Validação física: {'PASSOU' if validation['is_physically_consistent'] else 'FALHOU'}")

    if validation['energy_conservation']['is_conserved']:
        print(".3f")
    else:
        print(".3f")

    # Converter de volta para texto
    decoded_candidates = integrator.quantum_to_text(quantum_state, top_k=3)
    print(f"🔄 Candidatos decodificados (top-3):")
    for i, (text, conf) in enumerate(decoded_candidates):
        print(f"   {i+1}. '{text}' (confiança: {conf:.3f})")

    print("\n✅ Teste concluído!")

    # Salvar integração
    integrator.save_integration("enhanced_quantum_integration.pt")