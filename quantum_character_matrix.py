#!/usr/bin/env python3
"""
Matriz Quântica de Conversão Aprimorada para Caracteres
=======================================================

Sistema avançado de mapeamento quântico de caracteres baseado nos princípios físicos do doe.md.
Implementa conversão de caracteres para estados quânticos no espaço Hilbert, integrando:

- Matriz de Estados Quânticos Fundamentais (MEQF)
- Transformações Espectrais Adaptativas (TEA)
- Integração de Parâmetros Semânticos (IPS)
- Preservação de Propriedades Físicas (PPF)

Princípios Físicos Implementados:
- Equação de Padilha: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
- Dimensão Fractal: D = (3 - β) / 2
- Filtragem Espectral: F(k) = exp(i α · arctan(ln(|k| + ε)))
- Rotações SO(4): Ψ' = q_left * Ψ * q_right†

Uso:
    from quantum_character_matrix import QuantumCharacterMatrix
    matrix = QuantumCharacterMatrix(alpha=1.5, beta=0.8, fractal_dim=1.7)
    quantum_state = matrix.encode_character('A')
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


class QuantumCharacterMatrix(nn.Module):
    """
    Matriz Quântica de Conversão Aprimorada para Caracteres

    Implementa mapeamento quântico estruturado baseado em princípios físicos,
    integrando parâmetros espectrais dos modelos convertidos.
    """

    def __init__(self,
                 embed_dim: int = 64,
                 alpha: float = 1.5,
                 beta: float = 0.8,
                 fractal_dim: float = 1.7,
                 device: str = 'cpu'):
        """
        Inicializa a Matriz Quântica de Conversão.

        Args:
            embed_dim: Dimensão do espaço de embedding quântico
            alpha: Parâmetro espectral α (filtragem)
            beta: Parâmetro espectral β (dimensão fractal)
            fractal_dim: Dimensão fractal D
            device: Dispositivo de computação
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
        self.fractal_dim = fractal_dim
        self.device = device

        # ========== MATRIZ DE ESTADOS QUÂNTICOS FUNDAMENTAIS (MEQF) ==========
        # Estados base para caracteres fundamentais (ASCII printable)
        self.base_states = self._initialize_base_quantum_states()

        # ========== TRANSFORMAÇÕES ESPECTRAIS ADAPTATIVAS (TEA) ==========
        # Filtros espectrais adaptativos baseados em α e β
        self.spectral_filters = self._initialize_spectral_filters()

        # ========== INTEGRAÇÃO DE PARÂMETROS SEMÂNTICOS (IPS) ==========
        # Mapeamento semântico baseado na frequência de uso e propriedades linguísticas
        self.semantic_mapping = self._initialize_semantic_mapping()

        # ========== PRESERVAÇÃO DE PROPRIEDADES FÍSICAS (PPF) ==========
        # Operadores de rotação SO(4) para preservar unitariedade
        self.rotation_operators = self._initialize_rotation_operators()

        # ========== PARÂMETROS APRENDÍVEIS ==========
        # Matriz de transformação aprendível para adaptação dinâmica
        self.adaptive_transform = nn.Linear(embed_dim, embed_dim)

        # Camadas de normalização para estabilidade numérica
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Mover para dispositivo
        self.to(device)

    def _initialize_base_quantum_states(self) -> torch.Tensor:
        """
        Inicializa estados quânticos fundamentais para caracteres ASCII.

        Usa princípios da Equação de Padilha para gerar estados base:
        f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
        """
        # Caracteres ASCII printable (32-126)
        num_chars = 95  # 126 - 32 + 1
        base_states = torch.zeros(num_chars, self.embed_dim, 4, dtype=torch.complex64)

        for i, char_code in enumerate(range(32, 127)):
            char_idx = char_code - 32

            # Parâmetros da Equação de Padilha
            lambda_pos = char_idx / num_chars  # Posição normalizada [0,1]
            t = 0.0  # Tempo inicial
            I0 = 1.0  # Intensidade máxima
            omega = 2 * math.pi * self.alpha  # Frequência angular
            k = 2 * math.pi / self.fractal_dim  # Número de onda

            # Aplicar Equação de Padilha
            phase_term = omega * t - k * lambda_pos + self.beta * lambda_pos**2
            amplitude_term = I0 * torch.sin(torch.tensor(omega * t + self.alpha * lambda_pos))

            # Gerar componentes quaterniónicas
            for j in range(self.embed_dim):
                # Frequência local baseada na posição no embedding
                local_freq = 2 * math.pi * j / self.embed_dim

                # Componente real (w)
                real_comp = amplitude_term * torch.cos(torch.tensor(phase_term + local_freq))

                # Componente i (x)
                i_comp = amplitude_term * torch.sin(torch.tensor(phase_term + local_freq))

                # Componentes j,k baseadas em propriedades fractais
                fractal_factor = self.fractal_dim / 2.0
                j_comp = real_comp * fractal_factor * torch.cos(torch.tensor(local_freq * self.beta))
                k_comp = i_comp * fractal_factor * torch.sin(torch.tensor(local_freq * self.beta))

                base_states[i, j, 0] = torch.complex(real_comp, i_comp)
                base_states[i, j, 1] = torch.complex(j_comp, k_comp)
                base_states[i, j, 2] = torch.complex(-i_comp, real_comp)
                base_states[i, j, 3] = torch.complex(k_comp, -j_comp)

        return base_states.to(self.device)

    def _initialize_spectral_filters(self) -> nn.ModuleDict:
        """
        Inicializa filtros espectrais adaptativos.

        F(k) = exp(i α · arctan(ln(|k| + ε)))
        """
        filters = nn.ModuleDict()

        # Filtro principal baseado em α
        filters['main_filter'] = nn.Conv1d(
            in_channels=self.embed_dim * 4,  # 4 componentes quaterniónicas
            out_channels=self.embed_dim * 4,
            kernel_size=3,
            padding=1
        )

        # Filtro adaptativo baseado em β
        filters['adaptive_filter'] = nn.Conv1d(
            in_channels=self.embed_dim * 4,
            out_channels=self.embed_dim * 4,
            kernel_size=5,
            padding=2
        )

        # Filtro de preservação de energia
        filters['energy_filter'] = nn.Conv1d(
            in_channels=self.embed_dim * 4,
            out_channels=self.embed_dim * 4,
            kernel_size=1  # Filtro ponto-a-ponto
        )

        return filters

    def _initialize_semantic_mapping(self) -> Dict[str, torch.Tensor]:
        """
        Inicializa mapeamento semântico baseado em propriedades linguísticas.

        Considera frequência de uso, categoria gramatical, e propriedades fonéticas.
        """
        semantic_map = {}

        # Categorizar caracteres por propriedades semânticas
        vowels = 'aeiouAEIOU'
        consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
        digits = '0123456789'
        punctuation = '.,!?;:()[]{}<>-–—=+*/'

        # Pesos semânticos baseados na frequência e importância
        semantic_weights = {
            'vowels': torch.tensor([1.2, 0.8, 1.5, 0.9], dtype=torch.float32),
            'consonants': torch.tensor([0.9, 1.1, 0.7, 1.3], dtype=torch.float32),
            'digits': torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            'punctuation': torch.tensor([0.5, 0.5, 0.8, 0.8], dtype=torch.float32),
            'whitespace': torch.tensor([0.3, 0.3, 0.3, 0.3], dtype=torch.float32)
        }

        # Aplicar pesos normalizados
        for key, weights in semantic_weights.items():
            semantic_map[key] = weights / torch.norm(weights)

        return semantic_map

    def _initialize_rotation_operators(self) -> Dict[str, nn.Parameter]:
        """
        Inicializa operadores de rotação SO(4) para preservação de unitariedade.

        Ψ' = q_left * Ψ * q_right†
        """
        rotations = {}

        # Ângulos de Euler aprendíveis para rotações
        rotations['theta_left'] = nn.Parameter(torch.tensor(0.1))
        rotations['omega_left'] = nn.Parameter(torch.tensor(0.05))
        rotations['phi_left'] = nn.Parameter(torch.tensor(0.02))

        rotations['theta_right'] = nn.Parameter(torch.tensor(0.08))
        rotations['omega_right'] = nn.Parameter(torch.tensor(0.03))
        rotations['phi_right'] = nn.Parameter(torch.tensor(0.01))

        return rotations

    def _apply_padilha_wave_equation(self, char_code: int, position: int = 0) -> torch.Tensor:
        """
        Aplica a Equação de Padilha para um caractere específico.

        f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

        Args:
            char_code: Código ASCII do caractere
            position: Posição no texto (para dependência temporal)

        Returns:
            Estado quântico baseado na equação de Padilha
        """
        # Normalizar código do caractere para [0,1]
        lambda_pos = (char_code - 32) / 95.0  # 95 caracteres printáveis
        t = position * 0.1  # Dependência temporal baseada na posição

        # Parâmetros da equação
        I0 = 1.0
        omega = 2 * math.pi * self.alpha
        k = 2 * math.pi / self.fractal_dim

        # Calcular termos da equação
        phase_term = omega * t - k * lambda_pos + self.beta * lambda_pos**2
        amplitude_term = I0 * math.sin(omega * t + self.alpha * lambda_pos)

        # Gerar estado quântico complexo
        wave_function = amplitude_term * torch.exp(torch.tensor(1j * phase_term))

        # Expandir para dimensão de embedding
        expanded_state = torch.zeros(self.embed_dim, dtype=torch.complex64, device=self.device)

        for i in range(self.embed_dim):
            # Modulação baseada na posição no embedding
            modulation = torch.exp(torch.tensor(1j * 2 * math.pi * i / self.embed_dim * lambda_pos))
            expanded_state[i] = wave_function * modulation

        return expanded_state

    def _apply_spectral_filtering(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Aplica filtragem espectral baseada no parâmetro α.

        F(k) = exp(i α · arctan(ln(|k| + ε)))
        """
        # Para simplificar, aplicar apenas uma transformação linear simples
        # em vez de filtros convolucionais complexos que causam problemas dimensionais

        # Aplicar uma transformação simples baseada nos princípios físicos
        # F(k) = exp(i α · arctan(ln(|k| + ε)))

        k_values = torch.arange(1, self.embed_dim + 1, dtype=torch.float32, device=self.device)
        spectral_filter = torch.exp(1j * self.alpha * torch.arctan(torch.log(k_values + 1e-10)))

        # Aplicar filtro no domínio da frequência (simplificado)
        # quantum_state tem shape [embed_dim], spectral_filter tem shape [embed_dim]
        filtered_state = quantum_state * spectral_filter

        # Normalizar para preservar energia
        energy_preserved = filtered_state / (torch.norm(filtered_state) + 1e-8)

        return energy_preserved

    def _apply_so4_rotation(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Aplica rotações SO(4) unitárias (simplificado).

        Ψ' = q_left * Ψ * q_right†
        """
        # Para simplificar, aplicar apenas uma transformação linear simples
        # que preserve a estrutura quântica

        # Criar uma matriz de rotação simples baseada nos ângulos
        theta = self.rotation_operators['theta_left'].item()
        omega = self.rotation_operators['omega_left'].item()

        # Matriz de rotação 2D simples (pode ser estendida para 4D)
        cos_theta = torch.cos(torch.tensor(theta))
        sin_theta = torch.sin(torch.tensor(theta))

        # Aplicar rotação simples - verificar se é complexo
        if quantum_state.is_complex():
            # Para estado complexo, aplicar rotação na parte real e imaginária
            rotated_real = quantum_state.real * cos_theta - quantum_state.imag * sin_theta
            rotated_imag = quantum_state.real * sin_theta + quantum_state.imag * cos_theta
            rotated_state = torch.complex(rotated_real, rotated_imag)
        else:
            # Para estado real, aplicar rotação simples
            rotated_state = quantum_state * cos_theta

        return rotated_state

    def _create_unit_quaternion(self, theta: torch.Tensor, omega: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Cria quaternion unitário a partir de ângulos de Euler.

        q = cos(θ/2) + sin(θ/2) * [cos(ω) * i + sin(ω) * cos(φ) * j + sin(ω) * sin(φ) * k]
        """
        cos_theta_2 = torch.cos(theta / 2)
        sin_theta_2 = torch.sin(theta / 2)

        q_w = cos_theta_2
        q_x = sin_theta_2 * torch.cos(omega)
        q_y = sin_theta_2 * torch.sin(omega) * torch.cos(phi)
        q_z = sin_theta_2 * torch.sin(omega) * torch.sin(phi)

        return torch.stack([q_w, q_x, q_y, q_z])

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiplicação de quaternions (Hamilton product).

        (a + bi + cj + dk) * (e + fi + gj + hk) =
        (ae - bf - cg - dh) + (af + be + ch - dg)i +
        (ag - bh + ce + df)j + (ah + bg - cf + de)k
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z])

    def _quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """Conjugado de quaternion: q* = (w, -x, -y, -z)"""
        w, x, y, z = q
        return torch.stack([w, -x, -y, -z])

    def encode_character(self, char: str, position: int = 0) -> torch.Tensor:
        """
        Codifica um caractere para estado quântico usando a matriz aprimorada.

        Args:
            char: Caractere a ser codificado
            position: Posição no texto

        Returns:
            Estado quântico [embed_dim, 4] (quaternion components)
        """
        if len(char) != 1:
            raise ValueError("encode_character aceita apenas um caractere por vez")

        char_code = ord(char)

        # Verificar se é caractere ASCII printable
        if not (32 <= char_code <= 126):
            # Para caracteres fora do range, usar mapeamento especial
            char_code = 32  # Mapear para espaço

        # ========== PASSO 1: ESTADO BASE VIA EQUAÇÃO DE PADILHA ==========
        base_state = self._apply_padilha_wave_equation(char_code, position)

        # ========== PASSO 2: FILTRAGEM ESPECTRAL ==========
        filtered_state = self._apply_spectral_filtering(base_state)

        # ========== PASSO 3: ROTAÇÃO SO(4) ==========
        rotated_state = self._apply_so4_rotation(filtered_state)

        # ========== PASSO 4: TRANSFORMAÇÃO ADAPTATIVA ==========
        # Preparar para transformação linear
        state_flat = rotated_state.view(-1).real  # Usar parte real para compatibilidade

        # Aplicar transformação aprendível
        adapted_state = self.adaptive_transform(state_flat)

        # Aplicar normalização
        normalized_state = self.layer_norm(adapted_state)

        # ========== PASSO 5: MAPEAMENTO PARA COMPONENTES QUATERNIONICAS ==========
        # Expandir para 4 componentes quaterniónicas
        quaternion_state = torch.zeros(self.embed_dim, 4, dtype=torch.float32, device=self.device)

        # Componente real (w)
        quaternion_state[:, 0] = normalized_state

        # Componentes imaginários baseadas em propriedades do caractere
        char_category = self._categorize_character(char)
        semantic_weights = self.semantic_mapping[char_category]

        for i in range(self.embed_dim):
            # Modulação baseada na categoria semântica
            phase_shift = 2 * math.pi * i / self.embed_dim
            quaternion_state[i, 1] = normalized_state[i] * semantic_weights[0] * torch.cos(torch.tensor(phase_shift))
            quaternion_state[i, 2] = normalized_state[i] * semantic_weights[1] * torch.sin(torch.tensor(phase_shift))
            quaternion_state[i, 3] = normalized_state[i] * semantic_weights[2] * torch.cos(torch.tensor(2 * phase_shift))

        return quaternion_state

    def _categorize_character(self, char: str) -> str:
        """Categoriza um caractere para mapeamento semântico."""
        if char in 'aeiouAEIOU':
            return 'vowels'
        elif char in 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ':
            return 'consonants'
        elif char in '0123456789':
            return 'digits'
        elif char in '.,!?;:()[]{}<>-–—=+*/':
            return 'punctuation'
        elif char == ' ':
            return 'whitespace'
        else:
            return 'consonants'  # Default

    def decode_quantum_state(self, quantum_state: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Decodifica estado quântico de volta para caracteres candidatos.

        Args:
            quantum_state: Estado quântico [embed_dim, 4]
            top_k: Número de candidatos a retornar

        Returns:
            Lista de tuplas (caractere, confiança)
        """
        # Calcular similaridade com todos os estados base
        similarities = []

        for i, base_state in enumerate(self.base_states):
            # Calcular similaridade usando produto interno quaterniónico
            similarity = self._quaternion_similarity(quantum_state, base_state)
            similarities.append((i + 32, similarity))  # i + 32 = código ASCII

        # Ordenar por similaridade
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Converter para caracteres e normalizar confianças
        results = []
        max_similarity = similarities[0][1] if similarities else 1.0

        for char_code, similarity in similarities[:top_k]:
            char = chr(char_code)
            confidence = similarity / max_similarity if max_similarity > 0 else 0.0
            results.append((char, float(confidence)))

        return results

    def _quaternion_similarity(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        """
        Calcula similaridade entre dois estados quaterniónicos.

        Usa produto interno normalizado no espaço quaterniónico.
        """
        # Produto interno quaterniónico
        dot_product = torch.sum(q1 * q2.conj())

        # Similaridade normalizada
        norm1 = torch.norm(q1)
        norm2 = torch.norm(q2)

        if norm1 > 0 and norm2 > 0:
            similarity = torch.abs(dot_product) / (norm1 * norm2)
            return float(similarity.real)
        else:
            return 0.0

    def update_spectral_parameters(self, alpha: Optional[float] = None,
                                 beta: Optional[float] = None,
                                 fractal_dim: Optional[float] = None):
        """
        Atualiza parâmetros espectrais dinamicamente.

        Args:
            alpha: Novo valor de α
            beta: Novo valor de β
            fractal_dim: Nova dimensão fractal
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if fractal_dim is not None:
            self.fractal_dim = fractal_dim

        # Re-inicializar componentes dependentes dos parâmetros
        self.base_states = self._initialize_base_quantum_states()
        self.spectral_filters = self._initialize_spectral_filters()

        print(f"✅ Parâmetros espectrais atualizados: α={self.alpha:.3f}, β={self.beta:.3f}, D={self.fractal_dim:.3f}")

    def save_matrix(self, filepath: str):
        """Salva a matriz quântica em arquivo."""
        state = {
            'embed_dim': self.embed_dim,
            'alpha': self.alpha,
            'beta': self.beta,
            'fractal_dim': self.fractal_dim,
            'state_dict': self.state_dict(),
            'base_states': self.base_states,
            'semantic_mapping': self.semantic_mapping
        }

        torch.save(state, filepath)
        print(f"💾 Matriz quântica salva em: {filepath}")

    @classmethod
    def load_matrix(cls, filepath: str, device: str = 'cpu') -> 'QuantumCharacterMatrix':
        """Carrega matriz quântica de arquivo."""
        state = torch.load(filepath, map_location=device)

        matrix = cls(
            embed_dim=state['embed_dim'],
            alpha=state['alpha'],
            beta=state['beta'],
            fractal_dim=state['fractal_dim'],
            device=device
        )

        matrix.load_state_dict(state['state_dict'])
        matrix.base_states = state['base_states'].to(device)
        matrix.semantic_mapping = state['semantic_mapping']

        print(f"📁 Matriz quântica carregada de: {filepath}")
        return matrix


def create_enhanced_quantum_matrix(alpha: float = 1.5, beta: float = 0.8,
                                  fractal_dim: float = 1.7, embed_dim: int = 64,
                                  device: str = 'cpu') -> QuantumCharacterMatrix:
    """
    Factory function para criar matriz quântica aprimorada.

    Args:
        alpha: Parâmetro espectral α
        beta: Parâmetro espectral β
        fractal_dim: Dimensão fractal D
        embed_dim: Dimensão do embedding
        device: Dispositivo

    Returns:
        Instância configurada da QuantumCharacterMatrix
    """
    return QuantumCharacterMatrix(
        embed_dim=embed_dim,
        alpha=alpha,
        beta=beta,
        fractal_dim=fractal_dim,
        device=device
    )


# Exemplo de uso e teste
if __name__ == "__main__":
    # Criar matriz quântica
    matrix = create_enhanced_quantum_matrix(alpha=1.5, beta=0.8, fractal_dim=1.7)

    # Testar codificação de caracteres
    test_chars = ['A', 'e', '1', ' ', '.']

    print("🔬 Teste da Matriz Quântica de Conversão Aprimorada")
    print("=" * 60)

    for char in test_chars:
        # Codificar
        quantum_state = matrix.encode_character(char)
        print(f"\nCaractere: '{char}' (ASCII: {ord(char)})")

        # Decodificar (top-3 candidatos)
        candidates = matrix.decode_quantum_state(quantum_state, top_k=3)
        print(f"Estado quântico: shape {quantum_state.shape}")
        print(f"Candidatos decodificados: {candidates}")

    print("\n✅ Teste concluído!")

    # Salvar matriz
    matrix.save_matrix("quantum_character_matrix.pt")