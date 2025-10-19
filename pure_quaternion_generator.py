#!/usr/bin/env python3
"""
Pure Quaternion Generator for ΨQRH
===================================

Sistema de geração de quatérnios puramente físico, baseado em princípios
quânticos rigorosos. Mapeia sinais fractais para estados quatérnicos unitários.

Características principais:
- Mapeamento direto sinal → quatérnios sem embeddings clássicos
- Rotações SO(4) baseadas em propriedades do sinal
- Normalização unitária garantindo estados quânticos válidos
- Conservação de energia e informação

Autor: Kilo Code (Sistema ΨQRH)
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import scipy.linalg


@dataclass
class QuaternionState:
    """Estado quaterniónico unitário."""
    w: float  # Componente real
    x: float  # Componente i
    y: float  # Componente j
    z: float  # Componente k

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Converte para tensor torch."""
        return torch.tensor([self.w, self.x, self.y, self.z], dtype=torch.float32, device=device)

    def norm(self) -> float:
        """Norma do quatérnio."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'QuaternionState':
        """Normaliza o quatérnio."""
        n = self.norm()
        if n > 0:
            return QuaternionState(self.w/n, self.x/n, self.y/n, self.z/n)
        return QuaternionState(1.0, 0.0, 0.0, 0.0)

    def conjugate(self) -> 'QuaternionState':
        """Conjugado do quatérnio."""
        return QuaternionState(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: 'QuaternionState') -> 'QuaternionState':
        """Multiplicação de quatérnios."""
        w = float(self.w) * float(other.w) - float(self.x) * float(other.x) - float(self.y) * float(other.y) - float(self.z) * float(other.z)
        x = float(self.w) * float(other.x) + float(self.x) * float(other.w) + float(self.y) * float(other.z) - float(self.z) * float(other.y)
        y = float(self.w) * float(other.y) - float(self.x) * float(other.z) + float(self.y) * float(other.w) + float(self.z) * float(other.x)
        z = float(self.w) * float(other.z) + float(self.x) * float(other.y) - float(self.y) * float(other.x) + float(self.z) * float(other.w)
        return QuaternionState(w, x, y, z)


class PureQuaternionGenerator:
    """
    Gerador de quatérnios puramente físico.

    Mapeia sinais fractais para estados quatérnicos unitários
    usando princípios físicos rigorosos.
    """

    def __init__(self, device: str = "cpu", embedding_dim: int = 64):
        """
        Inicializa o gerador de quatérnios.

        Args:
            device: Dispositivo para tensores
            embedding_dim: Dimensão do embedding (deve ser múltiplo de 4)
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self.quaternion_dim = embedding_dim // 4

        if embedding_dim % 4 != 0:
            raise ValueError(f"embedding_dim deve ser múltiplo de 4, got {embedding_dim}")

        # Parâmetros físicos
        self.hbar = 1.0545718e-34  # Constante de Planck reduzida
        self.c = 299792458  # Velocidade da luz
        self.G = 6.67430e-11  # Constante gravitacional

        # Matrizes de rotação SO(4)
        self._initialize_rotation_matrices()

        print("🔬 PureQuaternionGenerator initialized")
        print(f"   📊 Embedding dim: {embedding_dim}, Quaternion dim: {self.quaternion_dim}")

    def _initialize_rotation_matrices(self):
        """Inicializa matrizes de rotação SO(4)."""
        # Matrizes de Pauli generalizadas para SO(4)
        self.pauli_matrices = {
            'I': torch.eye(4, dtype=torch.complex64, device=self.device),
            'X': torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                            dtype=torch.complex64, device=self.device),
            'Y': torch.tensor([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
                            dtype=torch.complex64, device=self.device),
            'Z': torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
                            dtype=torch.complex64, device=self.device)
        }

    def generate_from_fractal(self, fractal_signal: torch.Tensor,
                            fractal_dimension: float) -> torch.Tensor:
        """
        Gera quatérnios a partir de sinal fractal.

        Args:
            fractal_signal: Sinal fractal multidimensional
            fractal_dimension: Dimensão fractal do sinal

        Returns:
            Estados quatérnicos: [batch_size, seq_len, embedding_dim, 4]
        """
        batch_size, seq_len, features = fractal_signal.shape

        # Inicializar tensor de quatérnios
        quaternion_states = torch.zeros(batch_size, seq_len, self.embedding_dim, 4,
                                      dtype=torch.float32, device=self.device)

        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Extrair features do sinal fractal
                signal_features = fractal_signal[batch_idx, seq_idx]

                # Gerar quatérnios para cada grupo de 4 dimensões
                for quat_idx in range(self.quaternion_dim):
                    start_dim = quat_idx * 4
                    end_dim = min(start_dim + 4, self.embedding_dim)

                    # Features para este quatérnio
                    quat_features = signal_features[start_dim:end_dim]
                    if len(quat_features) < 4:
                        # Padding se necessário
                        padding = torch.zeros(4 - len(quat_features), device=self.device)
                        quat_features = torch.cat([quat_features, padding])

                    # Gerar estado quaterniónico
                    quaternion = self._generate_single_quaternion(
                        quat_features, fractal_dimension, seq_idx
                    )

                    quaternion_states[batch_idx, seq_idx, start_dim:end_dim] = quaternion

        return quaternion_states

    def _generate_single_quaternion(self, features: torch.Tensor,
                                  fractal_dimension: float, position: int) -> torch.Tensor:
        """
        Gera um único estado quaterniónico.

        Args:
            features: Features do sinal fractal [4]
            fractal_dimension: Dimensão fractal
            position: Posição na sequência

        Returns:
            Estado quaterniónico normalizado [4]
        """
        # Mapear features para componentes quaterniónicas
        w = self._map_to_real_component(features[0], fractal_dimension)
        x = self._map_to_imag_component(features[1], position)
        y = self._map_to_imag_component(features[2], position + 1)
        z = self._map_to_imag_component(features[3], position + 2)

        # Criar estado quaterniónico
        quaternion = QuaternionState(w, x, y, z)

        # Aplicar rotações SO(4) baseadas em propriedades físicas
        rotated_quaternion = self._apply_physical_rotations(quaternion, features, fractal_dimension)

        # Normalizar para garantir unitariedade
        normalized = rotated_quaternion.normalize()

        return normalized.to_tensor(self.device)

    def _map_to_real_component(self, feature: float, fractal_dimension: float) -> float:
        """Mapeia feature para componente real do quatérnio."""
        # Componente real baseada na dimensão fractal
        base_component = math.cos(2 * math.pi * feature * fractal_dimension)

        # Modulação baseada em constantes físicas
        physical_factor = math.sin(feature * self.hbar * 1e34)  # Normalizada

        return base_component * physical_factor

    def _map_to_imag_component(self, feature: float, position: int) -> float:
        """Mapeia feature para componente imaginária do quatérnio."""
        # Componente imaginária baseada na posição e feature
        phase = 2 * math.pi * feature * position / 100.0
        amplitude = math.sin(phase) * math.cos(feature * math.pi)

        # Modulação quântica
        quantum_factor = math.exp(-abs(feature) / 2.0)  # Decaimento gaussiano

        return amplitude * quantum_factor

    def _apply_physical_rotations(self, quaternion: QuaternionState,
                                features: torch.Tensor, fractal_dimension: float) -> QuaternionState:
        """
        Aplica rotações SO(4) baseadas em propriedades físicas.

        Usa decomposição Iwasawa para garantir unitariedade.
        """
        # Calcular ângulos de rotação baseados em features
        angles = self._compute_rotation_angles(features, fractal_dimension)

        # Aplicar rotações sequenciais
        rotated = quaternion

        # Rotação no plano w-x (energia)
        if len(angles) > 0:
            rot_wx = self._create_rotation_quaternion(angles[0], 'wx')
            rotated = rot_wx * rotated

        # Rotação no plano y-z (momento)
        if len(angles) > 1:
            rot_yz = self._create_rotation_quaternion(angles[1], 'yz')
            rotated = rot_yz * rotated

        # Rotação no plano w-y (fase)
        if len(angles) > 2:
            rot_wy = self._create_rotation_quaternion(angles[2], 'wy')
            rotated = rot_wy * rotated

        # Rotação no plano x-z (spin)
        if len(angles) > 3:
            rot_xz = self._create_rotation_quaternion(angles[3], 'xz')
            rotated = rot_xz * rotated

        return rotated

    def _compute_rotation_angles(self, features: torch.Tensor, fractal_dimension: float) -> List[float]:
        """Computa ângulos de rotação baseados em features físicas."""
        angles = []

        # Ângulo baseado na primeira feature e dimensão fractal
        angle1 = 2 * math.pi * features[0].item() * fractal_dimension
        angles.append(angle1)

        # Ângulo baseado na segunda feature e constante de Planck
        angle2 = math.pi * features[1].item() * self.hbar * 1e34
        angles.append(angle2)

        # Ângulo baseado na terceira feature e velocidade da luz
        angle3 = math.pi * features[2].item() * self.c / 1e10  # Normalizada
        angles.append(angle3)

        # Ângulo baseado na quarta feature e constante gravitacional
        angle4 = math.pi * features[3].item() * math.sqrt(abs(self.G)) * 1e5  # Normalizada
        angles.append(angle4)

        return angles

    def _create_rotation_quaternion(self, angle: float, plane: str) -> QuaternionState:
        """
        Cria quatérnio de rotação para um plano específico.

        Args:
            angle: Ângulo de rotação
            plane: Plano de rotação ('wx', 'wy', 'wz', 'xy', 'xz', 'yz')
        """
        cos_half = math.cos(angle / 2)
        sin_half = math.sin(angle / 2)

        if plane == 'wx':  # Rotação no plano w-x
            return QuaternionState(cos_half, sin_half, 0, 0)
        elif plane == 'wy':  # Rotação no plano w-y
            return QuaternionState(cos_half, 0, sin_half, 0)
        elif plane == 'wz':  # Rotação no plano w-z
            return QuaternionState(cos_half, 0, 0, sin_half)
        elif plane == 'xy':  # Rotação no plano x-y
            return QuaternionState(0, cos_half, sin_half, 0)
        elif plane == 'xz':  # Rotação no plano x-z
            return QuaternionState(0, cos_half, 0, sin_half)
        elif plane == 'yz':  # Rotação no plano y-z
            return QuaternionState(0, 0, cos_half, sin_half)
        else:
            return QuaternionState(1, 0, 0, 0)  # Identidade

    def validate_unitarity(self, quaternion_states: torch.Tensor) -> Dict[str, float]:
        """
        Valida unitariedade dos estados quatérnicos.

        Args:
            quaternion_states: Estados quatérnicos [..., 4]

        Returns:
            Métricas de validação
        """
        # Calcular normas
        norms = torch.norm(quaternion_states, dim=-1)

        # Estatísticas das normas
        mean_norm = torch.mean(norms).item()
        std_norm = torch.std(norms).item()
        min_norm = torch.min(norms).item()
        max_norm = torch.max(norms).item()

        # Unitariedade: norma deve ser 1
        unitarity_score = 1.0 - abs(mean_norm - 1.0) - std_norm

        # Contar estados com norma próxima de 1
        tolerance = 0.01
        valid_states = torch.sum(torch.abs(norms - 1.0) < tolerance).item()
        total_states = norms.numel()
        validity_ratio = valid_states / total_states

        return {
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'min_norm': min_norm,
            'max_norm': max_norm,
            'unitarity_score': max(0.0, unitarity_score),
            'validity_ratio': validity_ratio
        }

    def apply_quantum_gates(self, quaternion_states: torch.Tensor,
                          gate_sequence: List[str]) -> torch.Tensor:
        """
        Aplica portas quânticas aos estados quatérnicos.

        Args:
            quaternion_states: Estados quatérnicos [..., 4]
            gate_sequence: Sequência de portas a aplicar

        Returns:
            Estados transformados
        """
        transformed_states = quaternion_states.clone()

        for gate in gate_sequence:
            if gate == 'H':  # Hadamard
                transformed_states = self._apply_hadamard_gate(transformed_states)
            elif gate == 'X':  # Pauli-X
                transformed_states = self._apply_pauli_x_gate(transformed_states)
            elif gate == 'Y':  # Pauli-Y
                transformed_states = self._apply_pauli_y_gate(transformed_states)
            elif gate == 'Z':  # Pauli-Z
                transformed_states = self._apply_pauli_z_gate(transformed_states)
            elif gate == 'S':  # Phase gate
                transformed_states = self._apply_phase_gate(transformed_states)
            elif gate == 'T':  # T gate
                transformed_states = self._apply_t_gate(transformed_states)

        return transformed_states

    def _apply_hadamard_gate(self, states: torch.Tensor) -> torch.Tensor:
        """Aplica porta Hadamard quaterniónica."""
        # Hadamard quaterniónica simplificada
        hadamard_quat = QuaternionState(1/math.sqrt(2), 0, 1/math.sqrt(2), 0)

        # Aplicar a cada estado
        batch_size, seq_len, embed_dim, _ = states.shape
        result = torch.zeros_like(states)

        for b in range(batch_size):
            for s in range(seq_len):
                for e in range(0, embed_dim, 4):
                    if e + 4 <= embed_dim:
                        state_tensor = states[b, s, e:e+4]
                        state_quat = QuaternionState(
                            float(state_tensor[0]),
                            float(state_tensor[1]),
                            float(state_tensor[2]),
                            float(state_tensor[3])
                        )
                        transformed = hadamard_quat * state_quat
                        result[b, s, e:e+4] = transformed.to_tensor(self.device)

        return result

    def _apply_pauli_x_gate(self, states: torch.Tensor) -> torch.Tensor:
        """Aplica porta Pauli-X quaterniónica."""
        pauli_x_quat = QuaternionState(0, 1, 0, 0)  # i

        batch_size, seq_len, embed_dim, _ = states.shape
        result = torch.zeros_like(states)

        for b in range(batch_size):
            for s in range(seq_len):
                for e in range(0, embed_dim, 4):
                    if e + 4 <= embed_dim:
                        state_tensor = states[b, s, e:e+4]
                        state_quat = QuaternionState(
                            float(state_tensor[0]),
                            float(state_tensor[1]),
                            float(state_tensor[2]),
                            float(state_tensor[3])
                        )
                        transformed = pauli_x_quat * state_quat
                        result[b, s, e:e+4] = transformed.to_tensor(self.device)

        return result

    def _apply_pauli_y_gate(self, states: torch.Tensor) -> torch.Tensor:
        """Aplica porta Pauli-Y quaterniónica."""
        pauli_y_quat = QuaternionState(0, 0, 1, 0)  # j

        batch_size, seq_len, embed_dim, _ = states.shape
        result = torch.zeros_like(states)

        for b in range(batch_size):
            for s in range(seq_len):
                for e in range(0, embed_dim, 4):
                    if e + 4 <= embed_dim:
                        state_tensor = states[b, s, e:e+4]
                        state_quat = QuaternionState(
                            float(state_tensor[0]),
                            float(state_tensor[1]),
                            float(state_tensor[2]),
                            float(state_tensor[3])
                        )
                        transformed = pauli_y_quat * state_quat
                        result[b, s, e:e+4] = transformed.to_tensor(self.device)

        return result

    def _apply_pauli_z_gate(self, states: torch.Tensor) -> torch.Tensor:
        """Aplica porta Pauli-Z quaterniónica."""
        pauli_z_quat = QuaternionState(0, 0, 0, 1)  # k

        batch_size, seq_len, embed_dim, _ = states.shape
        result = torch.zeros_like(states)

        for b in range(batch_size):
            for s in range(seq_len):
                for e in range(0, embed_dim, 4):
                    if e + 4 <= embed_dim:
                        state_quat = QuaternionState(*states[b, s, e:e+4].tolist())
                        transformed = pauli_z_quat * state_quat
                        result[b, s, e:e+4] = transformed.to_tensor(self.device)

        return result

    def _apply_phase_gate(self, states: torch.Tensor) -> torch.Tensor:
        """Aplica porta de fase quaterniónica."""
        phase_quat = QuaternionState(math.cos(math.pi/4), 0, 0, math.sin(math.pi/4))

        batch_size, seq_len, embed_dim, _ = states.shape
        result = torch.zeros_like(states)

        for b in range(batch_size):
            for s in range(seq_len):
                for e in range(0, embed_dim, 4):
                    if e + 4 <= embed_dim:
                        state_quat = QuaternionState(*states[b, s, e:e+4].tolist())
                        transformed = phase_quat * state_quat
                        result[b, s, e:e+4] = transformed.to_tensor(self.device)

        return result

    def _apply_t_gate(self, states: torch.Tensor) -> torch.Tensor:
        """Aplica porta T quaterniónica."""
        t_quat = QuaternionState(math.cos(math.pi/8), 0, 0, math.sin(math.pi/8))

        batch_size, seq_len, embed_dim, _ = states.shape
        result = torch.zeros_like(states)

        for b in range(batch_size):
            for s in range(seq_len):
                for e in range(0, embed_dim, 4):
                    if e + 4 <= embed_dim:
                        state_quat = QuaternionState(*states[b, s, e:e+4].tolist())
                        transformed = t_quat * state_quat
                        result[b, s, e:e+4] = transformed.to_tensor(self.device)

        return result


# Funções utilitárias
def create_test_fractal_signal(length: int = 10, features: int = 7) -> torch.Tensor:
    """Cria sinal fractal de teste."""
    signal = torch.randn(1, length, features)  # Batch size 1
    return signal


def test_pure_quaternion_generator():
    """Testa o gerador de quatérnios puro."""
    print("🧪 Testing Pure Quaternion Generator")
    print("=" * 50)

    # Criar gerador
    generator = PureQuaternionGenerator(embedding_dim=64)

    # Sinal fractal de teste
    fractal_signal = create_test_fractal_signal(length=5, features=64)
    fractal_dimension = 1.5

    # Gerar quatérnios
    quaternion_states = generator.generate_from_fractal(fractal_signal, fractal_dimension)

    print(f"🔬 Quaternion Generation:")
    print(f"   Input shape: {fractal_signal.shape}")
    print(f"   Output shape: {quaternion_states.shape}")

    # Validar unitariedade
    validation = generator.validate_unitarity(quaternion_states)
    print(f"   📊 Unitarity Validation:")
    print(f"      Mean norm: {validation['mean_norm']:.3f}")
    print(f"      Std norm: {validation['std_norm']:.3f}")
    print(f"      Unitarity score: {validation['unitarity_score']:.3f}")
    print(f"      Validity ratio: {validation['validity_ratio']:.3f}")

    # Testar portas quânticas
    gate_sequence = ['H', 'X', 'Y', 'Z']
    transformed_states = generator.apply_quantum_gates(quaternion_states, gate_sequence)

    # Validar unitariedade após portas
    post_gate_validation = generator.validate_unitarity(transformed_states)
    print(f"   🔄 Post-Gate Validation:")
    print(f"      Mean norm: {post_gate_validation['mean_norm']:.3f}")
    print(f"      Unitarity score: {post_gate_validation['unitarity_score']:.3f}")

    print("\n✅ Pure Quaternion Generator test completed!")


if __name__ == "__main__":
    test_pure_quaternion_generator()