import torch
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
from core.TernaryLogicFramework import TernaryLogicFramework


class QuaternionOps:
    """
    Quaternion Operations - Operações quaterniônicas otimizadas

    Implementa operações fundamentais de quaternions para física ΨQRH:
    - Produto de Hamilton
    - Rotações SO(4)
    - Operações unitárias
    """

    def __init__(self, device: str = "cpu"):
        """
        Inicializa operações quaterniônicas com lógica ternária

        Args:
            device: Dispositivo de computação
        """
        self.device = device
        self.ternary_logic = TernaryLogicFramework(device=device)
        print(f"🔄 Quaternion Operations inicializadas no dispositivo: {device} com lógica ternária")

    def hamilton_product(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Produto de Hamilton entre quaternions com lógica ternária

        Args:
            q1: Primeiro quaternion [..., 4]
            q2: Segundo quaternion [..., 4]

        Returns:
            Produto q1 * q2 [..., 4]
        """
        # Desempacotar componentes
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        # Produto de Hamilton: (w1 + x1i + y1j + z1k) * (w2 + x2i + y2j + z2k)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        result = torch.stack([w, x, y, z], dim=-1)

        # Aplicar lógica ternária para estabilização
        # Converter para estados ternários e voltar para refinar o resultado
        ternary_w = self._tensor_to_ternary_states(w)
        ternary_x = self._tensor_to_ternary_states(x)
        ternary_y = self._tensor_to_ternary_states(y)
        ternary_z = self._tensor_to_ternary_states(z)

        # Aplicar operações ternárias para estabilização
        stabilized_w = self._apply_ternary_stabilization(w, ternary_w)
        stabilized_x = self._apply_ternary_stabilization(x, ternary_x)
        stabilized_y = self._apply_ternary_stabilization(y, ternary_y)
        stabilized_z = self._apply_ternary_stabilization(z, ternary_z)

        return torch.stack([stabilized_w, stabilized_x, stabilized_y, stabilized_z], dim=-1)

    def quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """
        Conjugado quaterniônico: q* = (w, -x, -y, -z)

        Args:
            q: Quaternion [..., 4]

        Returns:
            Conjugado [..., 4]
        """
        w, x, y, z = q.unbind(-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    def quaternion_norm(self, q: torch.Tensor) -> torch.Tensor:
        """
        Norma quaterniônica

        Args:
            q: Quaternion [..., 4]

        Returns:
            Norma [..., 1]
        """
        return torch.sqrt(torch.sum(q ** 2, dim=-1, keepdim=True))

    def normalize_quaternion(self, q: torch.Tensor) -> torch.Tensor:
        """
        Normaliza quaternion para norma unitária

        Args:
            q: Quaternion [..., 4]

        Returns:
            Quaternion normalizado [..., 4]
        """
        norm = self.quaternion_norm(q)
        return q / (norm + 1e-10)

    def so4_rotation(self, q: torch.Tensor, rotation_angles: torch.Tensor) -> torch.Tensor:
        """
        Aplica rotações SO(4) unitárias: Ψ' = q_left ⊗ Ψ ⊗ q_right†

        Args:
            q: Estado quântico quaterniônico [..., seq_len, embed_dim, 4]
            rotation_angles: Ângulos de rotação [..., 3] (theta, omega, phi)

        Returns:
            Estado rotacionado [..., seq_len, embed_dim, 4]
        """
        # Criar quaternions de rotação
        q_left, q_right = self._create_rotation_quaternions(rotation_angles)

        # Aplicar rotação: q_left * q * q_right†
        q_right_conj = self.quaternion_conjugate(q_right)

        # Produto esquerdo
        temp = self.hamilton_product(q_left, q)

        # Produto direito com conjugado
        result = self.hamilton_product(temp, q_right_conj)

        return result

    def _create_rotation_quaternions(self, angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cria quaternions de rotação para SO(4)

        Args:
            angles: Ângulos de rotação [..., 3]

        Returns:
            Tuple (q_left, q_right) para rotações SO(4)
        """
        theta, omega, phi = angles.unbind(-1)

        # Quaternions de rotação simplificados
        # Para implementação completa, seria necessário implementar rotações SO(4) gerais

        # Quaternion esquerdo (rotação temporal)
        q_left = torch.stack([
            torch.cos(theta / 2),
            torch.sin(theta / 2),
            torch.zeros_like(theta),
            torch.zeros_like(theta)
        ], dim=-1)

        # Quaternion direito (rotação espacial)
        q_right = torch.stack([
            torch.cos(phi / 2),
            torch.zeros_like(phi),
            torch.sin(phi / 2),
            torch.zeros_like(phi)
        ], dim=-1)

        return q_left, q_right

    def create_unit_quaternion(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Cria quaternion unitário (1, 0, 0, 0)

        Args:
            shape: Forma do tensor desejado

        Returns:
            Quaternion unitário
        """
        q = torch.zeros(*shape, 4, device=self.device)
        q[..., 0] = 1.0  # Componente real = 1
        return q

    def create_random_quaternion(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Cria quaternion aleatório normalizado

        Args:
            shape: Forma do tensor desejado

        Returns:
            Quaternion aleatório normalizado
        """
        q = torch.randn(*shape, 4, device=self.device)
        return self.normalize_quaternion(q)

    def quaternion_exponential(self, q: torch.Tensor) -> torch.Tensor:
        """
        Exponencial quaterniônico: exp(q) = e^a (cos|b| + (b/|b|) sin|b|)

        Args:
            q: Quaternion [..., 4]

        Returns:
            exp(q) [..., 4]
        """
        # Separar parte real e vetorial
        a = q[..., 0]  # Parte real
        b = q[..., 1:]  # Parte vetorial [x, y, z]

        # Norma da parte vetorial
        b_norm = torch.norm(b, dim=-1, keepdim=True)

        # Evitar divisão por zero
        b_norm_safe = torch.where(b_norm == 0, torch.ones_like(b_norm), b_norm)
        b_unit = b / b_norm_safe

        # Exponencial quaterniônico
        exp_a = torch.exp(a)
        cos_b = torch.cos(b_norm)
        sin_b = torch.sin(b_norm)

        # Resultado
        result = exp_a * torch.cat([cos_b, sin_b * b_unit], dim=-1)

        return result

    def quaternion_logarithm(self, q: torch.Tensor) -> torch.Tensor:
        """
        Logaritmo quaterniônico

        Args:
            q: Quaternion unitário [..., 4]

        Returns:
            log(q) [..., 4]
        """
        # Norma do quaternion
        q_norm = self.quaternion_norm(q)

        # Parte real do logaritmo
        log_norm = torch.log(q_norm)

        # Parte vetorial
        a = q[..., 0] / (q_norm + 1e-10)  # cos(theta)
        b = q[..., 1:]  # Parte vetorial normalizada

        # Ângulo theta
        theta = torch.acos(torch.clamp(a, -1.0, 1.0))

        # Logaritmo
        b_norm = torch.norm(b, dim=-1, keepdim=True)
        theta_safe = torch.where(b_norm == 0, torch.zeros_like(theta), theta / (b_norm + 1e-10))

        result = torch.cat([log_norm, theta_safe * b], dim=-1)

        return result

    def validate_unitarity(self, transformation: torch.Tensor) -> bool:
        """
        Valida unitariedade da transformação quaterniônica com lógica ternária

        Args:
            transformation: Matriz de transformação

        Returns:
            True se unitária
        """
        try:
            # Para quaternions, verificar se preserva a norma
            # Teste simplificado: aplicar transformação e verificar conservação de norma
            test_q = self.create_random_quaternion((10, 4))

            # Aplicar transformação (simplificada)
            transformed = self.hamilton_product(transformation, test_q)

            # Verificar conservação de norma
            norm_before = self.quaternion_norm(test_q)
            norm_after = self.quaternion_norm(transformed)

            conservation = torch.allclose(norm_before, norm_after, atol=1e-5)

            # Adicionar validação ternária: verificar se estados são consistentes
            ternary_consistency = self._validate_ternary_consistency(test_q, transformed)

            # Combinar validações usando lógica ternária
            binary_result = 1 if conservation else -1
            ternary_result = 1 if ternary_consistency else -1

            final_result = self.ternary_logic.ternary_and(binary_result, ternary_result)
            return final_result == 1

        except Exception:
            return False

    def _validate_ternary_consistency(self, input_q: torch.Tensor, output_q: torch.Tensor) -> bool:
        """
        Valida consistência ternária entre entrada e saída

        Args:
            input_q: Quaternion de entrada
            output_q: Quaternion de saída

        Returns:
            True se consistente
        """
        try:
            # Converter para estados ternários
            input_ternary = self._tensor_to_ternary_states(input_q)
            output_ternary = self._tensor_to_ternary_states(output_q)

            # Verificar se a distribuição de estados é similar
            input_counts = torch.bincount(input_ternary.flatten() + 1, minlength=3)  # Shift para 0,1,2
            output_counts = torch.bincount(output_ternary.flatten() + 1, minlength=3)

            # Calcular diferença relativa
            total_elements = input_q.numel()
            diff_ratio = torch.sum(torch.abs(input_counts - output_counts)) / (2 * total_elements)

            # Considerar consistente se diferença < 30%
            return diff_ratio < 0.3

        except Exception:
            return False

    def get_rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Converte ângulos de rotação para matriz SO(4)

        Args:
            angles: Ângulos [3]

        Returns:
            Matriz de rotação SO(4) [4, 4]
        """
        theta, omega, phi = angles

        # Implementação simplificada de matriz SO(4)
        # Para rotação completa, seria necessária implementação mais complexa

        c1, s1 = torch.cos(theta), torch.sin(theta)
        c2, s2 = torch.cos(omega), torch.sin(omega)
        c3, s3 = torch.cos(phi), torch.sin(phi)

        # Matriz SO(4) simplificada
        rotation_matrix = torch.tensor([
            [c1*c2, -s1*c3 + c1*s2*s3, s1*s3 + c1*s2*c3, 0],
            [s1*c2, c1*c3 + s1*s2*s3, -c1*s3 + s1*s2*c3, 0],
            [-s2, c2*s3, c2*c3, 0],
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float32)

        return rotation_matrix

    def apply_so4_transformation(self, q: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Aplica transformação SO(4) a quaternion

        Args:
            q: Quaternion [..., 4]
            rotation_matrix: Matriz SO(4) [4, 4]

        Returns:
            Quaternion transformado [..., 4]
        """
        # Aplicar transformação linear
        return torch.matmul(q, rotation_matrix.T)

    def _tensor_to_ternary_states(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Converte tensor para estados ternários (-1, 0, 1)

        Args:
            tensor: Tensor de entrada

        Returns:
            Estados ternários
        """
        # Classificar baseado na magnitude e sinal
        abs_tensor = torch.abs(tensor)
        max_val = torch.max(abs_tensor)

        if max_val == 0:
            return torch.zeros_like(tensor, dtype=torch.long)

        # Normalizar e classificar
        normalized = tensor / (max_val + 1e-10)

        # Converter para estados ternários
        ternary_states = torch.zeros_like(tensor, dtype=torch.long)
        ternary_states[normalized > 0.33] = 1
        ternary_states[normalized < -0.33] = -1
        # Valores entre -0.33 e 0.33 permanecem 0

        return ternary_states

    def _apply_ternary_stabilization(self, original: torch.Tensor, ternary_states: torch.Tensor) -> torch.Tensor:
        """
        Aplica estabilização baseada em estados ternários

        Args:
            original: Tensor original
            ternary_states: Estados ternários correspondentes

        Returns:
            Tensor estabilizado
        """
        # Aplicar operações ternárias para estabilização
        # Usar consenso ternário para valores próximos de transições
        stabilized = original.clone()

        # Para valores próximos de zero, aplicar estabilização ternária
        near_zero_mask = torch.abs(original) < 0.1
        if near_zero_mask.any():
            # Usar consenso ternário para decidir direção
            ternary_consensus = []
            for i in range(min(5, original.numel())):  # Amostra pequena para eficiência
                sample_idx = torch.randint(0, original.numel(), (1,))
                sample_val = original.flatten()[sample_idx]
                ternary_val = 1 if sample_val > 0.01 else (-1 if sample_val < -0.01 else 0)
                ternary_consensus.append(ternary_val)

            consensus_result = self.ternary_logic.ternary_majority_vote(ternary_consensus)
            stabilized[near_zero_mask] = consensus_result * 0.05  # Pequeno viés baseado no consenso

        return stabilized