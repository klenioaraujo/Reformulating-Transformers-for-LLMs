import torch
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
from ΨQRHSystem.core.TernaryLogicFramework import TernaryLogicFramework


class QuantumMasterEquation:
    """
    Equação Mestra Quântica: dρ/dt = -i[H,ρ] + 𝓛_fractal(ρ) + 𝓛_dissipative(ρ)

    Implementa a evolução temporal completa do estado quântico incluindo:
    - Evolução unitária: -i[H,ρ]
    - Decoerência fractal: 𝓛_fractal(ρ)
    - Dissipação: 𝓛_dissipative(ρ)
    """

    def __init__(self, device: str = "cpu", hbar: float = 1.0):
        self.device = device
        self.hbar = hbar
        self.ternary_logic = TernaryLogicFramework(device=device)

        # Parâmetros da equação mestra
        self.fractal_dimension = 1.26  # Dimensão fractal típica
        self.dissipation_rate = 0.1    # Taxa de dissipação
        self.fractal_coupling = 0.05   # Acoplamento fractal

        print(f"🔬 Quantum Master Equation inicializada: dρ/dt = -i[H,ρ] + 𝓛_fractal(ρ) + 𝓛_dissipative(ρ)")

    def evolve_quantum_state(self, rho: torch.Tensor, H: torch.Tensor,
                            dt: float, fractal_noise: torch.Tensor = None) -> torch.Tensor:
        """
        Evolui estado quântico usando equação mestra completa

        Args:
            rho: Matriz densidade [..., n, n]
            H: Hamiltoniano [..., n, n]
            dt: Passo temporal
            fractal_noise: Ruído fractal opcional

        Returns:
            Estado evoluído ρ(t+dt)
        """
        # 1. Evolução unitária: -i[H,ρ]
        unitary_evolution = self._unitary_evolution(rho, H, dt)

        # 2. Decoerência fractal: 𝓛_fractal(ρ)
        fractal_decoherence = self._fractal_lindblad(rho, fractal_noise)

        # 3. Dissipação: 𝓛_dissipative(ρ)
        dissipation = self._dissipative_lindblad(rho)

        # Evolução total
        drho_dt = unitary_evolution + fractal_decoherence + dissipation
        rho_evolved = rho + dt * drho_dt

        # Normalização e estabilização com lógica ternária
        rho_evolved = self._normalize_density_matrix(rho_evolved)

        return rho_evolved

    def _unitary_evolution(self, rho: torch.Tensor, H: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolução unitária: -i[H,ρ]"""
        # Comutador: [H,ρ] = Hρ - ρH
        commutator = torch.matmul(H, rho) - torch.matmul(rho, H)

        # Evolução: -i/ℏ [H,ρ]
        return -1j / self.hbar * commutator

    def _fractal_lindblad(self, rho: torch.Tensor, fractal_noise: torch.Tensor = None) -> torch.Tensor:
        """
        Superoperador Lindblad fractal: 𝓛_fractal(ρ)

        Modelo decoerência fractal baseada na dimensão fractal D
        """
        if rho.dim() < 2:
            return torch.zeros_like(rho)

        n = rho.shape[-1]

        # Gerar operadores Lindblad fractais
        if fractal_noise is None:
            # Ruído fractal baseado na dimensão D
            fractal_noise = self._generate_fractal_noise(n)

        # Aplicar decoerência fractal
        lindblad_term = torch.zeros_like(rho)

        for i in range(min(3, n//2)):  # Limitar para eficiência
            # Operador Lindblad fractal
            L_i = self._create_fractal_lindblad_operator(n, i, fractal_noise)

            # Termo Lindblad: LρL† - (1/2){L†L,ρ}
            L_rho = torch.matmul(L_i, rho)
            L_rho_L_dagger = torch.matmul(L_rho, L_i.conj().t())

            L_dagger_L = torch.matmul(L_i.conj().t(), L_i)
            anticommutator = torch.matmul(L_dagger_L, rho) + torch.matmul(rho, L_dagger_L)

            lindblad_term += self.fractal_coupling * (L_rho_L_dagger - 0.5 * anticommutator)

        return lindblad_term

    def _dissipative_lindblad(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Superoperador Lindblad dissipativo: 𝓛_dissipative(ρ)

        Modelo dissipação baseada em amplitude damping
        """
        if rho.dim() < 2:
            return torch.zeros_like(rho)

        n = rho.shape[-1]

        # Amplitude damping operators
        lindblad_term = torch.zeros_like(rho)

        for i in range(min(2, n-1)):  # Limitar para eficiência
            # Operador de dissipação
            gamma = self.dissipation_rate * (i + 1) / n  # Taxa dependente da dimensão

            # Amplitude damping: σ- = |0⟩⟨1|
            sigma_minus = torch.zeros((n, n), dtype=torch.complex64, device=self.device)
            if i+1 < n:
                sigma_minus[i, i+1] = 1.0

            # Aplicar termo Lindblad
            sigma_rho = torch.matmul(sigma_minus, rho)
            sigma_rho_sigma_dagger = torch.matmul(sigma_rho, sigma_minus.conj().t())

            sigma_dagger_sigma = torch.matmul(sigma_minus.conj().t(), sigma_minus)
            anticommutator = torch.matmul(sigma_dagger_sigma, rho) + torch.matmul(rho, sigma_dagger_sigma)

            lindblad_term += gamma * (sigma_rho_sigma_dagger - 0.5 * anticommutator)

        return lindblad_term

    def _generate_fractal_noise(self, n: int) -> torch.Tensor:
        """Gera ruído fractal baseado na dimensão D"""
        # Ruído 1/f baseado na dimensão fractal
        frequencies = torch.arange(1, n+1, dtype=torch.float32, device=self.device)
        fractal_spectrum = 1.0 / (frequencies ** self.fractal_dimension)

        # Normalizar e adicionar fase aleatória
        fractal_spectrum = fractal_spectrum / torch.sum(fractal_spectrum)
        phases = torch.rand(n, device=self.device) * 2 * math.pi

        return fractal_spectrum * torch.exp(1j * phases)

    def _create_fractal_lindblad_operator(self, n: int, index: int,
                                        fractal_noise: torch.Tensor) -> torch.Tensor:
        """Cria operador Lindblad fractal"""
        # Operador baseado no ruído fractal
        L = torch.zeros((n, n), dtype=torch.complex64, device=self.device)

        # Diagonal com ruído fractal
        for i in range(n):
            phase = torch.angle(fractal_noise[i])
            L[i, i] = torch.exp(1j * phase * (index + 1))

        # Off-diagonal limitado
        for i in range(min(2, n-1)):
            if i + index + 1 < n:
                L[i, i + index + 1] = 0.1 * fractal_noise[i]

        return L

    def _normalize_density_matrix(self, rho: torch.Tensor) -> torch.Tensor:
        """Normaliza matriz densidade com estabilização ternária"""
        # Traço = 1
        trace = torch.trace(rho)
        if trace != 0:
            rho = rho / trace

        # Hermitiana
        rho = (rho + rho.conj().t()) / 2

        # Semi-definida positiva (eigenvalues >= 0)
        eigenvalues, eigenvectors = torch.linalg.eigh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)  # Remover valores negativos pequenos

        # Reconstruir
        rho_normalized = torch.matmul(
            torch.matmul(eigenvectors, torch.diag(eigenvalues)),
            eigenvectors.conj().t()
        )

        # Estabilização ternária
        rho_normalized = self._apply_ternary_stabilization(rho_normalized)

        return rho_normalized

    def _apply_ternary_stabilization(self, rho: torch.Tensor) -> torch.Tensor:
        """Aplica estabilização ternária à matriz densidade"""
        # Converter para representação ternária e voltar
        ternary_real = self._tensor_to_ternary_states(rho.real)
        ternary_imag = self._tensor_to_ternary_states(rho.imag)

        # Aplicar consenso ternário
        stabilized_real = self._apply_ternary_consensus(rho.real, ternary_real)
        stabilized_imag = self._apply_ternary_consensus(rho.imag, ternary_imag)

        return stabilized_real + 1j * stabilized_imag

    def _tensor_to_ternary_states(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converte tensor para estados ternários"""
        abs_tensor = torch.abs(tensor)
        max_val = torch.max(abs_tensor)

        if max_val == 0:
            return torch.zeros_like(tensor, dtype=torch.long)

        normalized = tensor / (max_val + 1e-10)
        ternary_states = torch.zeros_like(tensor, dtype=torch.long)
        ternary_states[normalized > 0.33] = 1
        ternary_states[normalized < -0.33] = -1

        return ternary_states

    def _apply_ternary_consensus(self, original: torch.Tensor, ternary_states: torch.Tensor) -> torch.Tensor:
        """Aplica consenso ternário para estabilização"""
        # Para valores próximos de zero, usar consenso
        near_zero = torch.abs(original) < 0.1
        if near_zero.any():
            # Consenso baseado na vizinhança
            consensus_value = self.ternary_logic.ternary_majority_vote(
                ternary_states.flatten().tolist()[:10]  # Amostra pequena
            )
            original = torch.where(near_zero, consensus_value * 0.05, original)

        return original


class QuaternionOps:
    """
    Quaternion Operations - Operações quaterniônicas otimizadas com Equação Mestra

    Implementa operações fundamentais de quaternions para física ΨQRH:
    - Produto de Hamilton
    - Rotações SO(4)
    - Operações unitárias
    - Evolução temporal via Equação Mestra Quântica
    """

    def __init__(self, device: str = "cpu"):
        """
        Inicializa operações quaterniônicas com lógica ternária e equação mestra

        Args:
            device: Dispositivo de computação
        """
        self.device = device
        self.ternary_logic = TernaryLogicFramework(device=device)
        self.master_equation = QuantumMasterEquation(device=device)
        print(f"🔄 Quaternion Operations inicializadas no dispositivo: {device} com lógica ternária e equação mestra")

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

    def so4_rotation(self, q: torch.Tensor, rotation_angles: torch.Tensor,
                    time_step: float = 0.01, hamiltonian: torch.Tensor = None) -> torch.Tensor:
        """
        Aplica rotações SO(4) unitárias com evolução temporal via equação mestra:
        Ψ' = q_left ⊗ Ψ ⊗ q_right† + evolução temporal

        Args:
            q: Estado quântico quaterniônico [..., seq_len, embed_dim, 4]
            rotation_angles: Ângulos de rotação [..., 3] (theta, omega, phi)
            time_step: Passo temporal para evolução
            hamiltonian: Hamiltoniano opcional para evolução unitária

        Returns:
            Estado rotacionado e evoluído [..., seq_len, embed_dim, 4]
        """
        # Criar quaternions de rotação
        q_left, q_right = self._create_rotation_quaternions(rotation_angles)

        # Aplicar rotação: q_left * q * q_right†
        q_right_conj = self.quaternion_conjugate(q_right)

        # Produto esquerdo
        temp = self.hamilton_product(q_left, q)

        # Produto direito com conjugado
        rotated_state = self.hamilton_product(temp, q_right_conj)

        # Aplicar evolução temporal via equação mestra se Hamiltoniano fornecido
        if hamiltonian is not None:
            # Converter quaternion para matriz densidade aproximada
            rho = self._quaternion_to_density_matrix(rotated_state)

            # Aplicar evolução temporal
            rho_evolved = self.master_equation.evolve_quantum_state(rho, hamiltonian, time_step)

            # Converter de volta para quaternion
            rotated_state = self._density_matrix_to_quaternion(rho_evolved)

        return rotated_state

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

    def _quaternion_to_density_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        Converte quaternion para matriz densidade aproximada

        Args:
            q: Quaternion [..., 4]

        Returns:
            Matriz densidade [..., 2, 2] (aproximação para 2-level system)
        """
        # Para simplificar, mapear quaternion para estado puro 2x2
        # w + xi + yj + zk → |ψ⟩⟨ψ| onde |ψ⟩ = [w + xi, y + zk]

        if q.dim() == 1:
            q = q.unsqueeze(0)

        batch_shape = q.shape[:-1]
        w, x, y, z = q.unbind(-1)

        # Construir estado puro aproximado
        psi = torch.stack([w + 1j * x, y + 1j * z], dim=-1)  # [..., 2]

        # Normalizar
        psi = psi / (torch.norm(psi, dim=-1, keepdim=True) + 1e-10)

        # Matriz densidade |ψ⟩⟨ψ|
        rho = torch.matmul(psi.unsqueeze(-1), psi.conj().unsqueeze(-2))  # [..., 2, 2]

        return rho

    def _density_matrix_to_quaternion(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Converte matriz densidade de volta para quaternion aproximado

        Args:
            rho: Matriz densidade [..., 2, 2]

        Returns:
            Quaternion [..., 4]
        """
        # Extrair componentes do estado puro aproximado
        # Assumir estado puro: ρ = |ψ⟩⟨ψ|

        # Eigenvalores e eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(rho)

        # Usar eigenvector com maior eigenvalue
        max_eigenval_idx = torch.argmax(eigenvalues, dim=-1)
        psi = eigenvectors[..., max_eigenval_idx]  # [..., 2]

        # Mapear de volta para quaternion
        # |ψ⟩ = [α, β] → w + xi + yj + zk onde α = w + xi, β = y + zk
        alpha_real, alpha_imag = psi[..., 0].real, psi[..., 0].imag
        beta_real, beta_imag = psi[..., 1].real, psi[..., 1].imag

        w = alpha_real
        x = alpha_imag
        y = beta_real
        z = beta_imag

        return torch.stack([w, x, y, z], dim=-1)