porque isso é lento #!/usr/bin/env python3
"""
Matriz Quântica Dinâmica com Quarteniões e Primos
==================================================

Matriz quântica que se adapta dinamicamente aos parâmetros espectrais
dos modelos semânticos específicos, utilizando quarteniões e números primos.

Princípios Físicos Integrados:
- Equação de Padilha: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
- Filtragem Espectral: F(k) = exp(i α · arctan(ln(|k| + ε)))
- Rotações SO(4): Ψ' = q_left * Ψ * q_right†
- Quarteniões: Representação completa H = {a + bi + cj + dk}
- Números Primos: Ressonâncias e fatores primos nos parâmetros

Uso:
    from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix
    matrix = DynamicQuantumCharacterMatrix()
    matrix.adapt_to_model('gpt2')
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Importações locais do sistema ΨQRH
from spectral_parameters_integration import SpectralParametersIntegrator
from src.core.quaternion_operations import QuaternionOperations


class QuaternionRotationLayer(nn.Module):
    """
    Camada de rotação SO(4) que implementa multiplicação quaterniônica real.

    Esta camada agrupa os componentes quaterniônicos [w, x, y, z] e aplica
    rotações unitárias Ψ' = q_left * Ψ * q_right†, preservando a norma.
    """

    def __init__(self, quaternion_dim: int, device: str = "cpu"):
        super().__init__()
        self.quaternion_dim = quaternion_dim
        self.device = device

        # Parâmetros aprendíveis para rotações unitárias
        # Cada quaternion de rotação é parametrizado por 6 ângulos (theta1, omega1, phi1, theta2, omega2, phi2)
        # para rotações SO(4) verdadeiras com q_left e q_right
        self.rotation_angles = nn.Parameter(torch.randn(quaternion_dim, 6) * 0.1)

        # Inicializar operações quaterniônicas otimizadas
        from src.core.quaternion_operations import OptimizedQuaternionOperations
        self.quaternion_ops = OptimizedQuaternionOperations(device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica rotações SO(4) aos componentes quaterniônicos.

        Args:
            x: Tensor de entrada [batch, seq, hidden_size]

        Returns:
            Tensor rotacionado [batch, seq, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape

        # Reorganizar tensor para formato quaterniônico [batch, seq, quaternion_dim, 4]
        x_quaternion = x.view(batch_size, seq_len, self.quaternion_dim, 4)

        # Aplicar rotações SO(4) a cada componente quaterniônico
        rotated_quaternions = []
        for i in range(self.quaternion_dim):
            # Obter ângulos de rotação para este componente
            angles = self.rotation_angles[i]  # [6]

            # Expandir ângulos para o batch e sequência
            angles_expanded = angles.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)  # [batch, seq, 6]

            # Aplicar rotação SO(4)
            q_rotated = self.quaternion_ops.so4_rotation(x_quaternion[:, :, i, :], angles_expanded)  # [batch, seq, 4]

            rotated_quaternions.append(q_rotated)

        # Empilhar de volta
        x_rotated = torch.stack(rotated_quaternions, dim=2)  # [batch, seq, quaternion_dim, 4]

        # Reverter para formato original
        return x_rotated.view(batch_size, seq_len, hidden_size)


class DynamicQuantumCharacterMatrix(nn.Module):
    """
    Matriz quântica dinâmica com quarteniões e números primos.
    Adapta-se aos parâmetros espectrais dos modelos semânticos específicos.
    """

    def __init__(self, vocab_size: int = 50257, hidden_size: int = 256, device: str = "cpu"):
        """
        Inicializa a matriz quântica dinâmica com quarteniões.

        Args:
            vocab_size: Tamanho do vocabulário
            hidden_size: Dimensão do espaço latente (deve ser múltiplo de 4 para quarteniões)
            device: Dispositivo para computação
        """
        super().__init__()

        self.device = device
        self.vocab_size = vocab_size
        # Garantir que hidden_size seja múltiplo de 4 para quarteniões
        self.hidden_size = (hidden_size // 4) * 4
        self.quaternion_dim = self.hidden_size // 4  # Dimensão de cada componente quaterniónico

        self.spectral_integrator = SpectralParametersIntegrator()
        from src.core.quaternion_operations import OptimizedQuaternionOperations
        self.quaternion_ops = OptimizedQuaternionOperations(device=self.device)
        self.current_model_params = None

        # Gerar números primos para ressonâncias
        self.primes = self._generate_primes_up_to(100)
        self.prime_resonances = self._compute_prime_resonances()

        # Matriz quântica base (inicializada com valores padrão)
        self.quantum_matrix = self._initialize_quantum_matrix()

        # Ensure device attribute is accessible
        self.device = device

        # Camadas de adaptação dinâmica com quarteniões
        self.adaptation_layers = nn.ModuleDict({
            'spectral_filter': nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, dtype=torch.complex128, bias=False),
            'quaternion_rotator': self._create_quaternion_rotation_layer(),
            'prime_resonator': nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.complex128, bias=False)
        })

        # Mover para dispositivo
        self.to(device)

        print("🔬 Dynamic Quantum Character Matrix com Quarteniões inicializada")
        print(f"   📊 Vocab: {vocab_size}, Hidden: {self.hidden_size} (quaternion_dim: {self.quaternion_dim})")
        print(f"   🔢 Primos disponíveis: {len(self.primes)}")
        print(f"   🔄 Camada de rotação SO(4): Implementada com multiplicação quaterniônica")

    def _generate_primes_up_to(self, limit: int) -> List[int]:
        """Gera números primos até um limite usando Crivo de Eratóstenes."""
        if limit < 2:
            return []

        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False

        return [i for i in range(2, limit + 1) if is_prime[i]]

    def _create_quaternion_rotation_layer(self) -> nn.Module:
        """
        Cria uma camada de rotação SO(4) que implementa multiplicação quaterniônica real.

        Esta camada agrupa os componentes quaterniônicos e aplica rotações unitárias
        que preservam a norma, implementando Ψ' = q_left * Ψ * q_right†.
        """
        return QuaternionRotationLayer(self.quaternion_dim, self.device)

    def _compute_prime_resonances(self) -> Dict[int, float]:
        """Computa ressonâncias baseadas em números primos."""
        resonances = {}
        for prime in self.primes:
            # Ressonância baseada na distribuição de zeros da função zeta de Riemann
            # e propriedades dos números primos
            resonance = 1.0 / (math.log(prime) + 1e-8)
            resonances[prime] = resonance
        return resonances

    def _initialize_quantum_matrix(self) -> torch.Tensor:
        """
        Inicializa matriz quântica base com parâmetros padrão e quarteniões.
        """
        # Parâmetros padrão (serão sobrescritos pela adaptação)
        alpha_default = 1.5
        beta_default = 0.8
        fractal_dim_default = 1.7

        return self._compute_padilha_quantum_matrix(
            alpha_default, beta_default, fractal_dim_default
        )

    def adapt_to_model(self, model_name: str) -> bool:
        """
        Adapta a matriz quântica aos parâmetros de um modelo específico.

        Args:
            model_name: Nome do modelo semântico

        Returns:
            True se adaptação foi bem-sucedida
        """
        try:
            print(f"🔧 Adaptando matriz quântica para: {model_name}")

            # Extrair parâmetros espectrais
            model_params = self.spectral_integrator.extract_spectral_parameters(model_name)

            if not model_params:
                print(f"⚠️  Não foi possível extrair parâmetros de {model_name}")
                return False

            self.current_model_params = model_params

            # Atualizar matriz quântica com parâmetros do modelo
            self._update_quantum_matrix_with_model_params(model_params)

            print("✅ Adaptação concluída:")
            print(".3f")
            print(".3f")
            print(".3f")

            return True

        except Exception as e:
            print(f"❌ Erro adaptando matriz para {model_name}: {e}")
            return False

    def _update_quantum_matrix_with_model_params(self, model_params: Dict):
        """
        Atualiza matriz quântica com parâmetros específicos do modelo.
        """
        alpha = model_params.get('alpha_final', 1.5)
        beta = model_params.get('beta_final', 0.8)
        fractal_dim = model_params.get('fractal_dim_final', 1.7)

        # Computar nova matriz com Equação de Padilha
        self.quantum_matrix = self._compute_padilha_quantum_matrix(alpha, beta, fractal_dim)

        # Aplicar filtragem espectral adaptativa
        self._apply_adaptive_spectral_filtering(alpha, beta)

        # Atualizar camadas de adaptação
        self._update_adaptation_layers(alpha, beta, fractal_dim)

    def _compute_padilha_quantum_matrix(self, alpha: float, beta: float, D: float) -> torch.Tensor:
        """
        Computa matriz quântica usando Equação de Padilha com quarteniões e números primos.
        Versão simplificada para compatibilidade.

        f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

        Args:
            alpha: Parâmetro de filtragem espectral
            beta: Parâmetro de decaimento espectral
            D: Dimensão fractal

        Returns:
            Matriz quântica [vocab_size, hidden_size] com representação quaterniónica
        """
        # Matriz quaterniónica: [vocab_size, quaternion_dim, 4]
        # Cada posição do vocabulário tem um quaternion [w, x, y, z]
        # Inicializar como complexo para preservar informações de fase
        matrix = torch.zeros(self.vocab_size, self.quaternion_dim, 4, dtype=torch.complex64)

        I0 = 1.0  # Intensidade base
        omega = alpha  # Frequência angular relacionada a α
        k = beta      # Número de onda relacionado a β

        # Computar em lotes para eficiência (fallback para versão não-vetorizada)
        batch_size = min(50, self.vocab_size)

        for batch_start in range(0, self.vocab_size, batch_size):
            batch_end = min(batch_start + batch_size, self.vocab_size)

            for i in range(batch_start, batch_end):
                for j in range(self.quaternion_dim):
                    # Coordenadas normalizadas baseadas na dimensão fractal e primos
                    base_lambda = (i * j * D) / (self.vocab_size * self.quaternion_dim)

                    # Aplicar ressonâncias primas
                    prime_modulation = self._apply_prime_modulation(base_lambda, i, j)
                    lambda_val = base_lambda * prime_modulation

                    t = j / self.quaternion_dim

                    # Componente complexo único baseado na equação original
                    psi = I0 * torch.sin(torch.tensor(omega * t + alpha * lambda_val)) * \
                          torch.exp(1j * torch.tensor(omega * t - k * lambda_val + beta * lambda_val**2))

                    # Normalizar psi para ter norma unitária (quaternion unitário)
                    psi_norm = psi / (torch.abs(psi) + 1e-8)

                    # Aplicar modulações primas
                    prime_resonance = self._get_prime_resonance_for_position(i, j)

                    # Distribuir psi normalizado entre componentes quaterniônicos
                    # w (real): parte real de psi normalizado
                    w = psi_norm.real

                    # x (i): parte imaginária de psi normalizado modulada por ressonância prima
                    x = psi_norm.imag * prime_resonance

                    # y (j): sin(kλ) * parte real de psi normalizado * modulação prima
                    y = torch.sin(torch.tensor(k * lambda_val)) * psi_norm.real * prime_modulation

                    # z (k): e^(-βλ²) * parte imaginária de psi normalizado
                    z = torch.exp(torch.tensor(-beta * lambda_val**2)) * psi_norm.imag

                    # Criar tensores complexos para preservar informação de fase
                    w_complex = torch.complex(w, torch.zeros_like(w))
                    x_complex = torch.complex(x, torch.zeros_like(x))
                    y_complex = torch.complex(y, torch.zeros_like(y))
                    z_complex = torch.complex(z, torch.zeros_like(z))

                    matrix[i, j] = torch.stack([w_complex, x_complex, y_complex, z_complex])

        # Normalizar energia total da matriz quaterniónica
        total_energy = self._compute_quaternion_energy(matrix)
        target_energy = self.vocab_size * self.quaternion_dim

        if total_energy.real > 0:  # Check real part for complex energy
            normalization_factor = torch.sqrt(torch.tensor(target_energy / total_energy.real))
            matrix *= normalization_factor.clone().detach()

            print(".3f")

        return matrix

    def _apply_prime_modulation_vectorized(self, base_value: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        """Aplica modulação baseada em números primos (versão vetorizada)."""
        vocab_size = i_idx.shape[0]
        quaternion_dim = j_idx.shape[1]

        # Criar índices primos vetorizados de forma mais eficiente
        prime_idx_i = (i_idx.long() % len(self.primes))
        prime_idx_j = (j_idx.long() % len(self.primes))

        # Usar gather para obter primos de forma vetorizada
        primes_tensor = torch.tensor(self.primes, dtype=torch.float32)
        primes_i = primes_tensor[prime_idx_i]
        primes_j = primes_tensor[prime_idx_j]

        # Obter ressonâncias primas de forma vetorizada
        resonances_tensor = torch.tensor([self.prime_resonances[p] for p in self.primes], dtype=torch.float32)
        resonances_i = resonances_tensor[prime_idx_i]

        # Modulação baseada na razão de primos
        modulation = (primes_i / primes_j) * resonances_i

        return 1.0 + 0.1 * torch.sin(base_value * modulation)

    def _get_prime_resonance_vectorized(self, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        """Obtém ressonância prima para posições (versão vetorizada)."""
        vocab_size = i_idx.shape[0]
        quaternion_dim = j_idx.shape[1]

        # Calcular índices primos
        prime_idx = ((i_idx.long() * self.quaternion_dim + j_idx.long()) % len(self.primes))

        # Obter ressonâncias primas de forma vetorizada
        resonances_tensor = torch.tensor([self.prime_resonances[p] for p in self.primes], dtype=torch.float32)
        resonances = resonances_tensor[prime_idx]

        return resonances.unsqueeze(-1)

    def _apply_prime_modulation(self, base_value: float, i: int, j: int) -> float:
        """Aplica modulação baseada em números primos (versão compatibilidade)."""
        # Usar índices i,j para selecionar primos
        prime_idx_i = i % len(self.primes)
        prime_idx_j = j % len(self.primes)

        prime_i = self.primes[prime_idx_i]
        prime_j = self.primes[prime_idx_j]

        # Modulação baseada na razão de primos
        modulation = (prime_i / prime_j) * self.prime_resonances[prime_i]

        return 1.0 + 0.1 * torch.sin(torch.tensor(base_value * modulation)).item()

    def _get_prime_resonance_for_position(self, i: int, j: int) -> float:
        """Obtém ressonância prima para uma posição específica (versão compatibilidade)."""
        prime_idx = (i * self.quaternion_dim + j) % len(self.primes)
        prime = self.primes[prime_idx]
        return self.prime_resonances[prime]

    def _compute_quaternion_energy(self, quaternion_matrix: torch.Tensor) -> torch.Tensor:
        """Computa energia total de uma matriz quaterniónica."""
        # Norma de Frobenius para quarteniões: soma dos quadrados de todos os componentes
        # Para números complexos, usar |z|² = z * conj(z)
        return torch.sum(quaternion_matrix * quaternion_matrix.conj())

    def validate_physical_properties(self) -> Dict[str, bool]:
        """
        Valida propriedades físicas fundamentais do sistema ΨQRH.

        Returns:
            Dicionário com resultados das validações
        """
        results = {}

        # Teste 1: Norma preservada após rotação SO(4)
        q_test = torch.randn(100, 4, dtype=torch.complex64)
        q_test = q_test / torch.norm(q_test, dim=-1, keepdim=True)  # Normalizar

        rotation_angles = torch.randn(100, 6) * 2 * torch.pi
        q_rotated = self.quaternion_ops.so4_rotation(q_test, rotation_angles)

        original_norm = torch.norm(q_test, dim=-1)
        rotated_norm = torch.norm(q_rotated, dim=-1)

        results['norm_preservation'] = torch.allclose(original_norm, rotated_norm, atol=1e-5)

        # Teste 2: Energia preservada após filtragem espectral
        original_energy = torch.sum(self.quantum_matrix * self.quantum_matrix.conj()).real.item()
        # Simular filtragem (energia deve ser preservada)
        filtered_matrix = self.quantum_matrix.clone()
        self._apply_adaptive_spectral_filtering(1.5, 0.8)  # Aplicar filtragem
        filtered_energy = torch.sum(self.quantum_matrix * self.quantum_matrix.conj()).real.item()
        self.quantum_matrix = filtered_matrix  # Restaurar

        results['energy_conservation'] = abs(filtered_energy / original_energy - 1.0) < 0.01

        # Teste 3: Quaternions unitários gerados corretamente
        test_angles = torch.randn(50, 6) * 2 * torch.pi
        q_left = torch.stack([
            torch.cos(test_angles[:, 0]/2) * torch.cos(test_angles[:, 1]/2) * torch.cos(test_angles[:, 2]/2),
            torch.sin(test_angles[:, 0]/2) * torch.cos(test_angles[:, 1]/2) * torch.cos(test_angles[:, 2]/2),
            torch.cos(test_angles[:, 0]/2) * torch.sin(test_angles[:, 1]/2) * torch.cos(test_angles[:, 2]/2),
            torch.cos(test_angles[:, 0]/2) * torch.cos(test_angles[:, 1]/2) * torch.sin(test_angles[:, 2]/2)
        ], dim=-1)

        q_left_norm = torch.norm(q_left, dim=-1)
        results['unitary_quaternions'] = torch.allclose(q_left_norm, torch.ones_like(q_left_norm), atol=1e-6)

        return results

    def _apply_adaptive_spectral_filtering(self, alpha: float, beta: float):
        """
        Aplica filtragem espectral adaptativa com conservação de energia (Parseval).
        Preserva a estrutura quaterniónica aplicando FFT separadamente a cada componente.
        Versão vetorizada para melhor desempenho.

        F(k) = exp(i α · arctan(ln(|k| + ε)))
        """
        epsilon = 1e-8

        # Computar energia total antes da filtragem (Parseval)
        original_energy = torch.sum(self.quantum_matrix * self.quantum_matrix.conj()).real.item()

        # Aplicar filtragem espectral preservando estrutura quaterniónica (vetorizada)
        # self.quantum_matrix shape: [vocab_size, quaternion_dim, 4]

        # Aplicar FFT a todos os componentes de uma vez
        freq_domain = torch.fft.fft(self.quantum_matrix, dim=1)  # FFT ao longo da dimensão quaternion_dim

        # Aplicar filtro espectral adaptativo: F(k) = exp(i α · arctan(ln(|k| + ε)))
        k_magnitude = torch.abs(freq_domain)
        spectral_filter = torch.exp(1j * alpha * torch.arctan(torch.log(k_magnitude + epsilon)))

        # Aplicar filtro
        freq_domain_filtered = freq_domain * spectral_filter

        # Aplicar IFFT para voltar ao domínio do tempo
        time_domain = torch.fft.ifft(freq_domain_filtered, dim=1)

        # Conservar energia de cada quaternion individual
        original_quaternion_energy = torch.sum(self.quantum_matrix * self.quantum_matrix.conj(), dim=[1, 2]).real  # [vocab_size]
        filtered_quaternion_energy = torch.sum(time_domain * time_domain.conj(), dim=[1, 2]).real  # [vocab_size]

        # Aplicar correção de energia por quaternion
        energy_scale = torch.sqrt(original_quaternion_energy / (filtered_quaternion_energy + 1e-8))  # [vocab_size]
        time_domain = time_domain * energy_scale.unsqueeze(-1).unsqueeze(-1)  # Broadcast para [vocab_size, quaternion_dim, 4]

        # Atualizar matriz mantendo informação de fase completa
        self.quantum_matrix = time_domain

        # Verificar conservação de energia global
        final_energy = torch.sum(self.quantum_matrix * self.quantum_matrix.conj()).real.item()
        energy_ratio = final_energy / original_energy

        print(".3f")

        # Validação numérica: energia preservada após filtragem espectral
        assert abs(energy_ratio - 1.0) < 0.01, f"Energia não preservada: {energy_ratio:.6f}"

        # Correção final se necessário (backup)
        if abs(energy_ratio - 1.0) > 0.01:
            correction_factor = torch.sqrt(torch.tensor(original_energy / final_energy))
            self.quantum_matrix *= correction_factor
            print(".3f")

    def _update_adaptation_layers(self, alpha: float, beta: float, fractal_dim: float):
        """
        Atualiza as camadas de adaptação com os novos parâmetros.
        """
        # Aqui poderíamos ajustar pesos das camadas baseado nos parâmetros
        # Por simplicidade, mantemos as camadas como estão
        pass

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Codifica texto usando a matriz quântica quaterniónica adaptada.

        Args:
            text: Texto a ser codificado

        Returns:
            Tensor quântico [len(text), hidden_size] com representação quaterniónica
        """
        if not self.current_model_params:
            print("⚠️  Matriz não adaptada a nenhum modelo. Usando parâmetros padrão.")
            self.adapt_to_model('gpt2')  # Fallback

        # Converter texto para índices com modulação prima
        char_indices = []
        for c in text[:100]:  # Limitar tamanho
            base_idx = ord(c) % self.vocab_size
            # Aplicar modulação prima ao índice
            prime_mod = self._apply_prime_modulation_to_index(base_idx, len(char_indices))
            modulated_idx = int(base_idx * prime_mod) % self.vocab_size
            char_indices.append(modulated_idx)

        # Aplicar matriz quântica quaterniónica
        with torch.no_grad():
            # Obter quarteniões: [len(text), quaternion_dim, 4]
            quaternion_encoded = self.quantum_matrix[char_indices]

            # Achatar para [len(text), hidden_size]
            flattened = quaternion_encoded.reshape(len(char_indices), -1)

            # Aplicar camadas de adaptação
            # Converter para formato adequado para conv1d
            input_tensor = flattened.transpose(0, 1).unsqueeze(0).to(torch.complex128)  # [1, hidden_size, seq_len]
    
            # Aplicar filtros espectrais
            filtered = self.adaptation_layers['spectral_filter'](input_tensor)
    
            # Aplicar rotações quaterniónicas - input should be [seq_len, hidden_size]
            rotated = self.adaptation_layers['quaternion_rotator'](filtered.squeeze(0).transpose(0, 1).unsqueeze(0)).squeeze(0)
    
            # Aplicar ressonâncias primas
            resonated = self.adaptation_layers['prime_resonator'](rotated.to(torch.complex128))

            # Normalizar energia (usar normalização customizada para complexos)
            # Calcular média e desvio padrão das partes real e imaginária separadamente
            real_part = resonated.real
            imag_part = resonated.imag

            # Normalizar parte real
            real_mean = real_part.mean(dim=-1, keepdim=True)
            real_std = real_part.std(dim=-1, keepdim=True) + 1e-8
            real_normalized = (real_part - real_mean) / real_std

            # Normalizar parte imaginária
            imag_mean = imag_part.mean(dim=-1, keepdim=True)
            imag_std = imag_part.std(dim=-1, keepdim=True) + 1e-8
            imag_normalized = (imag_part - imag_mean) / imag_std

            # Recompor tensor complexo normalizado
            normalized = torch.complex(real_normalized, imag_normalized)

            return normalized

    def _apply_prime_modulation_to_index(self, base_idx: int, position: int) -> float:
        """Aplica modulação prima a um índice de caractere."""
        prime_idx = position % len(self.primes)
        prime = self.primes[prime_idx]

        # Modulação baseada na posição e primo
        modulation = 1.0 + 0.05 * torch.sin(torch.tensor(base_idx * self.prime_resonances[prime])).item()

        return modulation

    def get_current_parameters(self) -> Optional[Dict]:
        """
        Retorna os parâmetros atuais do modelo adaptado.
        """
        return self.current_model_params

    def save_adapted_matrix(self, filepath: str):
        """
        Salva a matriz adaptada em arquivo.
        """
        state = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'current_model_params': self.current_model_params,
            'quantum_matrix': self.quantum_matrix,
            'adaptation_layers': self.adaptation_layers.state_dict()
        }

        torch.save(state, filepath)
        print(f"💾 Matriz adaptada salva em: {filepath}")

    @classmethod
    def load_adapted_matrix(cls, filepath: str) -> 'DynamicQuantumCharacterMatrix':
        """
        Carrega matriz adaptada de arquivo.
        """
        state = torch.load(filepath, map_location='cpu')

        matrix = cls(
            vocab_size=state['vocab_size'],
            hidden_size=state['hidden_size']
        )

        matrix.current_model_params = state['current_model_params']
        matrix.quantum_matrix = state['quantum_matrix']
        matrix.adaptation_layers.load_state_dict(state['adaptation_layers'])

        print(f"📁 Matriz adaptada carregada de: {filepath}")
        return matrix


# Teste da implementação
if __name__ == "__main__":
    print("🔬 Teste da Dynamic Quantum Character Matrix")
    print("=" * 50)

    # Criar matriz dinâmica
    matrix = DynamicQuantumCharacterMatrix(vocab_size=1000, hidden_size=64)

    # Testar adaptação para modelo disponível
    integrator = SpectralParametersIntegrator()
    available_models = integrator.get_available_models()

    if available_models:
        test_model = available_models[0]
        print(f"🎯 Testando adaptação para: {test_model}")

        success = matrix.adapt_to_model(test_model)

        if success:
            # Testar codificação
            test_text = "Hello quantum world"
            encoded = matrix.encode_text(test_text)

            print("✅ Codificação bem-sucedida:")
            print(f"   Texto: '{test_text}'")
            print(f"   Shape: {encoded.shape}")
            print(".3f")
            print(f"   Valores finitos: {torch.isfinite(encoded).all().item()}")

            # Salvar matriz adaptada
            matrix.save_adapted_matrix("dynamic_quantum_matrix_adapted.pt")
        else:
            print("❌ Falha na adaptação")
    else:
        print("⚠️  Nenhum modelo semântico disponível para teste")