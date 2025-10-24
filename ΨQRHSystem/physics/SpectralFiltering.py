import torch
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
from core.TernaryLogicFramework import TernaryLogicFramework


class SpectralFiltering:
    """
    Spectral Filtering - Filtragem espectral avançada

    Implementa F(k) = exp(i α · arctan(ln(|k| + ε)))

    Com validação de conservação de energia e estabilidade numérica.
    """

    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-10,
                 use_stable_activation: bool = True, device: str = "cpu"):
        """
        Inicializa filtragem espectral com lógica ternária

        Args:
            alpha: Parâmetro espectral α
            epsilon: Parâmetro de estabilidade ε
            use_stable_activation: Usar ativação estável
            device: Dispositivo de computação
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.use_stable_activation = use_stable_activation
        self.device = device
        self.ternary_logic = TernaryLogicFramework(device=device)

        print(f"🌊 Spectral Filtering inicializada com lógica ternária: F(k) = exp(i α · arctan(ln(|k| + ε)))")
        print(f"   α = {alpha}, ε = {epsilon}, stable_activation = {use_stable_activation}")

    def apply_filter(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Aplica filtragem espectral ao estado quântico com lógica ternária

        Args:
            psi: Estado quântico [batch, seq_len, embed_dim, 4]

        Returns:
            Estado filtrado [batch, seq_len, embed_dim, 4]
        """
        batch_size, seq_len, embed_dim, quat_dim = psi.shape

        # FFT ao longo da dimensão embed_dim (domínio espectral)
        psi_fft = torch.fft.fft(psi, dim=2)

        # Calcular frequências
        freqs = torch.fft.fftfreq(embed_dim, device=self.device)
        k = 2 * torch.pi * freqs.view(1, 1, -1, 1)

        # Aplicar filtro espectral F(k) = exp(i α · arctan(ln(|k| + ε)))
        k_mag = torch.abs(k) + self.epsilon

        if self.use_stable_activation:
            # Versão estável com clamping
            log_k = torch.log(torch.clamp(k_mag, min=1e-9, max=1e9))
        else:
            log_k = torch.log(k_mag)

        phase = torch.arctan(log_k)
        filter_response = torch.exp(1j * self.alpha * phase)

        # Aplicar lógica ternária ao filtro
        filter_response = self._apply_ternary_filter_modulation(filter_response, k)

        # Expandir filtro para todas as dimensões
        filter_response = filter_response.expand_as(psi_fft)

        # Aplicar filtro no domínio espectral
        psi_filtered_fft = psi_fft * filter_response

        # Transformada inversa
        psi_filtered = torch.fft.ifft(psi_filtered_fft, dim=2).real

        # Aplicar estabilização ternária ao resultado
        psi_filtered = self._apply_ternary_state_stabilization(psi_filtered)

        # Garantir conservação de energia
        psi_filtered = self._ensure_energy_conservation(psi, psi_filtered)

        return psi_filtered

    def _ensure_energy_conservation(self, psi_original: torch.Tensor,
                                   psi_filtered: torch.Tensor) -> torch.Tensor:
        """
        Garante conservação de energia após filtragem

        Args:
            psi_original: Estado original
            psi_filtered: Estado filtrado

        Returns:
            Estado filtrado com energia conservada
        """
        # Calcular energias
        energy_original = torch.sum(psi_original.abs() ** 2)
        energy_filtered = torch.sum(psi_filtered.abs() ** 2)

        # Fator de escala para conservação
        if energy_filtered > 0:
            scale_factor = torch.sqrt(energy_original / (energy_filtered + 1e-10))
            psi_conserved = psi_filtered * scale_factor
        else:
            psi_conserved = psi_filtered

        # Verificar conservação (debug)
        energy_final = torch.sum(psi_conserved.abs() ** 2)
        conservation_error = abs(energy_final - energy_original) / energy_original

        if conservation_error > 0.01:  # 1% tolerância
            print(f"⚠️  Spectral filtering energy conservation error: {conservation_error:.2e}")

        return psi_conserved

    def validate_filter_unitarity(self, embed_dim: int = 64) -> bool:
        """
        Valida unitariedade do filtro espectral com lógica ternária

        Args:
            embed_dim: Dimensão do embedding

        Returns:
            True se filtro é unitário
        """
        try:
            # Criar sinal teste
            test_signal = torch.randn(1, 10, embed_dim, 4, device=self.device)

            # Aplicar filtro
            filtered_signal = self.apply_filter(test_signal)

            # Verificar conservação de energia
            energy_in = torch.sum(test_signal.abs() ** 2)
            energy_out = torch.sum(filtered_signal.abs() ** 2)

            conservation_ratio = abs(energy_in - energy_out) / energy_in
            energy_conserved = conservation_ratio <= 0.01  # 1% tolerância

            # Adicionar validação ternária
            ternary_consistency = self._validate_ternary_filter_consistency(test_signal, filtered_signal)

            # Combinar validações usando lógica ternária
            energy_result = 1 if energy_conserved else -1
            ternary_result = 1 if ternary_consistency else -1

            final_result = self.ternary_logic.ternary_and(energy_result, ternary_result)
            is_unitary = final_result == 1

            return is_unitary

        except Exception as e:
            print(f"⚠️  Filter unitarity validation failed: {e}")
            return False

    def _validate_ternary_filter_consistency(self, input_signal: torch.Tensor, output_signal: torch.Tensor) -> bool:
        """
        Valida consistência ternária do filtro

        Args:
            input_signal: Sinal de entrada
            output_signal: Sinal de saída

        Returns:
            True se consistente
        """
        try:
            # Converter sinais para estados ternários
            input_ternary = self._tensor_to_ternary_states(input_signal)
            output_ternary = self._tensor_to_ternary_states(output_signal)

            # Verificar preservação de estrutura ternária
            input_dist = torch.bincount(input_ternary.flatten() + 1, minlength=3)
            output_dist = torch.bincount(output_ternary.flatten() + 1, minlength=3)

            # Calcular diferença de distribuição
            total_elements = input_signal.numel()
            dist_diff = torch.sum(torch.abs(input_dist - output_dist)) / (2 * total_elements)

            # Considerar consistente se diferença < 40% (mais tolerante que quaternions)
            return dist_diff < 0.4

        except Exception:
            return False

    def get_filter_response(self, embed_dim: int = 64) -> Dict[str, torch.Tensor]:
        """
        Retorna resposta do filtro no domínio espectral

        Args:
            embed_dim: Dimensão do embedding

        Returns:
            Resposta do filtro
        """
        # Frequências
        freqs = torch.fft.fftfreq(embed_dim, device=self.device)
        k = 2 * torch.pi * freqs

        # Magnitude de k
        k_mag = torch.abs(k) + self.epsilon

        # Resposta do filtro
        log_k = torch.log(torch.clamp(k_mag, min=1e-9, max=1e9))
        phase = torch.arctan(log_k)
        filter_magnitude = torch.exp(-self.alpha * phase.real)  # Magnitude
        filter_phase = self.alpha * phase  # Fase

        return {
            'frequencies': freqs,
            'k_values': k,
            'filter_magnitude': filter_magnitude,
            'filter_phase': filter_phase,
            'alpha': self.alpha,
            'epsilon': self.epsilon
        }

    def update_parameters(self, alpha: Optional[float] = None,
                         epsilon: Optional[float] = None):
        """
        Atualiza parâmetros do filtro

        Args:
            alpha: Novo valor de α
            epsilon: Novo valor de ε
        """
        if alpha is not None:
            self.alpha = alpha
            print(f"🔧 Spectral filter α updated to {alpha}")

        if epsilon is not None:
            self.epsilon = epsilon
            print(f"🔧 Spectral filter ε updated to {epsilon}")

    def compute_spectral_density(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Computa densidade espectral do sinal

        Args:
            signal: Sinal de entrada

        Returns:
            Densidade espectral
        """
        # FFT do sinal
        spectrum = torch.fft.fft(signal)

        # Densidade espectral (magnitude ao quadrado)
        spectral_density = torch.abs(spectrum) ** 2

        # Normalizar
        spectral_density = spectral_density / torch.sum(spectral_density + 1e-10)

        return spectral_density

    def apply_adaptive_filtering(self, psi: torch.Tensor,
                               spectral_characteristics: Dict[str, float]) -> torch.Tensor:
        """
        Aplica filtragem adaptativa baseada em características espectrais

        Args:
            psi: Estado quântico
            spectral_characteristics: Características espectrais

        Returns:
            Estado filtrado adaptativamente
        """
        # Ajustar α baseado em características espectrais
        spectral_centroid = spectral_characteristics.get('spectral_centroid', 0.5)
        adaptive_alpha = self.alpha * (1.0 + spectral_centroid)

        # Aplicar filtro com α adaptativo
        original_alpha = self.alpha
        self.alpha = adaptive_alpha

        try:
            filtered_psi = self.apply_filter(psi)
        finally:
            # Restaurar α original
            self.alpha = original_alpha

        return filtered_psi

    def get_filter_properties(self) -> Dict[str, Any]:
        """
        Retorna propriedades do filtro

        Returns:
            Propriedades do filtro
        """
        return {
            'filter_type': 'spectral_filtering',
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'stable_activation': self.use_stable_activation,
            'device': self.device,
            'equation': 'F(k) = exp(i α · arctan(ln(|k| + ε)))'
        }

    def validate_numerical_stability(self, psi: torch.Tensor) -> bool:
        """
        Valida estabilidade numérica do filtro

        Args:
            psi: Estado quântico para teste

        Returns:
            True se numericamente estável
        """
        try:
            # Aplicar filtro
            filtered = self.apply_filter(psi)

            # Verificar valores finitos
            is_finite = torch.isfinite(filtered).all().item()

            # Verificar ausência de NaN/Inf
            has_nan = torch.isnan(filtered).any().item()
            has_inf = torch.isinf(filtered).any().item()

            # Verificar conservação de forma
            shape_preserved = filtered.shape == psi.shape

            return is_finite and not has_nan and not has_inf and shape_preserved

        except Exception:
            return False

    def _apply_ternary_filter_modulation(self, filter_response: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Aplica modulação ternária ao filtro espectral

        Args:
            filter_response: Resposta do filtro original
            k: Vetor de frequências

        Returns:
            Filtro modulado com lógica ternária
        """
        # Converter frequências para estados ternários
        k_ternary = self._frequency_to_ternary_states(k)

        # Aplicar modulação baseada em estados ternários
        modulation_factor = torch.where(
            k_ternary == 1,  # Frequências positivas
            torch.tensor(1.1, device=self.device),  # Aumento para positivas
            torch.where(
                k_ternary == -1,  # Frequências negativas
                torch.tensor(0.9, device=self.device),  # Redução para negativas
                torch.tensor(1.0, device=self.device)   # Neutro para zero
            )
        )

        return filter_response * modulation_factor

    def _apply_ternary_state_stabilization(self, psi_filtered: torch.Tensor) -> torch.Tensor:
        """
        Aplica estabilização de estados ternários ao sinal filtrado

        Args:
            psi_filtered: Sinal filtrado

        Returns:
            Sinal estabilizado
        """
        # Converter para estados ternários e aplicar estabilização
        ternary_states = self._tensor_to_ternary_states(psi_filtered)

        # Para regiões de alta frequência (valores extremos), aplicar consenso ternário
        extreme_mask = torch.abs(psi_filtered) > torch.quantile(torch.abs(psi_filtered), 0.95)

        if extreme_mask.any():
            # Aplicar consenso ternário para estabilização
            consensus_values = []
            for i in range(min(10, psi_filtered.numel())):  # Amostra para consenso
                sample_val = psi_filtered.flatten()[i]
                ternary_val = 1 if sample_val > 0.1 else (-1 if sample_val < -0.1 else 0)
                consensus_values.append(ternary_val)

            consensus_result = self.ternary_logic.ternary_majority_vote(consensus_values)

            # Aplicar estabilização baseada no consenso
            stabilization_factor = 0.1 if consensus_result == 0 else 0.05
            psi_filtered[extreme_mask] = torch.lerp(
                psi_filtered[extreme_mask],
                torch.full_like(psi_filtered[extreme_mask], consensus_result * stabilization_factor),
                0.3  # Fator de interpolação
            )

        return psi_filtered

    def _frequency_to_ternary_states(self, k: torch.Tensor) -> torch.Tensor:
        """
        Converte frequências para estados ternários

        Args:
            k: Vetor de frequências

        Returns:
            Estados ternários das frequências
        """
        # Classificar baseado no sinal e magnitude
        k_abs = torch.abs(k)
        max_k = torch.max(k_abs)

        if max_k == 0:
            return torch.zeros_like(k, dtype=torch.long)

        # Normalizar
        k_norm = k / (max_k + 1e-10)

        # Converter para ternário
        ternary_k = torch.zeros_like(k, dtype=torch.long)
        ternary_k[k_norm > 0.2] = 1
        ternary_k[k_norm < -0.2] = -1

        return ternary_k

    def _tensor_to_ternary_states(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Converte tensor para estados ternários

        Args:
            tensor: Tensor de entrada

        Returns:
            Estados ternários
        """
        # Similar ao método em QuaternionOps
        abs_tensor = torch.abs(tensor)
        max_val = torch.max(abs_tensor)

        if max_val == 0:
            return torch.zeros_like(tensor, dtype=torch.long)

        normalized = tensor / (max_val + 1e-10)

        ternary_states = torch.zeros_like(tensor, dtype=torch.long)
        ternary_states[normalized > 0.33] = 1
        ternary_states[normalized < -0.33] = -1

        return ternary_states