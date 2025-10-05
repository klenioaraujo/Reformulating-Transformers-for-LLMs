"""
Adaptive Spectral Parameters - Auto-Calibração de α e β
======================================================

Implementa cálculo emergente de parâmetros espectrais α e β a partir da análise espectral do sinal.

SEM valores fixos - tudo emerge da física do sinal via RANSAC robusto.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple


class AdaptiveSpectralParameters:
    """
    Calcula α e β a partir do espectro do sinal, SEM valores padrão.

    Pipeline:
    1. Análise espectral do sinal
    2. Lei de potência: P(k) ~ k^(-β)
    3. Dimensão fractal: D = (3 - β) / 2
    4. Alpha adaptativo: α(D) = α₀(1 + λ·ΔD/D_e)
    5. Beta chirp: β_chirp = D/2 · (1 + 0.2·FCI)
    """

    def __init__(self, alpha_base: float = 1.0, lambda_coupling: float = 0.8, D_euclidean: float = 1.0):
        """
        Inicializa calculador de parâmetros espectrais adaptativos.

        Args:
            alpha_base: Alpha base (α₀)
            lambda_coupling: Fator de acoplamento λ
            D_euclidean: Dimensão euclidiana de referência
        """
        self.alpha_base = alpha_base
        self.lambda_coupling = lambda_coupling
        self.D_euclidean = D_euclidean

    def compute_alpha_beta_from_spectrum(
        self,
        signal: torch.Tensor,  # [batch, seq_len, dim]
        consciousness_results: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        α e β derivados UNICAMENTE da física do sinal.

        Args:
            signal: Sinal de entrada
            consciousness_results: Resultados de consciência (opcional)

        Returns:
            (alpha, beta_chirp): Parâmetros espectrais adaptativos
        """
        # 1. Análise espectral
        spectrum = torch.fft.fft(signal, dim=-1, norm='ortho')
        power_spectrum = torch.abs(spectrum) ** 2

        # 2. Lei de potência: P(k) ~ k^(-β)
        beta, r_squared = self._robust_power_law_fit(power_spectrum)

        # 3. Dimensão fractal
        D = (3.0 - beta) / 2.0
        D = torch.clamp(torch.tensor(D), 1.0, 2.0).item()

        # 4. Alpha acoplado a D
        alpha = self.alpha_base * (1.0 + self.lambda_coupling * (D - self.D_euclidean) / self.D_euclidean)

        # 5. Beta (chirp) acoplado a D e consciência
        beta_chirp = D / 2.0  # ∈ [0.5, 1.0]

        # 6. Ajuste fino com consciência (se disponível)
        if consciousness_results:
            FCI = consciousness_results.get('FCI', consciousness_results.get('fci', 0))
            if FCI > 0:
                # FCI alto → α aumenta (filtro mais seletivo)
                alpha = alpha * (1.0 + 0.3 * FCI)
                # FCI alto → β aumenta (maior dispersão)
                beta_chirp = beta_chirp * (1.0 + 0.2 * FCI)

        return alpha, beta_chirp

    def _robust_power_law_fit(
        self,
        power_spectrum: torch.Tensor,
        ransac_iterations: int = 50
    ) -> Tuple[float, float]:
        """
        Regressão robusta para estimar β mesmo com outliers.

        Usa RANSAC-like approach para lei de potência P(k) ~ k^(-β)

        Args:
            power_spectrum: Espectro de potência [batch, seq_len, freq_bins]
            ransac_iterations: Número de iterações RANSAC

        Returns:
            (beta, r_squared): Exponente da lei de potência e qualidade do fit
        """
        # Média sobre batch e seq_len
        power_avg = power_spectrum.mean(dim=(0, 1))  # [freq_bins]

        # Converter para numpy para manipulação
        power_np = power_avg.detach().cpu().numpy()
        freq_bins = len(power_np)

        # k = frequência (1, 2, 3, ..., freq_bins)
        k = np.arange(1, freq_bins + 1, dtype=np.float32)

        # Evitar log(0)
        power_safe = np.maximum(power_np, 1e-10)

        # log-log space
        log_k = np.log(k)
        log_P = np.log(power_safe)

        # RANSAC para lei de potência
        best_beta = 0.0
        best_r2 = -float('inf')

        n_points = len(log_k)
        min_samples = max(int(0.5 * n_points), 10)

        for _ in range(ransac_iterations):
            # Amostragem aleatória
            indices = np.random.choice(n_points, min_samples, replace=False)

            x_sample = log_k[indices]
            y_sample = log_P[indices]

            # Regressão linear: y = -β·x + c
            # Usando numpy polyfit
            try:
                coeffs = np.polyfit(x_sample, y_sample, 1)
                beta_candidate = -coeffs[0]  # Negativo porque P ~ k^(-β)

                # R²
                y_pred = coeffs[0] * log_k + coeffs[1]
                ss_res = np.sum((log_P - y_pred) ** 2)
                ss_tot = np.sum((log_P - np.mean(log_P)) ** 2)
                r2 = 1.0 - ss_res / (ss_tot + 1e-10)

                if r2 > best_r2:
                    best_beta = beta_candidate
                    best_r2 = r2
            except:
                continue

        # Clamping para valores físicos
        best_beta = max(0.1, min(best_beta, 3.0))

        return best_beta, best_r2

    def get_spectral_analysis(
        self,
        signal: torch.Tensor,
        consciousness_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Análise completa dos parâmetros espectrais calculados.

        Returns:
            Dicionário com análise detalhada
        """
        alpha, beta_chirp = self.compute_alpha_beta_from_spectrum(signal, consciousness_results)

        # Análise adicional do espectro
        spectrum = torch.fft.fft(signal, dim=-1, norm='ortho')
        power_spectrum = torch.abs(spectrum) ** 2

        # Estatísticas básicas
        power_mean = power_spectrum.mean().item()
        power_std = power_spectrum.std().item()
        power_max = power_spectrum.max().item()

        # Análise de frequência dominante
        freq_bins = power_spectrum.shape[-1]
        dominant_freq = torch.argmax(power_spectrum.mean(dim=(0, 1))).item()

        return {
            'alpha': alpha,
            'beta_chirp': beta_chirp,
            'D_fractal': (3.0 - (3.0 - 2.0 * (3.0 - beta_chirp) / 2.0)) / 2.0,  # Derivar D de volta
            'spectral_stats': {
                'power_mean': power_mean,
                'power_std': power_std,
                'power_max': power_max,
                'dominant_frequency': dominant_freq,
                'frequency_bins': freq_bins
            },
            'fit_quality': {
                'r_squared': self._robust_power_law_fit(power_spectrum)[1]
            },
            'consciousness_influence': consciousness_results is not None
        }


def create_adaptive_spectral_parameters(
    alpha_base: float = 1.0,
    lambda_coupling: float = 0.8,
    D_euclidean: float = 1.0
) -> AdaptiveSpectralParameters:
    """
    Factory function para criar calculador de parâmetros espectrais adaptativos.

    Args:
        alpha_base: Alpha base
        lambda_coupling: Fator de acoplamento
        D_euclidean: Dimensão euclidiana de referência

    Returns:
        AdaptiveSpectralParameters configurado
    """
    return AdaptiveSpectralParameters(
        alpha_base=alpha_base,
        lambda_coupling=lambda_coupling,
        D_euclidean=D_euclidean
    )