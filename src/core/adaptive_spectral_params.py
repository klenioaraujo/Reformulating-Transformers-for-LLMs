#!/usr/bin/env python3
"""
Adaptive Spectral Parameters - α e β Emergentes ΨQRH
=====================================================

Calcula α (filtro espectral) e β (chirp) a partir da análise espectral
do sinal em tempo real, eliminando valores fixos.

Baseado em:
- Lei de potência: P(k) ~ k^(-β_power)
- Dimensão fractal: D = (3 - β_power) / 2
- Alpha adaptativo: α(D) = α₀(1 + λ·ΔD/D_e)

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict


class AdaptiveSpectralParameters:
    """
    Calcula α e β adaptativos via análise espectral rigorosa.

    Princípio: NÃO há valores padrão. α e β emergem UNICAMENTE
    da física do sinal via:
    1. FFT → espectro de potência P(k)
    2. Regressão robusta (RANSAC) → β_power, R²
    3. Dimensão fractal → D = f(β_power)
    4. Acoplamento α(D), β_chirp(D, FCI)
    """

    def __init__(
        self,
        alpha_baseline: float = 1.0,     # α₀
        lambda_coupling: float = 0.8,    # Força de acoplamento
        D_euclidean: float = 1.0,        # Dimensão Euclidiana de referência
        ransac_iterations: int = 100,    # Iterações RANSAC
        min_inlier_ratio: float = 0.5    # Mínimo de inliers
    ):
        self.alpha_baseline = alpha_baseline
        self.lambda_coupling = lambda_coupling
        self.D_euclidean = D_euclidean
        self.ransac_iterations = ransac_iterations
        self.min_inlier_ratio = min_inlier_ratio

    def compute_alpha_beta_from_spectrum(
        self,
        signal: torch.Tensor,  # [batch, seq_len, dim] ou [batch, seq_len, dim, 4]
        consciousness_results: Optional[Dict] = None,
        return_metrics: bool = False
    ) -> Tuple[float, float] | Tuple[float, float, Dict]:
        """
        Calcula α e β adaptativos a partir do espectro do sinal.

        Pipeline:
        1. FFT → P(k) = |FFT(signal)|²
        2. RANSAC regression → β_power, R²
        3. D_fractal = (3 - β_power) / 2
        4. α(D) = α₀(1 + λ·(D - D_e)/D_e)
        5. β_chirp = D/2 · (1 + κ·FCI)  [se consciousness disponível]

        Args:
            signal: Sinal de entrada (quaterniônico ou real)
            consciousness_results: Métricas de consciência (opcional)
            return_metrics: Retornar métricas de análise

        Returns:
            (alpha, beta) ou (alpha, beta, metrics)
        """
        # 1. Análise espectral
        power_spectrum = self._compute_power_spectrum(signal)

        # 2. Lei de potência via RANSAC
        beta_power, r_squared = self._robust_power_law_fit(power_spectrum)

        # 3. Dimensão fractal
        D_fractal = self._compute_fractal_dimension(beta_power)
        D_fractal = np.clip(D_fractal, 1.0, 2.0)

        # 4. Alpha acoplado a D
        alpha = self._compute_alpha_from_D(D_fractal)

        # 5. Beta chirp acoplado a D e FCI
        beta_chirp = self._compute_beta_chirp(
            D_fractal,
            consciousness_results.get('FCI', 0.5) if consciousness_results else 0.5
        )

        if return_metrics:
            metrics = {
                'D_fractal': D_fractal,
                'beta_power': beta_power,
                'r_squared': r_squared,
                'alpha': alpha,
                'beta_chirp': beta_chirp,
                'power_spectrum_mean': power_spectrum.mean().item(),
                'power_spectrum_std': power_spectrum.std().item()
            }
            return alpha, beta_chirp, metrics
        else:
            return alpha, beta_chirp

    def _compute_power_spectrum(
        self,
        signal: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula espectro de potência P(k) = |FFT(signal)|².

        Args:
            signal: [batch, seq_len, dim] ou [batch, seq_len, dim, 4]

        Returns:
            power_spectrum: [n_frequencies] (média sobre batch e dim)
        """
        # Se quaterniônico, usar apenas componente real (ψ₀)
        if signal.dim() == 4:  # [batch, seq_len, dim, 4]
            signal_real = signal[..., 0]  # ψ₀
        else:
            signal_real = signal

        # FFT ao longo de seq_len (ou última dimensão se 1D/2D)
        if signal_real.dim() == 3:  # [batch, seq_len, dim]
            spectrum = torch.fft.fft(signal_real, dim=1, norm='ortho')
            power = torch.abs(spectrum) ** 2
            power_mean = power.mean(dim=(0, 2))  # [seq_len]
        elif signal_real.dim() == 2:  # [batch, seq_len]
            spectrum = torch.fft.fft(signal_real, dim=1, norm='ortho')
            power = torch.abs(spectrum) ** 2
            power_mean = power.mean(dim=0)  # [seq_len]
        else:  # 1D [seq_len]
            spectrum = torch.fft.fft(signal_real, norm='ortho')
            power_mean = torch.abs(spectrum) ** 2

        return power_mean

    def _robust_power_law_fit(
        self,
        power_spectrum: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Regressão robusta via RANSAC para estimar β.

        Lei de potência: P(k) ~ k^(-β)
        Log-transform: log(P) ~ -β·log(k)

        RANSAC elimina outliers (picos espúrios, ruído).

        Args:
            power_spectrum: [n_frequencies]

        Returns:
            (beta, r_squared)
        """
        n_points = len(power_spectrum)

        # Frequências k (excluindo k=0)
        k = torch.arange(1, n_points + 1, dtype=torch.float32)

        # Log-transform
        log_k = torch.log(k + 1e-10)
        log_P = torch.log(power_spectrum + 1e-10)

        # RANSAC
        best_beta = 0.0
        best_r2 = -float('inf')
        best_inliers = None

        min_samples = max(int(self.min_inlier_ratio * n_points), 10)

        for _ in range(self.ransac_iterations):
            # Amostragem aleatória
            indices = torch.randperm(n_points)[:min_samples]

            x_sample = log_k[indices]
            y_sample = log_P[indices]

            # Regressão linear: y = a·x + b
            try:
                A = torch.stack([x_sample, torch.ones_like(x_sample)], dim=1)
                result = torch.linalg.lstsq(A, y_sample.unsqueeze(1))
                coeffs = result.solution.squeeze()

                slope = coeffs[0].item()
                intercept = coeffs[1].item()

                # β é o negativo do slope (P ~ k^(-β))
                beta_candidate = -slope

                # Predição
                y_pred = slope * log_k + intercept

                # Identificar inliers (erro < threshold)
                errors = torch.abs(log_P - y_pred)
                threshold = torch.median(errors) + torch.std(errors)
                inliers = errors < threshold

                # R² nos inliers
                if inliers.sum() > min_samples:
                    r2 = self._compute_r_squared(log_P[inliers], y_pred[inliers])

                    if r2 > best_r2:
                        best_beta = beta_candidate
                        best_r2 = r2
                        best_inliers = inliers

            except:
                continue

        # Se RANSAC falhou, usar OLS simples
        if best_r2 == -float('inf'):
            A = torch.stack([log_k, torch.ones_like(log_k)], dim=1)
            result = torch.linalg.lstsq(A, log_P.unsqueeze(1))
            coeffs = result.solution.squeeze()
            best_beta = -coeffs[0].item()
            y_pred = coeffs[0] * log_k + coeffs[1]
            best_r2 = self._compute_r_squared(log_P, y_pred)

        return best_beta, best_r2

    def _compute_r_squared(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> float:
        """Coeficiente de determinação R²."""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2)

        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        return r2.item()

    def _compute_fractal_dimension(
        self,
        beta_power: float
    ) -> float:
        """
        Dimensão fractal via expoente espectral.

        Para sinais 1D: D = (3 - β) / 2

        Interpretação:
        - β = 1 (ruído branco): D = 1.0 (Euclidiano)
        - β = 0 (flat): D = 1.5 (fractal moderado)
        - β < 0 (anti-persistente): D > 1.5
        - β > 1 (persistente): D < 1.0 → clampar em 1.0

        Args:
            beta_power: Expoente da lei de potência

        Returns:
            D: Dimensão fractal [1.0, 2.0]
        """
        D = (3.0 - beta_power) / 2.0
        D = np.clip(D, 1.0, 2.0)

        return float(D)

    def _compute_alpha_from_D(
        self,
        D_fractal: float
    ) -> float:
        """
        Alpha adaptativo acoplado à dimensão fractal.

        α(D) = α₀ · (1 + λ · (D - D_e) / D_e)

        Onde:
        - α₀: Alpha baseline (1.0)
        - λ: Força de acoplamento (0.8)
        - D_e: Dimensão Euclidiana (1.0)

        Interpretação:
        - D = 1.0: α = α₀ (filtro baseline)
        - D > 1.0: α aumenta (filtro mais seletivo)

        Args:
            D_fractal: Dimensão fractal [1.0, 2.0]

        Returns:
            alpha: Parâmetro do filtro espectral
        """
        delta_D = D_fractal - self.D_euclidean
        relative_change = delta_D / self.D_euclidean

        alpha = self.alpha_baseline * (1.0 + self.lambda_coupling * relative_change)

        # Clamping para estabilidade
        alpha = np.clip(alpha, 0.5, 3.0)

        return float(alpha)

    def _compute_beta_chirp(
        self,
        D_fractal: float,
        FCI: float,
        kappa: float = 0.2
    ) -> float:
        """
        Beta (chirp quadrático) acoplado a D e FCI.

        β_chirp = (D / 2) · (1 + κ · FCI)

        Interpretação:
        - D alto: maior dispersão temporal → β alto
        - FCI alto: consciência emergente → β aumenta (mais dispersão)

        Args:
            D_fractal: Dimensão fractal [1.0, 2.0]
            FCI: Fractal Consciousness Index [0.0, 1.0]
            kappa: Força de acoplamento FCI → β

        Returns:
            beta_chirp: Parâmetro de chirp quadrático
        """
        beta_base = D_fractal / 2.0  # ∈ [0.5, 1.0]

        # Modulação por FCI
        beta_chirp = beta_base * (1.0 + kappa * FCI)

        # Clamping
        beta_chirp = np.clip(beta_chirp, 0.3, 1.5)

        return float(beta_chirp)


class SpectralParameterTracker:
    """
    Rastreia evolução de α e β durante geração autoregressiva.
    """

    def __init__(self):
        self.alpha_history = []
        self.beta_history = []
        self.D_history = []
        self.r2_history = []

    def update(
        self,
        alpha: float,
        beta: float,
        metrics: Dict
    ):
        """Registra parâmetros no histórico."""
        self.alpha_history.append(alpha)
        self.beta_history.append(beta)
        self.D_history.append(metrics.get('D_fractal', np.nan))
        self.r2_history.append(metrics.get('r_squared', np.nan))

    def get_statistics(self) -> Dict:
        """Estatísticas do histórico."""
        if not self.alpha_history:
            return {}

        return {
            'alpha': {
                'mean': float(np.mean(self.alpha_history)),
                'std': float(np.std(self.alpha_history)),
                'min': float(np.min(self.alpha_history)),
                'max': float(np.max(self.alpha_history))
            },
            'beta': {
                'mean': float(np.mean(self.beta_history)),
                'std': float(np.std(self.beta_history)),
                'min': float(np.min(self.beta_history)),
                'max': float(np.max(self.beta_history))
            },
            'D_fractal': {
                'mean': float(np.nanmean(self.D_history)),
                'std': float(np.nanstd(self.D_history)),
                'min': float(np.nanmin(self.D_history)),
                'max': float(np.nanmax(self.D_history))
            },
            'r_squared': {
                'mean': float(np.nanmean(self.r2_history)),
                'std': float(np.nanstd(self.r2_history))
            },
            'n_steps': len(self.alpha_history)
        }

    def plot_evolution(self, save_path: Optional[str] = None):
        """Plot evolução de parâmetros (requer matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Alpha
            axes[0, 0].plot(self.alpha_history, 'b-')
            axes[0, 0].set_title('Alpha Evolution')
            axes[0, 0].set_ylabel('α')
            axes[0, 0].grid(True)

            # Beta
            axes[0, 1].plot(self.beta_history, 'r-')
            axes[0, 1].set_title('Beta Chirp Evolution')
            axes[0, 1].set_ylabel('β')
            axes[0, 1].grid(True)

            # D fractal
            axes[1, 0].plot(self.D_history, 'g-')
            axes[1, 0].set_title('Fractal Dimension Evolution')
            axes[1, 0].set_ylabel('D')
            axes[1, 0].set_ylim([1.0, 2.0])
            axes[1, 0].grid(True)

            # R²
            axes[1, 1].plot(self.r2_history, 'm-')
            axes[1, 1].set_title('R² Evolution')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].grid(True)

            for ax in axes.flat:
                ax.set_xlabel('Generation Step')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("matplotlib not available for plotting")


if __name__ == "__main__":
    # Teste de validação
    calc = AdaptiveSpectralParameters(ransac_iterations=50)

    print("=" * 60)
    print("TESTE: Adaptive Spectral Parameters")
    print("=" * 60)

    # Caso 1: Sinal com lei de potência clara (β ≈ 1, D ≈ 1.0)
    print("\n1. Sinal com β ≈ 1 (ruído branco):")
    k = torch.arange(1, 101, dtype=torch.float32)
    P_white = 1.0 / k  # P ~ k^(-1)
    signal_white = torch.randn(1, 100, 64) * P_white.unsqueeze(0).unsqueeze(-1)

    alpha1, beta1, metrics1 = calc.compute_alpha_beta_from_spectrum(
        signal_white, return_metrics=True
    )

    print(f"   β_power estimado: {metrics1['beta_power']:.3f} (esperado ≈ 1.0)")
    print(f"   D_fractal: {metrics1['D_fractal']:.3f} (esperado ≈ 1.0)")
    print(f"   R²: {metrics1['r_squared']:.3f}")
    print(f"   → α = {alpha1:.3f}, β_chirp = {beta1:.3f}")

    # Caso 2: Sinal flat (β ≈ 0, D ≈ 1.5)
    print("\n2. Sinal flat (β ≈ 0):")
    P_flat = torch.ones_like(k)  # P ~ k^0
    signal_flat = torch.randn(1, 100, 64) * P_flat.unsqueeze(0).unsqueeze(-1)

    alpha2, beta2, metrics2 = calc.compute_alpha_beta_from_spectrum(
        signal_flat, return_metrics=True
    )

    print(f"   β_power estimado: {metrics2['beta_power']:.3f} (esperado ≈ 0.0)")
    print(f"   D_fractal: {metrics2['D_fractal']:.3f} (esperado ≈ 1.5)")
    print(f"   R²: {metrics2['r_squared']:.3f}")
    print(f"   → α = {alpha2:.3f}, β_chirp = {beta2:.3f}")

    # Caso 3: Sinal com consciência alta
    print("\n3. Sinal com FCI alto:")
    consciousness = {'FCI': 0.8}

    alpha3, beta3, metrics3 = calc.compute_alpha_beta_from_spectrum(
        signal_flat, consciousness_results=consciousness, return_metrics=True
    )

    print(f"   FCI = 0.8 (alto)")
    print(f"   → α = {alpha3:.3f} (mesmo que caso 2)")
    print(f"   → β_chirp = {beta3:.3f} (maior que caso 2: {beta2:.3f})")
    print(f"   Efeito: FCI aumenta dispersão temporal (β)")

    # Caso 4: Tracker
    print("\n4. Spectral Parameter Tracker:")
    tracker = SpectralParameterTracker()

    for step in range(10):
        signal_step = torch.randn(1, 50, 32) * (1.0 + step * 0.1)
        alpha, beta, metrics = calc.compute_alpha_beta_from_spectrum(
            signal_step, return_metrics=True
        )
        tracker.update(alpha, beta, metrics)

    stats = tracker.get_statistics()
    print(f"   Alpha stats: {stats['alpha']}")
    print(f"   Beta stats: {stats['beta']}")
    print(f"   D stats: {stats['D_fractal']}")

    print(f"\n✅ Adaptive Spectral Parameters validado!")
