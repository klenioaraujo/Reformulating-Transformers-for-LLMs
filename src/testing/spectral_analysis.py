"""
Análise Espectral Avançada para ΨQRH - Implementação Real

Análise completa do domínio de frequência sem hardcoding:
- Resposta em frequência dinâmica
- Estabilidade numérica
- Mapeamento fractal-espectral
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
from datetime import datetime
import math
import random

from core.qrh_layer import QRHLayer, QRHConfig
# from ..fractal.needle_fractal_dimension import NeedleFractalDimension  # Temporarily disabled

logger = logging.getLogger(__name__)

@dataclass
class SpectralAnalysisResult:
    """Resultado de análise espectral"""
    analysis_id: str
    analysis_name: str
    parameters: Dict[str, Any]
    frequency_response: Dict[str, Any]
    stability_metrics: Dict[str, float]
    fractal_correlations: Dict[str, float]
    execution_time: float
    timestamp: str

class SpectralAnalyzer:
    """Analisador espectral avançado sem hardcoding"""

    def __init__(self, config: QRHConfig):
        self.config = config
        self.layer = QRHLayer(config)
        # self.fractal_analyzer = NeedleFractalDimension()  # Temporarily disabled

    def _generate_dynamic_frequency_range(self) -> np.ndarray:
        """Gera range de frequências dinamicamente"""
        # Baseado na dimensionalidade do modelo
        max_freq = min(20.0, self.config.embed_dim / 5.0)
        num_points = max(50, min(500, self.config.embed_dim))

        # Log spacing para melhor resolução em baixas frequências
        log_min = math.log10(0.01)
        log_max = math.log10(max_freq)
        log_freqs = np.linspace(log_min, log_max, num_points)

        return 10 ** log_freqs

    def analyze_frequency_response_dynamic(self) -> SpectralAnalysisResult:
        """Analisa resposta em frequência dinamicamente"""
        start_time = time.time()

        try:
            frequencies = self._generate_dynamic_frequency_range()
            magnitudes = []
            phases = []
            group_delays = []

            prev_phase = None

            for freq in frequencies:
                # Sinal dinâmico baseado na frequência
                signal_length = max(128, int(20 * max(frequencies) / freq))
                t = torch.linspace(0, 2 * math.pi, signal_length)
                signal = torch.sin(2 * math.pi * freq * t)

                # Preparar para QRHLayer
                signal_4d = signal.unsqueeze(0).unsqueeze(-1).repeat(
                    1, 1, 4 * self.config.embed_dim
                )

                # Aplicar camada
                with torch.no_grad():
                    processed = self.layer(signal_4d)

                # Extrair componente principal
                processed_signal = processed[0, :, 0].cpu()

                # Análise espectral
                input_fft = torch.fft.fft(signal)
                output_fft = torch.fft.fft(processed_signal)

                # Encontrar bin da frequência fundamental
                freq_bins = torch.fft.fftfreq(signal_length, d=1.0)
                target_idx = torch.argmin(torch.abs(freq_bins - freq))

                # Magnitude e fase
                input_mag = torch.abs(input_fft[target_idx]).item()
                output_mag = torch.abs(output_fft[target_idx]).item()

                if input_mag > 1e-10:
                    magnitude_ratio = output_mag / input_mag
                else:
                    magnitude_ratio = 0.0

                phase_diff = torch.angle(output_fft[target_idx]) - torch.angle(input_fft[target_idx])
                phase_diff = (phase_diff + math.pi) % (2 * math.pi) - math.pi  # Wrap to [-π, π]

                magnitudes.append(magnitude_ratio)
                phases.append(phase_diff.item())

                # Calcular group delay
                if prev_phase is not None and len(frequencies) > 1:
                    freq_diff = freq - frequencies[len(magnitudes)-2]
                    phase_diff_curr = phase_diff.item()
                    phase_diff_prev = prev_phase

                    if abs(freq_diff) > 1e-10:
                        group_delay = -(phase_diff_curr - phase_diff_prev) / (2 * math.pi * freq_diff)
                        group_delays.append(group_delay)

                prev_phase = phase_diff.item()

            # Métricas de estabilidade
            stability_metrics = self._calculate_spectral_stability(magnitudes, phases, group_delays)

            # Correlações fractais
            fractal_correlations = self._calculate_fractal_spectral_correlations(
                magnitudes, phases, frequencies
            )

            result = SpectralAnalysisResult(
                analysis_id="SPEC_001",
                analysis_name="Resposta em Frequência Dinâmica",
                parameters={
                    'frequency_range': [frequencies[0], frequencies[-1]],
                    'num_frequencies': len(frequencies),
                    'signal_length_dynamic': True
                },
                frequency_response={
                    'frequencies': frequencies.tolist(),
                    'magnitudes': magnitudes,
                    'phases': phases,
                    'group_delays': group_delays
                },
                stability_metrics=stability_metrics,
                fractal_correlations=fractal_correlations,
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )

            return result

        except Exception as e:
            logger.error(f"Erro na análise espectral dinâmica: {e}")
            # Retornar resultado de erro
            return SpectralAnalysisResult(
                analysis_id="SPEC_001",
                analysis_name="Resposta em Frequência Dinâmica",
                parameters={'error': str(e)},
                frequency_response={},
                stability_metrics={},
                fractal_correlations={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )

    def _calculate_spectral_stability(self, magnitudes: List[float], phases: List[float],
                                    group_delays: List[float]) -> Dict[str, float]:
        """Calcula métricas de estabilidade espectral"""
        try:
            mag_array = np.array(magnitudes)
            phase_array = np.array(phases)

            metrics = {}

            # Estabilidade de magnitude
            metrics['magnitude_stability'] = 1.0 / (1.0 + np.std(mag_array))
            metrics['magnitude_flatness'] = np.exp(np.mean(np.log(mag_array + 1e-10))) / np.mean(mag_array)

            # Estabilidade de fase
            phase_variance = np.var(phase_array)
            metrics['phase_stability'] = 1.0 / (1.0 + phase_variance)

            # Linearidade de fase (para minimum phase)
            if len(phases) > 1:
                phase_diff = np.diff(phase_array)
                metrics['phase_linearity'] = 1.0 / (1.0 + np.std(phase_diff))
            else:
                metrics['phase_linearity'] = 0.0

            # Estabilidade de group delay
            if group_delays:
                gd_array = np.array(group_delays)
                metrics['group_delay_stability'] = 1.0 / (1.0 + np.std(gd_array))
                metrics['group_delay_variation'] = np.ptp(gd_array)  # Peak-to-peak
            else:
                metrics['group_delay_stability'] = 0.0
                metrics['group_delay_variation'] = 0.0

            # Métrica geral de estabilidade
            stability_metrics = [v for k, v in metrics.items() if 'stability' in k]
            metrics['overall_stability'] = np.mean(stability_metrics) if stability_metrics else 0.0

            return metrics

        except Exception as e:
            logger.error(f"Erro no cálculo de estabilidade: {e}")
            return {'error': str(e)}

    def _calculate_fractal_spectral_correlations(self, magnitudes: List[float],
                                               phases: List[float], frequencies: np.ndarray) -> Dict[str, float]:
        """Calcula correlações entre propriedades fractais e espectrais"""
        try:
            # Converter dados espectrais para análise fractal
            spectral_data = np.column_stack([magnitudes, phases])

            # Calcular dimensão fractal dos dados espectrais
            fractal_dim = self.fractal_analyzer.calculate_fractal_dimension(spectral_data)

            # Análise de correlação
            correlations = {}

            # Correlação magnitude-frequência
            if len(magnitudes) > 1:
                freq_log = np.log10(frequencies + 1e-10)
                mag_corr = np.corrcoef(freq_log, magnitudes)[0, 1]
                correlations['magnitude_frequency_correlation'] = abs(mag_corr) if not np.isnan(mag_corr) else 0.0

            # Correlação fase-frequência
            if len(phases) > 1:
                phase_corr = np.corrcoef(frequencies, phases)[0, 1]
                correlations['phase_frequency_correlation'] = abs(phase_corr) if not np.isnan(phase_corr) else 0.0

            # Complexidade espectral (entropia)
            mag_entropy = -np.sum(magnitudes * np.log(magnitudes + 1e-10))
            correlations['spectral_entropy'] = mag_entropy

            # Dimensão fractal
            correlations['fractal_dimension'] = fractal_dim

            # Slope espectral (decay rate)
            if len(magnitudes) > 2:
                # Ajustar linha aos log-dados
                log_freq = np.log10(frequencies + 1e-10)
                log_mag = np.log10(np.array(magnitudes) + 1e-10)

                valid_idx = np.isfinite(log_mag) & np.isfinite(log_freq)
                if np.sum(valid_idx) > 1:
                    slope, _ = np.polyfit(log_freq[valid_idx], log_mag[valid_idx], 1)
                    correlations['spectral_slope'] = abs(slope)
                else:
                    correlations['spectral_slope'] = 0.0
            else:
                correlations['spectral_slope'] = 0.0

            return correlations

        except Exception as e:
            logger.error(f"Erro no cálculo de correlações fractais: {e}")
            return {'error': str(e)}

    def analyze_numerical_stability(self, num_tests: int = None) -> SpectralAnalysisResult:
        """Analisa estabilidade numérica do processamento espectral"""
        start_time = time.time()

        try:
            if num_tests is None:
                num_tests = max(10, min(100, self.config.embed_dim // 4))

            condition_numbers = []
            sensitivities = []
            robustness_scores = []

            for i in range(num_tests):
                # Gerar sinal de teste com ruído controlado
                signal_length = random.randint(64, 256)
                base_signal = torch.randn(signal_length)

                # Adicionar ruído em diferentes níveis
                noise_levels = [0.001, 0.01, 0.1, 1.0]

                for noise_level in noise_levels:
                    noisy_signal = base_signal + noise_level * torch.randn(signal_length)

                    # Preparar para processamento
                    signal_4d = noisy_signal.unsqueeze(0).unsqueeze(-1).repeat(
                        1, 1, 4 * self.config.embed_dim
                    )

                    # Processar
                    with torch.no_grad():
                        processed = self.layer(signal_4d)

                    # Calcular sensibilidade
                    input_norm = torch.norm(noisy_signal).item()
                    output_norm = torch.norm(processed).item()

                    if input_norm > 1e-10 and noise_level > 1e-10:
                        sensitivity = abs(output_norm / input_norm - 1.0) / noise_level
                        sensitivities.append(sensitivity)

                    # Verificar estabilidade numérica
                    has_nan = torch.isnan(processed).any().item()
                    has_inf = torch.isinf(processed).any().item()
                    robustness = 0.0 if (has_nan or has_inf) else 1.0
                    robustness_scores.append(robustness)

            # Métricas de estabilidade
            stability_metrics = {
                'mean_sensitivity': np.mean(sensitivities) if sensitivities else 0.0,
                'max_sensitivity': max(sensitivities) if sensitivities else 0.0,
                'robustness_score': np.mean(robustness_scores) if robustness_scores else 0.0,
                'condition_number_estimate': np.mean(condition_numbers) if condition_numbers else 0.0
            }

            result = SpectralAnalysisResult(
                analysis_id="SPEC_002",
                analysis_name="Análise de Estabilidade Numérica",
                parameters={
                    'num_tests': num_tests,
                    'noise_levels': noise_levels,
                    'signal_length_range': [64, 256]
                },
                frequency_response={},
                stability_metrics=stability_metrics,
                fractal_correlations={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )

            return result

        except Exception as e:
            logger.error(f"Erro na análise de estabilidade numérica: {e}")
            return SpectralAnalysisResult(
                analysis_id="SPEC_002",
                analysis_name="Análise de Estabilidade Numérica",
                parameters={'error': str(e)},
                frequency_response={},
                stability_metrics={},
                fractal_correlations={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )

    def run_comprehensive_spectral_analysis(self) -> Dict[str, Any]:
        """Executa análise espectral completa"""
        logger.info("Iniciando análise espectral completa...")

        start_time = time.time()

        analyses = {}

        # Análise de resposta em frequência
        freq_response = self.analyze_frequency_response_dynamic()
        analyses['frequency_response'] = freq_response

        # Análise de estabilidade numérica
        stability_analysis = self.analyze_numerical_stability()
        analyses['numerical_stability'] = stability_analysis

        # Consolidar resultados
        overall_metrics = {
            'frequency_response_score': freq_response.stability_metrics.get('overall_stability', 0.0),
            'numerical_stability_score': stability_analysis.stability_metrics.get('robustness_score', 0.0),
            'fractal_complexity': freq_response.fractal_correlations.get('fractal_dimension', 0.0)
        }

        overall_metrics['composite_score'] = np.mean(list(overall_metrics.values()))

        analysis_report = {
            'analysis_type': 'COMPREHENSIVE_SPECTRAL',
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': overall_metrics,
            'detailed_analyses': analyses,
            'execution_time': time.time() - start_time,
            'config_parameters': {
                'embed_dim': self.config.embed_dim,
                'alpha': self.config.alpha
            }
        }

        logger.info(f"Análise espectral completa concluída. Score: {overall_metrics['composite_score']:.3f}")
        return analysis_report