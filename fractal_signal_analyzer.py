#!/usr/bin/env python3
"""
Fractal Signal Analyzer for ΨQRH
=================================

Sistema de análise de sinais fractais para processamento físico puro.
Implementa análise espectral adaptativa e cálculo de dimensão fractal em tempo real.

Características principais:
- Análise espectral FFT adaptativa com janelas fractais
- Cálculo de dimensão fractal via box-counting
- Preservação de invariantes topológicos
- Integração com sistema quaterniónico

Autor: Kilo Code (Sistema ΨQRH)
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import scipy.signal
import scipy.fft


@dataclass
class FractalProperties:
    """Propriedades fractais de um sinal."""
    dimension: float
    lacunarity: float
    persistence: float
    spectral_exponent: float
    correlation_dimension: float
    multifractal_spectrum: Dict[str, float]


class FractalSignalAnalyzer:
    """
    Analisador de sinais fractais para processamento físico puro.

    Implementa análise espectral e cálculo de dimensão fractal
    com preservação de propriedades topológicas.
    """

    def __init__(self, device: str = "cpu", fractal_depth: int = 8):
        """
        Inicializa o analisador de sinais fractais.

        Args:
            device: Dispositivo para tensores
            fractal_depth: Profundidade máxima da análise fractal
        """
        self.device = device
        self.fractal_depth = fractal_depth

        # Parâmetros de análise
        self.min_box_size = 2
        self.max_box_size = 512
        self.fft_window_size = 1024

        # Cache para cálculos
        self.fractal_cache = {}

        print("🔬 FractalSignalAnalyzer initialized")
        print(f"   📊 Fractal depth: {fractal_depth}, FFT window: {self.fft_window_size}")

    def analyze_text_signal(self, tokens: List[str]) -> torch.Tensor:
        """
        Analisa sinal fractal de texto tokenizado.

        Args:
            tokens: Lista de tokens do texto

        Returns:
            Sinal fractal multidimensional
        """
        # Converter tokens para representação numérica
        token_signal = self._tokens_to_signal(tokens)

        # Aplicar análise fractal
        fractal_signal = self._compute_fractal_signal(token_signal)

        return fractal_signal

    def _tokens_to_signal(self, tokens: List[str]) -> torch.Tensor:
        """
        Converte tokens para sinal numérico multidimensional.

        Cada token é mapeado para um vetor baseado em suas propriedades quânticas.
        """
        signal_components = []

        for token in tokens:
            # Representação básica do token
            token_code = sum(ord(c) for c in token) / len(token) if token else 32.0
            normalized_code = token_code / 255.0

            # Componentes do sinal baseadas em propriedades físicas
            frequency = 1.0 / (normalized_code + 0.1)
            amplitude = math.sin(2 * math.pi * normalized_code)
            phase = math.cos(2 * math.pi * normalized_code * 2)
            coherence = 0.5 + 0.5 * math.sin(2 * math.pi * normalized_code * 3)

            # Vetor multidimensional
            token_vector = torch.tensor([
                normalized_code,      # Valor base
                frequency,           # Frequência
                amplitude,           # Amplitude
                phase,              # Fase
                coherence,          # Coerência
                len(token) / 10.0,  # Comprimento normalizado
                hash(token) % 100 / 100.0  # Hash normalizado para variabilidade
            ], dtype=torch.float32, device=self.device)

            signal_components.append(token_vector)

        # Concatenar em tensor
        if signal_components:
            signal = torch.stack(signal_components, dim=0)
        else:
            signal = torch.zeros(1, 7, dtype=torch.float32, device=self.device)

        return signal

    def _compute_fractal_signal(self, token_signal: torch.Tensor) -> torch.Tensor:
        """
        Computa sinal fractal a partir do sinal de tokens.

        Aplica transformadas fractais e análise espectral.
        """
        # Handle different tensor shapes
        if token_signal.dim() == 2:
            # Shape: [seq_len, features]
            seq_len, features = token_signal.shape
            batch_size = 1
            token_signal = token_signal.unsqueeze(0)  # Add batch dimension
        elif token_signal.dim() == 3:
            # Shape: [batch_size, seq_len, features]
            batch_size, seq_len, features = token_signal.shape
        else:
            # Unexpected shape, reshape to [1, length, 1]
            token_signal = token_signal.flatten().unsqueeze(0).unsqueeze(-1)
            batch_size, seq_len, features = token_signal.shape

        fractal_components = []

        for feature_idx in range(features):
            feature_signal = token_signal[:, :, feature_idx]

            # Análise fractal para cada feature
            fractal_feature = self._analyze_fractal_feature(feature_signal)
            fractal_components.append(fractal_feature)

        # Combinar componentes fractais
        if fractal_components:
            fractal_signal = torch.stack(fractal_components, dim=-1)
        else:
            fractal_signal = torch.zeros(batch_size, 5, features, dtype=torch.float32, device=self.device)

        return fractal_signal

    def _analyze_fractal_feature(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Analisa componente fractal individual.

        Aplica box-counting e análise espectral.
        """
        # Garantir que o sinal seja 1D para análise
        if signal.dim() > 1:
            signal = signal.flatten()

        signal_np = signal.detach().cpu().numpy()

        # Calcular dimensão fractal
        fractal_dimension = self.compute_fractal_dimension(signal_np)

        # Análise espectral
        spectral_features = self._compute_spectral_features(signal_np)

        # Combinar features
        fractal_features = torch.tensor([
            fractal_dimension,
            spectral_features['dominant_frequency'],
            spectral_features['spectral_centroid'],
            spectral_features['spectral_rolloff'],
            spectral_features['spectral_flux']
        ], dtype=torch.float32, device=self.device)

        return fractal_features

    def compute_fractal_dimension(self, signal: np.ndarray) -> float:
        """
        Computa dimensão fractal via box-counting.

        Implementa algoritmo de box-counting para sinais 1D.
        """
        if len(signal) < self.min_box_size:
            return 1.0  # Dimensão euclidiana mínima

        # Preparar sinal para análise
        signal = np.abs(signal)  # Usar magnitude
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)

        # Tamanhos de box
        box_sizes = []
        counts = []

        for box_size in range(self.min_box_size, min(self.max_box_size, len(signal) // 2)):
            # Box counting para sinal 1D
            count = self._box_count_1d(signal, box_size)
            if count > 0:
                box_sizes.append(box_size)
                counts.append(count)

        if len(box_sizes) < 2:
            return 1.0

        # Regressão linear para estimar dimensão fractal
        log_boxes = np.log(box_sizes)
        log_counts = np.log(counts)

        # Evitar valores infinitos
        valid_mask = np.isfinite(log_boxes) & np.isfinite(log_counts)
        if not np.any(valid_mask):
            return 1.0

        log_boxes = log_boxes[valid_mask]
        log_counts = log_counts[valid_mask]

        if len(log_boxes) < 2:
            return 1.0

        # Regressão linear: log(count) = -dimension * log(box_size) + c
        try:
            coeffs = np.polyfit(log_boxes, log_counts, 1)
            fractal_dimension = -coeffs[0]  # Dimensão é o coeficiente angular negativo
            fractal_dimension = np.clip(fractal_dimension, 1.0, 2.0)  # Limitar a intervalo físico
        except:
            fractal_dimension = 1.0

        return fractal_dimension

    def _box_count_1d(self, signal: np.ndarray, box_size: int) -> int:
        """
        Conta boxes não-vazios para sinal 1D.

        Divide o sinal em boxes e conta quantos contêm valores acima do threshold.
        """
        if box_size <= 0 or len(signal) < box_size:
            return 0

        # Threshold adaptativo baseado na amplitude do sinal
        threshold = np.mean(signal) + 0.1 * np.std(signal)

        count = 0
        for i in range(0, len(signal), box_size):
            box = signal[i:i + box_size]
            if len(box) > 0 and np.max(box) > threshold:
                count += 1

        return max(count, 1)  # Garantir pelo menos 1 para evitar log(0)

    def _compute_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Computa features espectrais do sinal.

        Usa FFT para análise espectral.
        """
        # Aplicar windowing
        try:
            window = scipy.signal.hann(len(signal))
        except AttributeError:
            try:
                # Fallback para hamming window se hann não estiver disponível
                window = scipy.signal.hamming(len(signal))
            except AttributeError:
                # Fallback para numpy se scipy.signal não tiver windows
                window = np.ones(len(signal))  # Rectangular window
        windowed_signal = signal * window

        # FFT
        fft_result = scipy.fft.fft(windowed_signal)
        freqs = scipy.fft.fftfreq(len(signal))

        # Magnitude do espectro
        magnitude = np.abs(fft_result)
        magnitude = magnitude[:len(magnitude)//2]  # Apenas frequências positivas
        freqs = freqs[:len(freqs)//2]

        # Features espectrais
        features = {}

        # Frequência dominante
        dominant_idx = np.argmax(magnitude)
        features['dominant_frequency'] = abs(freqs[dominant_idx]) if len(freqs) > 0 else 0.0

        # Centróide espectral
        if np.sum(magnitude) > 0:
            features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            features['spectral_centroid'] = 0.0

        # Spectral rolloff (frequência que contém 85% da energia)
        cumulative_energy = np.cumsum(magnitude**2)
        total_energy = cumulative_energy[-1]
        if total_energy > 0:
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        else:
            features['spectral_rolloff'] = 0.0

        # Spectral flux (mudança espectral)
        features['spectral_flux'] = np.mean(np.diff(magnitude))

        return features

    def compute_dimension(self, fractal_signal: torch.Tensor) -> float:
        """
        Computa dimensão fractal agregada do sinal.

        Args:
            fractal_signal: Sinal fractal multidimensional

        Returns:
            Dimensão fractal média
        """
        # Agregar dimensões de todas as componentes
        dimensions = []

        for i in range(fractal_signal.shape[-1]):
            if i == 0:  # Primeira componente é sempre a dimensão fractal
                dimension = fractal_signal[..., i].mean().item()
                dimensions.append(dimension)

        if dimensions:
            return np.mean(dimensions)
        else:
            return 1.0

    def get_fractal_properties(self, signal: torch.Tensor) -> FractalProperties:
        """
        Computa propriedades fractais completas do sinal.

        Args:
            signal: Sinal de entrada

        Returns:
            Propriedades fractais completas
        """
        signal_np = signal.detach().cpu().numpy().flatten()

        # Dimensão fractal
        dimension = self.compute_fractal_dimension(signal_np)

        # Lacunarity (gaps no fractal)
        lacunarity = self._compute_lacunarity(signal_np)

        # Persistence (autocorrelação)
        persistence = self._compute_persistence(signal_np)

        # Spectral exponent (decaimento espectral)
        spectral_exponent = self._compute_spectral_exponent(signal_np)

        # Correlation dimension
        correlation_dimension = self._compute_correlation_dimension(signal_np)

        # Multifractal spectrum (simplificado)
        multifractal_spectrum = {
            'width': abs(dimension - correlation_dimension),
            'asymmetry': (dimension - correlation_dimension) / (dimension + correlation_dimension + 1e-8),
            'peak_position': (dimension + correlation_dimension) / 2.0
        }

        return FractalProperties(
            dimension=dimension,
            lacunarity=lacunarity,
            persistence=persistence,
            spectral_exponent=spectral_exponent,
            correlation_dimension=correlation_dimension,
            multifractal_spectrum=multifractal_spectrum
        )

    def _compute_lacunarity(self, signal: np.ndarray) -> float:
        """Computa lacunarity do sinal."""
        if len(signal) < 10:
            return 1.0

        # Lacunarity baseada na variância local
        window_size = min(10, len(signal) // 3)
        local_vars = []

        for i in range(len(signal) - window_size + 1):
            window = signal[i:i + window_size]
            local_vars.append(np.var(window))

        if local_vars:
            mean_var = np.mean(local_vars)
            var_var = np.var(local_vars)
            lacunarity = var_var / (mean_var**2 + 1e-8) if mean_var > 0 else 1.0
            return min(lacunarity, 10.0)  # Limitar valor
        else:
            return 1.0

    def _compute_persistence(self, signal: np.ndarray) -> float:
        """Computa persistence (autocorrelação) do sinal."""
        if len(signal) < 5:
            return 0.0

        # Autocorrelação simples
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Apenas lags positivos

        if len(autocorr) > 1 and autocorr[0] > 0:
            persistence = autocorr[1] / autocorr[0]  # Primeiro lag
            return np.clip(persistence, -1.0, 1.0)
        else:
            return 0.0

    def _compute_spectral_exponent(self, signal: np.ndarray) -> float:
        """Computa expoente espectral (beta) do sinal."""
        if len(signal) < 32:
            return 0.0

        # FFT para obter espectro
        fft_result = scipy.fft.fft(signal)
        freqs = scipy.fft.fftfreq(len(signal))
        magnitude = np.abs(fft_result)

        # Apenas frequências positivas
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        magnitude_pos = magnitude[pos_mask]

        if len(freqs_pos) < 10:
            return 0.0

        # Regressão log-log para estimar expoente
        log_freqs = np.log(freqs_pos)
        log_magnitude = np.log(magnitude_pos + 1e-8)

        valid_mask = np.isfinite(log_freqs) & np.isfinite(log_magnitude)
        if not np.any(valid_mask) or np.sum(valid_mask) < 5:
            return 0.0

        try:
            coeffs = np.polyfit(log_freqs[valid_mask], log_magnitude[valid_mask], 1)
            spectral_exponent = -coeffs[0]  # Expoente negativo na lei de potência
            return np.clip(spectral_exponent, 0.0, 3.0)
        except:
            return 0.0

    def _compute_correlation_dimension(self, signal: np.ndarray) -> float:
        """Computa dimensão de correlação do sinal."""
        if len(signal) < 20:
            return 1.0

        # Simplified correlation dimension calculation
        # Usar abordagem de Grassberger-Procaccia simplificada
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        # Calcular correlação para diferentes distâncias
        max_r = min(len(signal) // 4, 20)
        correlations = []

        for r in range(1, max_r + 1):
            corr_sum = 0
            count = 0

            for i in range(len(signal) - r):
                for j in range(i + r, len(signal)):
                    dist = abs(signal_norm[i] - signal_norm[j])
                    if dist < 1.0:  # Threshold
                        corr_sum += 1.0
                    count += 1

            if count > 0:
                correlations.append(corr_sum / count)

        if len(correlations) < 3:
            return 1.0

        # Estimar dimensão via regressão
        r_values = np.arange(1, len(correlations) + 1)
        log_r = np.log(r_values)
        log_corr = np.log(np.array(correlations) + 1e-8)

        valid_mask = np.isfinite(log_r) & np.isfinite(log_corr)
        if not np.any(valid_mask) or np.sum(valid_mask) < 3:
            return 1.0

        try:
            coeffs = np.polyfit(log_r[valid_mask], log_corr[valid_mask], 1)
            correlation_dimension = coeffs[0]  # Dimensão é o coeficiente angular
            return np.clip(correlation_dimension, 1.0, 2.0)
        except:
            return 1.0

    def validate_topological_invariants(self, signal: torch.Tensor,
                                      transformed_signal: torch.Tensor) -> Dict[str, float]:
        """
        Valida preservação de invariantes topológicos.

        Args:
            signal: Sinal original
            transformed_signal: Sinal transformado

        Returns:
            Métricas de validação
        """
        validation_results = {
            'dimension_preservation': 0.0,
            'connectivity_preservation': 0.0,
            'homology_preservation': 0.0,
            'overall_topological_score': 0.0
        }

        # Preservação de dimensão fractal
        orig_props = self.get_fractal_properties(signal)
        trans_props = self.get_fractal_properties(transformed_signal)

        dim_diff = abs(orig_props.dimension - trans_props.dimension)
        validation_results['dimension_preservation'] = max(0.0, 1.0 - dim_diff)

        # Preservação de conectividade (simplificada)
        orig_signal_np = signal.detach().cpu().numpy().flatten()
        trans_signal_np = transformed_signal.detach().cpu().numpy().flatten()

        # Conectividade baseada em correlação cruzada
        if len(orig_signal_np) > 0 and len(trans_signal_np) > 0:
            min_len = min(len(orig_signal_np), len(trans_signal_np))
            cross_corr = np.correlate(orig_signal_np[:min_len], trans_signal_np[:min_len], mode='valid')
            if len(cross_corr) > 0:
                connectivity = abs(cross_corr[0]) / (np.linalg.norm(orig_signal_np[:min_len]) * np.linalg.norm(trans_signal_np[:min_len]) + 1e-8)
                validation_results['connectivity_preservation'] = connectivity

        # Homologia simplificada (preservação de extremos)
        orig_extrema = self._count_extrema(orig_signal_np)
        trans_extrema = self._count_extrema(trans_signal_np)

        extrema_ratio = min(orig_extrema, trans_extrema) / max(orig_extrema, trans_extrema) if max(orig_extrema, trans_extrema) > 0 else 1.0
        validation_results['homology_preservation'] = extrema_ratio

        # Score geral
        validation_results['overall_topological_score'] = np.mean([
            validation_results['dimension_preservation'],
            validation_results['connectivity_preservation'],
            validation_results['homology_preservation']
        ])

        return validation_results

    def _count_extrema(self, signal: np.ndarray) -> int:
        """Conta extremos locais no sinal."""
        if len(signal) < 3:
            return 0

        extrema_count = 0
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and signal[i] > signal[i+1]) or \
               (signal[i] < signal[i-1] and signal[i] < signal[i+1]):
                extrema_count += 1

        return extrema_count


# Funções utilitárias
def create_fractal_test_signal(length: int = 1000, fractal_dimension: float = 1.5) -> np.ndarray:
    """Cria sinal de teste com dimensão fractal específica."""
    # Sinal fractal simples baseado em random walk
    signal = np.zeros(length)
    step_size = 1.0

    for i in range(1, length):
        # Probabilidade baseada na dimensão fractal
        if np.random.random() < 0.5:
            signal[i] = signal[i-1] + step_size
        else:
            signal[i] = signal[i-1] - step_size

        # Ajustar step size baseado na dimensão
        step_size *= (1.0 + (fractal_dimension - 1.0) * 0.01)

    return signal


def test_fractal_analyzer():
    """Testa o analisador de sinais fractais."""
    print("🧪 Testing Fractal Signal Analyzer")
    print("=" * 50)

    # Criar analisador
    analyzer = FractalSignalAnalyzer()

    # Sinal de teste
    test_signal = torch.randn(100, 7)  # 100 tokens, 7 features cada

    # Análise fractal
    fractal_signal = analyzer.analyze_text_signal(['quantum', 'entanglement', 'test'])

    print(f"🔬 Fractal Analysis:")
    print(f"   Input shape: {test_signal.shape}")
    print(f"   Fractal signal shape: {fractal_signal.shape}")

    # Propriedades fractais
    dimension = analyzer.compute_dimension(fractal_signal)
    print(f"   Fractal dimension: {dimension:.3f}")

    # Propriedades completas
    properties = analyzer.get_fractal_properties(test_signal)
    print(f"   📊 Complete Properties:")
    print(f"      Dimension: {properties.dimension:.3f}")
    print(f"      Lacunarity: {properties.lacunarity:.3f}")
    print(f"      Persistence: {properties.persistence:.3f}")
    print(f"      Spectral exponent: {properties.spectral_exponent:.3f}")
    print(f"      Correlation dimension: {properties.correlation_dimension:.3f}")

    # Validação topológica
    validation = analyzer.validate_topological_invariants(test_signal, fractal_signal)
    print(f"   🔗 Topological Validation:")
    for key, value in validation.items():
        print(f"      {key}: {value:.3f}")

    print("\n✅ Fractal Signal Analyzer test completed!")


if __name__ == "__main__":
    test_fractal_analyzer()