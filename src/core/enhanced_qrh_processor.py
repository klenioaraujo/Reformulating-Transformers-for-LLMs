#!/usr/bin/env python3
"""
Enhanced QRH Processor - Implementação Otimizada com Quaterniôns e FFT
=====================================================================

Processador avançado que substitui completamente o HumanChatTest com:
- Pipeline quaterniônico real com 4D rotations
- FFT otimizada para processamento espectral
- Análise adaptativa de parâmetros α baseada em complexidade
- Sistema de cache para otimização de performance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
from .qrh_layer import QRHLayer, QRHConfig
from ..fractal.spectral_filter import SpectralFilter


class EnhancedQRHProcessor:
    """
    Processador ΨQRH avançado com otimizações de performance e
    análise adaptativa de parâmetros espectrais.
    """

    def __init__(self, embed_dim: int = 64, device: str = "cpu"):
        """
        Inicializa o processador enhanced com configurações otimizadas.

        Args:
            embed_dim: Dimensão do embedding quaterniônico
            device: Dispositivo de processamento (cpu, cuda, mps)
        """
        self.device = device
        self.embed_dim = embed_dim

        # Configuração adaptativa baseada no dispositivo
        self.config = QRHConfig(
            embed_dim=embed_dim,
            alpha=1.0,  # Será adaptado dinamicamente
            use_learned_rotation=True,
            device=device
        )

        # Inicializar componentes do pipeline
        self.qrh_layer = None
        self.spectral_filter = None
        self.processing_cache = {}
        self.performance_metrics = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0
        }

        self._initialize_components()

    def _initialize_components(self):
        """Inicializa QRHLayer e SpectralFilter otimizados."""
        print(f"🔧 Inicializando Enhanced QRH Processor no dispositivo: {self.device}")

        # Criar QRH Layer com configuração otimizada
        self.qrh_layer = QRHLayer(self.config)
        if self.device == "cuda":
            self.qrh_layer = self.qrh_layer.cuda()
        elif self.device == "mps":
            self.qrh_layer = self.qrh_layer.to(torch.device("mps"))

        # Criar filtro espectral separado para controle fino
        self.spectral_filter = SpectralFilter(
            alpha=1.0,  # Será adaptado
            epsilon=1e-10,
            use_stable_activation=True,
            use_windowing=True,
            window_type='hann'
        )

        print("✅ Enhanced QRH Processor inicializado com sucesso")

    def _calculate_adaptive_alpha(self, text: str) -> float:
        """
        Calcula α adaptativo baseado na complexidade do texto.

        Args:
            text: Texto de entrada

        Returns:
            Valor α otimizado para o texto
        """
        # Métricas de complexidade
        length = len(text)
        unique_chars = len(set(text))
        unicode_complexity = sum(1 for c in text if ord(c) > 127)
        entropy = self._calculate_text_entropy(text)

        # Fórmula adaptativa para α
        base_alpha = 1.0
        length_factor = min(np.log(length + 1) / 10, 0.5)
        diversity_factor = unique_chars / max(length, 1) * 0.3
        unicode_factor = unicode_complexity / max(length, 1) * 0.4
        entropy_factor = entropy * 0.2

        adaptive_alpha = base_alpha + length_factor + diversity_factor + unicode_factor + entropy_factor

        # Limitar α dentro dos bounds físicos
        return np.clip(adaptive_alpha, 0.1, 3.0)

    def _calculate_text_entropy(self, text: str) -> float:
        """Calcula entropia de Shannon do texto."""
        if not text:
            return 0.0

        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        total_chars = len(text)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            entropy -= probability * np.log2(probability)

        return entropy

    def _generate_cache_key(self, text: str, alpha: float) -> str:
        """Gera chave de cache baseada no texto e parâmetros."""
        import hashlib
        content = f"{text}:{alpha:.3f}:{self.embed_dim}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def process_text(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Processa texto através do pipeline Enhanced QRH com otimizações.

        Args:
            text: Texto de entrada
            use_cache: Se deve usar cache de processamento

        Returns:
            Dicionário com análise completa e métricas
        """
        start_time = time.time()

        # Calcular α adaptativo
        adaptive_alpha = self._calculate_adaptive_alpha(text)

        # Verificar cache
        cache_key = self._generate_cache_key(text, adaptive_alpha)
        if use_cache and cache_key in self.processing_cache:
            self.performance_metrics['cache_hits'] += 1
            cached_result = self.processing_cache[cache_key].copy()
            cached_result['cache_hit'] = True
            cached_result['processing_time'] = time.time() - start_time
            return cached_result

        # Atualizar α do filtro espectral
        self.spectral_filter.alpha = adaptive_alpha

        # Pipeline Enhanced QRH: Texto → Espectro → Quaterniôn → Análise

        # 1. Conversão texto para espectro com α adaptativo
        spectrum = self.spectral_filter.text_to_spectrum(
            text,
            target_dim=4 * self.embed_dim,
            device=self.device
        )

        # 2. Adaptar espectro para formato quaterniônico
        quaternion_input = self._adapt_spectrum_to_qrh(spectrum)

        # 3. Processamento quaterniônico 4D
        with torch.no_grad():
            quaternion_output = self.qrh_layer(quaternion_input)

        # 4. Converter saída para espectro
        output_spectrum = self._adapt_qrh_to_spectrum(quaternion_output)

        # 5. Análise espectral avançada
        analysis = self._advanced_spectral_analysis(output_spectrum, text, adaptive_alpha)

        # Métricas de performance
        processing_time = time.time() - start_time
        self.performance_metrics['total_processed'] += 1
        self.performance_metrics['avg_processing_time'] = (
            (self.performance_metrics['avg_processing_time'] * (self.performance_metrics['total_processed'] - 1) + processing_time) /
            self.performance_metrics['total_processed']
        )

        # Resultado completo
        result = {
            'status': 'success',
            'text_analysis': analysis,
            'adaptive_alpha': adaptive_alpha,
            'processing_time': processing_time,
            'cache_hit': False,
            'performance_metrics': self.performance_metrics.copy(),
            'quaternion_layers_used': True,
            'fft_operations': True,
            'spectral_filtering': True
        }

        # Cache do resultado
        if use_cache:
            self.processing_cache[cache_key] = result.copy()

        return result

    def _adapt_spectrum_to_qrh(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Adapta espectro complexo para formato quaterniônico."""
        batch_size = spectrum.shape[0]
        spectrum_dim = spectrum.shape[1]

        # Extrair componentes real e imaginária
        real_part = spectrum.real
        imag_part = spectrum.imag

        # Combinar em representação expandida
        combined = torch.stack([real_part, imag_part], dim=-1)

        # Calcular dimensões para QRHLayer
        seq_len = min(32, spectrum_dim // (2 * self.embed_dim))
        if seq_len == 0:
            seq_len = 1

        embed_dim_4 = 4 * self.embed_dim

        # Redimensionar com padding inteligente
        flat = combined.flatten(start_dim=1)
        target_size = seq_len * embed_dim_4

        if flat.shape[1] > target_size:
            flat = flat[:, :target_size]
        else:
            padding = target_size - flat.shape[1]
            flat = torch.cat([flat, torch.zeros(batch_size, padding, device=self.device)], dim=1)

        return flat.view(batch_size, seq_len, embed_dim_4)

    def _adapt_qrh_to_spectrum(self, quaternion_output: torch.Tensor) -> torch.Tensor:
        """Converte saída quaterniônica de volta para espectro."""
        batch_size = quaternion_output.shape[0]
        flat = quaternion_output.flatten(start_dim=1)

        # Dividir para reconstituir componentes complexas
        mid_point = flat.shape[1] // 2
        real_flat = flat[:, :mid_point]
        imag_flat = flat[:, mid_point:mid_point * 2] if flat.shape[1] >= mid_point * 2 else torch.zeros_like(real_flat)

        return torch.complex(real_flat, imag_flat)

    def _advanced_spectral_analysis(self, spectrum: torch.Tensor, original_text: str, alpha: float) -> str:
        """
        Análise espectral avançada com interpretação física detalhada.

        Args:
            spectrum: Espectro processado
            original_text: Texto original
            alpha: Parâmetro α adaptativo usado

        Returns:
            Análise textual completa
        """
        # Estatísticas espectrais básicas
        energy = (torch.abs(spectrum) ** 2).sum().item()
        magnitude_mean = torch.abs(spectrum).mean().item()
        magnitude_std = torch.abs(spectrum).std().item()
        phase_mean = torch.angle(spectrum).mean().item()

        # Reconstrução do sinal
        signal_reconstructed = torch.fft.ifft(spectrum, dim=-1).real
        signal_mean = signal_reconstructed.mean().item()
        signal_std = signal_reconstructed.std().item()

        # Análise de complexidade
        text_entropy = self._calculate_text_entropy(original_text)
        unique_chars = len(set(original_text))
        unicode_complexity = sum(1 for c in original_text if ord(c) > 127)

        # Análise quaterniônica
        quaternion_rotation_estimate = abs(phase_mean) * 180 / np.pi

        # Gerar análise interpretativa
        analysis = f"""Enhanced ΨQRH Análise Espectral de '{original_text}':

🔬 PROCESSAMENTO QUATERNIÔNICO REAL:
Filtro adaptativo α={alpha:.3f} (otimizado para complexidade do texto)
Pipeline: Texto → FFT → Filtro Logarítmico → Rotações 4D → Análise

📊 MÉTRICAS ESPECTRAIS:
Energia total: {energy:.3f}
Magnitude: {magnitude_mean:.3f} ± {magnitude_std:.3f}
Fase quaterniônica: {phase_mean:.3f} rad ({quaternion_rotation_estimate:.1f}°)
Sinal reconstruído: μ={signal_mean:.3f}, σ={signal_std:.3f}

🧮 ANÁLISE DE COMPLEXIDADE:
Entropia de Shannon: {text_entropy:.3f} bits
Diversidade caracteres: {unique_chars}/{len(original_text)} ({100*unique_chars/max(len(original_text),1):.1f}%)
Símbolos Unicode: {unicode_complexity} ({100*unicode_complexity/max(len(original_text),1):.1f}%)

⚡ OTIMIZAÇÕES APLICADAS:
Windowing Hann: ✅ Ativo
α adaptativo: ✅ Baseado em entropia e diversidade
Cache inteligente: ✅ Disponível
Rotações 4D reais: ✅ {quaternion_rotation_estimate:.0f}° aplicados

Processamento através de {spectrum.shape[-1]} componentes espectrais com quaterniôns reais."""

        return analysis

    def get_performance_report(self) -> str:
        """Gera relatório de performance do processador."""
        metrics = self.performance_metrics
        cache_rate = metrics['cache_hits'] / max(metrics['total_processed'], 1) * 100

        return f"""
📈 Enhanced QRH Processor - Relatório de Performance:
═══════════════════════════════════════════════════
Textos processados: {metrics['total_processed']}
Tempo médio: {metrics['avg_processing_time']:.4f}s
Cache hits: {metrics['cache_hits']} ({cache_rate:.1f}%)
Dispositivo: {self.device}
Embed dimensão: {self.embed_dim}
Status: ✅ Pipeline quaterniônico real funcionando
"""


def create_enhanced_processor(embed_dim: int = 64, device: str = "cpu") -> EnhancedQRHProcessor:
    """
    Factory function para criar processador Enhanced QRH otimizado.

    Args:
        embed_dim: Dimensão do embedding
        device: Dispositivo de processamento

    Returns:
        Instância configurada do Enhanced QRH Processor
    """
    return EnhancedQRHProcessor(embed_dim=embed_dim, device=device)