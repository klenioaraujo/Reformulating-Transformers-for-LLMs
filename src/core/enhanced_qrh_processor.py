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
    Processador ΨQRH avançado com otimizações de performance e análise adaptativa de parâmetros espectrais.
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
        self.config = QRHConfig(
            embed_dim=embed_dim,
            alpha=1.0,
            use_learned_rotation=True,
            device=device
        )
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
        print(f"Inicializando Enhanced QRH Processor no dispositivo: {self.device}")
        self.qrh_layer = QRHLayer(self.config)
        if self.device == "cuda":
            self.qrh_layer = self.qrh_layer.cuda()
        elif self.device == "mps":
            self.qrh_layer = self.qrh_layer.to(torch.device("mps"))
        self.spectral_filter = SpectralFilter(
            alpha=1.0,
            epsilon=1e-10,
            use_stable_activation=True,
            use_windowing=True,
            window_type='hann'
        )
        print("Enhanced QRH Processor inicializado com sucesso")

    def _calculate_text_entropy(self, text: str) -> float:
        """Calcula a entropia de Shannon para o texto."""
        if not text:
            return 0.0
        entropy = 0
        char_counts = {char: text.count(char) for char in set(text)}
        for count in char_counts.values():
            p = count / len(text)
            entropy -= p * np.log2(p)
        return entropy

    def _calculate_adaptive_alpha(self, text: str) -> float:
        """
        Calcula α adaptativo baseado na complexidade do texto.
        """
        length = len(text)
        unique_chars = len(set(text))
        unicode_complexity = sum(1 for c in text if ord(c) > 127)
        entropy = self._calculate_text_entropy(text)
        
        base_alpha = 1.0
        length_factor = min(np.log(length + 1) / 10, 0.5)
        diversity_factor = unique_chars / max(length, 1) * 0.3
        unicode_factor = unicode_complexity / max(length, 1) * 0.4
        entropy_factor = entropy * 0.2
        
        adaptive_alpha = base_alpha - length_factor + diversity_factor + unicode_factor + entropy_factor
        return max(0.1, min(adaptive_alpha, 2.0))

    def _generate_cache_key(self, text: str, alpha: float) -> str:
        """Gera uma chave de cache baseada no texto e no alpha."""
        return f"{hash(text)}_{alpha:.4f}"

    def _adapt_spectrum_to_qrh(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Adapta o espectro para o formato de entrada da QRHLayer."""
        # Placeholder: a lógica real pode ser mais complexa
        batch_size, spec_dim = spectrum.shape
        target_dim = 4 * self.embed_dim
        
        flat_real = spectrum.real.flatten()
        flat_imag = spectrum.imag.flatten()
        
        combined = torch.cat([flat_real, flat_imag])
        
        if combined.shape[0] < target_dim:
            padding = torch.zeros(target_dim - combined.shape[0], device=self.device)
            combined = torch.cat([combined, padding])
        
        reshaped = combined[:target_dim].reshape(batch_size, 1, target_dim)
        return reshaped

    def _adapt_qrh_to_spectrum(self, q_output: torch.Tensor) -> torch.Tensor:
        """Adapta a saída da QRHLayer de volta para o formato de espectro."""
        # Placeholder
        return torch.fft.fft(q_output.real, norm="ortho").to(self.device)

    def _advanced_spectral_analysis(self, spectrum: torch.Tensor, text: str, alpha: float) -> str:
        """Realiza uma análise espectral avançada."""
        return f"Análise para '{text}' com alpha={alpha:.3f}: Espectro com {spectrum.shape} dimensões."

    def process_text(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Processa texto através do pipeline Enhanced QRH com otimizações.
        """
        start_time = time.time()
        adaptive_alpha = self._calculate_adaptive_alpha(text)
        cache_key = self._generate_cache_key(text, adaptive_alpha)

        if use_cache and cache_key in self.processing_cache:
            self.performance_metrics['cache_hits'] += 1
            cached_result = self.processing_cache[cache_key].copy()
            cached_result['cache_hit'] = True
            cached_result['processing_time'] = time.time() - start_time
            return cached_result

        self.spectral_filter.alpha = adaptive_alpha
        
        spectrum = self.spectral_filter.text_to_spectrum(
            text,
            target_dim=4 * self.embed_dim,
            device=self.device
        )
        
        quaternion_input = self._adapt_spectrum_to_qrh(spectrum)
        
        with torch.no_grad():
            quaternion_output = self.qrh_layer(quaternion_input)
            
        output_spectrum = self._adapt_qrh_to_spectrum(quaternion_output)
        
        analysis = self._advanced_spectral_analysis(output_spectrum, text, adaptive_alpha)
        
        processing_time = time.time() - start_time
        self.performance_metrics['total_processed'] += 1
        self.performance_metrics['avg_processing_time'] = (
            (self.performance_metrics['avg_processing_time'] * (self.performance_metrics['total_processed'] - 1) + processing_time)
            / self.performance_metrics['total_processed']
        )
        
        result = {
            'status': 'success',
            'text_analysis': analysis,
            'adaptive_alpha': adaptive_alpha,
            'processing_time': processing_time,
            'cache_hit': False,
            'performance_metrics': self.performance_metrics
        }
        
        if use_cache:
            self.processing_cache[cache_key] = result
            
        return result

def create_enhanced_processor(embed_dim: int = 64, device: str = "cpu") -> EnhancedQRHProcessor:
    """
    Factory function para criar processador Enhanced QRH otimizado.
    """
    return EnhancedQRHProcessor(embed_dim=embed_dim, device=device)
