#!/usr/bin/env python3
"""
Enhanced QRH Processor - Implementação Otimizada com Quaterniôns e FFT
=====================================================================

Processador avançado que substitui completamente o HumanChatTest com:
- Pipeline quaterniônico real com 4D rotations
- FFT otimizada para processamento espectral
- Análise adaptativa de parâmetros α baseada em complexidade
- Sistema de cache para otimização de performance
- Filtros cognitivos adaptativos integrados
"""
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
from .qrh_layer import QRHLayer, QRHConfig
from ..fractal.spectral_filter import SpectralFilter
from ..cognitive.semantic_adaptive_filters import (
    SemanticAdaptiveFilter,
    SemanticFilterConfig
)

class EnhancedQRHProcessor:
    """
    Processador ΨQRH avançado com otimizações de performance e análise adaptativa de parâmetros espectrais.

    Pipeline completo: Input → QRHLayer (spectral) → SemanticAdaptiveFilter (cognitive) → Output
    """

    def __init__(self,
                 embed_dim: int = 64,
                 device: str = "cpu",
                 enable_cognitive_filters: bool = True,
                 cognitive_config_path: Optional[str] = None):
        """
        Inicializa o processador enhanced com configurações otimizadas.

        Args:
            embed_dim: Dimensão do embedding quaterniônico
            device: Dispositivo de processamento (cpu, cuda, mps)
            enable_cognitive_filters: Habilitar filtros cognitivos adaptativos
            cognitive_config_path: Caminho para config dos filtros cognitivos
        """
        self.device = device
        self.embed_dim = embed_dim
        self.enable_cognitive_filters = enable_cognitive_filters

        # QRH Config
        self.config = QRHConfig(
            embed_dim=embed_dim,
            alpha=1.0,
            use_learned_rotation=True,
            device=device
        )

        # Cognitive Config
        self.cognitive_config = self._load_cognitive_config(cognitive_config_path)

        self.qrh_layer = None
        self.spectral_filter = None
        self.semantic_filter = None
        self.processing_cache = {}
        self.performance_metrics = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cognitive_filters_applied': 0
        }
        self._initialize_components()

    def _load_cognitive_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Carrega configuração dos filtros cognitivos."""
        if config_path is None:
            # Caminho padrão relativo ao arquivo atual
            current_dir = Path(__file__).parent
            config_path = current_dir.parent.parent / "configs" / "cognitive_filters_config.yaml"

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️ Erro ao carregar cognitive config: {e}")
            print("Usando configuração padrão")
            return self._get_default_cognitive_config()

    def _get_default_cognitive_config(self) -> Dict[str, Any]:
        """Retorna configuração padrão dos filtros cognitivos."""
        return {
            'semantic_filter': {
                'embed_dim': self.embed_dim,
                'num_heads': 8,
                'learning_rate': 1e-4,
                'temperature': 0.5,
                'epsilon': 1e-8
            },
            'contradiction_detector': {
                'contradiction_threshold': 0.3,
                'contradiction_sensitivity': 2.0,
                'phase_rotation_strength': 0.5
            },
            'irrelevance_filter': {
                'irrelevance_threshold': 0.4
            },
            'bias_filter': {
                'bias_threshold': 0.6
            },
            'filter_coordination': {
                'residual_weight': 0.1
            }
        }

    def _initialize_components(self):
        """Inicializa QRHLayer, SpectralFilter e SemanticAdaptiveFilter."""
        print(f"🚀 Inicializando Enhanced QRH Processor no dispositivo: {self.device}")

        # QRH Layer
        self.qrh_layer = QRHLayer(self.config)
        if self.device == "cuda":
            self.qrh_layer = self.qrh_layer.cuda()
        elif self.device == "mps":
            self.qrh_layer = self.qrh_layer.to(torch.device("mps"))

        # Spectral Filter
        self.spectral_filter = SpectralFilter(
            alpha=1.0,
            epsilon=1e-10,
            use_stable_activation=True,
            use_windowing=True,
            window_type='hann'
        )

        # Semantic Adaptive Filter (Cognitive)
        if self.enable_cognitive_filters:
            semantic_config = SemanticFilterConfig(
                embed_dim=self.cognitive_config['semantic_filter']['embed_dim'],
                num_heads=self.cognitive_config['semantic_filter']['num_heads'],
                contradiction_threshold=self.cognitive_config['contradiction_detector']['contradiction_threshold'],
                irrelevance_threshold=self.cognitive_config['irrelevance_filter']['irrelevance_threshold'],
                bias_threshold=self.cognitive_config['bias_filter']['bias_threshold'],
                learning_rate=self.cognitive_config['semantic_filter']['learning_rate'],
                temperature=self.cognitive_config['semantic_filter']['temperature'],
                epsilon=self.cognitive_config['semantic_filter']['epsilon'],
                contradiction_sensitivity=self.cognitive_config['contradiction_detector']['contradiction_sensitivity'],
                phase_rotation_strength=self.cognitive_config['contradiction_detector']['phase_rotation_strength']
            )
            self.semantic_filter = SemanticAdaptiveFilter(semantic_config)

            if self.device == "cuda":
                self.semantic_filter = self.semantic_filter.cuda()
            elif self.device == "mps":
                self.semantic_filter = self.semantic_filter.to(torch.device("mps"))

            print("✅ Filtros cognitivos inicializados")

        print("✅ Enhanced QRH Processor inicializado com sucesso")

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

        Retorna α entre 0.1 e 2.0 baseado em:
        - Comprimento do texto (mais longo → α menor para compressão)
        - Diversidade de caracteres (mais diverso → α maior para preservação)
        - Complexidade Unicode (caracteres especiais → α maior)
        - Entropia de Shannon (mais entropia → α maior)
        """
        length = len(text)
        unique_chars = len(set(text))
        unicode_complexity = sum(1 for c in text if ord(c) > 127)
        entropy = self._calculate_text_entropy(text)

        base_alpha = 1.0

        # Fatores adaptativos
        length_factor = min(np.log(length + 1) / 10, 0.5)  # Textos longos → menor α
        diversity_factor = unique_chars / max(length, 1) * 0.3  # Mais diverso → maior α
        unicode_factor = unicode_complexity / max(length, 1) * 0.4  # Unicode → maior α
        entropy_factor = entropy * 0.2  # Mais entropia → maior α

        adaptive_alpha = base_alpha - length_factor + diversity_factor + unicode_factor + entropy_factor

        # Log para debug
        print(f"🔧 Alpha adaptativo: texto='{text[:20]}...' len={length} unique={unique_chars} "
              f"unicode={unicode_complexity} entropy={entropy:.3f} → α={adaptive_alpha:.3f}")

        return max(0.1, min(adaptive_alpha, 2.0))

    def _generate_cache_key(self, text: str, alpha: float) -> str:
        """Gera uma chave de cache baseada no texto e no alpha."""
        return f"{hash(text)}_{alpha:.4f}"

    def _adapt_spectrum_to_qrh(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Adapta o espectro para o formato de entrada da QRHLayer."""
        # Placeholder: a lógica real pode ser mais complexa
        batch_size, spec_dim = spectrum.shape
        target_dim = 4 * self.embed_dim

        # Handle both complex and real tensors safely
        if spectrum.is_complex():
            flat_real = spectrum.real.flatten()
            flat_imag = spectrum.imag.flatten()
        else:
            # For real tensors, imaginary part is zero
            flat_real = spectrum.flatten()
            flat_imag = torch.zeros_like(flat_real)

        combined = torch.cat([flat_real, flat_imag])

        if combined.shape[0] < target_dim:
            padding = torch.zeros(target_dim - combined.shape[0], device=self.device)
            combined = torch.cat([combined, padding])

        reshaped = combined[:target_dim].reshape(batch_size, 1, target_dim)
        return reshaped

    def _adapt_qrh_to_spectrum(self, q_output: torch.Tensor) -> torch.Tensor:
        """Adapta a saída da QRHLayer de volta para o formato de espectro."""
        # PRESERVE FULL QUATERNION STATE - ensure proper 4D shape for wave_to_text
        if q_output.dim() == 3:
            # For quaternion output [batch, seq, 4*embed_dim], reshape to [batch, seq, embed_dim, 4]
            batch_size, seq_len, total_dim = q_output.shape
            if total_dim % 4 == 0:
                embed_dim = total_dim // 4
                q_reshaped = q_output.view(batch_size, seq_len, embed_dim, 4)
                # Return the full 4D quaternion state for wave_to_text
                return q_reshaped

        # Fallback: return as-is for other dimensions
        return q_output

    def _extract_layer1_fractal(self, spectrum: torch.Tensor, text: str, alpha: float) -> Dict[str, Any]:
        """
        LAYER1-FRACTAL: Extrai valores espectrais quaterniônicos para análise.

        Retorna valores numéricos reais do espectro para posterior aplicação de cálculos.
        """
        # Converter para numpy para facilitar manipulação (detach primeiro para evitar erros de grad)
        spectrum_magnitude = torch.abs(spectrum).detach().cpu().numpy()
        spectrum_phase = torch.angle(spectrum).detach().cpu().numpy()

        # Handle both complex and real tensors safely
        if spectrum.is_complex():
            spectrum_real = spectrum.real.detach().cpu().numpy()
            spectrum_imag = spectrum.imag.detach().cpu().numpy()
        else:
            # For real tensors, imaginary part is zero
            spectrum_real = spectrum.detach().cpu().numpy()
            spectrum_imag = np.zeros_like(spectrum_real)

        # Flatten para obter array 1D
        mag_flat = spectrum_magnitude.flatten()
        phase_flat = spectrum_phase.flatten()
        real_flat = spectrum_real.flatten()
        imag_flat = spectrum_imag.flatten()

        return {
            'input_text': text,
            'alpha': float(alpha),
            'shape': list(spectrum.shape),
            'values': {
                'magnitude': mag_flat.tolist(),
                'phase': phase_flat.tolist(),
                'real': real_flat.tolist(),
                'imaginary': imag_flat.tolist()
            },
            'statistics': {
                'magnitude_mean': float(mag_flat.mean()),
                'magnitude_std': float(mag_flat.std()),
                'magnitude_max': float(mag_flat.max()),
                'magnitude_min': float(mag_flat.min()),
                'phase_mean': float(phase_flat.mean()),
                'phase_std': float(phase_flat.std()),
                'real_mean': float(real_flat.mean()),
                'imag_mean': float(imag_flat.mean()),
                'energy': float((mag_flat ** 2).sum()),
                'sparsity': float((np.abs(mag_flat) < 0.01).sum() / len(mag_flat))
            }
        }

    def extract_consciousness_coupling_data(self, spectrum: torch.Tensor,
                                           quaternion_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NOVO: Extrai spectral_energy e quaternion_phase como TENSORES para acoplamento com consciência.

        Args:
            spectrum: Espectro complexo do sinal [batch, spec_dim]
            quaternion_output: Saída quaterniônica da QRHLayer [batch, seq, 4*embed_dim]

        Returns:
            spectral_energy: Energia espectral |q|² [batch, embed_dim]
            quaternion_phase: Fase quaterniônica atan2(||v||, r) [batch, embed_dim]
        """
        batch_size = spectrum.shape[0]

        # 1. SPECTRAL ENERGY: |spectrum|² (energia do espectro complexo)
        spectral_magnitude = torch.abs(spectrum)  # [batch, spec_dim] ou [batch, ...]
        spectral_energy_raw = spectral_magnitude ** 2  # |q|²

        # Flatten se multidimensional
        if spectral_energy_raw.dim() > 2:
            spectral_energy_raw = spectral_energy_raw.reshape(batch_size, -1)

        # Adaptar para embed_dim
        spec_dim = spectral_energy_raw.shape[-1]

        if spec_dim > self.embed_dim:
            # Downsampling via average pooling (batch, 1, spec_dim) -> (batch, 1, embed_dim)
            spectral_energy = torch.nn.functional.adaptive_avg_pool1d(
                spectral_energy_raw.unsqueeze(1),
                self.embed_dim
            ).squeeze(1)
        elif spec_dim < self.embed_dim:
            # Upsampling via interpolação linear
            spectral_energy = torch.nn.functional.interpolate(
                spectral_energy_raw.unsqueeze(1),
                size=self.embed_dim,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        else:
            spectral_energy = spectral_energy_raw

        # 2. QUATERNION PHASE: Usar fase do ESPECTRO COMPLEXO diretamente
        # CORREÇÃO: torch.angle(spectrum) captura a fase do sinal complexo original
        # Isso preserva melhor as diferenças entre textos
        quaternion_phase_raw = torch.angle(spectrum)  # [batch, spec_dim]

        # Adaptar para embed_dim (mesmo processo que spectral_energy)
        if quaternion_phase_raw.dim() > 2:
            quaternion_phase_raw = quaternion_phase_raw.reshape(batch_size, -1)

        phase_dim = quaternion_phase_raw.shape[-1]

        if phase_dim > self.embed_dim:
            quaternion_phase = torch.nn.functional.adaptive_avg_pool1d(
                quaternion_phase_raw.unsqueeze(1),
                self.embed_dim
            ).squeeze(1)
        elif phase_dim < self.embed_dim:
            quaternion_phase = torch.nn.functional.interpolate(
                quaternion_phase_raw.unsqueeze(1),
                size=self.embed_dim,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        else:
            quaternion_phase = quaternion_phase_raw

        print(f"🔗 Dados de acoplamento extraídos:")
        print(f"   spectral_energy: shape={spectral_energy.shape}, "
              f"mean={spectral_energy.mean().item():.6f}, std={spectral_energy.std().item():.6f}")
        print(f"   quaternion_phase: shape={quaternion_phase.shape}, "
              f"mean={quaternion_phase.mean().item():.6f}, std={quaternion_phase.std().item():.6f}")

        return spectral_energy, quaternion_phase

    def _advanced_spectral_analysis(self, spectrum: torch.Tensor, text: str, alpha: float) -> str:
        """Realiza uma análise espectral avançada."""
        return f"Análise para '{text}' com alpha={alpha:.3f}: Espectro com {spectrum.shape} dimensões."

    def process_text(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Processa texto através do pipeline Enhanced QRH com filtros cognitivos.

        Pipeline: Input → Spectral → QRHLayer → CognitiveFilters → Output
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

        # STEP 1: Spectral Processing
        self.spectral_filter.alpha = adaptive_alpha
        spectrum = self.spectral_filter.text_to_spectrum(
            text,
            target_dim=4 * self.embed_dim,
            device=self.device
        )

        # STEP 2: QRH Layer (Quaternionic + Spectral Filtering)
        quaternion_input = self._adapt_spectrum_to_qrh(spectrum)

        with torch.no_grad():
            qrh_output = self.qrh_layer(quaternion_input)

        # STEP 3: Cognitive Filters (SemanticAdaptiveFilter)
        cognitive_metrics = {}
        if self.enable_cognitive_filters and self.semantic_filter is not None:
            filtered_output, cognitive_metrics = self.semantic_filter(qrh_output)
            self.performance_metrics['cognitive_filters_applied'] += 1

            # Gerar relatório de saúde semântica
            semantic_health = self.semantic_filter.get_semantic_health_report(cognitive_metrics)
            cognitive_metrics['semantic_health'] = semantic_health
        else:
            filtered_output = qrh_output

        # STEP 4: Output Processing
        output_spectrum = self._adapt_qrh_to_spectrum(filtered_output)

        # LAYER1-FRACTAL: Extração de valores espectrais
        layer1_fractal = self._extract_layer1_fractal(output_spectrum, text, adaptive_alpha)

        # NOVO: Extrair dados de acoplamento para consciência (spectral_energy, quaternion_phase)
        # IMPORTANTE: Usar spectrum ORIGINAL (antes do QRHLayer) para preservar diferenças entre textos
        spectral_energy, quaternion_phase = self.extract_consciousness_coupling_data(
            spectrum,  # ← CORREÇÃO: usar spectrum original ao invés de output_spectrum
            qrh_output  # ← CORREÇÃO: usar qrh_output ao invés de filtered_output
        )

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
            'layer1_fractal': layer1_fractal,
            'adaptive_alpha': adaptive_alpha,
            'processing_time': processing_time,
            'cache_hit': False,
            'performance_metrics': self.performance_metrics,
            'cognitive_metrics': self._format_cognitive_metrics(cognitive_metrics) if cognitive_metrics else None,
            'pipeline_stages': {
                'spectral_processing': True,
                'qrh_layer': True,
                'cognitive_filters': self.enable_cognitive_filters
            },
            # NOVO: Dados de acoplamento para consciência
            'consciousness_coupling': {
                'spectral_energy': spectral_energy,
                'quaternion_phase': quaternion_phase
            },
            # NOVO: Adicionar qrh_output para geração de texto
            # PRESERVE FULL QUATERNION STATE - ensure proper 4D shape
            'qrh_output': qrh_output
        }

        if use_cache:
            self.processing_cache[cache_key] = result

        return result

    def _format_cognitive_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Formata métricas cognitivas para output legível."""
        if not metrics:
            return {}

        formatted = {}

        # Contradiction scores
        if 'contradiction_scores' in metrics:
            scores = metrics['contradiction_scores']
            # Safe handling of std for short sequences
            std_val = scores.std().item() if scores.numel() > 1 else 0.0
            formatted['contradiction'] = {
                'mean': float(scores.mean().item()),
                'max': float(scores.max().item()),
                'min': float(scores.min().item()),
                'std': std_val
            }

        # Relevance scores
        if 'relevance_scores' in metrics:
            scores = metrics['relevance_scores']
            std_val = scores.std().item() if scores.numel() > 1 else 0.0
            formatted['relevance'] = {
                'mean': float(scores.mean().item()),
                'max': float(scores.max().item()),
                'min': float(scores.min().item()),
                'std': std_val
            }

        # Bias magnitude
        if 'bias_magnitude' in metrics:
            magnitude = metrics['bias_magnitude']
            std_val = magnitude.std().item() if magnitude.numel() > 1 else 0.0
            formatted['bias'] = {
                'mean': float(magnitude.mean().item()),
                'max': float(magnitude.max().item()),
                'min': float(magnitude.min().item()),
                'std': std_val
            }

        # Semantic health
        if 'semantic_health' in metrics:
            formatted['semantic_health'] = metrics['semantic_health']

        # Filter weights
        if 'filter_weights' in metrics:
            weights = metrics['filter_weights']
            formatted['filter_weights'] = {
                'contradiction_avg': float(weights[:, :, 0].mean().item()),
                'irrelevance_avg': float(weights[:, :, 1].mean().item()),
                'bias_avg': float(weights[:, :, 2].mean().item())
            }

        return formatted

def create_enhanced_processor(embed_dim: int = 64,
                             device: str = "cpu",
                             enable_cognitive_filters: bool = True,
                             cognitive_config_path: Optional[str] = None) -> EnhancedQRHProcessor:
    """
    Factory function para criar processador Enhanced QRH otimizado com filtros cognitivos.

    Args:
        embed_dim: Dimensão do embedding quaterniônico
        device: Dispositivo (cpu, cuda, mps)
        enable_cognitive_filters: Habilitar filtros cognitivos
        cognitive_config_path: Caminho para configuração customizada

    Returns:
        EnhancedQRHProcessor configurado e pronto para uso
    """
    return EnhancedQRHProcessor(
        embed_dim=embed_dim,
        device=device,
        enable_cognitive_filters=enable_cognitive_filters,
        cognitive_config_path=cognitive_config_path
    )
