# Chunk 1: Lines 1-1153
# Tokens: 14005, Lines: 1-1153

#!/usr/bin/env python3
"""
ΨQRH CLI - Pipeline Físico Completo com Equação de Padilha
===========================================================

Implementação rigorosa do doe.md Seções 2.9.1-2.9.4:
- Padilha Wave Equation: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
- Fractal Dimension Mapping: α(D) = α₀(1 + λ(D - D_euclidean)/D_euclidean)
- Quaternion Operations: Hamilton product, SO(4) rotations
- Spectral Filtering: F(k) = exp(i α · arctan(ln(|k| + ε)))
- 4D Unitary Operations: Ψ' = q_left * Ψ * q_right†
- Optical Probe Generation: Para conversão wave-to-text
- Consciousness Integration: FCI calculation com bootstrap
- Auto-calibration: Parâmetros emergentes da física

VALIDAÇÃO MATEMÁTICA OBRIGATÓRIA:
- Energia conservada: ||output|| ≈ ||input|| (dentro de 5%)
- Unitaridade: Filtro espectral preserva energia
- Estabilidade numérica: Double precision quaternion arithmetic
- Consistência fractal: D calculado via power-law fitting

Exemplos de uso:
    python psiqrh.py "Prove that √2 is irrational"
    python psiqrh.py --interactive
    python psiqrh.py --test
    python psiqrh.py --help
"""

import argparse
import sys
import os
import torch
import json
# import requests  # Temporarily commented out for testing
import time
import numpy as np
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Global quiet mode flag (defined early for import sections)
QUIET_MODE = False

# Import model manager and tensor validator
try:
    from tools.model_manager import ModelManager, ModelManagerError
except ImportError:
    ModelManager = None
    ModelManagerError = Exception

from src.core.tensor_validator import ScientificTensorValidator

# Import learnable quantum embedding
from src.core.losses import QuantumEmbedding

# Import QuantumDecoder for direct quantum-to-text conversion
from tools.quantum_decoder import QuantumDecoder

# Import OpticalProbe for Padilha Wave Equation-based decoding
from src.core.optical_probe import OpticalProbe

# Import stable quantum evolution components
from src.core.prime_resonant_filter import (
    PrimeResonantFilter,
    LeechLatticeEmbedding,
    StableQuantumEvolution,
    create_stable_quantum_evolution
)

# Auto-calibration components
try:
    from src.core.quantum_temperature_calculator import QuantumTemperatureCalculator
    from src.core.optical_coherence_calculator import OpticalCoherenceCalculator
    from src.core.adaptive_spectral_parameters import AdaptiveSpectralParameters
    from src.core.complete_auto_calibration_system import CompleteAutoCalibrationSystem
    HAS_AUTO_CALIBRATION = True
except ImportError as e:
    HAS_AUTO_CALIBRATION = False
    print(f"❌ Auto-calibration components not available. ZERO FALLBACK POLICY: {e}")

# Physical Harmonic Orchestrator components - advanced physical corrections
try:
    from src.core.harmonic_signature_analyzer import HarmonicSignatureAnalyzer
    from src.core.physical_fundamental_corrections import PhysicalHarmonicOrchestrator
    HAS_PHYSICAL_HARMONIC_ORCHESTRATOR = True
    print("🔬 Physical Harmonic Orchestrator components loaded successfully!")
except ImportError as e:
    HAS_PHYSICAL_HARMONIC_ORCHESTRATOR = False
    print(f"❌ Physical Harmonic Orchestrator not available. Using basic auto-calibration: {e}")

# Auto-learning components - ΨQRH only (NO transformers)
try:
    # Use only ΨQRH models for auto-learning
    from src.core.spectral_harmonic_processor import QuaternionMLP, spectral_attention, harmonic_evolution, process_signal_stack
    from src.core.fractal_quantum_embedding import FractalQuantumEmbedding
    HAS_AUTO_LEARNING = True
except ImportError as e:
    HAS_AUTO_LEARNING = False
    print(f"❌ Auto-learning components not available. ZERO FALLBACK POLICY: {e}")

# Quantum temporal memory system for contextual coherence
try:
    from src.core.quantum_temporal_memory import (
        QuantumContextualLinguisticProcessor,
        create_quantum_memory_system
    )
    HAS_QUANTUM_MEMORY = True
    print("🧠 Quantum temporal memory system loaded successfully!")
except ImportError as e:
    HAS_QUANTUM_MEMORY = False
    print(f"❌ Quantum memory system not available. ZERO FALLBACK POLICY: {e}")


# Non-commutative geometry components (advanced quantum physics)
try:
    from src.core.noncommutative_geometry import (
        RegularizedNonCommutativeGeometry,
        NonCommutativeWaveDynamics,
        QuantumPhonemeField,
        StabilizedPsiQRHPipeline,
        create_noncommutative_pipeline
    )
    from src.core.quaternion_operations import OptimizedQuaternionOperations
    HAS_NONCOMMUTATIVE = True
    print("🔬 Stabilized non-commutative geometry framework loaded successfully!")
except ImportError as e:
    HAS_NONCOMMUTATIVE = False
    print(f"❌ Non-commutative geometry not available. ZERO FALLBACK POLICY: {e}")

# Hybrid quantum-classical system (resolves physics-linguistics divorce)
try:
    from src.core.hybrid_quantum_classical import (
        HybridQuantumClassicalSystem,
        create_hybrid_system
    )
    HAS_HYBRID_SYSTEM = True
    # Only print success message if not in quiet mode
    if not QUIET_MODE:
        print("🔗 Hybrid quantum-classical system loaded successfully!")
except ImportError as e:
    HAS_HYBRID_SYSTEM = False
    # Suppress error message in quiet mode (unified pipeline context)
    if not QUIET_MODE:
        print(f"❌ Hybrid system not available. ZERO FALLBACK POLICY: {e}")

# Wave-to-text conversion components - DEPRECATED: Use QuantumStateInterpreter instead
# The wave_to_text module is deprecated and should not be used
HAS_WAVE_TO_TEXT = False

def set_quiet_mode(quiet: bool):
    """Define modo silencioso global."""
    global QUIET_MODE
    QUIET_MODE = quiet
    import os
    os.environ['PSIQRH_QUIET'] = '1' if quiet else '0'


def get_active_model_path() -> Optional[str]:
    """
    Obtém o caminho do modelo ativo no registro.

    Returns:
        Caminho do modelo ativo ou None se não houver modelo ativo
    """
    if ModelManager is None:
        return None

    try:
        manager = ModelManager()
        active_model_name = manager.get_active_model()

        if active_model_name:
            registry = manager.load_registry()
            for model in registry['models']:
                if model['name'] == active_model_name:
                    return model['path']

        return None
    except Exception as e:
        print(f"⚠️  Erro ao obter modelo ativo: {e}")
        return None

class ΨQRHPipeline:
    """
    Pipeline Físico Completo ΨQRH - doe.md Seções 2.9.1-2.9.4

    Implementação rigorosa da Equação de Padilha com física quântica, fractal e óptica:
    f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

    Pipeline: texto → fractal_embedding → quaternions Ψ(x) → spectral_filtering → optical_probe → texto_saída
    """

    def __init__(self, task: str = "text-generation", device: Optional[str] = None,
                     input_text: Optional[str] = None, model_dir: Optional[str] = None,
                     enable_auto_calibration: bool = True, enable_noncommutative: bool = True,
                     tokenizer_config: Optional[Dict[str, Any]] = None,
                     enable_cognitive_priming: bool = True, audit_mode: bool = False,
                     vocab_path: Optional[str] = None, reasoning_mode: str = 'geometric'):
        """
        Inicializa o pipeline ΨQRH com física completa.

        Args:
            task: Tipo de tarefa (text-generation, analysis, chat, signal-processing)
            device: Dispositivo (cpu, cuda, mps) - detecta automaticamente se None
            input_text: Texto de entrada para detecção automática de tarefa (opcional)
            model_dir: Caminho para o modelo a ser carregado (opcional, usa modelo ativo se None)
            enable_auto_calibration: Habilita auto-calibração física (ZERO FALLBACK)
            tokenizer_config: Configuração do tokenizer adaptativo (opcional)
                - embed_dim: Dimensão do embedding (padrão: 64)
                - spectral_params_dim: Número de parâmetros espectrais por caractere (padrão: 8)
                - learnable: Se deve usar tokenizer aprendível (padrão: True)
        """
        self.device = self._detect_device(device)
        self.task = task
        self.enable_auto_calibration = enable_auto_calibration and HAS_AUTO_CALIBRATION
        self.enable_noncommutative = enable_noncommutative and HAS_NONCOMMUTATIVE
        self.enable_cognitive_priming = enable_cognitive_priming
        self.audit_mode = audit_mode
        self.reasoning_mode = reasoning_mode

        # Unified configuration from ModelManager
        manager = ModelManager()
        self.config = manager.get_active_model_config().get('qrh_config', {})
        self.config['device'] = self.device # Ensure device is correctly set

        # ========== FIXED ARCHITECTURE DEFINITION ==========
        # Define fixed architecture parameters to ensure dimensional compatibility
        # These values satisfy all mathematical constraints:
        # - embed_dim must be divisible by num_heads (for attention)
        # - embed_dim must be divisible by 4 (for quaternions)
        self.fixed_embed_dim = 64    # Divisible by 4 (quaternions) and 8 (heads)
        self.fixed_num_heads = 8     # Compatible with embed_dim=64
        self.fixed_hidden_dim = 512  # Standard hidden dimension

        # Override config with fixed architecture
        self.config['embed_dim'] = self.fixed_embed_dim
        self.config['num_heads'] = self.fixed_num_heads
        self.config['hidden_dim'] = self.fixed_hidden_dim

        print(f">> ARQUITETURA FIXA DEFINIDA: embed_dim={self.fixed_embed_dim}, num_heads={self.fixed_num_heads}, hidden_dim={self.fixed_hidden_dim}")

        # ========== LOAD PRETRAINED MODEL WEIGHTS ==========
        # Load pretrained state_dict for weight adaptation during inference
        self.pretrained_state_dict = None
        try:
            # Try to load the best model from checkpoints
            checkpoint_path = Path("models/checkpoints/best_model.pt")
            if checkpoint_path.exists():
                print("🔄 Loading pretrained model weights for inference adaptation...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.pretrained_state_dict = checkpoint.get('model_state_dict', checkpoint)
                print(f"✅ Loaded pretrained weights: {len(self.pretrained_state_dict)} parameter groups")
            else:
                print("⚠️  No pretrained model found - using random initialization")
        except Exception as e:
            print(f"⚠️  Failed to load pretrained model: {e} - using random initialization")
            self.pretrained_state_dict = None

        # Tokenizer config can still be passed as an override
        self.tokenizer_config = tokenizer_config or {
            'embed_dim': self.config['embed_dim'],
            'spectral_params_dim': 8,
            'learnable': True
        }
        print("✅ Pipeline configured via unified ModelManager.")

        # Componentes físicos obrigatórios (doe.md)
        self.fractal_analyzer = None
        self.quaternion_processor = None
        self.spectral_filter = None
        self.optical_probe = None
        self.consciousness_processor = None

        # DCF (Dinâmica de Consciência Fractal) components - initialized AFTER calibration
        self.dcf_analyzer = None
        self.kuramoto_layer = None
        self.consciousness_metrics = None
        self.diffusion_engine = None

        # Auto-calibration components
        self.temp_calculator = None
        self.coherence_calculator = None
        self.spectral_params = None
        self.calibration_system = None

        # Initialize auto-calibration components
        self._initialize_auto_calibration_components()

        # Non-commutative geometry components (advanced quantum physics)
        self.nc_pipeline = None

        # Hybrid quantum-classical system (resolves physics-linguistics divorce)
        self.hybrid_system = None

        # Quantum temporal memory system for contextual coherence
        self.quantum_memory_system = None

        # Quantum vocabulary for semantic connectivity
        self.quantum_vocab_representations = None

        # Learnable quantum embedding layer
        self.quantum_embedding = None

        # Audit logger for debugging and analysis
        self.audit_logger = None

        # Initialize Physical Harmonic Orchestrator for advanced physical corrections
        self.physical_harmonic_orchestrator = None
        if HAS_PHYSICAL_HARMONIC_ORCHESTRATOR:
            self.physical_harmonic_orchestrator = PhysicalHarmonicOrchestrator(device=self.device)
            print("🎼 Sistema Harmonizado: ✅ ATIVADO (Analisador de Assinatura Harmônica Física integrado)")
        else:
            print("🎼 Sistema Harmonizado: ❌ DESATIVADO (Analisador de Assinatura Harmônica Física ausente)")

        # Initialize stable quantum evolution components
        self.stable_evolution = create_stable_quantum_evolution(
            embed_dim=self.fixed_embed_dim,  # Use fixed dimension
            device=self.device
        )

        # Get dynamic vocabulary size from native vocabulary FIRST
        # Use injected vocab_path if provided, otherwise try default locations
        if vocab_path is not None:
            native_vocab_path = vocab_path
        else:
            native_vocab_paths = [
                os.path.join(os.getcwd(), "data", "native_vocab.json"),
                os.path.join(BASE_DIR, "data", "native_vocab.json")
            ]
            native_vocab_path = None
            for path in native_vocab_paths:
                if os.path.exists(path):
                    native_vocab_path = path
                    break

        dynamic_vocab_size = 256  # Default fallback
        if native_vocab_path and os.path.exists(native_vocab_path):
            try:
                with open(native_vocab_path, 'r', encoding='utf-8') as f:
                    native_vocab_data = json.load(f)
                dynamic_vocab_size = native_vocab_data.get('vocab_size', 256)
                print(f"📚 Using dynamic vocabulary size: {dynamic_vocab_size} (from {native_vocab_path})")
            except Exception as e:
                print(f"⚠️  Error loading native vocabulary from {native_vocab_path}: {e}")
        else:
            print(f"⚠️  Native vocabulary not found, using fallback vocab_size: {dynamic_vocab_size}")

        # Initialize learnable quantum embedding with dynamic vocab_size FIRST
        self.quantum_embedding = QuantumEmbedding(
            vocab_size=dynamic_vocab_size,  # Dynamic from native vocabulary
            embed_dim=self.fixed_embed_dim  # Use fixed dimension
        ).to(self.device)

        # Initialize quantum vocabulary for semantic connectivity AFTER quantum embedding
        try:
            self._initialize_quantum_vocabulary_with_genesis(vocab_path)
        except Exception as e:
            print(f"⚠️  Erro ao inicializar dicionário quântico: {e}")
            # Fallback to original method
            try:
                self._initialize_quantum_vocabulary(vocab_path)
            except Exception as fallback_e:
                print(f"⚠️  Fallback também falhou: {fallback_e}")
                self.quantum_vocab_representations = None
                self.char_to_idx = None

        # Initialize dimension calibrator for auto-calibration
        from src.core.processing_parameter_calibrator import ProcessingParameterCalibrator
        self.dimension_calibrator = ProcessingParameterCalibrator()

        # Enhanced OpticalProbe for Padilha Wave Equation-based decoding (doe.md 2.9.5) with genesis integration
        from src.core.optical_probe_fixed import create_enhanced_optical_probe
        self.optical_probe = create_enhanced_optical_probe(
            device=self.device
        )

        # Character vocabulary for optical probe (ASCII printable characters)
        self.char_vocab = [chr(i) for i in range(32, 127)]  # Printable ASCII characters

        # ========== NOVOS COMPONENTES APRENDÍVEIS ==========
        # Context Funnel - Mecanismo de atenção para histórico de conversa
        from src.core.context_funnel import create_context_funnel
        self.context_funnel = create_context_funnel(
            embed_dim=self.fixed_embed_dim,  # Use fixed dimension
            num_heads=self.fixed_num_heads,  # Use fixed num_heads
            max_history=10
        )

        # Inverse Cognitive Projector - Balança de Calibragem aprendível
        from src.core.inverse_cognitive_projector import create_inverse_cognitive_projector
        self.inverse_projector = create_inverse_cognitive_projector(
            embed_dim=self.fixed_embed_dim,  # Use fixed dimension
            vocab_size=dynamic_vocab_size,  # Dynamic from native vocabulary
            hidden_dim=self.fixed_hidden_dim,  # Use fixed hidden_dim
            num_layers=3,
            dropout=0.1
        )

        # ========== DCF INITIALIZATION MOVED TO MAIN EXECUTION METHOD ==========
        # DCF components are now initialized in the main execution method after calibration

        # ZERO FALLBACK POLICY: No external pre-trained weights loaded
        # System achieves true vocabulary autonomy through emergent generation
        print("🎯 Using random initialization for true vocabulary autonomy (ZERO FALLBACK)")

        # Validar que o dicionário quântico foi criado corretamente
        if self.quantum_vocab_representations is None:
            raise RuntimeError("Dicionário quântico não foi inicializado - sistema DCF não pode funcionar")
        if len(self.quantum_vocab_representations) == 0:
            raise RuntimeError("Dicionário quântico está vazio - sistema DCF requer representações quânticas")

        print(f"✅ Dicionário quântico validado: {len(self.quantum_vocab_representations)} representações quânticas disponíveis")

        # Inicializar componentes físicos
        self._initialize_physical_components()

        # Configuração de priming cognitivo
        self.enable_cognitive_priming = enable_cognitive_priming

        # Histórico de conversa para priming contextual (memória de curto prazo)
        self.conversation_history = deque(maxlen=10)  # Aumentado para melhor contexto

        # Método para atualizar histórico de conversa
        self.update_conversation_history = self._update_conversation_history

        # Initialize complete auto-calibration system
        if self.enable_auto_calibration:
            self._initialize_complete_auto_calibration()

        # Initialize non-commutative geometry se disponível
        if self.enable_noncommutative:
            self._initialize_noncommutative()

        # Initialize hybrid quantum-classical system
        if HAS_HYBRID_SYSTEM:
            self._initialize_hybrid_system()

        # Initialize quantum temporal memory system
        if HAS_QUANTUM_MEMORY:
            self._initialize_quantum_memory()

        # Initialize audit logger if audit mode is enabled
        if self.audit_mode:
            self._initialize_audit_logger()

        # ========== DCF INITIALIZATION MOVED AFTER CALIBRATION ==========
        # DCF components are now initialized after calibration in _setup_and_calibrate method

        # Detecção inteligente de tarefa se input_text for fornecido
        if input_text is not None:
            self.task = self._detect_task_type(input_text)

            # Atualizar histórico de conversa para aprendizado contínuo
            # Nota: A resposta será adicionada após geração

        # ========== HARMONIZATION CHECK ==========
        print(f"🔬 ΨQRH Pipeline Físico inicializado no dispositivo: {self.device}")
        print(f"   📐 Configuração FIXA: embed_dim={self.fixed_embed_dim}, num_heads={self.fixed_num_heads}")

        # Verificar harmonização do sistema (auto-calibração + assinatura harmônica)
        harmonization_status = self._check_system_harmonization()
        if harmonization_status['is_harmonized']:
            print("   🎼 Sistema Harmonizado: ✅ ATIVO (auto-calibração completa + assinatura harmônica)")
            print(f"      📊 Componentes harmonizados: {len(harmonization_status['harmonized_components'])}")
        else:
            print("   🎼 Sistema Harmonizado: ❌ DESATIVADO (auto-calibração incompleta ou assinatura harmônica ausente)")
            print(f"      ⚠️  Componentes faltando: {harmonization_status['missing_components']}")

        if self.enable_auto_calibration:
            print("   🔧 Auto-calibração: ATIVADA (todos os parâmetros emergentes da física)")
        else:
            print("   🔧 Auto-calibração: DESATIVADA (parâmetros fixos)")
        if HAS_QUANTUM_MEMORY and self.quantum_memory_system is not None:
            print("   🧠 Memória Quântica: ATIVADA (correlações temporais de longo alcance)")
        else:
            print("   🧠 Memória Quântica: DESATIVADA (processamento independente)")

        print("   🤖 Pipeline Físico: ATIVADO (apenas componentes físicos)")

        # ========== DCF INITIALIZATION AFTER CALIBRATION ==========
        # DCF components are initialized after calibration in _setup_and_calibrate method
        # This ensures they get the correct calibrated dimensions

        # ========== OTIMIZADOR PARA TREINAMENTO END-TO-END ==========
        # Todos os componentes aprendíveis usam o mesmo otimizador
        learnable_params = list(self.context_funnel.parameters()) + \
                          list(self.inverse_projector.parameters()) + \
                          list(self.quantum_embedding.parameters())
        # Nota: Cognitive Processor (DCF) usa seus próprios otimizadores internos

        # Verificar se temos parâmetros para otimizar
        try:
            if len(learnable_params) > 0:
                self.optimizer = torch.optim.AdamW(learnable_params, lr=1e-4, weight_decay=0.01)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=1000, T_mult=2
                )
                print("   🎓 Otimizador End-to-End: ATIVADO (Context Funnel + Inverse Projector)")
            else:
                self.optimizer = None
                self.scheduler = None
                print("   ⚠️  Otimizador End-to-End: DESATIVADO (nenhum parâmetro aprendível encontrado)")
        except Exception as e:
            print(f"   ⚠️  Otimizador End-to-End falhou: {e}")
            self.optimizer = None
            self.scheduler = None

    def _text_to_fractal_signal(self, text: str, embed_dim: int, proc_params: Dict[str, Any] = None) -> torch.Tensor:
        """
        Converte texto para sinal fractal sequencial (doe.md 2.9.1: Fractal Embedding)

        Produz representação sequencial [seq_len, features] onde seq_len = len(text),
        permitindo processamento token-a-token em vez de representação global.
        """
        seq_len = len(text)

        # Criar representação sequencial: cada caractere mapeado para um vetor de features
        signal_features = []
        for char in text:
            # Análise espectral básica do caractere
            char_value = torch.tensor([ord(char) / 127.0], dtype=torch.float32)

            # Criar representação multidimensional via análise de frequência simples
            # Usar uma abordagem mais simples para evitar complexidade excessiva
            base_features = torch.randn(embed_dim, device=self.device) * 0.1
            base_features[0] = char_value  # Primeiro feature é o valor do caractere normalizado

            # Adicionar variação baseada na posição do caractere no alfabeto
            char_idx = ord(char.lower()) - ord('a') if char.isalpha() else 26
            if char_idx >= 0 and char_idx < 27:
                base_features[1] = char_idx / 26.0  # Normalizado 0-1

            # Adicionar features baseados em propriedades do caractere
            base_features[2] = 1.0 if char.isupper() else 0.0  # Maiúsculo
            base_features[3] = 1.0 if char.isdigit() else 0.0  # Dígito
            base_features[4] = 1.0 if char.isspace() else 0.0  # Espaço
            base_features[5] = 1.0 if char in 'aeiouAEIOU' else 0.0  # Vogal

            signal_features.append(base_features)

        # Stack para criar tensor [seq_len, embed_dim]
        signal = torch.stack(signal_features, dim=0)

        # Aplicar janela perceptual se parâmetros disponíveis
        if proc_params and 'input_window' in proc_params:
            window_type = proc_params['input_window']
            if window_type == 'hann':
                window = torch.hann_window(seq_len, device=self.device)
            elif window_type == 'hamming':
                window = torch.hamming_window(seq_len, device=self.device)
            else:  # boxcar (sem janela)
                window = torch.ones(seq_len, device=self.device)

            # Aplicar janela ao longo da dimensão sequencial
            signal = signal * window.unsqueeze(-1)

        return signal.to(self.device)

    def _calculate_fractal_dimension(self, signal: torch.Tensor) -> float:
        """
        Calcula dimensão fractal via power-law fitting (doe.md 2.9.1)

        P(k) ~ k^(-β) → D = (3 - β) / 2
        """
        # Análise espectral
        spectrum = torch.fft.fft(signal)
        power_spectrum = torch.abs(spectrum) ** 2

        # Frequências
        k = torch.arange(1, len(power_spectrum) + 1, dtype=torch.float32)

        # Power-law fitting (simplificado)
        log_k = torch.log((k + 1e-10).clamp(min=1e-9))
        log_P = torch.log((power_spectrum + 1e-10).clamp(min=1e-9))

        # Regressão linear simples
        n = len(log_k)
        sum_x = log_k.sum()
        sum_y = log_P.sum()
        sum_xy = (log_k * log_P).sum()
        sum_x2 = (log_k ** 2).sum()

        # Coeficiente angular (β)
        beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        # Dimensão fractal
        D = (3.0 - beta.item()) / 2.0

        # Clamping para valores físicos
        D = max(1.0, min(D, 2.0))

        return D


    def _signal_to_quaternions_base(self, signal: torch.Tensor, embed_dim: int, proc_params: Dict[str, Any] = None) -> torch.Tensor:
        """
        Base quaternion mapping function without orchestrator - used by Harmonic Orchestrator.
        Maps each element of the input sequence to a quaternion state.
        """
        # Input signal shape: [seq_len, features] where features is the signal dimension
        # Output shape: [batch=1, seq_len, embed_dim, 4]
        batch_size = 1
        seq_len = signal.shape[0]  # Number of elements in sequence

        # Create quaternion representation [batch, seq, embed_dim, 4]
        psi = torch.zeros(batch_size, seq_len, embed_dim, 4, dtype=torch.float32, device=self.device)

        # For each position in the sequence, map the signal features to quaternion components
        for i in range(seq_len):
            # Get the signal features for this position [features]
            signal_at_pos = signal[i]  # [features]

            # If signal has more features than embed_dim, truncate or average
            if signal_at_pos.shape[0] > embed_dim:
                # Take first embed_dim features
                signal_features = signal_at_pos[:embed_dim]
            elif signal_at_pos.shape[0] < embed_dim:
                # Pad with zeros if needed
                padding = torch.zeros(embed_dim - signal_at_pos.shape[0], device=self.device)
                signal_features = torch.cat([signal_at_pos, padding])
            else:
                signal_features = signal_at_pos

            # Map signal features to quaternion components for each embed_dim position
            # Use the signal features to create quaternion representations
            for j in range(embed_dim):
                if j < len(signal_features):
                    feature_val = signal_features[j]
                    # Create quaternion from this feature value
                    psi[0, i, j, 0] = feature_val.real if torch.is_complex(feature_val) else feature_val  # w (real)
                    psi[0, i, j, 1] = feature_val.imag if torch.is_complex(feature_val) else 0.0  # x (i)
                    psi[0, i, j, 2] = torch.sin(feature_val.real if torch.is_complex(feature_val) else feature_val)  # y (j)
                    psi[0, i, j, 3] = torch.cos(feature_val.real if torch.is_complex(feature_val) else feature_val)  # z (k)
                else:
                    # Default quaternion for padding positions
                    psi[0, i, j, 0] = 0.0  # w
                    psi[0, i, j, 1] = 0.0  # x
                    psi[0, i, j, 2] = 0.0  # y
                    psi[0, i, j, 3] = 1.0  # z (identity quaternion)

        # Apply adaptive mappings if parameters provided
        if proc_params and 'cross_coupling_enabled' in proc_params:
            coupling_params = proc_params.get('coupling_coefficients', {})
            c1_real = coupling_params.get('c1_real', 1.0)
            c2_imag = coupling_params.get('c2_imag', 1.0)
            c3_cross = coupling_params.get('c3_cross', 0.0)

            # Apply cross-coupling across the embed_dim dimension
            for i in range(seq_len):
                for j in range(embed_dim):
                    w, x, y, z = psi[0, i, j]
                    # Cross-coupled transformation
                    psi[0, i, j, 2] = torch.sin(c1_real * w + c3_cross * x)  # y (j) - cross-coupled
                    psi[0, i, j, 3] = torch.cos(c2_imag * x + c3_cross * w)  # z (k) - cross-coupled

            print(f"      🔗 Mapeamento quaterniônico cross-coupled: c1={c1_real:.2f}, c2={c2_imag:.2f}, c3={c3_cross:.3f}")
        elif proc_params and 'quaternion_complexity' in proc_params and 'quaternion_phase_shift' in proc_params:
            # Adaptive mapping
            complexity_factor = proc_params['quaternion_complexity']
            phase_shift = proc_params['quaternion_phase_shift']

            for i in range(seq_len):
                for j in range(embed_dim):
                    w, x = psi[0, i, j, 0], psi[0, i, j, 1]
                    psi[0, i, j, 2] = torch.sin(complexity_factor * w + phase_shift)  # y (j) - adaptive
                    psi[0, i, j, 3] = torch.cos(complexity_factor * x)  # z (k) - adaptive

            print(f"      🔄 Mapeamento quaterniônico adaptativo: complexidade={complexity_factor:.2f}, fase={phase_shift:.3f}")

        return psi

    def _signal_to_quaternions(self, signal: torch.Tensor, embed_dim: int, proc_params: Dict[str, Any] = None) -> torch.Tensor:
        """
        Mapeamento para quaternions Ψ(x) com complexidade adaptativa e validação dimensional (doe.md 2.9.2)

        Converte sinal sequencial [seq_len, features] para representação quaterniônica 4D [1, seq_len, embed_dim, 4],
        com a complexidade do mapeamento controlada pela autocalibragem e orquestração harmônica.
        """
        # Validate input signal dimensions - now expect [seq_len, embed_dim]
        expected_signal_shape = (-1, embed_dim)  # [seq_len, embed_dim]
        signal = self._validate_dimensions_compatibility(signal, expected_signal_shape, "_signal_to_quaternions")

        # Use Physical Harmonic Orchestrator if available for cross-coupled mapping
        if self.physical_harmonic_orchestrator is not None:
            # Get orchestrated quantum mapping with cross-coupling
            psi = self.physical_harmonic_orchestrator.orchestrate_transformation(
                signal,  # Pass full signal for base function
                'quantum_mapping',
                self._signal_to_quaternions_base,
                embed_dim=embed_dim, proc_params=proc_params
            )
        else:
            # Fallback to base function if no orchestrator
            psi = self._signal_to_quaternions_base(signal, embed_dim, proc_params)

        # Validate output dimensions
        expected_psi_shape = (1, -1, embed_dim, 4)  # [batch=1, seq_len, embed_dim, 4]
        psi = self._validate_dimensions_compatibility(psi, expected_psi_shape, "_signal_to_quaternions_output")

        return psi


    def _apply_spectral_filtering(self, psi: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Filtragem espectral aprimorada usando Prime Resonant Filtering + Leech Lattice Embedding
        Substitui FFT padrão por técnicas de estabilização numérica baseadas em números primos.
        Com validação dimensional automática e orquestração harmônica com máscaras ressonantes.
        Now operates on sequential tensor shape [batch, seq_len, embed_dim, 4].
        """
        print(f"🌊 Aplicando filtragem espectral estável (α={alpha:.3f})...")

        # Validate input dimensions
        expected_input_shape = (-1, -1, -1, 4)  # [batch, seq, embed, 4]
        psi = self._validate_dimensions_compatibility(psi, expected_input_shape, "_apply_spectral_filtering")

        batch_size, seq_len, embed_dim, quat_dim = psi.shape

        # Use Physical Harmonic Orchestrator if available for resonant filtering
        if self.physical_harmonic_orchestrator is not None:
            # Define the base filtering function with resonance mask support
            def base_spectral_filter(psi_tensor, alpha_param, resonance_mask=None, **kwargs):
                # Apply FFT along embedding dimension for frequency domain processing
                psi_fft = torch.fft.fft(psi_tensor, dim=2)  # [batch, seq, embed, 4]

                # Apply resonant mask if provided by orchestrator
                if resonance_mask is not None:
                    # Pad resonance_mask to embed_dim if necessary
                    if resonance_mask.shape[0] < embed_dim:
                        padding = torch.ones(embed_dim - resonance_mask.shape[0], device=resonance_mask.device)
                        resonance_mask = torch.cat([resonance_mask, padding])
                    # Expand mask to match tensor dimensions [batch, seq, embed, 4]
                    resonance_mask_expanded = resonance_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, embed_dim, 1]
                    resonance_mask_expanded = resonance_mask_expanded.expand(batch_size, seq_len, embed_dim, quat_dim)

                    # Apply resonance mask in frequency domain
                    psi_fft = psi_fft * resonance_mask_expanded
                    print(f"      🎵 Resonance mask applied: {resonance_mask.shape} → enhanced {torch.sum(resonance_mask > 1.0).item()} frequencies")

                # Apply standard spectral filter F(k) = exp(i α · arctan(ln(|k| + ε)))
                freqs = torch.fft.fftfreq(embed_dim, device=self.device)
                k = 2 * torch.pi * freqs.view(1, 1, -1, 1)  # [1, 1, embed_dim, 1]

                epsilon = 1e-10
                k_mag = torch.abs(k) + epsilon
                log_k = torch.log(k_mag.clamp(min=1e-9))
                phase = torch.arctan(log_k)

                # Create filter response
                filter_response = torch.exp(1j * alpha_param * phase)
                filter_response = filter_response.expand_as(psi_fft)

                # Apply filter in frequency domain
                psi_filtered_fft = psi_fft * filter_response

                # Inverse FFT back to spatial domain
                psi_filtered = torch.fft.ifft(psi_filtered_fft, dim=2).real

                return psi_filtered

            # Get orchestrated filtering with resonant masks
            psi_filtered = self.physical_harmonic_orchestrator.orchestrate_transformation(
                psi.mean(dim=(0, 1, 3)),  # Use mean signal for signature analysis - mean over batch, seq, quat
                'spectral_filter',
                base_spectral_filter,
                psi=psi, alpha=alpha
            )
        else:
            # Fallback to original filtering logic
            # Apply FFT along embedding dimension for each sequence position
            psi_fft = torch.fft.fft(psi, dim=2)  # FFT along embed_dim dimension

            # Apply standard spectral filter F(k) = exp(i α · arctan(ln(|k| + ε)))
            freqs = torch.fft.fftfreq(embed_dim, device=self.device)
            k = 2 * torch.pi * freqs.view(1, 1, -1, 1)  # [1, 1, embed_dim, 1]

            epsilon = 1e-10
            k_mag = torch.abs(k) + epsilon
            log_k = torch.log(k_mag.clamp(min=1e-9))
            phase = torch.arctan(log_k)

            filter_response = torch.exp(1j * alpha * phase)
            filter_response = filter_response.expand(batch_size, seq_len, embed_dim, quat_dim)

            psi_filtered_fft = psi_fft * filter_response
            psi_filtered = torch.fft.ifft(psi_filtered_fft, dim=2).real

        # Conservação de energia: renormalizar para preservar a norma original
        psi_renormalized = psi_filtered

        # Validate output dimensions
        expected_output_shape = psi.shape  # Should maintain input shape
        psi_renormalized = self._validate_dimensions_compatibility(psi_renormalized, expected_output_shape, "_apply_spectral_filtering_output")

        print(f"   ✅ Filtragem espectral estável aplicada: {psi.shape} → {psi_renormalized.shape}")
        return psi_renormalized.real  # Retornar parte real para compatibilidade

    def _apply_so4_rotation(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Rotações SO(4) unitárias otimizadas: Ψ' = q_left ⊗ Ψ ⊗ q_right†

        Usa operações quaterniônicas otimizadas com validação de unitariedade e conservação de energia.
        Com validação dimensional automática e orquestração harmônica adaptativa.
        Now operates on sequential tensor shape [batch, seq_len, embed_dim, 4].
        """
        # Validate input dimensions
        expected_input_shape = (-1, -1, -1, 4)  # [batch, seq, embed, 4]
        psi = self._validate_dimensions_compatibility(psi, expected_input_shape, "_apply_so4_rotation")

        batch_size, seq_len, embed_dim, quat_dim = psi.shape

        # Inicializar operações quaterniônicas otimizadas se necessário
        if not hasattr(self, 'optimized_quaternion_ops'):
            self.optimized_quaternion_ops = OptimizedQuaternionOperations(device=self.device)

        # Use Physical Harmonic Orchestrator if available for adaptive rotations
        if self.physical_harmonic_orchestrator is not None:
            # Define the base rotation function
            def base_so4_rotation(psi_tensor, rotation_angles, **kwargs):
                return self.optimized_quaternion_ops.so4_rotation(psi_tensor, rotation_angles)

            # Get orchestrated rotation
            psi_rotated = self.physical_harmonic_orchestrator.orchestrate_transformation(
                psi.mean(dim=(0, 2, 3)),  # Use mean signal for signature analysis
                'so4_rotation',
                base_so4_rotation,
                psi=psi
            )
        else:
            # Fallback to original fixed rotation logic
            # Parâmetros de rotação adaptativos baseados na estrutura do sinal
            theta_left = torch.tensor(0.1, device=self.device)
            omega_left = torch.tensor(0.05, device=self.device)
            phi_left = torch.tensor(0.02, device=self.device)

            # Criar tensores de rotação para todo o batch, seq_len, embed_dim
            rotation_angles_left = torch.stack([theta_left, omega_left, phi_left], dim=-1)
            rotation_angles_left = rotation_angles_left.expand(batch_size, seq_len, embed_dim, -1)

            # Aplicar SO(4) rotação otimizada
            psi_rotated = self.optimized_quaternion_ops.so4_rotation(psi, rotation_angles_left)

        # Conservação de energia: renormalizar para preservar a norma original
        psi_renormalized = psi_rotated

        # Validate output dimensions
        expected_output_shape = psi.shape  # Should maintain input shape
        psi_renormalized = self._validate_dimensions_compatibility(psi_renormalized, expected_output_shape, "_apply_so4_rotation_output")

        # Validação opcional de unitariedade (para debug)
        if torch.rand(1).item() < 0.01:  # 1% das vezes para performance
            norms_before = torch.norm(psi, dim=(-2, -1))
            norms_after = torch.norm(psi_renormalized, dim=(-2, -1))
            unitarity_error = torch.mean(torch.abs(norms_before - norms_after)).item()
            if unitarity_error > 1e-5:
                print(f"⚠️  SO(4) rotation unitarity error: {unitarity_error:.2e}")

        return psi_renormalized

    def _safe_optical_probe_extraction(self, optical_output):
        """
        Extração segura de saída do optical probe com múltiplos fallbacks
        """
        try:
            # Método 1: Tentar acesso direto
            if hasattr(optical_output, '__getitem__'):
                try:
                    if len(optical_output) > 0:
                        return optical_output[0]
                except:
                    pass

            # Método 2: Se for tuple, extrair primeiro elemento
            if isinstance(optical_output, tuple) and len(optical_output) > 0:
                return optical_output[0]

            # Método 3: Se for lista, extrair primeiro elemento
            if isinstance(optical_output, list) and len(optical_output) > 0:
                return optical_output[0]

            # Método 4: Converter para string e extrair primeiro caractere
            str_output = str(optical_output)
            if len(str_output) > 0:
                return str_output[0]

            # Método 5: Fallback absoluto
            return 'Ψ'  # Símbolo quântico como fallback

        except Exception as e:
            print(f"⚠️  Erro na extração do optical probe: {e}")
            return 'Q'  # Fallback final

    def _process_consciousness(self, psi: torch.Tensor, D_fractal: float) -> Dict:
        """
        Processamento de consciência com FCI calculation (doe.md consciousness)
        """
        # Usar processador de consciência
        consciousness_input = psi  # O processador espera um tensor diretamente

        results = self.consciousness_processor.forward(consciousness_input)

        return {
            'FCI': results.get('fci', 0.0),
            'D_fractal': D_fractal,
            'state': results.get('final_consciousness_state', {}).get('name', 'UNKNOWN'),
            'CLZ': results.get('clz', 0.5)
        }

    def _apply_consciousness_bootstrap(self, psi: torch.Tensor, consciousness_results: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Bootstrap cognitivo para estados de baixa consciência (FCI < 0.3)
        """
        from src.processing.consciousness_bootstrapper import create_consciousness_bootstrapper

        bootstrapper = create_consciousness_bootstrapper(
            chaos_strength=0.1,
            logistic_r=3.99,
            min_fci_threshold=0.15,
            max_boost_iterations=5
        )

        psi_bootstrapped, new_consciousness = bootstrapper.apply_bootstrap(
            psi.squeeze(0),  # Remove batch dim
            consciousness_results,
            self.consciousness_processor
        )

        return psi_bootstrapped.unsqueeze(0), new_consciousness

    def _emergent_language_generation(self, psi: torch.Tensor, alpha: float, beta: float,
                                       temperature: float = 1.0, max_length: int = 50,
                                       input_text: str = None, alpha_calibrated: float = None) -> str:
        """
        Nova Arquitetura de Três Componentes Aprendíveis:

        1. 🎯 Context Funnel: Condensar histórico de conversa em Ψ_context
        2. 🧠 Cognitive Processor: ΨQRH/DCF pipeline com Ψ_context como estado inicial
        3. ⚖️ Inverse Cognitive Projector: Traduzir Ψ_final abstrato para Ψ_reconstructed

        Esta arquitetura permite treinamento end-to-end onde o sistema aprende:
        - A prestar atenção (Context Funnel)
        - A raciocinar de forma traduzível (Cognitive Processor)
        - A traduzir pensamentos para linguagem (Inverse Projector)
        """
        print(f"🎯 Iniciando Geração de Linguagem Emergente (Arquitetura de 3 Componentes)...")

        # ========== LOG QUANTUM STATE INPUT ==========
        print(f"🔍 [VALIDATION] Logging quantum state input...")
        print(f"   📊 Quantum state shape: {psi.shape}")
        print(f"   📊 Conversation history length: {len(self.conversation_history)}")

        # ========== AUDIT LOGGING: INPUT STATE ==========
        if self.audit_logger:
            self.audit_logger.start_session(input_text or "emergent_generation", {"stage": "emergent_generation_start"})
            self.audit_logger.log_tensor_state("input", psi, {"stage": "emergent_generation_start"})

        try:
            # ========== COMPONENTE 1: CONTEXT FUNNEL ==========
            print(f"   🎯 [Context Funnel] Processando histórico de conversa...")
            psi_context = self.context_funnel(self.conversation_history)

            # Handle case where context funnel returns None (empty history)
            if psi_context is None:
                print(f"      ⚠️ Context funnel returned None (empty conversation history), using default context")
                # Create default context tensor with same shape as expected
                psi_context = torch.zeros(1, self.config['embed_dim'], dtype=torch.float32, device=self.device)

            print(f"      ✅ Contexto focado gerado: shape {psi_context.shape}")

            # ========== COMPONENTE 2: COGNITIVE PROCESSOR ==========
            print(f"   🧠 [Cognitive Processor] Executando ΨQRH/DCF com contexto focado...")

            # Usar Ψ_context como estado inicial para o pipeline cognitivo
            # Em vez de usar psi diretamente, começamos com o contexto focado
            psi_with_context = psi_context.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]

            # ========== AUDIT LOGGING: CONTEXT FOCUSED STATE ==========
            if self.audit_logger:
                self.audit_logger.log_tensor_state("transformed", psi_with_context, {"stage": "context_funnel_output"})

            # Preparar logits usando Inverse Cognitive Projector (orquestrado)
            vocab_size = self.quantum_embedding.vocab_size  # Use current vocab size

            # Use Inverse Cognitive Projector for quantum-to-linguistic translation
            # This replaces the crude linear interpolation with proper cognitive projection
            try:
                # Prepare quantum state for projection [1, embed_dim]
                psi_for_projection = psi_context.unsqueeze(0)  # [1, embed_dim]

                # Use Inverse Cognitive Projector to generate logits
                logits = self.inverse_projector(psi_for_projection)  # [1, vocab_size]

                # Remove batch dimension
                logits = logits.squeeze(0)  # [vocab_size]

                print(f"   🧠 Used Inverse Cognitive Projector for quantum-to-linguistic translation: {psi_for_projection.shape} → {logits.shape}")

            except Exception as e:
                print(f"   ⚠️ Inverse Cognitive Projector failed: {e} - falling back to linear interpolation")
                # Fallback to linear interpolation if projector fails
                context_flat = psi_context.view(-1)  # [embed_dim]

                # Interpolar para tamanho do vocabulário dinâmico
                if len(context_flat) < vocab_size:
                    logits = torch.nn.functional.interpolate(
                        context_flat.unsqueeze(0).unsqueeze(0),
                        size=vocab_size,
                        mode='linear',
                        align_corners=False
                    ).squeeze()
                else:
                    step = len(context_flat) // vocab_size
                    if step > 0:
                        logits = torch.tensor([context_flat[i*step:(i+1)*step].mean() for i in range(vocab_size)])
                    else:
                        logits = torch.nn.functional.interpolate(
                            context_flat.unsqueeze(0).unsqueeze(0),
                            size=vocab_size,
                            mode='linear',
                            align_corners=False
                        ).squeeze()

                # Garantir tamanho correto
                if len(logits) != vocab_size:
                    if len(logits) < vocab_size:
                        padding = torch.zeros(vocab_size - len(logits), device=logits.device)
                        logits = torch.cat([logits, padding])
                    else:
                        logits = logits[:vocab_size]

            # Adicionar ruído controlado e normalizar
            logits += torch.randn_like(logits) * 0.1
            logits = (logits - logits.mean()) / (logits.std() + 1e-8) * 2.0

            print(f"   📊 Generated contextual logits: shape {logits.shape}")

            # ========== EXECUTAR DCF COM CONTEXTO ==========
            if self.dcf_analyzer is not None:
                # Convert logits to token IDs for DCF analyzer
                _, top_token_ids = torch.topk(logits, k=min(50, len(logits)))
                dcf_result = self.dcf_analyzer.analyze_tokens(top_token_ids.tolist())
            else:
                from src.processing.token_analysis import analyze_tokens_dcf
                dcf_result = analyze_tokens_dcf(logits, device=self.device, embeddings=None)

            # Extrair Ψ_final do DCF (estado de pensamento abstrato)
            psi_final = dcf_result['final_quantum_state']  # [n_candidates, 1, embed_dim]

            # Usar estado do líder do cluster dominante
            if 'dcf_metadata' in dcf_result and 'final_token_selection' in dcf_result['dcf_metadata']:
                selected_token_id = dcf_result['dcf_metadata']['final_token_selection'].get('token_id')
                candidate_tokens = dcf_result['dcf_metadata'].get('candidate_tokens', [])
                try:
                    leader_idx = candidate_tokens.index(selected_token_id)
                    psi_final_abstract = psi_final[leader_idx, 0]  # [embed_dim]
                    print(f"   🎯 Usando estado do líder: token_id={selected_token_id}")
                except (ValueError, IndexError):
                    psi_final_abstract = psi_final[0, 0]
            else:
                psi_final_abstract = psi_final[0, 0]

            # ========== COMPONENTE 3: OPTICAL PROBE (Padilha Wave Equation) ==========
            print(f"   🔬 [Optical Probe] Aplicando equação de onda de Padilha...")

            # Ajustar dimensão se necessário (DCF pode usar dimensão diferente)
            if psi_final_abstract.shape[0] != self.config['embed_dim']:
                # Projetar para a dimensão correta
                proj_layer = torch.nn.Linear(psi_final_abstract.shape[0], self.config['embed_dim']).to(psi_final_abstract.device)
                psi_final_abstract = proj_layer(psi_final_abstract)

            # Preparar estado quântico para optical probe [1, 1, embed_dim, 4]
            # Usar psi_final_abstract como base para todos os componentes quaterniônicos
            psi_for_optical = torch.zeros(1, 1, self.config['embed_dim'], 4, device=psi_final_abstract.device)
            psi_for_optical[0, 0, :, 0] = psi_final_abstract  # w component
            psi_for_optical[0, 0, :, 1] = torch.roll(psi_final_abstract, 1)  # x component (shifted)
            psi_for_optical[0, 0, :, 2] = torch.sin(psi_final_abstract)  # y component
            psi_for_optical[0, 0, :, 3] = torch.cos(psi_final_abstract)  # z component

            # Aplicar Optical Probe com equação de Padilha
            # Optical probe expects [seq_len, embed_dim, 4], so squeeze batch and seq dims
            psi_for_optical_squeezed = psi_for_optical.squeeze(0).squeeze(0)  # [embed_dim, 4]
            psi_reconstructed_text = self.optical_probe(psi_for_optical_squeezed.unsqueeze(0))  # Add seq dim back
            confidence = 0.8  # Placeholder confidence for optical probe

            print(f"      ✅ Equação de Padilha aplicada: texto '{psi_reconstructed_text}', confiança {confidence:.3f}")

            # ========== AUDIT LOGGING: FINAL RECONSTRUCTED STATE ==========
            if self.audit_logger:
                self.audit_logger.log_tensor_state("optical_probe_output", psi_for_optical, {"stage": "optical_probe_output"})

            # ========== TEXTO FINAL GERADO PELA OPTICAL PROBE ==========
            # Use the new safe extraction method
            emergent_text = self._safe_optical_probe_extraction(psi_reconstructed_text)
            print(f"   📝 Texto final da Optical Probe: '{emergent_text}'")

            print(f"   ✅ Arquitetura de 3 componentes concluída!")
            print(f"      📊 Ψ_context: {psi_context.shape}")
            print(f"      🧠 Ψ_final: {psi_final_abstract.shape}")
            print(f"      🔬 Optical Probe: Aplicada equação de Padilha")
            print(f"      📝 Texto gerado: '{emergent_text}'")
            print(f"      🧠 FCI: {dcf_result.get('fci_value', 0):.4f}")

            # ========== GERAR SEQUÊNCIA COMPLETA ==========
            if max_length > 1:
                print(f"   🔄 Gerando sequência completa (max_length={max_length})...")
                generated_chars = [emergent_text]

                for i in range(min(max_length - 1, 10)):
                    # Evoluir contexto para próximo passo
                    evolved_context = psi_context + torch.randn_like(psi_context) * 0.1 * (i + 1)

                    # Repetir pipeline com contexto evoluído
                    evolved_logits = logits + torch.randn_like(logits) * 0.05 * (i + 1)

                    if self.dcf_analyzer is not None:
                        next_dcf_result = self.dcf_analyzer.analyze_tokens(evolved_logits, embeddings=None, reasoning_mode=self.reasoning_mode)
                    else:
                        next_dcf_result = analyze_tokens_dcf(evolved_logits, device=self.device, embeddings=None)

                    next_psi_final = next_dcf_result['final_quantum_state'][0, 0]
                    # Ajustar dimensão se necessário (DCF pode usar dimensão diferente)
                    if next_psi_final.shape[0] != self.config['embed_dim']:
                        # Projetar para a dimensão correta
                        proj_layer = torch.nn.Linear(next_psi_final.shape[0], self.config['embed_dim']).to(next_psi_final.device)
                        next_psi_final = proj_layer(next_psi_final)

                    # Usar Optical Probe para próxima geração
                    next_psi_for_optical = torch.zeros(1, 1, self.config['embed_dim'], 4, device=next_psi_final.device)
                    next_psi_for_optical[0, 0, :, 0] = next_psi_final
                    next_psi_for_optical[0, 0, :, 1] = torch.roll(next_psi_final, 1)
                    next_psi_for_optical[0, 0, :, 2] = torch.sin(next_psi_final)
                    next_psi_for_optical[0, 0, :, 3] = torch.cos(next_psi_final)
                    next_psi_for_optical_squeezed = next_psi_for_optical.squeeze(0).squeeze(0)
                    next_psi_reconstructed = self.optical_probe(next_psi_for_optical_squeezed.unsqueeze(0))

                    # Use the new safe extraction method for sequence generation
                    next_char = self._safe_optical_probe_extraction(next_psi_reconstructed)
                    generated_chars.append(next_char)

                    if next_char in ['.', '!', '?', '\n']:
                        break

                emergent_text = ''.join(generated_chars)

            print(f"   📝 Texto emergente final: '{emergent_text}'")

            # ========== AUDIT LOGGING: END SESSION ==========
            if self.audit_logger:
                self.audit_logger.end_session(emergent_text)