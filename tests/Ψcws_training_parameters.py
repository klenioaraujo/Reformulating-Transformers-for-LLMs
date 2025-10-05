#!/usr/bin/env python3
"""
ΨCWS Training Parameters - Parâmetros para Treinamento de Modelos ΨCWS
========================================================================

Este arquivo define os parâmetros de treinamento para o sistema ΨCWS que converte:
TEXT → ESPECTRO → ESPECTRO SAÍDA → ESPECTRO ENTRADA → CONVERSÃO TEXT

O sistema usa:
- Modelos open-source como base
- Camada de criptografia (7 camadas)
- Máscara científica para garantir padrão
- Conversão espectral

Parâmetros otimizados para treinamento eficiente.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ΨCWSTrainingConfig:
    """Configuração completa para treinamento ΨCWS."""

    # ===== PARÂMETROS GERAIS =====
    model_name: str = "ΨCWS-Spectral-Transformer"
    version: str = "1.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== DIMENSÕES DO MODELO =====
    vocab_size: int = 50000  # Tamanho do vocabulário
    embedding_dim: int = 512  # Dimensão de embedding
    hidden_dim: int = 1024  # Dimensão oculta
    num_layers: int = 6  # Número de camadas
    num_heads: int = 8  # Número de cabeças de atenção
    dropout: float = 0.1  # Dropout rate

    # ===== PARÂMETROS ESPECTRAIS =====
    spectral_dim: int = 256  # Dimensão espectral
    fft_bins: int = 128  # Número de bins FFT
    window_size: int = 64  # Tamanho da janela
    hop_length: int = 32  # Hop length para STFT

    # ===== PARÂMETROS DE TREINAMENTO =====
    batch_size: int = 32  # Batch size
    learning_rate: float = 1e-4  # Taxa de aprendizado
    warmup_steps: int = 1000  # Passos de warmup
    max_epochs: int = 100  # Máximo de épocas
    gradient_clip: float = 1.0  # Clipping de gradiente
    weight_decay: float = 1e-6  # Decaimento de peso

    # ===== PARÂMETROS DE CRIPTOGRAFIA =====
    encryption_layers: int = 7  # 7 camadas de criptografia
    encryption_key_size: int = 32  # Tamanho da chave (bytes)
    salt_size: int = 16  # Tamanho do salt

    # ===== PARÂMETROS DE MÁSCARA CIENTÍFICA =====
    scientific_mask_enabled: bool = True
    mask_pattern: str = "fractal_gaussian"  # Padrão da máscara
    mask_entropy_threshold: float = 0.8  # Threshold de entropia

    # ===== PARÂMETROS DE CONVERSÃO TEXT-ESPECTRO =====
    text_encoding: str = "utf-8"  # Encoding de texto
    max_sequence_length: int = 1024  # Comprimento máximo da sequência
    spectral_compression_ratio: float = 0.5  # Taxa de compressão espectral

    # ===== PARÂMETROS DE VALIDAÇÃO =====
    validation_frequency: int = 1000  # Frequência de validação
    early_stopping_patience: int = 10  # Paciência para early stopping
    checkpoint_frequency: int = 5000  # Frequência de checkpoint

    # ===== PARÂMETROS AVANÇADOS =====
    use_mixed_precision: bool = True  # Precisão mista
    gradient_accumulation_steps: int = 4  # Passos de acumulação
    use_spectral_norm: bool = True  # Normalização espectral
    use_layer_norm: bool = True  # Normalização de camada

    # ===== PARÂMETROS DE OTIMIZAÇÃO =====
    optimizer: str = "AdamW"  # Otimizador
    scheduler: str = "cosine"  # Scheduler
    beta1: float = 0.9  # Beta1 para Adam
    beta2: float = 0.999  # Beta2 para Adam
    epsilon: float = 1e-8  # Epsilon para Adam

    # ===== PARÂMETROS DE MEMÓRIA =====
    memory_efficient: bool = True  # Modo eficiente em memória
    gradient_checkpointing: bool = True  # Gradient checkpointing

    # ===== PARÂMETROS DE LOGGING =====
    log_frequency: int = 100  # Frequência de logging
    save_frequency: int = 1000  # Frequência de salvamento

    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class ΨCWSSpectralConfig:
    """Configuração específica para processamento espectral."""

    # Transformações espectrais
    use_stft: bool = True  # Usar Short-Time Fourier Transform
    use_mel_scale: bool = True  # Usar escala Mel
    use_mfcc: bool = False  # Usar MFCC

    # Parâmetros STFT
    sample_rate: int = 16000  # Taxa de amostragem
    n_fft: int = 1024  # Número de pontos FFT
    win_length: int = 1024  # Comprimento da janela
    hop_length: int = 256  # Hop length
    window: str = "hann"  # Tipo de janela

    # Parâmetros Mel
    n_mels: int = 80  # Número de bandas Mel
    fmin: float = 80.0  # Frequência mínima
    fmax: float = 8000.0  # Frequência máxima

    # Normalização espectral
    spectral_norm_type: str = "per_channel"  # Tipo de normalização
    spectral_mean: float = 0.0  # Média para normalização
    spectral_std: float = 1.0  # Desvio padrão para normalização

    # Compressão espectral
    compression_method: str = "log"  # Método de compressão
    compression_factor: float = 0.3  # Fator de compressão

    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class ΨCWSEncryptionConfig:
    """Configuração para sistema de criptografia."""

    # Camadas de criptografia
    layer_1: str = "AES-256-GCM"
    layer_2: str = "ChaCha20-Poly1305"
    layer_3: str = "Fernet"
    layer_4: str = "XOR-Custom"
    layer_5: str = "Transposition"
    layer_6: str = "HMAC-AES"
    layer_7: str = "Obfuscation"

    # Parâmetros de segurança
    key_derivation_iterations: int = 1000000  # Iterações para derivação de chave
    salt: str = "PSIQRH_SECURE_SALT_v1.0"  # Salt para derivação
    hmac_algorithm: str = "SHA256"  # Algoritmo HMAC

    # Política anti-violacao
    max_access_attempts: int = 3  # Máximo de tentativas de acesso
    violation_threshold: float = 0.9  # Threshold para detecção de violação

    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class ΨCWSModelArchitecture:
    """Arquitetura do modelo ΨCWS."""

    # Encoder (Texto → Espectro)
    encoder_layers: int = 6
    encoder_heads: int = 8
    encoder_dim: int = 512
    encoder_ff_dim: int = 2048

    # Decoder (Espectro → Texto)
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_dim: int = 512
    decoder_ff_dim: int = 2048

    # Camada Espectral
    spectral_encoder_layers: int = 4
    spectral_decoder_layers: int = 4
    spectral_attention_heads: int = 4

    # Conexões Residuais
    use_residual_connections: bool = True
    residual_scaling: float = 0.1

    # Normalização
    use_layer_norm: bool = True
    use_spectral_norm: bool = True
    norm_epsilon: float = 1e-5

    # Ativações
    activation_function: str = "gelu"
    output_activation: str = "softmax"

    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class ΨCWSTrainingParameters:
    """Classe principal para gerenciar parâmetros de treinamento ΨCWS."""

    def __init__(self):
        self.training_config = ΨCWSTrainingConfig()
        self.spectral_config = ΨCWSSpectralConfig()
        self.encryption_config = ΨCWSEncryptionConfig()
        self.model_architecture = ΨCWSModelArchitecture()

    def get_all_parameters(self) -> Dict[str, Any]:
        """Retorna todos os parâmetros em um dicionário."""
        return {
            "training": self.training_config.to_dict(),
            "spectral": self.spectral_config.to_dict(),
            "encryption": self.encryption_config.to_dict(),
            "architecture": self.model_architecture.to_dict()
        }

    def get_training_hyperparameters(self) -> Dict[str, Any]:
        """Retorna hiperparâmetros específicos para treinamento."""
        return {
            "batch_size": self.training_config.batch_size,
            "learning_rate": self.training_config.learning_rate,
            "warmup_steps": self.training_config.warmup_steps,
            "max_epochs": self.training_config.max_epochs,
            "gradient_clip": self.training_config.gradient_clip,
            "weight_decay": self.training_config.weight_decay,
            "optimizer": self.training_config.optimizer,
            "scheduler": self.training_config.scheduler
        }

    def get_model_parameters(self) -> Dict[str, Any]:
        """Retorna parâmetros específicos do modelo."""
        return {
            "vocab_size": self.training_config.vocab_size,
            "embedding_dim": self.training_config.embedding_dim,
            "hidden_dim": self.training_config.hidden_dim,
            "num_layers": self.training_config.num_layers,
            "num_heads": self.training_config.num_heads,
            "dropout": self.training_config.dropout,
            "spectral_dim": self.training_config.spectral_dim
        }

    def get_spectral_parameters(self) -> Dict[str, Any]:
        """Retorna parâmetros específicos para processamento espectral."""
        return {
            "fft_bins": self.training_config.fft_bins,
            "window_size": self.training_config.window_size,
            "hop_length": self.training_config.hop_length,
            "sample_rate": self.spectral_config.sample_rate,
            "n_fft": self.spectral_config.n_fft,
            "n_mels": self.spectral_config.n_mels,
            "compression_method": self.spectral_config.compression_method
        }

    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """Valida todos os parâmetros e retorna status e lista de erros."""
        errors = []

        # Validações de treinamento
        if self.training_config.batch_size <= 0:
            errors.append("batch_size deve ser > 0")

        if self.training_config.learning_rate <= 0:
            errors.append("learning_rate deve ser > 0")

        if self.training_config.embedding_dim % self.training_config.num_heads != 0:
            errors.append("embedding_dim deve ser divisível por num_heads")

        # Validações espectrais
        if self.spectral_config.n_fft <= 0:
            errors.append("n_fft deve ser > 0")

        if self.spectral_config.hop_length <= 0:
            errors.append("hop_length deve ser > 0")

        if self.spectral_config.fmin >= self.spectral_config.fmax:
            errors.append("fmin deve ser < fmax")

        return len(errors) == 0, errors

    def optimize_for_hardware(self, device_type: str = "auto") -> None:
        """Otimiza parâmetros para hardware específico."""

        if device_type == "gpu" or (device_type == "auto" and torch.cuda.is_available()):
            # Otimizações para GPU
            self.training_config.batch_size = 64
            self.training_config.use_mixed_precision = True
            self.training_config.gradient_accumulation_steps = 2

        elif device_type == "cpu":
            # Otimizações para CPU
            self.training_config.batch_size = 16
            self.training_config.use_mixed_precision = False
            self.training_config.gradient_accumulation_steps = 8

        elif device_type == "tpu":
            # Otimizações para TPU
            self.training_config.batch_size = 128
            self.training_config.use_mixed_precision = True
            self.training_config.gradient_accumulation_steps = 1


# Funções utilitárias para treinamento
def create_optimizer(model: nn.Module, config: ΨCWSTrainingConfig):
    """Cria otimizador baseado na configuração."""

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Otimizador não suportado: {config.optimizer}")


def create_scheduler(optimizer, config: ΨCWSTrainingConfig, total_steps: int):
    """Cria scheduler baseado na configuração."""

    if config.scheduler.lower() == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=total_steps)

    elif config.scheduler.lower() == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    elif config.scheduler.lower() == "exponential":
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, gamma=0.95)

    else:
        # Scheduler padrão: sem scheduler
        return None


# Configurações predefinidas
def get_preset_config(preset_name: str) -> ΨCWSTrainingParameters:
    """Retorna configuração predefinida."""

    params = ΨCWSTrainingParameters()

    if preset_name == "small":
        # Configuração pequena para teste rápido
        params.training_config.batch_size = 8
        params.training_config.embedding_dim = 256
        params.training_config.hidden_dim = 512
        params.training_config.num_layers = 4
        params.training_config.num_heads = 4
        params.training_config.max_epochs = 10

    elif preset_name == "medium":
        # Configuração média para desenvolvimento
        params.training_config.batch_size = 16
        params.training_config.embedding_dim = 384
        params.training_config.hidden_dim = 768
        params.training_config.num_layers = 6
        params.training_config.num_heads = 6
        params.training_config.max_epochs = 50

    elif preset_name == "large":
        # Configuração grande para produção
        params.training_config.batch_size = 32
        params.training_config.embedding_dim = 512
        params.training_config.hidden_dim = 1024
        params.training_config.num_layers = 8
        params.training_config.num_heads = 8
        params.training_config.max_epochs = 100

    elif preset_name == "spectral_focus":
        # Configuração focada em processamento espectral
        params.training_config.spectral_dim = 512
        params.training_config.fft_bins = 256
        params.spectral_config.n_mels = 128
        params.spectral_config.use_mfcc = True

    else:
        raise ValueError(f"Preset não encontrado: {preset_name}")

    return params


if __name__ == "__main__":
    # Exemplo de uso
    params = ΨCWSTrainingParameters()

    print("🔧 Parâmetros de Treinamento ΨCWS")
    print("=" * 50)

    all_params = params.get_all_parameters()

    for category, config in all_params.items():
        print(f"\n📋 {category.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Validação
    is_valid, errors = params.validate_parameters()
    if is_valid:
        print("\n✅ Todos os parâmetros são válidos!")
    else:
        print(f"\n❌ Erros encontrados: {errors}")

    # Hiperparâmetros de treinamento
    hyperparams = params.get_training_hyperparameters()
    print(f"\n🎯 Hiperparâmetros de Treinamento:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")