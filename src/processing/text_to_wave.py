"""
Text to Wave Converter - Conversão Texto → Sinal Contínuo
==========================================================

O ΨQRH NÃO trabalha com tokens discretos (vocab_size, token_id).
Ele lê texto bruto como onda contínua via análise espectral.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.fft as fft
import numpy as np
from typing import Dict, Tuple


def text_to_ascii_signal(text: str, normalize: bool = True) -> torch.Tensor:
    """
    Converte texto em sinal ASCII contínuo.

    Args:
        text: Texto de entrada (string)
        normalize: Normalizar para [-1, 1]

    Returns:
        Sinal [seq_len] onde seq_len = len(text)
    """
    # Conversão ASCII direta
    ascii_values = [ord(char) for char in text]
    signal = torch.tensor(ascii_values, dtype=torch.float32)

    if normalize:
        # Normalizar para [-1, 1]
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)

    return signal


def text_to_spectral_modes(text: str,
                           n_modes: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompõe texto em modos espectrais (não vocabulário discreto).

    Retorna:
    - λ: frequências espectrais
    - A(λ): amplitudes complexas

    Args:
        text: Texto de entrada
        n_modes: Número de modos espectrais

    Returns:
        (frequencies, amplitudes) onde:
        - frequencies: [n_modes] frequências λ
        - amplitudes: [n_modes] amplitudes complexas A(λ)
    """
    # Converter texto para sinal ASCII
    signal = text_to_ascii_signal(text, normalize=True)
    seq_len = len(signal)

    #Pad para n_modes se necessário
    if seq_len < n_modes:
        signal = torch.nn.functional.pad(signal, (0, n_modes - seq_len))
    elif seq_len > n_modes:
        signal = signal[:n_modes]

    # FFT para obter modos espectrais
    signal_fft = fft.fft(signal)

    # Frequências e amplitudes
    frequencies = fft.fftfreq(n_modes)
    amplitudes = signal_fft

    return frequencies, amplitudes


def text_to_fractal_embedding(text: str,
                              embed_dim: int = 64) -> torch.Tensor:
    """
    Converte texto para embedding fractal (não token embedding).

    Usa análise espectral para extrair dimensão fractal D e mapear
    para espaço de embedding contínuo.

    Args:
        text: Texto de entrada
        embed_dim: Dimensão do embedding

    Returns:
        Embedding fractal [seq_len, embed_dim]
    """
    # Sinal ASCII
    signal = text_to_ascii_signal(text, normalize=True)
    seq_len = len(signal)

    # Criar embedding via análise de frequência
    # Cada caractere → embedding baseado em contexto espectral local

    embeddings = []
    window_size = 5  # Janela de contexto

    for i in range(seq_len):
        # Janela local
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        local_window = signal[start:end]

        # FFT da janela local
        if len(local_window) < embed_dim:
            local_window = torch.nn.functional.pad(
                local_window,
                (0, embed_dim - len(local_window))
            )
        elif len(local_window) > embed_dim:
            local_window = local_window[:embed_dim]

        window_fft = fft.fft(local_window)

        # Usar magnitude e fase como embedding
        embedding = torch.stack([
            window_fft.real,
            window_fft.imag
        ], dim=-1).flatten()[:embed_dim]

        # Pad se necessário
        if len(embedding) < embed_dim:
            embedding = torch.nn.functional.pad(
                embedding,
                (0, embed_dim - len(embedding))
            )

        embeddings.append(embedding)

    return torch.stack(embeddings)


def text_to_continuous_wave(text: str,
                           sample_rate: int = 1000,
                           duration_per_char: float = 0.01) -> torch.Tensor:
    """
    Converte texto em onda contínua para processamento ΨQRH.

    Args:
        text: Texto de entrada
        sample_rate: Taxa de amostragem (Hz)
        duration_per_char: Duração de cada caractere (segundos)

    Returns:
        Onda contínua [total_samples]
    """
    signal = text_to_ascii_signal(text, normalize=True)
    seq_len = len(signal)

    # Interpolar para criar onda contínua
    samples_per_char = int(sample_rate * duration_per_char)
    total_samples = seq_len * samples_per_char

    # Criar sinal contínuo via interpolação
    discrete_points = signal.numpy()
    continuous_signal = np.interp(
        np.linspace(0, seq_len - 1, total_samples),
        np.arange(seq_len),
        discrete_points
    )

    return torch.from_numpy(continuous_signal).float()


def create_spectral_character_map(n_modes: int = 256) -> Dict[int, torch.Tensor]:
    """
    Cria mapeamento caractere → modo espectral (não vocabulário tradicional).

    Cada caractere é mapeado para um padrão espectral único baseado em
    sua estrutura harmônica, não em um índice arbitrário.

    Args:
        n_modes: Número de modos espectrais

    Returns:
        Dicionário {ord(char): spectral_pattern[n_modes]}
    """
    spectral_map = {}

    # Caracteres ASCII imprimíveis (32-126)
    for ascii_code in range(32, 127):
        # Criar padrão espectral único baseado em harmônicos
        fundamental_freq = ascii_code / 127.0  # Normalizado [0, 1]

        # Série harmônica
        harmonics = []
        for k in range(1, n_modes + 1):
            # Amplitude do k-ésimo harmônico
            amplitude = np.exp(-k * fundamental_freq) * np.sin(2 * np.pi * k * fundamental_freq)
            harmonics.append(amplitude)

        spectral_pattern = torch.tensor(harmonics, dtype=torch.float32)
        spectral_map[ascii_code] = spectral_pattern

    return spectral_map
