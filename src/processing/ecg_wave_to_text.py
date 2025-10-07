"""
ECG-like Wave to Text Converter - Processamento de Ondas como ECG
===================================================================

Implementa geração de texto via processamento de ondas similares a ECG:
- Converte estados quaterniônicos em padrões de onda
- Analisa características de ECG (P, QRS, T waves)
- Mapeia padrões de onda para caracteres semânticos

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple
from scipy.signal import find_peaks, peak_widths


def create_ecg_waveform_from_quaternion(psi: torch.Tensor,
                                       sample_rate: int = 256,
                                       duration: float = 1.0) -> torch.Tensor:
    """
    Converte estado quaterniônico em padrão de onda ECG-like.

    Args:
        psi: Estado quaterniônico [embed_dim, 4]
        sample_rate: Taxa de amostragem (Hz)
        duration: Duração do sinal (segundos)

    Returns:
        Sinal ECG-like [n_samples]
    """
    n_samples = int(sample_rate * duration)
    time = torch.linspace(0, duration, n_samples)

    # Extrair componentes quaterniônicas
    w, x, y, z = psi[:, 0], psi[:, 1], psi[:, 2], psi[:, 3]

    # Criar padrão ECG usando as componentes quaterniônicas
    ecg_signal = torch.zeros(n_samples)

    # Componente P (atrial depolarization)
    p_amplitude = torch.mean(torch.abs(w)).item()
    p_duration = 0.08 + 0.04 * torch.mean(torch.abs(x)).item()  # 80-120ms
    p_position = 0.15 + 0.1 * torch.mean(torch.abs(y)).item()

    # Componente QRS (ventricular depolarization) - mais complexa
    qrs_amplitude = torch.mean(torch.abs(x)).item() * 2.0
    qrs_duration = 0.06 + 0.04 * torch.mean(torch.abs(z)).item()  # 60-100ms
    qrs_position = 0.35 + 0.1 * torch.mean(torch.abs(w)).item()

    # Componente T (ventricular repolarization)
    t_amplitude = torch.mean(torch.abs(y)).item() * 1.5
    t_duration = 0.12 + 0.08 * torch.mean(torch.abs(z)).item()  # 120-200ms
    t_position = 0.55 + 0.1 * torch.mean(torch.abs(x)).item()

    # Gerar componentes ECG
    for i, t in enumerate(time):
        # Onda P
        if p_position <= t <= p_position + p_duration:
            ecg_signal[i] += p_amplitude * math.sin(math.pi * (t - p_position) / p_duration)

        # Complexo QRS
        if qrs_position <= t <= qrs_position + qrs_duration:
            # Forma característica do QRS (triângulo com picos)
            t_rel = (t - qrs_position) / qrs_duration
            if t_rel < 0.33:  # Onda Q
                ecg_signal[i] -= qrs_amplitude * 0.3 * math.sin(math.pi * t_rel * 3)
            elif t_rel < 0.66:  # Onda R
                ecg_signal[i] += qrs_amplitude * math.sin(math.pi * (t_rel - 0.33) * 3)
            else:  # Onda S
                ecg_signal[i] -= qrs_amplitude * 0.4 * math.sin(math.pi * (t_rel - 0.66) * 3)

        # Onda T
        if t_position <= t <= t_position + t_duration:
            ecg_signal[i] += t_amplitude * math.sin(math.pi * (t - t_position) / t_duration)

    # Adicionar linha de base com pequeno ruído
    baseline_noise = torch.randn(n_samples) * 0.05
    ecg_signal += baseline_noise

    return ecg_signal


def extract_ecg_features(ecg_signal: torch.Tensor) -> Dict[str, float]:
    """
    Extrai características de ECG do sinal gerado.

    Args:
        ecg_signal: Sinal ECG-like [n_samples]

    Returns:
        Dicionário com características extraídas
    """
    signal_np = ecg_signal.numpy()

    # Encontrar picos principais
    peaks, properties = find_peaks(signal_np, height=0.1, distance=50)

    features = {
        'num_peaks': len(peaks),
        'max_amplitude': float(torch.max(ecg_signal)),
        'min_amplitude': float(torch.min(ecg_signal)),
        'mean_amplitude': float(torch.mean(ecg_signal)),
        'std_amplitude': float(torch.std(ecg_signal)),
        'signal_energy': float(torch.sum(ecg_signal ** 2)),
    }

    # Características baseadas em picos
    if len(peaks) > 0:
        peak_heights = properties['peak_heights']
        features.update({
            'avg_peak_height': float(np.mean(peak_heights)),
            'max_peak_height': float(np.max(peak_heights)),
            'peak_variability': float(np.std(peak_heights)),
        })

        # Calcular larguras dos picos
        widths, _, _, _ = peak_widths(signal_np, peaks, rel_height=0.5)
        features.update({
            'avg_peak_width': float(np.mean(widths)),
            'peak_width_variability': float(np.std(widths)),
        })

    # Análise de frequência
    fft_signal = torch.fft.fft(ecg_signal)
    fft_magnitude = torch.abs(fft_signal)
    dominant_freq = torch.argmax(fft_magnitude[1:len(fft_signal)//2]).item() + 1

    features.update({
        'dominant_frequency': dominant_freq,
        'spectral_entropy': float(-torch.sum(fft_magnitude * torch.log(fft_magnitude + 1e-8))),
    })

    return features


def map_ecg_features_to_character(ecg_features: Dict[str, float],
                                 character_map: Dict[str, List[float]]) -> str:
    """
    Mapeia características de ECG para caracteres usando similaridade.

    Args:
        ecg_features: Características extraídas do ECG
        character_map: Mapeamento de caracteres para vetores de características

    Returns:
        Caractere mais similar
    """
    # Converter características em vetor
    feature_vector = np.array([
        ecg_features['max_amplitude'],
        ecg_features['mean_amplitude'],
        ecg_features['std_amplitude'],
        ecg_features['signal_energy'],
        ecg_features['num_peaks'],
        ecg_features['avg_peak_height'],
        ecg_features['peak_variability'],
        ecg_features['dominant_frequency'],
        ecg_features['spectral_entropy'],
    ])

    # Normalizar vetor
    feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)

    # Encontrar caractere mais similar
    best_char = ' '
    best_similarity = -1.0

    for char, char_vector in character_map.items():
        char_vector_np = np.array(char_vector)
        char_vector_np = char_vector_np / (np.linalg.norm(char_vector_np) + 1e-8)

        similarity = np.dot(feature_vector, char_vector_np)

        if similarity > best_similarity:
            best_similarity = similarity
            best_char = char

    return best_char


def create_ecg_character_map() -> Dict[str, List[float]]:
    """
    Cria mapeamento de caracteres para vetores de características ECG.

    Returns:
        Dicionário {caractere: vetor_de_características}
    """
    # Características baseadas em frequência linguística e padrões ECG
    character_map = {
        # Espaço e pontuação (baixa complexidade)
        ' ': [0.1, 0.1, 0.05, 0.1, 1, 0.1, 0.05, 2, 0.5],
        '.': [0.2, 0.15, 0.08, 0.2, 2, 0.15, 0.08, 3, 0.6],
        ',': [0.18, 0.14, 0.07, 0.18, 2, 0.14, 0.07, 3, 0.6],

        # Vogais (alta frequência, padrões mais complexos)
        'a': [0.8, 0.6, 0.3, 0.9, 4, 0.6, 0.25, 8, 1.2],
        'e': [0.9, 0.7, 0.35, 1.0, 5, 0.7, 0.3, 10, 1.4],
        'i': [0.7, 0.5, 0.25, 0.8, 3, 0.5, 0.2, 6, 1.0],
        'o': [0.85, 0.65, 0.32, 0.95, 4, 0.65, 0.28, 9, 1.3],
        'u': [0.75, 0.55, 0.28, 0.85, 4, 0.55, 0.22, 7, 1.1],

        # Consoantes comuns
        's': [0.6, 0.4, 0.2, 0.7, 3, 0.4, 0.18, 5, 0.9],
        'r': [0.65, 0.45, 0.22, 0.75, 3, 0.45, 0.2, 6, 1.0],
        'n': [0.55, 0.35, 0.18, 0.65, 2, 0.35, 0.15, 4, 0.8],
        'd': [0.7, 0.5, 0.25, 0.8, 3, 0.5, 0.2, 6, 1.0],
        'm': [0.8, 0.6, 0.3, 0.9, 4, 0.6, 0.25, 8, 1.2],
        't': [0.6, 0.4, 0.2, 0.7, 3, 0.4, 0.18, 5, 0.9],

        # Consoantes menos comuns
        'c': [0.5, 0.3, 0.15, 0.6, 2, 0.3, 0.12, 3, 0.7],
        'l': [0.45, 0.25, 0.12, 0.55, 2, 0.25, 0.1, 3, 0.6],
        'p': [0.55, 0.35, 0.18, 0.65, 2, 0.35, 0.15, 4, 0.8],
        'v': [0.4, 0.2, 0.1, 0.5, 1, 0.2, 0.08, 2, 0.5],
        'g': [0.35, 0.15, 0.08, 0.45, 1, 0.15, 0.06, 2, 0.4],

        # Caracteres raros
        'k': [0.2, 0.1, 0.05, 0.3, 1, 0.1, 0.04, 1, 0.3],
        'w': [0.25, 0.12, 0.06, 0.35, 1, 0.12, 0.05, 1, 0.4],
        'y': [0.15, 0.08, 0.04, 0.25, 1, 0.08, 0.03, 1, 0.2],
    }

    return character_map


def ecg_wave_to_character(psi: torch.Tensor,
                         character_map: Dict[str, List[float]] = None,
                         verbose: bool = True) -> str:
    """
    Converte estado quaterniônico em caractere via análise ECG.

    Args:
        psi: Estado quaterniônico [embed_dim, 4]
        character_map: Mapeamento de caracteres (opcional)
        verbose: Se deve imprimir logs detalhados

    Returns:
        Caractere gerado
    """
    if verbose:
        print(f"    🫀 [ecg_wave_to_character] Convertendo estado quaterniônico em padrão ECG...")

    if character_map is None:
        character_map = create_ecg_character_map()

    # Gerar sinal ECG-like
    ecg_signal = create_ecg_waveform_from_quaternion(psi)
    if verbose:
        print(f"    📈 [ecg_wave_to_character] Sinal ECG gerado: {len(ecg_signal)} amostras")

    # Extrair características
    ecg_features = extract_ecg_features(ecg_signal)
    if verbose:
        print(f"    🔍 [ecg_wave_to_character] Características extraídas: {len(ecg_features)} features")

    # Mapear para caractere
    character = map_ecg_features_to_character(ecg_features, character_map)
    if verbose:
        print(f"    ✅ [ecg_wave_to_character] Caractere selecionado: '{character}'")

    return character


def ecg_wave_to_text(psi_sequence: torch.Tensor,
                    character_map: Dict[str, List[float]] = None,
                    min_seq_len: int = 5) -> str:
    """
    Converte sequência de estados quaterniônicos em texto via análise ECG.

    Args:
        psi_sequence: Sequência de estados [seq_len, embed_dim, 4]
        character_map: Mapeamento de caracteres (opcional)
        min_seq_len: Comprimento mínimo da sequência

    Returns:
        Texto gerado
    """
    print(f"🫀 [ecg_wave_to_text] Iniciando decodificação ECG: seq_len={len(psi_sequence)}")

    if character_map is None:
        character_map = create_ecg_character_map()

    characters = []

    # Garantir comprimento mínimo
    target_seq_len = max(len(psi_sequence), min_seq_len)

    if len(psi_sequence) < target_seq_len:
        print(f"  🔄 [ecg_wave_to_text] Estendendo sequência de {len(psi_sequence)} para {target_seq_len} caracteres")
        extended_sequence = []
        for i in range(target_seq_len):
            base_idx = i % len(psi_sequence)
            base_psi = psi_sequence[base_idx]
            noise = torch.randn_like(base_psi) * 0.01
            extended_sequence.append(base_psi + noise)
        psi_sequence = torch.stack(extended_sequence)

    for i in range(len(psi_sequence)):
        psi = psi_sequence[i]
        # Show progress every 10 characters to reduce verbosity
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  📝 [ecg_wave_to_text] Processando caractere {i+1}/{len(psi_sequence)} via ECG")
        char = ecg_wave_to_character(psi, character_map, verbose=(i < 2))  # Only verbose for first 2 characters
        characters.append(char)

    result = ''.join(characters)
    print(f"🎯 [ecg_wave_to_text] Texto final gerado: '{result}'")
    return result


def analyze_ecg_signal_quality(ecg_signal: torch.Tensor) -> Dict[str, str]:
    """
    Analisa a qualidade do sinal ECG gerado.

    Args:
        ecg_signal: Sinal ECG-like

    Returns:
        Dicionário com métricas de qualidade
    """
    features = extract_ecg_features(ecg_signal)

    quality_metrics = {}

    # Avaliar amplitude
    if features['max_amplitude'] > 1.0:
        quality_metrics['amplitude'] = 'ALTA'
    elif features['max_amplitude'] > 0.5:
        quality_metrics['amplitude'] = 'MÉDIA'
    else:
        quality_metrics['amplitude'] = 'BAIXA'

    # Avaliar complexidade
    if features['num_peaks'] > 3:
        quality_metrics['complexity'] = 'ALTA'
    elif features['num_peaks'] > 1:
        quality_metrics['complexity'] = 'MÉDIA'
    else:
        quality_metrics['complexity'] = 'BAIXA'

    # Avaliar estabilidade
    if features['peak_variability'] < 0.1:
        quality_metrics['stability'] = 'ALTA'
    elif features['peak_variability'] < 0.2:
        quality_metrics['stability'] = 'MÉDIA'
    else:
        quality_metrics['stability'] = 'BAIXA'

    # Classificação geral
    good_metrics = sum([1 for metric in quality_metrics.values() if metric == 'ALTA'])
    if good_metrics >= 2:
        quality_metrics['overall'] = 'EXCELENTE'
    elif good_metrics >= 1:
        quality_metrics['overall'] = 'BOA'
    else:
        quality_metrics['overall'] = 'RUIM'

    return quality_metrics