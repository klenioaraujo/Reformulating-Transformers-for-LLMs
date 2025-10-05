"""
Teste Simples do Processamento ECG
===================================

Teste direto do processamento ECG sem dependências complexas.
"""

import torch
import numpy as np
from .ecg_wave_to_text import (
    create_ecg_waveform_from_quaternion,
    extract_ecg_features,
    create_ecg_character_map,
    map_ecg_features_to_character
)


def test_ecg_processing():
    """Testa o processamento ECG com dados simulados."""
    print("🧪 Testando processamento ECG...")

    # Criar estado quaterniônico simulado
    embed_dim = 64
    psi = torch.randn(embed_dim, 4)  # [embed_dim, 4] para quaterniões
    print(f"  📊 Estado quaterniônico criado: {psi.shape}")

    # Gerar sinal ECG
    ecg_signal = create_ecg_waveform_from_quaternion(psi)
    print(f"  📈 Sinal ECG gerado: {len(ecg_signal)} amostras")

    # Extrair características
    features = extract_ecg_features(ecg_signal)
    print(f"  🔍 Características extraídas: {len(features)} features")
    print(f"     - Picos: {features.get('num_peaks', 0)}")
    print(f"     - Amplitude máxima: {features.get('max_amplitude', 0):.3f}")
    print(f"     - Energia: {features.get('signal_energy', 0):.3f}")

    # Criar mapeamento de caracteres
    character_map = create_ecg_character_map()
    print(f"  🗺️  Mapeamento criado: {len(character_map)} caracteres")

    # Mapear para caractere
    character = map_ecg_features_to_character(features, character_map)
    print(f"  ✅ Caractere gerado: '{character}'")

    return {
        'character': character,
        'features': features,
        'signal_length': len(ecg_signal)
    }


def test_ecg_text_generation():
    """Testa geração de texto completo via ECG."""
    print("\n🧪 Testando geração de texto ECG...")

    # Criar sequência de estados quaterniônicos
    seq_len = 10
    embed_dim = 64
    psi_sequence = torch.randn(seq_len, embed_dim, 4)
    print(f"  📊 Sequência criada: {psi_sequence.shape}")

    # Gerar texto caractere por caractere
    character_map = create_ecg_character_map()
    characters = []

    for i in range(seq_len):
        psi = psi_sequence[i]
        ecg_signal = create_ecg_waveform_from_quaternion(psi)
        features = extract_ecg_features(ecg_signal)
        char = map_ecg_features_to_character(features, character_map)
        characters.append(char)
        print(f"  📝 Caractere {i+1}: '{char}'")

    result_text = ''.join(characters)
    print(f"  ✅ Texto gerado: '{result_text}'")

    return {
        'text': result_text,
        'sequence_length': seq_len,
        'characters': characters
    }


if __name__ == "__main__":
    # Teste básico
    result1 = test_ecg_processing()

    # Teste de texto
    result2 = test_ecg_text_generation()

    print(f"\n🎯 Resumo dos testes:")
    print(f"  - Caractere único: '{result1['character']}'")
    print(f"  - Texto completo: '{result2['text']}'")
    print(f"  - Comprimento do sinal: {result1['signal_length']} amostras")
    print(f"  - Sequência processada: {result2['sequence_length']} caracteres")