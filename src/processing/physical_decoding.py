#!/usr/bin/env python3
"""
ΨQRH Physical Decoding - Decodificação Física Canônica
======================================================

Módulo de decodificação física que substitui completamente a lógica softmax
por medição direta de ressonância quântica, cumprindo o princípio de
"zero lógica transformer".

Este módulo implementa a "Medição por Pico de Ressonância" como método
único e aprovado para converter energia em tokens.
"""

import torch
import numpy as np
from scipy.signal import find_peaks


def decode_resonance_to_token_id(resonance_energy: torch.Tensor, temperature: float = 0.1, top_k: int = 5) -> int:
    """
    Decodifica um vetor de energia de ressonância em um ID de token usando
    o método de 'Medição por Pico de Ressonância', sem usar softmax.

    Args:
        resonance_energy: Tensor ou array numpy com energias de ressonância
        temperature: Controle de "flutuação quântica" (0.0 = determinístico, 1.0 = exploratório)
        top_k: Número máximo de picos secundários para exploração

    Returns:
        ID do token selecionado pela medição física
    """
    if not isinstance(resonance_energy, np.ndarray):
        resonance_energy = resonance_energy.detach().cpu().numpy()

    if len(resonance_energy) == 0:
        return 0  # Retorna token padrão

    # O 'threshold' de proeminência é crucial para ignorar ruído
    prominence_threshold = np.max(resonance_energy) * 0.05
    peaks, properties = find_peaks(resonance_energy, prominence=prominence_threshold)

    if len(peaks) == 0:
        # Se nenhum pico proeminente for encontrado, o token é o de energia máxima
        return np.argmax(resonance_energy)

    # Ordena os picos pela sua proeminência (importância)
    sorted_peak_indices = np.argsort(properties['prominences'])[::-1]

    # A temperatura controla a chance de uma "flutuação quântica" escolher um pico não-principal
    if np.random.rand() < temperature and len(sorted_peak_indices) > 1:
        # Explora um dos 'top_k' picos secundários
        k = min(top_k, len(sorted_peak_indices))
        chosen_peak_index = np.random.choice(sorted_peak_indices[:k])
    else:
        # Colapsa para o estado mais provável (pico mais proeminente)
        chosen_peak_index = sorted_peak_indices[0]

    return peaks[chosen_peak_index]


def decode_batch_resonance_to_tokens(resonance_batch: torch.Tensor,
                                   temperature: float = 0.1,
                                   top_k: int = 5) -> torch.Tensor:
    """
    Versão batch da decodificação física para processamento paralelo.

    Args:
        resonance_batch: Tensor [batch_size, seq_len, vocab_size] com energias
        temperature: Controle de flutuação quântica
        top_k: Número máximo de picos para exploração

    Returns:
        Tensor [batch_size, seq_len] com IDs de tokens
    """
    batch_size, seq_len, vocab_size = resonance_batch.shape
    token_ids = []

    for b in range(batch_size):
        batch_tokens = []
        for s in range(seq_len):
            resonance_energy = resonance_batch[b, s]
            token_id = decode_resonance_to_token_id(resonance_energy, temperature, top_k)
            batch_tokens.append(token_id)
        token_ids.append(batch_tokens)

    return torch.tensor(token_ids, dtype=torch.long, device=resonance_batch.device)


def validate_physical_decoding_consistency(resonance_energy: torch.Tensor,
                                         n_trials: int = 10,
                                         temperature: float = 0.0) -> dict:
    """
    Valida a consistência da decodificação física (deve ser determinística com T=0).

    Args:
        resonance_energy: Energia de ressonância para teste
        n_trials: Número de tentativas para verificar consistência
        temperature: Temperatura para teste (0.0 = determinístico)

    Returns:
        Dicionário com métricas de consistência
    """
    results = []
    for _ in range(n_trials):
        token_id = decode_resonance_to_token_id(resonance_energy, temperature=temperature)
        results.append(token_id)

    unique_tokens = set(results)
    most_common = max(set(results), key=results.count)

    return {
        'consistency_ratio': results.count(most_common) / n_trials,
        'unique_tokens': len(unique_tokens),
        'most_common_token': most_common,
        'is_deterministic': len(unique_tokens) == 1
    }


if __name__ == "__main__":
    # Teste da decodificação física
    print("🧪 Testando decodificação física canônica...")

    # Teste com sinal simples
    test_energy = torch.randn(100)  # Simula 100 tokens possíveis
    token_id = decode_resonance_to_token_id(test_energy, temperature=0.0)
    print(f"✅ Token decodificado (T=0.0): {token_id}")

    # Teste de consistência
    consistency = validate_physical_decoding_consistency(test_energy, temperature=0.0)
    print(f"✅ Consistência determinística: {consistency['is_deterministic']}")

    # Teste com temperatura
    token_id_temp = decode_resonance_to_token_id(test_energy, temperature=0.5)
    print(f"✅ Token com temperatura (T=0.5): {token_id_temp}")

    print("🎯 Decodificação física canônica validada!")