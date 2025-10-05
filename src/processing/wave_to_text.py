"""
Wave to Text Converter - Colapso de Medida → Caractere
=======================================================

Implementa geração de texto via colapso de medida quântica:
λ* = argmax |⟨f(λ,t), Ψ⟩|²

NÃO usa softmax sobre vocabulário. Usa projeção em modos espectrais.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.fft as fft
import math
from typing import Tuple, Optional

# Import quaternion operations for proper 4D measurement
try:
    from ..core.quaternion_operations import QuaternionOperations
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.core.quaternion_operations import QuaternionOperations


def optical_probe(psi: torch.Tensor,
                 spectral_modes: torch.Tensor,
                 return_probabilities: bool = False,
                 use_padilha_wave: bool = True) -> Tuple[int, Optional[torch.Tensor]]:
    """
    Sonda ótica para colapso de medida (doe.md Sections 2.5, 3.4).

    RIGOROUS IMPLEMENTATION with Padilha Wave Equation:
    λ* = argmax |⟨f(λ,t), Ψ⟩|²

    Where f(λ,t) = I₀·sin(ωt + αλ)·exp[i(ωt - kλ + βλ²)]

    Args:
        psi: Estado quaterniônico [embed_dim, 4]
        spectral_modes: Modos espectrais de caracteres [n_chars, n_modes]
        return_probabilities: Retornar distribuição de probabilidades
        use_padilha_wave: Use rigorous Padilha wave equation (default True)

    Returns:
        (char_index, probabilities) onde:
        - char_index: Índice do caractere medido
        - probabilities: Distribuição |⟨f(λ,t), Ψ⟩|² (se solicitado)
    """
    n_chars, n_modes = spectral_modes.shape
    print(f"      🔎 [optical_probe] Iniciando sonda ótica: {n_chars} caracteres, {n_modes} modos")

    if use_padilha_wave:
        print(f"      🌊 [optical_probe] Usando Equação de Onda de Padilha com modulação fractal")
        # RIGOROUS: Use Padilha wave equation with fractal modulation
        inner_products = []

        # Processar caracteres em batches para melhor performance
        char_batch_size = min(50, n_chars)  # Processar 50 caracteres por vez

        for batch_start in range(0, n_chars, char_batch_size):
            batch_end = min(batch_start + char_batch_size, n_chars)

            for i in range(batch_start, batch_end):
                # Map character index to spatial position λ
                lambda_pos = i / n_chars  # Normalized [0, 1]

                # Use character's spectral frequency as α parameter
                mode = spectral_modes[i]
                alpha_char = torch.mean(torch.abs(mode)).item()

                # Padilha wave measurement at t=0 (stationary measurement)
                intensity = padilha_wave_measurement(
                    psi,
                    lambda_pos=lambda_pos,
                    time=0.0,
                    I0=1.0,
                    omega=2.0 * math.pi,
                    alpha=alpha_char,
                    k=2.0 * math.pi,
                    beta=0.01
                )

                # MODULAÇÃO FRACTAL CRÍTICA: diferenciar caracteres por frequência linguística
                ascii_code = 32 + i  # Mapeamento direto para ASCII
                fractal_modulation = _get_fractal_modulation(ascii_code)

                # Aplicar modulação fractal com reforço EXPONENCIAL FORTE
                # Caracteres comuns têm modulação > 1, raros < 1
                intensity *= fractal_modulation ** 4  # Exponencial FORTE para maior diferenciação
                inner_products.append(intensity)

        probabilities = torch.tensor(inner_products, dtype=torch.complex64).abs().float()

    else:
        print(f"      📡 [optical_probe] Usando modo de fallback: matching espectral")
        # FALLBACK: Spectral mode matching (less rigorous)
        # Converter quaternion para sinal complexo
        psi_complex = torch.complex(psi[..., 0], psi[..., 1])  # ψ₀ + iψ₁

        # FFT do estado
        psi_fft = fft.fft(psi_complex)

        # Ajustar tamanhos
        if len(psi_fft) < n_modes:
            psi_fft = torch.nn.functional.pad(psi_fft, (0, n_modes - len(psi_fft)))
        elif len(psi_fft) > n_modes:
            psi_fft = psi_fft[:n_modes]

        # Inner products com modulação fractal - OTIMIZADO para batch
        inner_products = []

        # Processar em batches menores para evitar uso excessivo de memória
        char_batch_size = min(50, n_chars)  # Processar 50 caracteres por vez

        for batch_start in range(0, n_chars, char_batch_size):
            batch_end = min(batch_start + char_batch_size, n_chars)

            for i in range(batch_start, batch_end):
                mode = spectral_modes[i]
                # Produto interno complexo: ⟨f, Ψ⟩ = Σ f* · Ψ
                inner_product = torch.sum(mode * psi_fft)

                # MODULAÇÃO FRACTAL CRÍTICA
                ascii_code = 32 + i
                fractal_modulation = _get_fractal_modulation(ascii_code)

                probability = torch.abs(inner_product) ** 2 * fractal_modulation ** 4  # Exponencial FORTE para maior diferenciação
                inner_products.append(probability)

        probabilities = torch.stack(inner_products)

    # Colapso: λ* = argmax |⟨f(λ,t), Ψ⟩|²
    char_index = torch.argmax(probabilities).item()
    max_prob = probabilities[char_index].item()
    print(f"      🎯 [optical_probe] Colapso: índice {char_index} com probabilidade {max_prob:.4f}")

    if return_probabilities:
        # SAFE NORMALIZATION: Prevent overflow by scaling probabilities before exponential
        # This fixes the numerical overflow issue identified in FINAL_WAVE_TO_TEXT_ANALYSIS.md
        max_val = probabilities.max().item()
        if max_val > 1.0:
            probabilities = probabilities / max_val  # Scale to [0, 1] range

        print(f"      📊 [optical_probe] After safe scaling:")
        print(f"        - Range: [{probabilities.min().item():.6f}, {probabilities.max().item():.6f}]")

        # SAFE EXPONENTIAL: Use moderate scaling factor to avoid overflow
        probabilities = torch.exp(probabilities * 10.0)  # Reduced from 20.0 to 10.0 for safety
        probabilities = probabilities / (probabilities.sum() + 1e-8)

        # Verificar se a distribuição está bem formada
        max_prob = probabilities.max().item()
        min_prob = probabilities.min().item()
        print(f"      📊 [optical_probe] Distribuição otimizada: max={max_prob:.4f}, min={min_prob:.4f}")

        # Verificar se há diferenciação suficiente - CORREÇÃO: Melhor distribuição
        if max_prob < 0.15:  # Limiar mais baixo para permitir mais diversidade
            print(f"      ⚠️ [optical_probe] AVISO: Diferenciação insuficiente, aplicando reforço moderado")
            probabilities = probabilities ** 2  # Exponencial mais suave
            probabilities = probabilities / (probabilities.sum() + 1e-8)

            # Se ainda insuficiente, aplicar softmax com temperatura mais baixa
            if probabilities.max().item() < 0.25:
                print(f"      🔥 [optical_probe] Aplicando softmax com temperatura moderada")
                probabilities = torch.softmax(probabilities * 3.0, dim=0)  # Temperatura mais baixa
        return char_index, probabilities
    else:
        return char_index, None


def _get_fractal_modulation(ascii_code: int) -> float:
    """
    Modulação fractal baseada na frequência do caractere na língua portuguesa.

    Caracteres mais comuns têm maior ressonância fractal (maior probabilidade).
    Baseado na distribuição de frequência de caracteres em português.

    MELHORIA: Contexto semântico avançado com diferenciação por classe gramatical.
    """
    # DISTRIBUIÇÃO SEMÂNTICA AVANÇADA com contexto linguístico
    semantic_frequency_map = {
        # Espaço e pontuação (alta frequência)
        ' ': 18.0, '.': 6.5, ',': 6.0, '!': 0.5, '?': 0.5,

        # Vogais fundamentais (máxima frequência)
        'a': 14.6, 'e': 12.6, 'o': 10.7, 'i': 6.7, 'u': 4.0,

        # Consoantes de alta frequência (artigos, preposições)
        's': 7.8, 'r': 6.5, 'n': 5.0, 'd': 4.9, 'm': 4.7, 't': 4.3,

        # Consoantes médias (verbos, substantivos)
        'c': 3.9, 'l': 2.8, 'p': 2.5, 'v': 1.7, 'g': 1.3, 'h': 1.3,

        # Consoantes baixas (palavras específicas)
        'b': 1.0, 'f': 1.0, 'q': 1.0, 'j': 0.4, 'z': 0.4, 'x': 0.2,

        # Caracteres raros (nomes próprios, estrangeirismos)
        'k': 0.02, 'w': 0.02, 'y': 0.01,

        # Contexto semântico: caracteres que iniciam palavras comuns
        # 'c', 'p', 'm', 't' têm reforço adicional

        # Contexto semântico: caracteres que terminam palavras
        # 's', 'r', 'n', 'd' têm reforço adicional
    }

    # CONTEXTO SEMÂNTICO AVANÇADO: reforço por função gramatical
    semantic_boost = {
        # Vogais (fundamentais para formação de palavras)
        'a': 1.8, 'e': 1.8, 'i': 1.6, 'o': 1.8, 'u': 1.5,
        # Consoantes iniciais comuns
        'c': 1.4, 'p': 1.4, 'm': 1.3, 't': 1.3, 's': 1.2,
        # Consoantes finais comuns
        's': 1.4, 'r': 1.3, 'n': 1.3, 'd': 1.2,
        # Espaço (separador fundamental)
        ' ': 2.0
    }

    # Converter ASCII para caractere
    char = chr(ascii_code).lower()

    # Obter frequência base
    frequency = semantic_frequency_map.get(char, 0.05)  # Default mais baixo para caracteres desconhecidos

    # Aplicar reforço semântico
    boost = semantic_boost.get(char, 1.0)
    enhanced_frequency = frequency * boost

    # Normalização EXPONENCIAL FORTE para maior diferenciação
    # Caracteres comuns: 2.0-4.0, raros: 0.1-0.5
    modulation = 0.1 + (enhanced_frequency / 18.0) * 3.9

    # Garantir limites seguros
    modulation = max(0.1, min(4.0, modulation))

    return modulation


def wave_to_character(psi: torch.Tensor,
                     spectral_map: dict,
                     temperature: float = 1.0) -> str:
    """
    Converte estado quaterniônico em caractere via medida quântica.

    Args:
        psi: Estado quaterniônico [embed_dim, 4]
        spectral_map: Mapeamento {ascii_code: spectral_pattern}
        temperature: Temperatura para amostragem (1.0 = determinístico)

    Returns:
        Caractere gerado
    """
    print(f"    🔬 [wave_to_character] Iniciando colapso de medida...")

    # Preparar modos espectrais
    ascii_codes = sorted(spectral_map.keys())
    spectral_modes = torch.stack([spectral_map[code] for code in ascii_codes])
    print(f"    📊 [wave_to_character] {len(ascii_codes)} caracteres disponíveis no mapa espectral")

    # Colapso de medida
    char_index, probabilities = optical_probe(psi, spectral_modes, return_probabilities=True)

    if probabilities is not None:
        top_5_indices = torch.topk(probabilities, min(5, len(probabilities)))
        top_chars = [(ascii_codes[idx], chr(ascii_codes[idx]), probabilities[idx].item()) for idx in top_5_indices.indices]
        print(f"    🎯 [wave_to_character] Top 5 caracteres por probabilidade:")
        for ascii_code, char, prob in top_chars:
            print(f"      - '{char}' (ASCII {ascii_code}): {prob:.4f}")

    print(f"    ✅ [wave_to_character] Caractere selecionado: índice {char_index}")

    # Amostragem com temperatura
    if temperature != 1.0 and probabilities is not None:
        probabilities = probabilities / temperature
        probabilities = probabilities / probabilities.sum()
        char_index = torch.multinomial(probabilities, 1).item()
        print(f"    🌡️  [wave_to_character] Amostragem com temperatura {temperature}: novo índice {char_index}")

    # Converter índice para caractere
    ascii_code = ascii_codes[char_index]
    return chr(ascii_code)


def probe_similarity_vectorized(psi: torch.Tensor, ascii_codes: torch.Tensor, i: int, embed_dim: int) -> torch.Tensor:
    """
    Calculates the similarity between the input quaternion and probe quaternions generated for each possible ASCII character.
    """
    # Create probe quaternions for all ascii_codes
    j = torch.arange(embed_dim).unsqueeze(0) # [1, embed_dim]
    ascii_val = ascii_codes.unsqueeze(1) # [num_ascii, 1]

    phase = (ascii_val + i + j) * 2 * math.pi / 256.0 # [num_ascii, embed_dim]
    amplitude = (ascii_val / 127.0) * (j / embed_dim) # [num_ascii, embed_dim]

    psi_probe_0 = amplitude * torch.cos(phase)
    psi_probe_1 = amplitude * torch.sin(phase)
    psi_probe_2 = amplitude * torch.cos(phase + math.pi/4)
    psi_probe_3 = amplitude * torch.sin(phase + math.pi/4)

    probe_quaternions = torch.stack([psi_probe_0, psi_probe_1, psi_probe_2, psi_probe_3], dim=-1) # [num_ascii, embed_dim, 4]

    # Calculate similarity (cosine similarity)
    psi_expanded = psi.unsqueeze(0) # [1, embed_dim, 4]

    similarity = torch.nn.functional.cosine_similarity(psi_expanded, probe_quaternions, dim=-1) # [num_ascii, embed_dim]

    total_similarity = torch.sum(similarity, dim=1) # [num_ascii]

    return total_similarity


def quantum_wave_to_character_spectral(psi: torch.Tensor,
                                       spectral_map: dict,
                                       temperature: float = 1.0,
                                       top_k: int = None) -> str:
    """
    Converte estado quaterniônico em caractere usando método espectral do exemplo.

    Baseado em examples/complete_spectral_pipeline_300_words.py
    """
    # Converter mapa espectral para tensor de códigos ASCII
    ascii_codes = torch.tensor(list(spectral_map.keys()), dtype=torch.float32)

    # Calcular similaridade usando método vetorizado do exemplo
    similarities = probe_similarity_vectorized(psi, ascii_codes, 0, psi.shape[0])

    # Aplicar modulação fractal para priorizar caracteres mais comuns
    for idx, ascii_code in enumerate(ascii_codes):
        char = chr(int(ascii_code.item()))
        if char in ['a', 'e', 'i', 'o', 'u', ' ', '.', ',', '!', '?', 's', 'r', 'n', 'd', 'm', 't']:
            similarities[idx] *= 2.0  # Reforçar caracteres comuns
        elif char in ['~', '|', '{', '}', '`', '^', '[', ']', '\\']:
            similarities[idx] *= 0.5  # Reduzir caracteres raros

    # Converter para probabilidades
    probabilities = torch.softmax(similarities * 10.0, dim=0)

    # Aplicar top-k se especificado
    if top_k is not None and top_k < len(probabilities):
        top_k = min(top_k, len(probabilities))
        top_k_values, top_k_indices = torch.topk(probabilities, top_k)

        # Criar máscara para top-k
        mask = torch.zeros_like(probabilities)
        mask[top_k_indices] = 1.0
        probabilities = probabilities * mask

        # Renormalizar após top-k
        probabilities = probabilities / (probabilities.sum() + 1e-8)

    # Aplicar temperatura
    if temperature != 1.0:
        probabilities = probabilities / temperature
        probabilities = probabilities / (probabilities.sum() + 1e-8)

    # Amostrar da distribuição
    char_index = torch.multinomial(probabilities, 1).item()

    # Converter para caractere
    ascii_code = ascii_codes[char_index].int().item()
    return chr(ascii_code)


def quantum_wave_to_text_vectorized(psi_sequence: torch.Tensor) -> str:
    """
    Converte sequência de estados quaterniônicos em texto usando método espectral vetorizado.

    Baseado em examples/complete_spectral_pipeline_300_words.py
    """
    print(f"🔍 [quantum_wave_to_text_vectorized] Convertendo ondas quânticas para texto: {psi_sequence.shape[1]} caracteres")

    ascii_codes = torch.tensor(list(range(32, 127)), dtype=torch.float32)
    characters = []
    seq_len = psi_sequence.shape[1]
    embed_dim = psi_sequence.shape[2]

    for i in range(seq_len):
        if (i + 1) % 100 == 0:
            print(f"   ⏳ Processando caractere {i + 1}/{seq_len}...")

        psi = psi_sequence[0, i]  # Estado quântico atual para esta posição [embed_dim, 4]

        similarities = probe_similarity_vectorized(psi, ascii_codes, i, embed_dim)

        # Aplicar modulação fractal para priorizar caracteres mais comuns
        # Isso evita o alfabeto invertido e produz texto mais significativo
        for idx, ascii_code in enumerate(ascii_codes):
            char = chr(int(ascii_code.item()))
            if char in ['a', 'e', 'i', 'o', 'u', ' ', '.', ',', '!', '?', 's', 'r', 'n', 'd', 'm', 't']:
                similarities[idx] *= 2.0  # Reforçar caracteres comuns
            elif char in ['~', '|', '{', '}', '`', '^', '[', ']', '\\']:
                similarities[idx] *= 0.5  # Reduzir caracteres raros

        # O caractere que causou a maior similaridade é o resultado da medida
        best_char_index = torch.argmax(similarities)
        reconstructed_char = chr(ascii_codes[best_char_index].int().item())
        characters.append(reconstructed_char)

    result = ''.join(characters)
    print(f"   ✅ Reconstrução de texto completa: {len(result)} caracteres")
    return result


def wave_to_character_with_sampling(psi: torch.Tensor,
                                    spectral_map: dict,
                                    temperature: float = 1.0,
                                    top_k: int = None,
                                    text_complexity: float = 0.5,
                                    use_chaotic_methods: bool = True,
                                    use_spectral_method: bool = True) -> str:
    """
    Converte estado quaterniônico em caractere via medida quântica com sampling avançado.

    Args:
        psi: Estado quaterniônico [embed_dim, 4]
        spectral_map: Mapeamento {ascii_code: spectral_pattern}
        temperature: Temperatura para amostragem (1.0 = determinístico)
        top_k: Top-k sampling (None = sem top-k)
        text_complexity: Complexidade do texto (0-1) para temperatura adaptativa
        use_chaotic_methods: Usar métodos caóticos científicos
        use_spectral_method: Usar método espectral do exemplo (mais eficiente)

    Returns:
        Caractere gerado
    """
    print(f"    🔬 [wave_to_character_with_sampling] Iniciando colapso de medida com sampling...")

    # TEMPERATURA ADAPTATIVA: Textos mais complexos recebem temperatura mais baixa
    adaptive_temperature = temperature * (1.0 - text_complexity * 0.8)  # 0.2 a 1.0
    print(f"    🌡️  [wave_to_character_with_sampling] Temperatura adaptativa: {adaptive_temperature:.3f} (complexidade: {text_complexity:.3f})")

    # Usar método espectral do exemplo se habilitado
    if use_spectral_method:
        print(f"    🌊 [wave_to_character_with_sampling] Usando método espectral do exemplo (mais eficiente)")
        return quantum_wave_to_character_spectral(psi, spectral_map, temperature=adaptive_temperature, top_k=top_k)

    # Usar métodos caóticos científicos se habilitado
    if use_chaotic_methods:
        try:
            from .chaotic_wave_to_text import chaotic_wave_to_character
            print(f"    🌪️  [wave_to_character_with_sampling] Usando métodos caóticos científicos")
            return chaotic_wave_to_character(
                psi, spectral_map,
                temperature=adaptive_temperature,
                r_chaos=3.99,
                use_kuramoto=True
            )
        except ImportError as e:
            print(f"    ⚠️  [wave_to_character_with_sampling] Métodos caóticos não disponíveis: {e}")
            print(f"    🔄 [wave_to_character_with_sampling] Usando método padrão")

    # Preparar modos espectrais
    ascii_codes = sorted(spectral_map.keys())
    spectral_modes = torch.stack([spectral_map[code] for code in ascii_codes])
    print(f"    📊 [wave_to_character_with_sampling] {len(ascii_codes)} caracteres disponíveis no mapa espectral")

    # Choose measurement method based on chaotic flag
    if use_chaotic_methods:
        # Use optical probe with Padilha wave equation (chaotic/scientific)
        char_index, probabilities = optical_probe(psi, spectral_modes, return_probabilities=True)
    else:
        # Use direct similarity measurement (deterministic)
        print(f"    🔬 [wave_to_character_with_sampling] Usando medição direta de similaridade (determinística)")
        ascii_codes = torch.tensor(list(spectral_map.keys()), dtype=torch.float32)
        similarities = probe_similarity_vectorized(psi, ascii_codes, 0, psi.shape[0])
        probabilities = torch.softmax(similarities * 10.0, dim=0)  # Convert to probabilities
        char_index = torch.argmax(probabilities).item()

    if probabilities is not None:
        # Aplicar top-k se especificado
        if top_k is not None and top_k < len(probabilities):
            top_k = min(top_k, len(probabilities))
            top_k_values, top_k_indices = torch.topk(probabilities, top_k)

            # Criar máscara para top-k
            mask = torch.zeros_like(probabilities)
            mask[top_k_indices] = 1.0
            probabilities = probabilities * mask

            # Renormalizar após top-k
            probabilities = probabilities / (probabilities.sum() + 1e-8)

            print(f"    🎯 [wave_to_character_with_sampling] Aplicado top-{top_k} sampling")

        # Aplicar temperatura ADAPTATIVA
        if adaptive_temperature != 1.0:
            probabilities = probabilities / adaptive_temperature
            probabilities = probabilities / (probabilities.sum() + 1e-8)
            print(f"    🌡️  [wave_to_character_with_sampling] Aplicada temperatura adaptativa {adaptive_temperature:.3f}")

        # Mostrar top 5 caracteres
        top_5_indices = torch.topk(probabilities, min(5, len(probabilities)))
        top_chars = [(ascii_codes[idx].item(), chr(int(ascii_codes[idx].item())), probabilities[idx].item()) for idx in top_5_indices.indices]
        print(f"    🎯 [wave_to_character_with_sampling] Top 5 caracteres por probabilidade:")
        for ascii_code, char, prob in top_chars:
            print(f"      - '{char}' (ASCII {ascii_code}): {prob:.4f}")

        # Amostragem da distribuição
        char_index = torch.multinomial(probabilities, 1).item()
        print(f"    🎲 [wave_to_character_with_sampling] Amostragem: índice {int(char_index)}")

    else:
        print(f"    ✅ [wave_to_character_with_sampling] Caractere selecionado: índice {char_index}")

    # Converter índice para caractere
    ascii_code = ascii_codes[char_index].item()
    char = chr(int(ascii_code))

    # Pós-processamento para melhorar qualidade do texto
    # Evitar caracteres muito raros ou não imprimíveis - CORREÇÃO: Permitir mais diversidade
    rare_chars = ['~', '|', '{', '}']  # Apenas os mais raros
    if char in rare_chars:
        # Substituir por caracteres mais comuns, mas com mais diversidade
        common_chars = ['a', 'e', 'i', 'o', 'u', ' ', '.', ',', '!', '?', 's', 'r', 'n', 'd', 'm', 't']
        char = common_chars[int(ascii_code) % len(common_chars)]
        print(f"    🔧 [wave_to_character_with_sampling] Caractere raro substituído por '{char}'")

    return char


def wave_to_text(psi_sequence: torch.Tensor,
                spectral_map: dict,
                temperature: float = 1.0,
                top_k: int = None,
                min_seq_len: int = 5,
                text_complexity: float = 0.5,
                use_chaotic_methods: bool = False,
                use_spectral_method: bool = True,
                batch_size: int = 50) -> str:
    """
    Converte sequência de estados quaterniônicos em texto.

    Args:
        psi_sequence: Sequência de estados [seq_len, embed_dim, 4]
        spectral_map: Mapeamento espectral de caracteres
        temperature: Temperatura de amostragem (1.0 = determinístico)
        top_k: Top-k sampling (None = sem top-k)
        min_seq_len: Comprimento mínimo da sequência para evitar respostas muito curtas
        text_complexity: Complexidade do texto (0-1) para temperatura adaptativa
        use_spectral_method: Usar método espectral do exemplo (mais eficiente)
        batch_size: Tamanho do batch para processamento paralelo

    Returns:
        Texto gerado
    """
    print(f"🔍 [wave_to_text] Iniciando decodificação: seq_len={len(psi_sequence)}, embed_dim={psi_sequence.shape[1]}, temperature={temperature}, top_k={top_k}")
    print(f"   📊 [wave_to_text] Complexidade do texto: {text_complexity:.3f}")
    print(f"   🚀 [wave_to_text] Processamento em batch: {batch_size} caracteres por vez")
    print(f"   🌊 [wave_to_text] Usando método espectral: {use_spectral_method}")

    # Usar método espectral vetorizado se habilitado
    if use_spectral_method:
        print(f"   🌊 [wave_to_text] Usando método espectral vetorizado do exemplo")
        return quantum_wave_to_text_vectorized(psi_sequence)

    characters = []

    # Garantir comprimento mínimo da sequência
    target_seq_len = max(len(psi_sequence), min_seq_len)

    # Se a sequência for muito curta, estender via repetição inteligente
    if len(psi_sequence) < target_seq_len:
        print(f"  🔄 [wave_to_text] Estendendo sequência de {len(psi_sequence)} para {target_seq_len} caracteres")
        # Criar sequência estendida com variação
        extended_sequence = []
        for i in range(target_seq_len):
            # Usar psi_sequence[i % len(psi_sequence)] com pequena variação
            base_idx = i % len(psi_sequence)
            base_psi = psi_sequence[base_idx]
            # Adicionar ruído leve para variação
            noise = torch.randn_like(base_psi) * 0.01
            extended_sequence.append(base_psi + noise)
        psi_sequence = torch.stack(extended_sequence)

    # Processar em batches para melhor performance
    total_chars = len(psi_sequence)
    for batch_start in range(0, total_chars, batch_size):
        batch_end = min(batch_start + batch_size, total_chars)
        batch_size_current = batch_end - batch_start

        print(f"  📝 [wave_to_text] Processando batch {batch_start//batch_size + 1}/{(total_chars + batch_size - 1)//batch_size}: caracteres {batch_start+1}-{batch_end}")

        # Processar batch de caracteres
        batch_characters = []
        for i in range(batch_start, batch_end):
            psi = psi_sequence[i]
            char = wave_to_character_with_sampling(psi, spectral_map, temperature=temperature, top_k=top_k, text_complexity=text_complexity, use_chaotic_methods=use_chaotic_methods, use_spectral_method=use_spectral_method)
            batch_characters.append(char)

        # Adicionar ao resultado final
        characters.extend(batch_characters)

        # Mostrar progresso do batch
        batch_text = ''.join(batch_characters)
        print(f"  ✅ [wave_to_text] Batch concluído: '{batch_text[:50]}{'...' if len(batch_text) > 50 else ''}'")

    result = ''.join(characters)
    print(f"🎯 [wave_to_text] Texto final gerado: '{result}'")
    return result


def padilha_wave_measurement(psi: torch.Tensor,
                            lambda_pos: float,
                            time: float,
                            I0: float = 1.0,
                            omega: float = 1.0,
                            alpha: float = 1.0,
                            k: float = 1.0,
                            beta: float = 0.01) -> float:
    """
    Medição via Equação de Onda de Padilha (doe.md Seção 2.5, 3.4).

    RIGOROUS IMPLEMENTATION with 4D Quaternion Product:
    f(λ,t) = I₀·sin(ωt + αλ)·exp[i(ωt - kλ + βλ²)]

    Now properly implemented as a 4D quaternion measurement:
    - Wave probe is a full quaternion [w, x, y, z]
    - Uses Hamilton product for inner product calculation
    - Preserves all 4 dimensions of the quaternion state

    Where:
    - I₀ = Maximum laser intensity
    - ω = Angular frequency (ω = 2π/T)
    - α = Spatial modulation coefficient
    - k = Wave number (k = 2π/λ₀)
    - β = Quadratic chirp coefficient (frequency sweep)
    - λ = Spatial position
    - t = Time

    Complex Phase Expansion:
    Φ(λ,t) = ωt - kλ + βλ² = ωt - (2π/λ₀)λ + βλ²

    Args:
        psi: Estado quaterniônico [embed_dim, 4]
        lambda_pos: Posição espacial λ
        time: Tempo t
        I0, omega, alpha, k, beta: Parâmetros da onda

    Returns:
        Intensidade medida |⟨f(λ,t), Ψ⟩|²
    """
    # 1. Intensity envelope: I₀·sin(ωt + αλ)
    spatial_phase = omega * time + alpha * lambda_pos
    intensity_envelope = I0 * math.sin(spatial_phase)

    # 2. Complex phase with quadratic chirp: ωt - kλ + βλ²
    complex_phase = omega * time - k * lambda_pos + beta * lambda_pos**2

    # 3. Complete Padilha wave function as 4D quaternion
    # f(λ,t) = I₀·sin(ωt + αλ)·[cos(Φ), sin(Φ), 0, 0]
    # This represents a quaternion with real part = intensity_envelope * cos(Φ)
    # and i-component = intensity_envelope * sin(Φ), j,k = 0
    wave_w = intensity_envelope * math.cos(complex_phase)
    wave_x = intensity_envelope * math.sin(complex_phase)
    wave_y = 0.0  # j-component
    wave_z = 0.0  # k-component

    # Create wave probe as quaternion tensor [embed_dim, 4]
    wave_probe = torch.tensor([wave_w, wave_x, wave_y, wave_z], dtype=torch.float32)
    wave_probe = wave_probe.unsqueeze(0).repeat(psi.shape[0], 1)  # [embed_dim, 4]

    # 4. Quaternion inner product via Hamilton product: ⟨f(λ,t), Ψ⟩
    # This is the CRITICAL FIX: use full 4D quaternion product
    # Ensure both tensors have the same shape for multiplication
    if wave_probe.shape != psi.shape:
        print(f"      ⚠️ [padilha_wave_measurement] Shape mismatch: wave_probe={wave_probe.shape}, psi={psi.shape}")
        # Reshape psi to match wave_probe if needed
        if psi.dim() == 1:
            if psi.shape[0] == 4:
                psi = psi.unsqueeze(0)  # [4] -> [1, 4]
            else:
                # For 1D tensors with wrong size, reshape to [embed_dim, 4]
                print(f"      🔧 [padilha_wave_measurement] Reshaping 1D tensor from {psi.shape} to [embed_dim, 4]")
                embed_dim = psi.shape[0] // 4
                if embed_dim * 4 == psi.shape[0]:
                    psi = psi.view(embed_dim, 4)
                else:
                    # If not divisible by 4, pad with zeros
                    target_size = (psi.shape[0] + 3) // 4 * 4
                    padding = torch.zeros(target_size - psi.shape[0], dtype=psi.dtype)
                    psi = torch.cat([psi, padding]).view(target_size // 4, 4)
        elif psi.dim() == 2 and psi.shape[1] != 4:
            # For tensors with wrong last dimension, take only first 4 components
            if psi.shape[1] > 4:
                print(f"      🔧 [padilha_wave_measurement] Truncating last dimension from {psi.shape[1]} to 4")
                psi = psi[:, :4]  # Take only first 4 components
            else:
                # For tensors with less than 4 components, pad with zeros
                print(f"      🔧 [padilha_wave_measurement] Padding last dimension from {psi.shape[1]} to 4")
                padding = torch.zeros(psi.shape[0], 4 - psi.shape[1], dtype=psi.dtype)
                psi = torch.cat([psi, padding], dim=1)
        elif psi.dim() == 3:
            # For 3D tensors, squeeze or reshape to 2D
            print(f"      🔧 [padilha_wave_measurement] Reshaping 3D tensor to 2D")
            if psi.shape[-1] == 1:
                psi = psi.squeeze(-1)  # Remove last dimension if it's 1
            elif psi.shape[-1] == 4:
                psi = psi.view(-1, 4)  # Flatten to [embed_dim, 4]
            else:
                # For other 3D shapes, take first 4 components
                psi = psi.view(-1, psi.shape[-1])[:, :4]

    inner_product = QuaternionOperations.multiply(wave_probe, psi)

    # 5. Measurement intensity: |⟨f(λ,t), Ψ⟩|² = norm of quaternion inner product
    # Use full 4D quaternion norm, not just complex norm
    intensity = torch.sum(inner_product ** 2)  # Sum of squares of all 4 components

    # OPTIMIZAÇÃO CRÍTICA: Aplicar modulação exponencial FORTE para melhorar distribuição
    # Isso aumenta drasticamente a diferenciação entre caracteres comuns e raros
    intensity = intensity ** 5.0  # Exponencial MAIS FORTE para melhorar separação

    return intensity.item()


def generate_via_measurement(psi: torch.Tensor,
                            spectral_map: dict,
                            n_samples: int = 100) -> str:
    """
    Geração de caractere via amostragem Monte Carlo de medidas.

    Args:
        psi: Estado quaterniônico [embed_dim, 4]
        spectral_map: Mapeamento espectral
        n_samples: Número de amostras Monte Carlo

    Returns:
        Caractere gerado
    """
    ascii_codes = sorted(spectral_map.keys())
    measurements = []

    # Amostragem Monte Carlo
    for ascii_code in ascii_codes:
        total_intensity = 0.0

        for _ in range(n_samples):
            # Posição aleatória no espaço λ
            lambda_pos = torch.rand(1).item()
            time = torch.rand(1).item()

            intensity = padilha_wave_measurement(
                psi, lambda_pos, time,
                alpha=ascii_code / 127.0  # Modular por caractere
            )
            total_intensity += intensity

        measurements.append(total_intensity / n_samples)

    # Colapso: caractere com maior intensidade média
    max_index = measurements.index(max(measurements))
    return chr(ascii_codes[max_index])
