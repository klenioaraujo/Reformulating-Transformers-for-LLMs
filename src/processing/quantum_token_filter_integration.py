#!/usr/bin/env python3
"""
Quantum Token Filter Integration - Integração do filtro quântico no sistema principal

Este módulo fornece funções para integrar o filtro quântico de tokens
no sistema psiqrh.py principal sem modificar o código existente.
"""

import torch
from typing import Optional, List, Dict, Any


def integrate_quantum_filter(psiqrh_instance) -> bool:
    """
    Integra o filtro quântico em uma instância do PsiQRH.

    Args:
        psiqrh_instance: Instância do sistema PsiQRH

    Returns:
        bool: True se a integração foi bem-sucedida
    """
    try:
        # Verificar se o módulo de filtro quântico está disponível
        from src.processing.quantum_token_filter import create_quantum_token_filter

        # Verificar se temos representações quânticas válidas
        if not hasattr(psiqrh_instance, 'quantum_vocab_representations') or psiqrh_instance.quantum_vocab_representations is None:
            print("⚠️  Não foi possível integrar filtro quântico: representações quânticas não disponíveis")
            return False

        # Criar filtro quântico
        embed_dim = psiqrh_instance.quantum_vocab_representations.shape[1] if len(psiqrh_instance.quantum_vocab_representations.shape) > 1 else 256
        vocab_size = psiqrh_instance.quantum_vocab_representations.shape[0]

        psiqrh_instance.quantum_token_filter = create_quantum_token_filter(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            quantum_vocab_representations=psiqrh_instance.quantum_vocab_representations,
            device=psiqrh_instance.device
        )

        print("✅ Filtro quântico integrado com sucesso!")
        print(f"   📊 Embed dim: {embed_dim}")
        print(f"   📚 Vocab size: {vocab_size}")
        print(f"   🔧 Filtro: {psiqrh_instance.quantum_token_filter is not None}")

        return True

    except ImportError as e:
        print(f"⚠️  Módulo de filtro quântico não disponível: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro ao integrar filtro quântico: {e}")
        return False


def enhanced_quantum_text_generation(psiqrh_instance, psi_final_abstract: torch.Tensor, input_text: str) -> str:
    """
    Geração de texto quântico aprimorada com filtro quântico.

    Args:
        psiqrh_instance: Instância do sistema PsiQRH
        psi_final_abstract: Estado quântico final
        input_text: Texto de entrada

    Returns:
        str: Texto gerado
    """
    # Verificar se temos representações quânticas válidas
    if psi_final_abstract.numel() == 0 or not hasattr(psiqrh_instance, 'quantum_vocab_representations') or psiqrh_instance.quantum_vocab_representations is None:
        raise ValueError("Estado quântico final ou vocabulário quântico não disponível")

    try:
        # ========== OPERAÇÃO NO ESPAÇO DE HILBERT COM FILTRO QUÂNTICO ==========
        # Projetar estado final no espaço de palavras usando operadores de projeção
        # com filtro quântico para prevenir loops e melhorar qualidade

        # 1. Normalizar o estado quântico final
        psi_normalized = psi_final_abstract / torch.norm(psi_final_abstract)

        # 2. Calcular amplitudes de transição para cada palavra no vocabulário
        transition_amplitudes = []
        for word_idx in range(len(psiqrh_instance.quantum_vocab_representations)):
            word_state = psiqrh_instance.quantum_vocab_representations[word_idx]  # [embed_dim, 4]

            # Normalizar estado da palavra
            word_state_normalized = word_state / torch.norm(word_state)

            # Calcular amplitude de transição (produto interno no espaço de Hilbert)
            # <ψ_final|ψ_word> = amplitude de transição
            # Ajustar dimensões para compatibilidade
            psi_flat = psi_normalized.flatten()
            word_flat = word_state_normalized.flatten()

            # Verificar e ajustar dimensões se necessário
            min_dim = min(psi_flat.shape[0], word_flat.shape[0])
            if psi_flat.shape[0] != word_flat.shape[0]:
                # Ajustar para a dimensão menor
                psi_flat = psi_flat[:min_dim]
                word_flat = word_flat[:min_dim]

            amplitude = torch.vdot(psi_flat, word_flat)
            transition_amplitudes.append((amplitude.abs().item(), word_idx))

        # 3. Aplicar filtro quântico para seleção robusta
        if hasattr(psiqrh_instance, 'quantum_token_filter'):
            # Converter amplitudes em logits simulados
            vocab_size = len(psiqrh_instance.quantum_vocab_representations)
            logits = torch.zeros(vocab_size)
            for amplitude, word_idx in transition_amplitudes:
                if word_idx < vocab_size:
                    logits[word_idx] = amplitude

            # Usar filtro quântico para seleção
            previous_tokens = []  # Contexto vazio para primeira palavra
            prediction_result = psiqrh_instance.quantum_token_filter.predict_next_token(
                logits, previous_tokens, temperature=0.8, top_k=50
            )

            best_word_idx = prediction_result['selected_token']
            best_amplitude = prediction_result['selected_probability']

            print(f"      🎯 Palavra selecionada via filtro quântico: ID {best_word_idx} (probabilidade: {best_amplitude:.4f})")
        else:
            # Fallback: seleção por amplitude máxima
            transition_amplitudes.sort(reverse=True)
            best_word_idx = transition_amplitudes[0][1]
            best_amplitude = transition_amplitudes[0][0]

            print(f"      🎯 Palavra selecionada via espaço de Hilbert: ID {best_word_idx} (amplitude: {best_amplitude:.4f})")

        # 4. Mapear índice para palavra usando id_to_word
        if hasattr(psiqrh_instance, 'id_to_word') and psiqrh_instance.id_to_word:
            selected_word = psiqrh_instance.id_to_word.get(best_word_idx)
            if selected_word:
                print(f"      📝 Palavra decodificada: '{selected_word}'")
                return selected_word

        # 5. Se não encontrou, usar projeção contextual baseada no input
        if input_text:
            # Projetar palavras do input no espaço quântico e encontrar similaridade contextual
            input_words = input_text.lower().split()
            contextual_scores = []

            for word in input_words:
                if hasattr(psiqrh_instance, 'word_to_id') and word in psiqrh_instance.word_to_id:
                    word_id = psiqrh_instance.word_to_id[word]
                    word_state = psiqrh_instance.quantum_vocab_representations[word_id]
                    word_state_normalized = word_state / torch.norm(word_state)
                    # Ajustar dimensões para compatibilidade
                    psi_flat = psi_normalized.flatten()
                    word_flat = word_state_normalized.flatten()

                    # Verificar e ajustar dimensões se necessário
                    min_dim = min(psi_flat.shape[0], word_flat.shape[0])
                    if psi_flat.shape[0] != word_flat.shape[0]:
                        # Ajustar para a dimensão menor
                        psi_flat = psi_flat[:min_dim]
                        word_flat = word_flat[:min_dim]

                    contextual_amplitude = torch.vdot(psi_flat, word_flat)
                    contextual_scores.append((contextual_amplitude.abs().item(), word))

            if contextual_scores:
                contextual_scores.sort(reverse=True)
                contextual_word = contextual_scores[0][1]
                print(f"      🔄 Usando palavra contextual: '{contextual_word}'")
                return contextual_word

        # 6. Fallback final: usar primeira palavra do vocabulário
        if hasattr(psiqrh_instance, 'id_to_word') and psiqrh_instance.id_to_word:
            first_word = list(psiqrh_instance.id_to_word.values())[0]
            print(f"      ⚠️  Fallback para primeira palavra: '{first_word}'")
            return first_word

        # 7. Fallback extremo
        print(f"      ❌ Nenhuma palavra encontrada, usando 'the'")
        return "the"

    except Exception as e:
        print(f"      ❌ Erro na geração quântica aprimorada: {e}")
        # Fallback para geração básica
        return _basic_quantum_fallback(psiqrh_instance, psi_final_abstract, input_text)


def _basic_quantum_fallback(psiqrh_instance, psi_final_abstract: torch.Tensor, input_text: str) -> str:
    """Fallback básico para geração quântica"""
    try:
        # Implementação básica de fallback
        if hasattr(psiqrh_instance, 'id_to_word') and psiqrh_instance.id_to_word:
            # Usar primeira palavra disponível
            first_word = list(psiqrh_instance.id_to_word.values())[0]
            return first_word
        else:
            return "the"
    except:
        return "the"


def test_quantum_filter_integration():
    """Testa a integração do filtro quântico"""
    print("🧪 Testando integração do filtro quântico...")

    # Simular uma instância básica para teste
    class MockPsiQRH:
        def __init__(self):
            self.device = "cpu"
            # Criar representações quânticas simuladas
            self.quantum_vocab_representations = torch.randn(100, 256, 4)
            self.id_to_word = {i: f"word_{i}" for i in range(100)}
            self.word_to_id = {f"word_{i}": i for i in range(100)}

    # Criar instância mock
    mock_instance = MockPsiQRH()

    # Testar integração
    success = integrate_quantum_filter(mock_instance)

    if success:
        print("✅ Integração do filtro quântico testada com sucesso!")

        # Testar geração de texto
        psi_final = torch.randn(256)
        input_text = "test input"

        result = enhanced_quantum_text_generation(mock_instance, psi_final, input_text)
        print(f"📝 Texto gerado: '{result}'")
    else:
        print("❌ Falha na integração do filtro quântico")

    return success


if __name__ == "__main__":
    test_quantum_filter_integration()