"""
ΨQRH Pipeline - Processamento Físico Completo (SEM forward())
==============================================================

Pipeline autônomo de processamento espectral-quaterniônico:

    texto_bruto → sinal_contínuo → Ψ(x) → processamento_espectral → medida → texto_saída

NÃO é uma classe nn.Module. É uma função pura process(text: str) -> str.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
from typing import Dict, Optional
import sys
from pathlib import Path

# Adicionar src/ ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.spectral_harmonic_processor import (
    quaternion_from_signal,
    spectral_attention,
    harmonic_evolution,
    process_signal_stack,
    QuaternionMLP
)
from processing.text_to_wave import (
    text_to_fractal_embedding,
    create_spectral_character_map
)
from processing.wave_to_text import (
    wave_to_text,
    optical_probe
)


def process(text: str,
           n_layers: int = 6,
           alpha: float = 1.0,
           embed_dim: int = 64,
           temperature: float = 1.0,
           return_metrics: bool = False) -> str:
    """
    Pipeline físico RIGOROSO ΨQRH (doe.md Seções 2.9.1-2.9.4).

    RIGOROUS IMPLEMENTATION:
    - Uses MLP for quaternion mapping (NOT FFT) - doe.md 2.9.1
    - Uses Hamilton product for attention - doe.md 2.9.2
    - Uses unit quaternion R for evolution - doe.md 2.9.3

    Fluxo:
    1. Texto → Embedding Fractal (análise espectral)
    2. Embedding → Quaternions Ψ(x) via MLP (RIGOROUS)
    3. Processamento Espectral-Harmônico (n camadas)
    4. Colapso de Medida → Caracteres
    5. Caracteres → Texto

    Args:
        text: Texto de entrada (string bruta)
        n_layers: Número de camadas de processamento
        alpha: Parâmetro do filtro espectral F(k)
        embed_dim: Dimensão do embedding (deve ser divisível por 4)
        temperature: Temperatura de amostragem
        return_metrics: Retornar métricas de processamento

    Returns:
        Texto processado (string)
    """
    metrics = {}

    # 1. CONVERSÃO TEXTO → SINAL CONTÍNUO
    # NÃO usa tokenização - análise espectral direta
    signal = text_to_fractal_embedding(text, embed_dim=embed_dim)  # [seq_len, embed_dim]
    seq_len = signal.shape[0]
    metrics['input_length'] = seq_len

    # Adicionar dimensão de batch
    signal = signal.unsqueeze(0)  # [1, seq_len, embed_dim]

    # 2. CRIAR MLP PARA QUATERNION MAPPING (RIGOROUS - doe.md 2.9.1)
    mlp = QuaternionMLP(embed_dim=embed_dim)

    # 3. PROCESSAMENTO ESPECTRAL-HARMÔNICO RIGOROSO
    # Substitui transformer.forward() por pipeline físico
    processed_signal = process_signal_stack(signal, n_layers=n_layers, alpha=alpha, mlp=mlp)

    # Remover batch dimension
    processed_signal = processed_signal.squeeze(0)  # [seq_len, embed_dim]

    # 4. CONVERSÃO SINAL → QUATERNIONS (via MLP rigoroso)
    # Para medição quântica
    processed_signal_batch = processed_signal.unsqueeze(0)  # [1, seq_len, embed_dim]
    psi_sequence = quaternion_from_signal(processed_signal_batch, mlp=mlp)
    psi_sequence = psi_sequence.squeeze(0)  # [seq_len, embed_dim, 4]

    # 5. COLAPSO DE MEDIDA → TEXTO
    # NÃO usa softmax sobre vocab - usa projeção em modos espectrais
    spectral_map = create_spectral_character_map(n_modes=embed_dim)
    output_text = wave_to_text(psi_sequence, spectral_map, temperature=temperature, min_seq_len=10)

    metrics['output_length'] = len(output_text)
    metrics['rigorous_mode'] = 'MLP-based quaternion mapping (doe.md 2.9.1)'

    if return_metrics:
        return output_text, metrics
    else:
        return output_text


def process_with_consciousness(text: str,
                               n_layers: int = 6,
                               alpha: float = 1.0,
                               embed_dim: int = 64) -> Dict:
    """
    Processamento ΨQRH com análise de consciência via AUTOCALIBRAÇÃO.

    Usa o sistema de autocalibração já implementado:
    - FractalConsciousnessProcessor (autocalibragem de D)
    - ConsciousnessMetrics (cálculo de FCI)
    - Enhanced QRH Processor (alpha adaptativo)

    Args:
        text: Texto de entrada
        n_layers: Número de camadas
        alpha: Parâmetro espectral (inicial, será autocaliobrado)
        embed_dim: Dimensão do embedding

    Returns:
        Dicionário com texto processado e métricas de consciência
    """
    # Usar sistema de autocalibração completo via Enhanced QRH Processor
    try:
        from ..core.enhanced_qrh_processor import EnhancedQRHProcessor
        from ..conscience.fractal_consciousness_processor import FractalConsciousnessProcessor

        # Criar processador enhanced (com autocalibragem)
        processor = EnhancedQRHProcessor(device='cpu')

        # Processar texto primeiro
        result = processor.process_text(text)

        # Processar consciência SEPARADAMENTE usando dados de acoplamento
        from ..conscience.fractal_consciousness_processor import create_consciousness_processor
        consciousness_processor = create_consciousness_processor(embedding_dim=64, device='cpu')

        # Extrair dados de acoplamento para processamento de consciência
        spectral_energy = result['consciousness_coupling']['spectral_energy']
        quaternion_phase = result['consciousness_coupling']['quaternion_phase']

        # Processar consciência fractal usando forward()
        # Criar tensor de entrada dummy para forward()
        dummy_input = torch.randn(1, 64, 64)  # [batch, seq_len, embed_dim]

        consciousness_results = consciousness_processor(
            dummy_input,
            spectral_energy=spectral_energy,
            quaternion_phase=quaternion_phase
        )

        # Adicionar resultados de consciência ao resultado
        result['consciousness_results'] = consciousness_results

        # EXTRA: Gerar texto real via wave_to_text se FCI for alto o suficiente
        generated_text = text  # fallback

        # Aplicar bootstrap cognitivo se FCI estiver baixo
        # Use lowercase 'fci' key for consistency across the system
        current_fci = consciousness_results.get('fci', consciousness_results.get('FCI', 0)) if consciousness_results else 0
        if consciousness_results and current_fci < 0.25:
            try:
                from .consciousness_bootstrapper import create_consciousness_bootstrapper

                print(f"🧠 [psiqrh_pipeline] FCI baixo ({current_fci:.3f}), aplicando bootstrap cognitivo...")

                # Criar bootstrapper com parâmetros SUPER agressivos
                bootstrapper = create_consciousness_bootstrapper(
                    chaos_strength=0.8,  # MUITO mais agressivo
                    logistic_r=3.99,
                    min_fci_threshold=0.25,
                    max_boost_iterations=10  # Mais iterações
                )

                # Aplicar bootstrap
                if 'qrh_output' in result:
                    print(f"🔍 [psiqrh_pipeline] DEBUG - qrh_output encontrado, shape: {result['qrh_output'].shape}")
                    psi_sequence = result['qrh_output']  # [batch, seq_len, embed_dim, 4]
                    print(f"🔍 [psiqrh_pipeline] DEBUG - psi_sequence shape: {psi_sequence.shape}")
                    print(f"🔍 [psiqrh_pipeline] DEBUG - consciousness_results keys: {consciousness_results.keys()}")

                    psi_boosted, consciousness_results = bootstrapper.apply_bootstrap(
                        psi_sequence.squeeze(0),  # Remove batch dimension
                        consciousness_results,
                        consciousness_processor
                    )

                    # Use lowercase 'fci' key for consistency across the system
                    current_fci = consciousness_results.get('fci', consciousness_results.get('FCI', 0))
                    print(f"✅ [psiqrh_pipeline] Bootstrap aplicado: FCI = {current_fci:.3f}")
                else:
                    print(f"⚠️  [psiqrh_pipeline] qrh_output não encontrado no resultado")
            except Exception as e:
                print(f"⚠️  [psiqrh_pipeline] Erro no bootstrap cognitivo: {e}")

        # Gerar texto se FCI for alto o suficiente
        # Use lowercase 'fci' key for consistency across the system
        current_fci = consciousness_results.get('fci', consciousness_results.get('FCI', 0)) if consciousness_results else 0
        if consciousness_results and current_fci > 0.25:
            try:
                from .wave_to_text import wave_to_text
                from .text_to_wave import create_spectral_character_map

                # Converter estado quântico processado para texto
                # Usar qrh_output do EnhancedQRHProcessor
                if 'qrh_output' in result:
                    psi_sequence = result['qrh_output']  # [batch, seq_len, embed_dim, 4]
                    print(f"🔍 [psiqrh_pipeline] qrh_output shape: {psi_sequence.shape}")
                    # Ensure proper shape for wave_to_text: [seq_len, embed_dim, 4]
                    if psi_sequence.dim() == 4:
                        psi_sequence = psi_sequence.squeeze(0)  # Remove batch dimension
                    print(f"🔍 [psiqrh_pipeline] psi_sequence shape after squeeze: {psi_sequence.shape}")
                    spectral_map = create_spectral_character_map(n_modes=psi_sequence.shape[-2])
                    generated_text = wave_to_text(psi_sequence, spectral_map, min_seq_len=10)
                    print(f"🎯 [psiqrh_pipeline] Texto gerado via wave_to_text: '{generated_text}'")
            except Exception as e:
                print(f"⚠️  [psiqrh_pipeline] Erro na geração de texto: {e}")
                generated_text = result.get('text_analysis', text)
        else:
            # FCI ainda baixo - usar análise espectral
            generated_text = result.get('text_analysis', text)
            print(f"ℹ️  [psiqrh_pipeline] FCI ainda baixo ({current_fci:.3f}), usando análise espectral")

        # Extrair métricas autocalibradas
        if consciousness_results:
            # Get FCI from multiple possible locations in consciousness results
            fci_value = None
            if 'fci' in consciousness_results:
                fci_value = consciousness_results['fci']
            elif 'FCI' in consciousness_results:
                fci_value = consciousness_results['FCI']
            elif 'fci_evolution' in consciousness_results and len(consciousness_results['fci_evolution']) > 0:
                fci_value = consciousness_results['fci_evolution'][-1]
                if isinstance(fci_value, torch.Tensor):
                    fci_value = fci_value.item()

            metrics = {
                'fractal_dimension': consciousness_results.get('fractal_dimension'),
                'fci': fci_value,
                'consciousness_state': consciousness_results.get('consciousness_state', {}).get('name'),
                'alpha_adapted': consciousness_results.get('alpha_adapted'),
                'autocalibracao': 'ATIVADA'
            }
        else:
            metrics = {
                'fractal_dimension': None,
                'fci': None,
                'autocalibracao': 'ERRO',
                'error': 'consciousness_results não disponível'
            }

        return {
            'output': generated_text,
            'metrics': metrics,
            'full_result': result
        }

    except Exception as e:
        # Se enhanced processor falhar, retornar processamento básico
        output_text, basic_metrics = process(
            text, n_layers=n_layers, alpha=alpha,
            embed_dim=embed_dim, return_metrics=True
        )

        return {
            'output': output_text,
            'metrics': {
                **basic_metrics,
                'autocalibracao': 'DESATIVADA',
                'error': str(e)
            }
        }


def batch_process(texts: list,
                 n_layers: int = 6,
                 alpha: float = 1.0,
                 embed_dim: int = 64) -> list:
    """
    Processamento em lote de múltiplos textos.

    Args:
        texts: Lista de strings
        n_layers: Número de camadas
        alpha: Parâmetro espectral
        embed_dim: Dimensão do embedding

    Returns:
        Lista de textos processados
    """
    outputs = []

    for text in texts:
        output = process(text, n_layers=n_layers, alpha=alpha, embed_dim=embed_dim)
        outputs.append(output)

    return outputs


def interactive_process(n_layers: int = 6,
                       alpha: float = 1.0,
                       embed_dim: int = 64):
    """
    Modo interativo para processamento ΨQRH.

    Loop:
    1. Usuário digita texto
    2. Sistema processa via pipeline físico
    3. Exibe resultado e métricas
    4. Repete até 'quit'
    """
    print("=" * 60)
    print("ΨQRH Interactive Pipeline")
    print("=" * 60)
    print(f"Configuração: {n_layers} camadas, α={alpha}, dim={embed_dim}")
    print("Digite 'quit' para sair\n")

    while True:
        try:
            text = input(">>> ")

            if text.lower() in ['quit', 'exit', 'q']:
                print("Encerrando...")
                break

            if not text.strip():
                continue

            # Processar
            result = process_with_consciousness(
                text, n_layers=n_layers,
                alpha=alpha, embed_dim=embed_dim
            )

            # Exibir resultado
            print(f"\nSaída: {result['output']}")
            print(f"Métricas:")
            for key, value in result['metrics'].items():
                print(f"  {key}: {value}")
            print()

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
            break
        except Exception as e:
            print(f"Erro: {e}")
            continue


if __name__ == "__main__":
    # Teste básico
    test_text = "Hello PSIQRH"
    print(f"Input: {test_text}")

    output = process(test_text, n_layers=3, embed_dim=64)
    print(f"Output: {output}")

    # Modo interativo
    # interactive_process(n_layers=3, alpha=1.0, embed_dim=64)
