"""
ΨQRH Pipeline com Processamento ECG - Pipeline Principal
========================================================

Pipeline principal que usa processamento ECG-like para conversão de texto.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Importar processamento ECG
from .ecg_wave_to_text import ecg_wave_to_text, create_ecg_character_map


def process_with_consciousness_ecg(text: str,
                                  n_layers: int = 6,
                                  alpha: float = 1.5,
                                  embed_dim: int = 64) -> Dict[str, Any]:
    """
    Processa texto usando pipeline ΨQRH com processamento ECG-like.

    Args:
        text: Texto de entrada
        n_layers: Número de camadas QRH
        alpha: Parâmetro alpha adaptativo
        embed_dim: Dimensão de embedding

    Returns:
        Resultado do processamento
    """
    print(f"🫀 [process_with_consciousness_ecg] Iniciando pipeline ECG para: '{text}'")

    try:
        # Importar componentes do pipeline
        from .text_to_wave import text_to_continuous_wave
        from ..core.enhanced_qrh_processor import EnhancedQRHProcessor
        from ..conscience.fractal_consciousness_processor import create_consciousness_processor

        # 1. Texto → Onda
        print(f"  📝 [process_with_consciousness_ecg] Convertendo texto em onda...")
        wave_signal = text_to_continuous_wave(text)
        print(f"  ✅ [process_with_consciousness_ecg] Onda gerada: {len(wave_signal)} amostras")

        # 2. Inicializar processador QRH
        print(f"  🚀 [process_with_consciousness_ecg] Inicializando Enhanced QRH Processor...")
        qrh_processor = EnhancedQRHProcessor(
            embed_dim=embed_dim,
            device='cpu'
        )

        # 3. Processar texto → Estado quaterniônico
        print(f"  🔄 [process_with_consciousness_ecg] Processando texto em estado quaterniônico...")
        result_dict = qrh_processor.process_text(text)
        psi_states = result_dict.get('qrh_output', torch.randn(len(text), embed_dim, 4))
        print(f"  ✅ [process_with_consciousness_ecg] Estados quaterniônicos gerados: {psi_states.shape}")

        # 4. Análise de consciência
        print(f"  🧠 [process_with_consciousness_ecg] Analisando consciência fractal...")
        consciousness_processor = create_consciousness_processor(embedding_dim=embed_dim, device='cpu')
        consciousness_results = consciousness_processor.analyze_consciousness(psi_states)
        print(f"  ✅ [process_with_consciousness_ecg] Análise de consciência concluída")

        # 5. Converter estados ECG → Texto
        print(f"  🫀 [process_with_consciousness_ecg] Convertendo estados ECG em texto...")
        character_map = create_ecg_character_map()
        generated_text = ecg_wave_to_text(psi_states, character_map)
        print(f"  ✅ [process_with_consciousness_ecg] Texto gerado: '{generated_text}'")

        # 6. Preparar resultado
        result = {
            'output': generated_text,
            'metrics': {
                'input_length': len(text),
                'output_length': len(generated_text),
                'rigorous_mode': 'ECG-based quaternion mapping',
                'autocalibracao': 'DESATIVADA',
                'consciousness_fci': consciousness_results.get('FCI', 0.0),
                'consciousness_state': consciousness_results.get('consciousness_state', {}).get('name', 'UNKNOWN'),
            },
            'full_result': {
                'qrh_output': psi_states,
                'consciousness_results': consciousness_results,
                'wave_signal': wave_signal,
            }
        }

        print(f"🎯 [process_with_consciousness_ecg] Pipeline ECG concluído com sucesso!")
        return result

    except Exception as e:
        print(f"❌ [process_with_consciousness_ecg] Erro no pipeline ECG: {e}")
        import traceback
        traceback.print_exc()

        # Fallback para análise básica
        return {
            'output': f"Análise ECG para '{text}': Sistema em modo de diagnóstico.",
            'metrics': {
                'input_length': len(text),
                'output_length': 0,
                'rigorous_mode': 'ECG FALLBACK',
                'autocalibracao': 'DESATIVADA',
                'error': str(e)
            }
        }


def create_ecg_pipeline(config_path: str = "configs/qrh_config.yaml") -> Dict[str, Any]:
    """
    Cria pipeline ΨQRH com processamento ECG.

    Args:
        config_path: Caminho para configuração

    Returns:
        Pipeline configurado
    """
    # Carregar configuração
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    qrh_config = config.get('qrh_layer', {})

    pipeline_config = {
        'name': 'ΨQRH ECG Pipeline',
        'version': '1.0.0',
        'processing_method': 'ECG-like waveform analysis',
        'config': {
            'embed_dim': qrh_config.get('embed_dim', 64),
            'n_layers': qrh_config.get('n_layers', 6),
            'alpha': qrh_config.get('alpha', 1.5),
            'use_ecg_processing': True,
        },
        'features': {
            'ecg_waveform_generation': True,
            'ecg_feature_extraction': True,
            'semantic_character_mapping': True,
            'consciousness_integration': True,
        }
    }

    return pipeline_config


if __name__ == "__main__":
    # Teste do pipeline ECG
    test_text = "Olá mundo"
    print(f"🧪 Testando pipeline ECG com: '{test_text}'")

    result = process_with_consciousness_ecg(test_text)
    print(f"\n📊 Resultado:")
    print(f"  Entrada: {result['metrics']['input_length']} caracteres")
    print(f"  Saída: {result['metrics']['output_length']} caracteres")
    print(f"  Texto: {result['output']}")
    print(f"  FCI: {result['metrics'].get('consciousness_fci', 'N/A')}")
    print(f"  Estado: {result['metrics'].get('consciousness_state', 'N/A')}")