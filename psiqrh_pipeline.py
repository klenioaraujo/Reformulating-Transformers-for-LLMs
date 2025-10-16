#!/usr/bin/env python3
"""
ΨQRH Pipeline Configurado e Robusto
====================================

Implementação final que utiliza a QuantumCharacterMatrix como motor central e exclusivo
para todo o processamento de texto, com todos os parâmetros externalizados em configuração.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from typing import List

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Importar componentes essenciais do sistema ΨQRH
from quantum_character_matrix import QuantumCharacterMatrix
from src.core.context_funnel import ContextFunnel

# Importar gerenciador de configuração
try:
    from src.utils.config_manager import get_config_manager
except ImportError:
    # Fallback simples se o config manager não estiver disponível
    class SimpleConfigManager:
        def load_config(self, config_name):
            import yaml
            config_path = f"configs/{config_name}.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return {}

    def get_config_manager():
        return SimpleConfigManager()

class ΨQRHPipeline:
    """
    Pipeline configurado que orquestra a QuantumCharacterMatrix para gerar respostas
    de forma algorítmica, coesa e totalmente configurável via arquivos.
    """

    def __init__(self):
        # Carregar configuração
        config_mgr = get_config_manager()
        self.config = config_mgr.load_config('pipeline_config')

        # Parâmetros do pipeline
        self.device = self.config.get('pipeline', {}).get('device', 'cpu')

        # Parâmetros da QuantumCharacterMatrix
        qm_config = self.config.get('quantum_matrix', {})
        vocabulary = self.config.get('vocabulary')

        # Inicializar QuantumCharacterMatrix com parâmetros do config
        self.qcm = QuantumCharacterMatrix(
            embed_dim=qm_config.get('embed_dim', 64),
            alpha=qm_config.get('alpha', 1.5),
            beta=qm_config.get('beta', 0.8),
            fractal_dim=qm_config.get('fractal_dim', 1.7),
            device=self.device,
            vocabulary=vocabulary
        )

        # Inicializar ContextFunnel com parâmetros do config
        cf_config = self.config.get('context_funnel', {})
        self.context_funnel = ContextFunnel(
            embed_dim=self.qcm.embed_dim * 4,  # Opera sobre o quaternião achatado
            num_heads=cf_config.get('num_heads', 8),
            max_history=cf_config.get('max_history', 50)
        ).to(self.device)

        print("✅ ΨQRH Pipeline Configurado inicializado com sucesso.")
        print(f"   🔩 Usando QuantumCharacterMatrix como motor principal.")
        print(f"   📚 Vocabulário: {len(self.qcm.vocabulary)} caracteres.")

    def process(self, input_text: str) -> str:
        """
        PIPELINE CORRIGIDO com decodificação posicional e preservação de contexto.
        """
        max_length = self.config.get('pipeline', {}).get('max_generation_length', 20)
        context_blend_ratio = 0.7  # Preserva 70% do contexto anterior

        print(f"\n🔄 Processando: '{input_text}'")

        # --- Etapa 1: Codificação do Input via QuantumCharacterMatrix ---
        with torch.no_grad():
            input_states = [self.qcm.encode_character(char, position=i) for i, char in enumerate(input_text)]
            flattened_input_states = [s.flatten() for s in input_states]
            current_context = self.context_funnel(flattened_input_states)

        # --- Etapa 2: Loop de Geração Auto-Regressivo CORRIGIDO ---
        generated_chars = []
        current_position = len(input_text)  # Começa após o input

        for i in range(max_length):
            with torch.no_grad():
                context_to_decode = current_context.view(self.qcm.embed_dim, 4)

                # 🔥 DECODIFICAÇÃO COM POSIÇÃO CORRETA
                decoded_results = self.qcm.decode_quantum_state(
                    context_to_decode, top_k=1, position=current_position
                )

                if not decoded_results:
                    break

                next_char, confidence = decoded_results[0]

                # Critério de parada mais inteligente
                if next_char == '<UNK>' or confidence < 0.3:
                    break

                generated_chars.append(next_char)

                # 🔥 ATUALIZAÇÃO PONDERADA DO CONTEXTO
                new_char_state = self.qcm.encode_character(next_char, position=current_position)
                current_context = (
                    context_blend_ratio * current_context +
                    (1 - context_blend_ratio) * new_char_state.flatten()
                )

                current_position += 1

        generated_text = "".join(generated_chars)
        print(f"   🔬 Resposta Gerada: '{generated_text}'")
        return generated_text

def main():
    """Função principal para lidar com argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description="ΨQRH Pipeline Configurado - Geração de Texto com QuantumCharacterMatrix",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'text',
        nargs='?',
        default=None,
        help='Texto a ser processado pelo pipeline.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Semente de aleatoriedade para garantir resultados reproduzíveis.'
    )

    args = parser.parse_args()

    if args.seed is not None:
        print(f"🌱 Usando semente de aleatoriedade: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        print("🌱 Executando em modo aleatório (sem semente).")

    # Se nenhum texto for passado, usa o default do config
    text_to_process = args.text
    if text_to_process is None:
        config_mgr = get_config_manager()
        try:
            app_config = config_mgr.load_config('pipeline_config')
            text_to_process = app_config.get('pipeline', {}).get('default_prompt', 'life is beautiful')
        except FileNotFoundError:
            text_to_process = 'life is beautiful'

    # Inicializa e executa o pipeline
    pipeline = ΨQRHPipeline()
    result = pipeline.process(text_to_process)

    print(f"\n🎯 Input: {text_to_process}")
    print(f"🎯 Output: {result}")

if __name__ == "__main__":
    main()