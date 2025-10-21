import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from psi_qrh_semantic_pipeline import PsiQRHSemanticPipeline

class SemanticProcessingPipeline:
    def __init__(self, psiqrh_pipeline: PsiQRHSemanticPipeline):
        self.pipeline = psiqrh_pipeline

    def process_text(self, text: str, task_type: str) -> Dict:
        """Processa texto através do pipeline completo"""

        # 1. Tokenização e pré-processamento
        tokens = self._tokenize_text(text)

        # 2. Análise semântica bilateral
        semantic_analysis = self._bilateral_semantic_analysis(tokens)

        # 3. Processamento PsiQRH tradicional
        with torch.no_grad():  # Desabilitar gradientes para inferência
            psiqrh_output = self.pipeline.model(tokens)

        # 4. Fusão adaptativa
        # O modelo PsiQRH retorna logits diretamente, não um dicionário
        # Simular probabilidades softmax
        probabilities = torch.softmax(psiqrh_output, dim=-1)
        # Para simplificação, usar a média das probabilidades como "probabilidades gerais"
        avg_probabilities = torch.mean(probabilities, dim=1)  # [batch_size, vocab_size] -> [vocab_size]

        final_prediction = self.pipeline.semantic_bases['fusion_engine'].fuse_analysis(
            semantic_analysis['left_features'],
            semantic_analysis['right_features'],
            avg_probabilities
        )

        # 5. Pós-processamento baseado na tarefa
        return self._post_process(final_prediction, task_type)

    def _bilateral_semantic_analysis(self, tokens: torch.Tensor) -> Dict:
        """Executa análise semântica bilateral"""
        # Converter tensor para lista de strings para análise semântica
        token_list = tokens.squeeze(0).tolist()  # Remover batch dimension
        vocab_reverse = {1: 'o', 2: 'banco', 3: 'tem', 4: 'capital', 5: 'suficiente', 6: 'para', 7: 'investimentos',
                        8: 'gato', 9: 'persegue', 10: 'rato', 11: 'no', 12: 'jardim'}
        token_strings = [vocab_reverse.get(token, '<unk>') for token in token_list]

        left_features = []
        right_features = []

        for i, token in enumerate(token_strings):
            # Contexto esquerdo
            left_context = token_strings[max(0, i-2):i]  # 2 tokens anteriores
            left_feat = self.pipeline.semantic_bases['left_base'].get_semantic_features(
                token, left_context
            )
            left_features.append(left_feat)

            # Contexto direito
            right_context = token_strings[i+1:min(len(token_strings), i+3)]  # 2 tokens posteriores
            right_feat = self.pipeline.semantic_bases['right_base'].get_semantic_features(
                token, right_context
            )
            right_features.append(right_feat)

        return {
            'left_features': torch.stack(left_features),
            'right_features': torch.stack(right_features),
            'tokens': token_strings
        }

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenização adaptada para análise semântica"""
        # Implementação específica para o PsiQRH
        tokens = text.lower().split()
        # Simulação de tokenização - converter para IDs (simples)
        vocab = {'o': 1, 'banco': 2, 'tem': 3, 'capital': 4, 'suficiente': 5, 'para': 6, 'investimentos': 7,
                'gato': 8, 'persegue': 9, 'rato': 10, 'no': 11, 'jardim': 12}
        token_ids = [vocab.get(token, 0) for token in tokens]  # 0 para tokens desconhecidos
        return torch.tensor(token_ids).unsqueeze(0)  # [1, seq_len]

    def _post_process(self, prediction: Dict, task_type: str) -> Dict:
        """Pós-processamento baseado no tipo de tarefa"""
        if task_type == 'ag_news':
            # Mapeamento para classes do AG News
            class_mapping = {
                0: 'World',
                1: 'Sports',
                2: 'Business',
                3: 'Sci/Tech'
            }
            prediction['class_name'] = class_mapping.get(prediction['prediction'], 'Unknown')

        elif task_type == 'imdb':
            # Mapeamento para sentimento
            prediction['sentiment'] = 'positive' if prediction['prediction'] == 1 else 'negative'

        return prediction