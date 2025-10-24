import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class SemanticFusionEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence_threshold = 0.5  # Reduzido para permitir mais fusão híbrida
        self.fusion_rules = self._load_fusion_rules()

        # Pesos adaptativos para fusão
        self.left_weight = nn.Parameter(torch.tensor(0.5))
        self.right_weight = nn.Parameter(torch.tensor(0.5))
        self.softmax_weight = nn.Parameter(torch.tensor(0.3))

        # Mecanismo de atenção para fusão
        self.attention_fusion = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

    def fuse_analysis(self, left_features: torch.Tensor,
                     right_features: torch.Tensor,
                     softmax_probs: torch.Tensor) -> Dict:
        """Fusão inteligente com pesos adaptativos e atenção"""

        semantic_confidence = self._calculate_semantic_confidence(left_features, right_features)

        # Fusão adaptativa com pesos aprendíveis
        if semantic_confidence > self.confidence_threshold:
            # Fusão semântica com atenção
            fused_semantic = self._attention_based_fusion(left_features, right_features)
            semantic_result = self._resolve_semantic_ambiguity(fused_semantic, right_features)

            return {
                'prediction': semantic_result['prediction'],
                'confidence': semantic_confidence,
                'method': 'semantic',
                'explanation': semantic_result['explanation']
            }
        else:
            # Fusão híbrida: semântica + softmax com pesos adaptativos
            semantic_fused = self._weighted_fusion(left_features, right_features)
            combined_prediction = self._combine_with_softmax(semantic_fused, softmax_probs)

            return {
                'prediction': combined_prediction['prediction'],
                'confidence': combined_prediction['confidence'],
                'method': 'hybrid',
                'explanation': 'Fusão híbrida semântica + estatística'
            }

    def _calculate_semantic_confidence(self, left: torch.Tensor, right: torch.Tensor) -> float:
        """Calcula confiança na análise semântica"""
        similarity = torch.cosine_similarity(left.mean(dim=0), right.mean(dim=0), dim=0)
        consistency = 1.0 - torch.std(torch.stack([left.mean(dim=0), right.mean(dim=0)]))
        return float((similarity + consistency) / 2)

    def _attention_based_fusion(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Fusão baseada em atenção entre features esquerda e direita"""
        # Combinar features para atenção
        combined = torch.stack([left, right], dim=1)  # [batch, 2, features]

        # Aplicar atenção multi-head
        attended, _ = self.attention_fusion(combined, combined, combined)

        # Agregar resultado
        return attended.mean(dim=1)  # [batch, features]

    def _weighted_fusion(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Fusão com pesos adaptativos"""
        # Normalizar pesos
        total_weight = torch.sigmoid(self.left_weight) + torch.sigmoid(self.right_weight)
        left_norm = torch.sigmoid(self.left_weight) / total_weight
        right_norm = torch.sigmoid(self.right_weight) / total_weight

        return left_norm * left + right_norm * right

    def _combine_with_softmax(self, semantic_fused: torch.Tensor, softmax_probs: torch.Tensor) -> Dict:
        """Combina features semânticas com probabilidades softmax"""
        # Usar similaridade com classes para determinar predição
        # Simplificação: usar norma do vetor semântico como confiança
        confidence = torch.norm(semantic_fused).item()

        # Predição baseada na classe mais provável do softmax, mas ajustada pela confiança semântica
        softmax_pred = torch.argmax(softmax_probs).item()
        softmax_conf = torch.max(softmax_probs).item()

        # Combinação ponderada
        combined_conf = self.softmax_weight * softmax_conf + (1 - self.softmax_weight) * confidence

        return {
            'prediction': softmax_pred,
            'confidence': combined_conf
        }

    def _resolve_semantic_ambiguity(self, fused: torch.Tensor, right: torch.Tensor) -> Dict:
        """Resolve ambiguidades usando features fusionadas"""
        # Implementação das regras de desambiguização com features fusionadas
        left_dominant = torch.norm(fused) > torch.norm(right)

        if left_dominant:
            return self._apply_left_dominant_rules(fused, right)
        else:
            return self._apply_right_dominant_rules(fused, right)

    def _apply_left_dominant_rules(self, left: torch.Tensor, right: torch.Tensor) -> Dict:
        """Aplica regras quando contexto esquerdo é dominante"""
        # Exemplo: banco + tem → instituição financeira
        return {
            'prediction': 0,  # Classe para instituição financeira
            'explanation': 'Contexto esquerdo dominante: instituição financeira'
        }

    def _apply_right_dominant_rules(self, left: torch.Tensor, right: torch.Tensor) -> Dict:
        """Aplica regras quando contexto direito é dominante"""
        # Exemplo: banco + perigo → risco financeiro
        return {
            'prediction': 1,  # Classe para risco financeiro
            'explanation': 'Contexto direito dominante: risco financeiro'
        }

    def _load_fusion_rules(self) -> Dict:
        """Carrega regras de fusão semântica"""
        return {
            'left_dominant': {
                'banco_tem': 'instituição_financeira',
                'gato_persegue': 'predador_doméstico'
            },
            'right_dominant': {
                'banco_perigo': 'risco_financeiro',
                'rato_perigo': 'situação_risco'
            }
        }