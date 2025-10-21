import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class SemanticFusionEngine:
    def __init__(self):
        self.confidence_threshold = 0.7
        self.fusion_rules = self._load_fusion_rules()

    def fuse_analysis(self, left_features: torch.Tensor,
                     right_features: torch.Tensor,
                     softmax_probs: torch.Tensor) -> Dict:
        """Fusão inteligente entre análise semântica e softmax"""

        semantic_confidence = self._calculate_semantic_confidence(left_features, right_features)

        if semantic_confidence > self.confidence_threshold:
            # Prioridade para análise semântica
            semantic_result = self._resolve_semantic_ambiguity(left_features, right_features)
            return {
                'prediction': semantic_result['prediction'],
                'confidence': semantic_confidence,
                'method': 'semantic',
                'explanation': semantic_result['explanation']
            }
        else:
            # Fallback para softmax estatístico
            return {
                'prediction': torch.argmax(softmax_probs).item(),
                'confidence': torch.max(softmax_probs).item(),
                'method': 'softmax',
                'explanation': 'Estatístico - baixa confiança semântica'
            }

    def _calculate_semantic_confidence(self, left: torch.Tensor, right: torch.Tensor) -> float:
        """Calcula confiança na análise semântica"""
        similarity = torch.cosine_similarity(left.mean(dim=0), right.mean(dim=0), dim=0)
        consistency = 1.0 - torch.std(torch.stack([left.mean(dim=0), right.mean(dim=0)]))
        return float((similarity + consistency) / 2)

    def _resolve_semantic_ambiguity(self, left: torch.Tensor, right: torch.Tensor) -> Dict:
        """Resolve ambiguidades usando regras de fusão"""
        # Implementação das regras de desambiguização
        left_dominant = torch.norm(left) > torch.norm(right)

        if left_dominant:
            return self._apply_left_dominant_rules(left, right)
        else:
            return self._apply_right_dominant_rules(left, right)

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