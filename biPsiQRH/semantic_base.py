import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class SemanticBase:
    def __init__(self, direction: str):
        self.direction = direction
        self.ontology = self._load_ontology(direction)
        self.word_vectors = {}

    def _load_ontology(self, direction: str) -> Dict:
        """Carrega ontologia específica para direção"""
        if direction == 'left':
            return {
                'banco': ['instituição_financeira', 'local_sentarse', 'organização'],
                'tem': ['posse', 'existência', 'característica'],
                'gato': ['animal', 'predador', 'doméstico'],
                'persegue': ['caça', 'segue', 'procura'],
                # ... expandir base
            }
        else:  # right
            return {
                'banco': ['financeiro', 'móvel_parque', 'rio', 'dados'],
                'tem': ['possui', 'contém', 'experimenta'],
                'rato': ['animal', 'presa', 'roedor'],
                'perigo': ['risco', 'ameaça', 'situação_ruim'],
                # ... expandir base
            }

    def get_semantic_features(self, word: str, context: List[str]) -> torch.Tensor:
        """Extrai features semânticas baseadas no contexto"""
        if word not in self.ontology:
            return torch.zeros(128)  # embedding padrão

        semantic_tags = self.ontology[word]
        context_influence = self._calculate_context_influence(context, semantic_tags)

        return self._encode_semantic_vector(semantic_tags, context_influence)

    def _calculate_context_influence(self, context: List[str], tags: List[str]) -> Dict:
        """Calcula influência do contexto nas tags semânticas"""
        influence = {}
        for tag in tags:
            influence[tag] = sum(1 for ctx_word in context if self._has_semantic_relation(ctx_word, tag))
        return influence

    def _has_semantic_relation(self, word: str, tag: str) -> bool:
        """Verifica se palavra tem relação com tag semântica"""
        if word not in self.ontology:
            return False
        return tag in self.ontology[word]

    def _encode_semantic_vector(self, tags: List[str], context_influence: Dict) -> torch.Tensor:
        """Codifica tags semânticas em vetor"""
        # Embedding simples baseado na frequência das tags
        vector = torch.zeros(128)
        for i, tag in enumerate(tags):
            if i < 128:
                influence = context_influence.get(tag, 0)
                vector[i] = influence + 1  # +1 para evitar zeros
        return vector