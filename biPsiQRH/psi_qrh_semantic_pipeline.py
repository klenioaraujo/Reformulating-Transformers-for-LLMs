import torch
import torch.nn as nn
import json
from typing import Dict, List, Tuple
import numpy as np
from semantic_base import SemanticBase
from semantic_fusion_engine import SemanticFusionEngine

class PsiQRHSemanticPipeline(nn.Module):
    def __init__(self, config_path: str):
        """
        Pipeline PsiQRH com análise semântica bilateral aprendível
        """
        super().__init__()
        self.config = self._load_config(config_path)
        self.semantic_bases = self._initialize_semantic_bases()
        self.model = self._initialize_psiqrh_model()

    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração do PsiQRH"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _initialize_semantic_bases(self) -> nn.ModuleDict:
        """Inicializa bases semânticas bilaterais aprendíveis"""
        return nn.ModuleDict({
            'left_base': SemanticBase(direction='left'),
            'right_base': SemanticBase(direction='right'),
            'fusion_engine': SemanticFusionEngine()
        })

    def _initialize_psiqrh_model(self):
        """Inicializa o modelo PsiQRH base"""
        # Baseado na arquitetura do repositório
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

        from src.architecture.psiqrh_transformer import PsiQRHTransformer
        return PsiQRHTransformer(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['model_dim'],
            n_layers=self.config['num_layers'],
            n_heads=self.config['num_heads']
        )

    def parameters(self):
        """Retorna todos os parâmetros aprendíveis do pipeline"""
        params = list(self.semantic_bases.parameters())
        params.extend(list(self.model.parameters()))
        return params