import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class SemanticBase(nn.Module):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction
        self.ontology = self._load_ontology(direction)
        self.word_vectors = {}

        # Parâmetros aprendíveis para embeddings semânticos
        self.embedding_dim = 128
        self.semantic_embeddings = nn.Embedding(len(self.ontology), self.embedding_dim)
        self.context_encoder = nn.Linear(self.embedding_dim * 2, self.embedding_dim)  # Para contexto

        # Mapeamento palavra -> índice
        self.word_to_idx = {word: i for i, word in enumerate(self.ontology.keys())}

    def _load_ontology(self, direction: str) -> Dict:
        """Carrega ontologia específica para direção com base expandida"""
        if direction == 'left':
            return {
                # Palavras originais expandidas
                'banco': ['instituição_financeira', 'local_sentarse', 'organização', 'empresa', 'estabelecimento'],
                'tem': ['posse', 'existência', 'característica', 'contém', 'possui', 'apresenta'],
                'gato': ['animal', 'predador', 'doméstico', 'felino', 'mamífero', 'pet'],
                'persegue': ['caça', 'segue', 'procura', 'persegue', 'busca', 'acompanha'],

                # Novas palavras para contexto esquerdo (concreto/físico)
                'casa': ['habitação', 'residência', 'lar', 'construção', 'moradia'],
                'carro': ['veículo', 'automóvel', 'transporte', 'meio_locomoção'],
                'livro': ['leitura', 'conhecimento', 'obra', 'publicação', 'texto'],
                'comida': ['alimento', 'nutrição', 'refeição', 'sustento', 'ingestão'],
                'água': ['líquido', 'hidratação', 'elemento', 'fluido', 'bebida'],
                'sol': ['astro', 'luz', 'calor', 'energia', 'iluminação'],
                'árvore': ['planta', 'vegetal', 'natureza', 'tronco', 'folhas'],
                'rio': ['curso_água', 'fluxo', 'natureza', 'corrente', 'líquido'],
                'montanha': ['elevação', 'terreno', 'natureza', 'pico', 'rocha'],
                'mar': ['oceano', 'água_salgada', 'natureza', 'imensidão', 'onda'],

                # Verbos de ação física
                'corre': ['movimento', 'velocidade', 'deslocamento', 'rapidez'],
                'anda': ['locomoção', 'movimento', 'passo', 'deslocamento'],
                'come': ['ingestão', 'alimentação', 'mastigação', 'consumo'],
                'bebe': ['ingestão', 'líquido', 'hidratação', 'consumo'],
                'dorme': ['repouso', 'descanso', 'sono', 'inatividade'],
                'trabalha': ['atividade', 'labor', 'esforço', 'ocupação'],
                'estuda': ['aprendizado', 'conhecimento', 'educação', 'análise'],
                'joga': ['diversão', 'esporte', 'recreação', 'competição'],
            }
        else:  # right - contexto abstrato/conceitual
            return {
                # Palavras originais expandidas
                'banco': ['financeiro', 'móvel_parque', 'rio', 'dados', 'instituição', 'crédito'],
                'tem': ['possui', 'contém', 'experimenta', 'sente', 'sofre', 'enfrenta'],
                'rato': ['animal', 'presa', 'roedor', 'praga', 'medo', 'fobia'],
                'perigo': ['risco', 'ameaça', 'situação_ruim', 'insegurança', 'vulnerabilidade'],

                # Novas palavras para contexto direito (abstrato/emocional)
                'amor': ['sentimento', 'afeição', 'paixão', 'emoção', 'carinho'],
                'medo': ['emoção', 'ansiedade', 'fobia', 'insegurança', 'pavor'],
                'felicidade': ['alegria', 'contentamento', 'satisfação', 'bem_estar'],
                'tristeza': ['melancolia', 'depressão', 'sofrimento', 'infelicidade'],
                'esperança': ['otimismo', 'confiança', 'expectativa', 'fé'],
                'ódio': ['raiva', 'ressentimento', 'aversão', 'antipatia'],
                'paz': ['tranquilidade', 'harmonia', 'calma', 'equilíbrio'],
                'guerra': ['conflito', 'batalha', 'violência', 'destruição'],
                'justiça': ['direito', 'equidade', 'fairness', 'moralidade'],
                'liberdade': ['autonomia', 'independência', 'direito', 'autodeterminação'],

                # Conceitos abstratos
                'tempo': ['duração', 'eternidade', 'momento', 'passagem', 'fluxo'],
                'espaço': ['dimensão', 'ambiente', 'localização', 'extensão'],
                'pensamento': ['ideia', 'reflexão', 'cognição', 'mente', 'intelecto'],
                'sonho': ['imaginação', 'desejo', 'aspiracao', 'ilusão', 'meta'],
                'realidade': ['verdade', 'existência', 'factual', 'concreto'],
                'ilusão': ['enganar', 'aparência', 'falsidade', 'ilusório'],
                'verdade': ['realidade', 'facticidade', 'autenticidade', 'certeza'],
                'mentira': ['falsidade', 'engano', 'decepção', 'inverdade'],
            }

    def get_semantic_features(self, word: str, context: List[str]) -> torch.Tensor:
        """Extrai features semânticas aprendíveis baseadas no contexto"""
        if word not in self.word_to_idx:
            return torch.zeros(self.embedding_dim)  # embedding padrão

        # Obter embedding base da palavra
        word_idx = self.word_to_idx[word]
        word_embedding = self.semantic_embeddings(torch.tensor(word_idx))

        # Processar contexto se disponível
        if context:
            # Criar representação de contexto (média dos embeddings das palavras de contexto)
            context_embeddings = []
            for ctx_word in context:
                if ctx_word in self.word_to_idx:
                    ctx_idx = self.word_to_idx[ctx_word]
                    ctx_emb = self.semantic_embeddings(torch.tensor(ctx_idx))
                    context_embeddings.append(ctx_emb)

            if context_embeddings:
                context_mean = torch.stack(context_embeddings).mean(dim=0)
                # Combinar palavra + contexto
                combined = torch.cat([word_embedding, context_mean])
                contextualized = self.context_encoder(combined)
                return contextualized
            else:
                return word_embedding
        else:
            return word_embedding

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

    def parameters(self):
        """Retorna parâmetros aprendíveis"""
        return list(self.semantic_embeddings.parameters()) + list(self.context_encoder.parameters())