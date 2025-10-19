#!/usr/bin/env python3
"""
Quantum Token Filter - Filtro Quântico para Predição de Tokens e Análise Espectral

Implementa um filtro quântico que prevê caracteres problemáticos e identifica seu espectral,
com funções matemáticas de gramática para decodificação dentro do espaço quântico.

Componentes principais:
- QuantumSpectralAnalyzer: Análise espectral de tokens
- MathematicalGrammarFunctions: Funções de gramática matemática
- QuantumDecodingFilter: Filtro de decodificação quântica
- TokenPredictionEngine: Motor de predição de tokens
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
from pathlib import Path
import json


class QuantumSpectralAnalyzer:
    """
    Analisador Espectral Quântico para tokens

    Analisa o espectro quântico de tokens para identificar padrões problemáticos
    e caracteres que podem causar loops de geração.
    """

    def __init__(self, embed_dim: int = 256, spectral_bands: int = 8):
        """
        Inicializa o analisador espectral quântico.

        Args:
            embed_dim: Dimensão do embedding
            spectral_bands: Número de bandas espectrais para análise
        """
        self.embed_dim = embed_dim
        self.spectral_bands = spectral_bands

        # Mapeamento de caracteres problemáticos conhecidos
        self.problematic_chars = {
            'ĠFriday': 'repetitive_token',
            'ii': 'repetitive_token',
            '--------': 'dash_pattern',
            'Ġkept': 'low_entropy',
            'Ġreb': 'low_entropy',
            'Ġfan': 'low_entropy'
        }

        # Limiares espectrais para detecção de problemas
        self.spectral_thresholds = {
            'repetition': 0.8,      # Alta correlação espectral
            'low_entropy': 0.1,     # Baixa entropia espectral
            'pattern': 0.6,         # Padrões repetitivos
            'anomaly': 3.0          # Anomalia espectral (desvio padrão)
        }

        print("🔬 QuantumSpectralAnalyzer inicializado")
        print(f"   📊 Embed dim: {embed_dim}")
        print(f"   🎵 Spectral bands: {spectral_bands}")

    def analyze_token_spectrum(self, token_quantum: torch.Tensor, token_id: int) -> Dict[str, Any]:
        """
        Analisa o espectro quântico de um token.

        Args:
            token_quantum: Representação quântica do token [embed_dim, 4]
            token_id: ID do token para análise

        Returns:
            spectral_analysis: Análise espectral completa
        """
        # Converter para tensor 1D para análise espectral
        token_flat = token_quantum.view(-1)  # [embed_dim * 4]

        # Calcular transformada de Fourier
        spectrum = torch.fft.fft(token_flat)
        magnitudes = torch.abs(spectrum)

        # Análise de entropia espectral
        spectral_entropy = self._compute_spectral_entropy(magnitudes)

        # Detecção de padrões repetitivos
        repetition_score = self._detect_repetitive_patterns(magnitudes)

        # Análise de anomalias
        anomaly_score = self._detect_spectral_anomalies(magnitudes)

        # Classificação do token
        token_class = self._classify_token_spectrum(
            spectral_entropy, repetition_score, anomaly_score
        )

        return {
            'token_id': token_id,
            'spectral_entropy': spectral_entropy.item(),
            'repetition_score': repetition_score.item(),
            'anomaly_score': anomaly_score.item(),
            'token_class': token_class,
            'magnitude_stats': {
                'mean': magnitudes.mean().item(),
                'std': magnitudes.std().item(),
                'max': magnitudes.max().item(),
                'min': magnitudes.min().item()
            },
            'dominant_frequencies': self._extract_dominant_frequencies(magnitudes)
        }

    def _compute_spectral_entropy(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Calcula a entropia espectral (medida de diversidade frequencial)"""
        # Normalizar magnitudes para distribuição de probabilidade
        prob_dist = magnitudes / magnitudes.sum()

        # Calcular entropia: H = -∑ p_i log(p_i)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8))

        # Normalizar para [0, 1]
        max_entropy = math.log(len(magnitudes))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def _detect_repetitive_patterns(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Detecta padrões repetitivos no espectro"""
        # Calcular autocorrelação usando FFT (mais eficiente e compatível)
        # autocorr = F^{-1}(|F(signal)|^2)
        spectrum = torch.fft.fft(magnitudes)
        power_spectrum = torch.abs(spectrum) ** 2
        autocorr = torch.fft.ifft(power_spectrum).real

        # Normalizar autocorrelação
        autocorr_norm = autocorr / (autocorr.max() + 1e-8)

        # Score baseado na autocorrelação (alta = repetitivo)
        repetition_score = autocorr_norm.mean()

        return repetition_score

    def _detect_spectral_anomalies(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Detecta anomalias no espectro"""
        # Calcular z-score para cada frequência
        mean = magnitudes.mean()
        std = magnitudes.std()
        z_scores = (magnitudes - mean) / (std + 1e-8)

        # Contar outliers (z-score > 2)
        outliers = torch.sum(torch.abs(z_scores) > 2.0)
        anomaly_score = outliers / len(magnitudes)

        return anomaly_score

    def _extract_dominant_frequencies(self, magnitudes: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """Extrai as frequências dominantes do espectro"""
        # Encontrar top-K magnitudes
        top_magnitudes, top_indices = torch.topk(magnitudes, top_k)

        dominant_freqs = []
        for i in range(top_k):
            freq_info = {
                'frequency_index': top_indices[i].item(),
                'magnitude': top_magnitudes[i].item(),
                'normalized_frequency': top_indices[i].item() / len(magnitudes)
            }
            dominant_freqs.append(freq_info)

        return dominant_freqs

    def _classify_token_spectrum(self, entropy: float, repetition: float, anomaly: float) -> str:
        """Classifica o token baseado na análise espectral"""
        if repetition > self.spectral_thresholds['repetition']:
            return 'repetitive_token'
        elif entropy < self.spectral_thresholds['low_entropy']:
            return 'low_entropy_token'
        elif anomaly > self.spectral_thresholds['anomaly']:
            return 'anomalous_token'
        else:
            return 'normal_token'


class MathematicalGrammarFunctions:
    """
    Funções de Gramática Matemática para Decodificação

    Implementa funções matemáticas que simulam regras gramaticais
    dentro do espaço quântico para melhor decodificação.
    """

    def __init__(self, vocab_size: int = 50257):
        """
        Inicializa as funções de gramática matemática.

        Args:
            vocab_size: Tamanho do vocabulário
        """
        self.vocab_size = vocab_size

        # Regras gramaticais básicas (pesos para diferentes tipos de tokens)
        self.grammar_rules = {
            'space_after_punctuation': 0.8,
            'capital_after_period': 0.9,
            'verb_after_pronoun': 0.7,
            'noun_after_article': 0.8,
            'consistent_tense': 0.6
        }

        # Mapeamento de categorias gramaticais (simplificado)
        self.grammar_categories = self._initialize_grammar_categories()

        print("📐 MathematicalGrammarFunctions inicializado")
        print(f"   📚 Vocab size: {vocab_size}")
        print(f"   📝 Grammar rules: {len(self.grammar_rules)} regras")

    def _initialize_grammar_categories(self) -> Dict[str, List[int]]:
        """Inicializa categorias gramaticais básicas"""
        # Mapeamento simplificado - em produção seria mais detalhado
        return {
            'punctuation': [13, 25, 26, 30, 366, 705],  # . : ; ? " '
            'articles': [1004, 1028],                   # a, an
            'pronouns': [1013, 1014, 1015, 1030, 1050], # i, you, it, we, they
            'verbs': [1007, 1018, 1019, 1025, 1033],   # is, be, are, have, can
            'nouns': [1000, 1002, 1009, 1029, 1049]    # the, of, that, was, time
        }

    def apply_grammar_constraints(self,
                                 candidate_logits: torch.Tensor,
                                 previous_tokens: List[int],
                                 grammar_strength: float = 0.5) -> torch.Tensor:
        """
        Aplica restrições gramaticais aos logits candidatos.

        Args:
            candidate_logits: Logits dos tokens candidatos [n_candidates]
            previous_tokens: Lista de tokens anteriores no contexto
            grammar_strength: Força das restrições gramaticais [0, 1]

        Returns:
            constrained_logits: Logits com restrições gramaticais aplicadas
        """
        if len(previous_tokens) == 0:
            return candidate_logits  # Sem contexto para aplicar gramática

        # Obter último token do contexto
        last_token = previous_tokens[-1]

        # Inicializar penalidades gramaticais
        grammar_penalties = torch.zeros_like(candidate_logits)

        # Aplicar regras gramaticais baseadas no último token
        if self._is_punctuation(last_token):
            # Após pontuação: favorecer espaços e maiúsculas
            space_boost = self._boost_spaces_after_punctuation(candidate_logits)
            capital_boost = self._boost_capitals_after_punctuation(candidate_logits)
            grammar_penalties += space_boost + capital_boost

        elif self._is_pronoun(last_token):
            # Após pronome: favorecer verbos
            verb_boost = self._boost_verbs_after_pronoun(candidate_logits)
            grammar_penalties += verb_boost

        elif self._is_article(last_token):
            # Após artigo: favorecer substantivos
            noun_boost = self._boost_nouns_after_article(candidate_logits)
            grammar_penalties += noun_boost

        # Aplicar penalidades com força controlada
        constrained_logits = candidate_logits + grammar_strength * grammar_penalties

        return constrained_logits

    def _is_punctuation(self, token_id: int) -> bool:
        """Verifica se o token é pontuação"""
        return token_id in self.grammar_categories['punctuation']

    def _is_pronoun(self, token_id: int) -> bool:
        """Verifica se o token é pronome"""
        return token_id in self.grammar_categories['pronouns']

    def _is_article(self, token_id: int) -> bool:
        """Verifica se o token é artigo"""
        return token_id in self.grammar_categories['articles']

    def _boost_spaces_after_punctuation(self, candidate_logits: torch.Tensor) -> torch.Tensor:
        """Aumenta logits de espaços após pontuação"""
        # Identificar tokens de espaço (simplificado)
        space_tokens = [220]  # Ġ (espaço no GPT-2)
        boost = torch.zeros_like(candidate_logits)

        for token_id in space_tokens:
            if token_id < len(candidate_logits):
                boost[token_id] = self.grammar_rules['space_after_punctuation']

        return boost

    def _boost_capitals_after_punctuation(self, candidate_logits: torch.Tensor) -> torch.Tensor:
        """Aumenta logits de maiúsculas após pontuação"""
        # Identificar tokens que começam com maiúscula (simplificado)
        capital_tokens = [309, 311, 327, 337, 360, 367, 376, 399, 402, 412, 440, 449, 509, 569, 575]  # T, S, C, M, D, H, F, N, G, E, O, J, K, V, Y
        boost = torch.zeros_like(candidate_logits)

        for token_id in capital_tokens:
            if token_id < len(candidate_logits):
                boost[token_id] = self.grammar_rules['capital_after_period']

        return boost

    def _boost_verbs_after_pronoun(self, candidate_logits: torch.Tensor) -> torch.Tensor:
        """Aumenta logits de verbos após pronomes"""
        verb_boost = torch.zeros_like(candidate_logits)

        for token_id in self.grammar_categories['verbs']:
            if token_id < len(candidate_logits):
                verb_boost[token_id] = self.grammar_rules['verb_after_pronoun']

        return verb_boost

    def _boost_nouns_after_article(self, candidate_logits: torch.Tensor) -> torch.Tensor:
        """Aumenta logits de substantivos após artigos"""
        noun_boost = torch.zeros_like(candidate_logits)

        for token_id in self.grammar_categories['nouns']:
            if token_id < len(candidate_logits):
                noun_boost[token_id] = self.grammar_rules['noun_after_article']

        return noun_boost


class QuantumDecodingFilter:
    """
    Filtro de Decodificação Quântica

    Combina análise espectral e funções gramaticais para filtrar
    tokens problemáticos durante a decodificação.
    """

    def __init__(self,
                 embed_dim: int = 256,
                 vocab_size: int = 50257,
                 quantum_vocab_representations: Optional[torch.Tensor] = None):
        """
        Inicializa o filtro de decodificação quântica.

        Args:
            embed_dim: Dimensão do embedding
            vocab_size: Tamanho do vocabulário
            quantum_vocab_representations: Representações quânticas do vocabulário
        """
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.quantum_vocab_representations = quantum_vocab_representations

        # Inicializar componentes
        self.spectral_analyzer = QuantumSpectralAnalyzer(embed_dim)
        self.grammar_functions = MathematicalGrammarFunctions(vocab_size)

        # Configurações do filtro
        self.filter_settings = {
            'max_repetition': 3,           # Máximo de repetições consecutivas
            'min_entropy': 0.3,            # Entropia espectral mínima
            'grammar_strength': 0.7,       # Força das restrições gramaticais
            'problematic_penalty': -10.0   # Penalidade para tokens problemáticos
        }

        # Estado do filtro
        self.repetition_history = []
        self.problematic_tokens_detected = []

        print("🔧 QuantumDecodingFilter inicializado")
        print(f"   📊 Embed dim: {embed_dim}")
        print(f"   📚 Vocab size: {vocab_size}")
        print(f"   🔬 Spectral analyzer: {self.spectral_analyzer is not None}")
        print(f"   📐 Grammar functions: {self.grammar_functions is not None}")

    def filter_candidates(self,
                         candidate_logits: torch.Tensor,
                         candidate_indices: torch.Tensor,
                         previous_tokens: List[int] = None,
                         context_quantum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Filtra tokens candidatos usando análise quântica e gramatical.

        Args:
            candidate_logits: Logits dos candidatos [n_candidates]
            candidate_indices: Índices dos candidatos [n_candidates]
            previous_tokens: Lista de tokens anteriores (para contexto gramatical)
            context_quantum: Estado quântico do contexto (opcional)

        Returns:
            filtered_logits: Logits filtrados
        """
        if previous_tokens is None:
            previous_tokens = []

        # Copiar logits originais
        filtered_logits = candidate_logits.clone()

        # 1. Aplicar análise espectral para detectar tokens problemáticos
        spectral_penalties = self._apply_spectral_filtering(
            candidate_indices, candidate_logits
        )

        # 2. Aplicar restrições gramaticais
        grammar_constrained = self.grammar_functions.apply_grammar_constraints(
            filtered_logits, previous_tokens, self.filter_settings['grammar_strength']
        )

        # 3. Combinar penalidades
        filtered_logits = grammar_constrained + spectral_penalties

        # 4. Atualizar histórico de repetição
        self._update_repetition_history(candidate_indices, filtered_logits)

        # 5. Aplicar penalidades por repetição excessiva
        repetition_penalties = self._apply_repetition_penalties(candidate_indices)
        filtered_logits += repetition_penalties

        return filtered_logits

    def _apply_spectral_filtering(self,
                                 candidate_indices: torch.Tensor,
                                 candidate_logits: torch.Tensor) -> torch.Tensor:
        """Aplica filtragem espectral aos candidatos"""
        penalties = torch.zeros_like(candidate_logits)

        if self.quantum_vocab_representations is None:
            return penalties  # Sem representações quânticas disponíveis

        for i, token_idx in enumerate(candidate_indices):
            if token_idx >= self.quantum_vocab_representations.shape[0]:
                continue  # Índice fora do range

            # Obter representação quântica do token
            token_quantum = self.quantum_vocab_representations[token_idx]

            # Analisar espectro do token
            spectral_analysis = self.spectral_analyzer.analyze_token_spectrum(
                token_quantum, token_idx.item()
            )

            # Aplicar penalidades baseadas na análise espectral
            if spectral_analysis['token_class'] == 'repetitive_token':
                penalties[i] += self.filter_settings['problematic_penalty'] * 0.5

            elif spectral_analysis['token_class'] == 'low_entropy_token':
                penalties[i] += self.filter_settings['problematic_penalty'] * 0.3

            elif spectral_analysis['token_class'] == 'anomalous_token':
                penalties[i] += self.filter_settings['problematic_penalty'] * 0.7

            # Registrar tokens problemáticos
            if spectral_analysis['token_class'] != 'normal_token':
                self.problematic_tokens_detected.append({
                    'token_id': token_idx.item(),
                    'analysis': spectral_analysis,
                    'timestamp': len(self.repetition_history)
                })

        return penalties

    def _update_repetition_history(self,
                                  candidate_indices: torch.Tensor,
                                  candidate_logits: torch.Tensor):
        """Atualiza o histórico de repetição"""
        # Encontrar token mais provável
        best_idx = torch.argmax(candidate_logits).item()
        best_token = candidate_indices[best_idx].item()

        # Adicionar ao histórico
        self.repetition_history.append(best_token)

        # Manter apenas histórico recente
        if len(self.repetition_history) > 10:
            self.repetition_history = self.repetition_history[-10:]

    def _apply_repetition_penalties(self, candidate_indices: torch.Tensor) -> torch.Tensor:
        """Aplica penalidades por repetição excessiva"""
        penalties = torch.zeros(len(candidate_indices))

        if len(self.repetition_history) < 2:
            return penalties  # Histórico insuficiente

        # Verificar repetições recentes
        recent_tokens = self.repetition_history[-3:]  # Últimos 3 tokens

        for i, token_idx in enumerate(candidate_indices):
            token_id = token_idx.item()

            # Contar ocorrências recentes deste token
            recent_count = recent_tokens.count(token_id)

            if recent_count >= self.filter_settings['max_repetition']:
                # Penalidade severa para repetição excessiva
                penalties[i] = self.filter_settings['problematic_penalty'] * 2.0
            elif recent_count > 0:
                # Penalidade moderada para repetição
                penalties[i] = self.filter_settings['problematic_penalty'] * 0.5 * recent_count

        return penalties

    def get_filter_report(self) -> Dict[str, Any]:
        """Retorna relatório do filtro"""
        return {
            'repetition_history': self.repetition_history.copy(),
            'problematic_tokens_detected': self.problematic_tokens_detected.copy(),
            'filter_settings': self.filter_settings.copy(),
            'total_filter_applications': len(self.repetition_history)
        }


class TokenPredictionEngine:
    """
    Motor de Predição de Tokens com Filtro Quântico

    Combina todos os componentes para predição robusta de tokens
    com prevenção de loops e melhor qualidade gramatical.
    """

    def __init__(self,
                 embed_dim: int = 256,
                 vocab_size: int = 50257,
                 quantum_vocab_representations: Optional[torch.Tensor] = None,
                 device: str = "cpu"):
        """
        Inicializa o motor de predição de tokens.

        Args:
            embed_dim: Dimensão do embedding
            vocab_size: Tamanho do vocabulário
            quantum_vocab_representations: Representações quânticas do vocabulário
            device: Dispositivo para computação
        """
        self.device = device
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Inicializar filtro quântico
        self.quantum_filter = QuantumDecodingFilter(
            embed_dim, vocab_size, quantum_vocab_representations
        )

        # Estado da geração
        self.generation_history = []

        print("🚀 TokenPredictionEngine inicializado")
        print(f"   📊 Embed dim: {embed_dim}")
        print(f"   📚 Vocab size: {vocab_size}")
        print(f"   🔧 Quantum filter: {self.quantum_filter is not None}")

    def predict_next_token(self,
                          logits: torch.Tensor,
                          previous_tokens: List[int],
                          temperature: float = 1.0,
                          top_k: int = 50) -> Dict[str, Any]:
        """
        Prediz o próximo token com filtro quântico.

        Args:
            logits: Logits do modelo [vocab_size]
            previous_tokens: Lista de tokens anteriores
            temperature: Temperatura para sampling
            top_k: Número de candidatos top-K

        Returns:
            prediction_result: Resultado da predição
        """
        # Selecionar top-K candidatos
        top_logits, top_indices = torch.topk(logits, top_k)

        # Aplicar filtro quântico
        filtered_logits = self.quantum_filter.filter_candidates(
            top_logits, top_indices, previous_tokens
        )

        # Aplicar temperatura
        if temperature > 0:
            filtered_logits = filtered_logits / temperature
            probs = torch.softmax(filtered_logits, dim=0)

            # Sample do distribution
            selected_idx = torch.multinomial(probs, 1).item()
        else:
            # Seleção greedy
            selected_idx = torch.argmax(filtered_logits).item()

        # Obter token selecionado
        selected_token = top_indices[selected_idx].item()
        selected_prob = torch.softmax(filtered_logits, dim=0)[selected_idx].item()

        # Atualizar histórico
        self.generation_history.append({
            'token': selected_token,
            'probability': selected_prob,
            'timestamp': len(self.generation_history)
        })

        # Gerar relatório
        filter_report = self.quantum_filter.get_filter_report()

        return {
            'selected_token': selected_token,
            'selected_probability': selected_prob,
            'filtered_logits': filtered_logits.tolist(),
            'candidate_indices': top_indices.tolist(),
            'filter_report': filter_report,
            'generation_step': len(self.generation_history),
            'method': 'quantum_filtered_prediction'
        }

    def reset_state(self):
        """Reseta o estado do motor"""
        self.generation_history = []
        # O filtro mantém seu próprio estado interno


# Função de interface principal
def create_quantum_token_filter(embed_dim: int = 256,
                               vocab_size: int = 50257,
                               quantum_vocab_representations: Optional[torch.Tensor] = None,
                               device: str = "cpu") -> TokenPredictionEngine:
    """
    Cria uma instância do motor de predição de tokens com filtro quântico.

    Args:
        embed_dim: Dimensão do embedding
        vocab_size: Tamanho do vocabulário
        quantum_vocab_representations: Representações quânticas do vocabulário
        device: Dispositivo para computação

    Returns:
        TokenPredictionEngine: Motor de predição configurado
    """
    return TokenPredictionEngine(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        quantum_vocab_representations=quantum_vocab_representations,
        device=device
    )


if __name__ == "__main__":
    # Teste do sistema
    print("🧪 Testando Quantum Token Filter...")

    # Criar motor de predição
    prediction_engine = create_quantum_token_filter()

    # Simular logits
    vocab_size = 1000
    logits = torch.randn(vocab_size)
    previous_tokens = [1000, 1007, 1013]  # the, is, i

    # Predizer próximo token
    result = prediction_engine.predict_next_token(
        logits, previous_tokens, temperature=0.8
    )

    print("\n" + "="*60)
    print("RESULTADO DA PREDIÇÃO:")
    print("="*60)
    print(f"Token Selecionado: {result['selected_token']}")
    print(f"Probabilidade: {result['selected_probability']:.4f}")
    print(f"Método: {result['method']}")
    print(f"Passo de Geração: {result['generation_step']}")
    print("\n📋 Relatório do Filtro:")
    print(f"Histórico de Repetição: {result['filter_report']['repetition_history']}")
    print(f"Tokens Problemáticos Detectados: {len(result['filter_report']['problematic_tokens_detected'])}")
    print("="*60)