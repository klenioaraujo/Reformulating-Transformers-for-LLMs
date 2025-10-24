import torch
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter, defaultdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ΨQRHSystem.configs.SystemConfig import SystemConfig


class VocabularyMaker:
    """
    VocabularyMaker - Cria vocabulários nativos dinamicamente

    Permite criação programática de vocabulários ΨQRH com diferentes
    estratégias: de modelos fonte, semânticos, quânticos, etc.
    """

    def __init__(self, base_vocab_path: Optional[str] = None):
        """
        Inicializa VocabularyMaker

        Args:
            base_vocab_path: Caminho para vocabulário base (opcional)
        """
        self.base_vocab_path = base_vocab_path or "data/native_vocab.json"
        self.created_vocabularies = []
        self.quantum_features_cache = {}

        print("🔧 VocabularyMaker inicializado - criação dinâmica de vocabulários ΨQRH")

    def create_from_source_model(self, source_model: str, vocab_size: int = 512) -> Dict[str, Any]:
        """
        Cria vocabulário a partir de modelo fonte existente

        Args:
            source_model: Nome ou caminho do modelo fonte
            vocab_size: Tamanho desejado do vocabulário

        Returns:
            Vocabulário criado
        """
        try:
            print(f"🔧 Criando vocabulário do modelo fonte: {source_model}")

            # Estratégias de extração baseadas no tipo de modelo
            if "gpt2" in source_model.lower():
                vocab = self._extract_gpt2_vocab(source_model, vocab_size)
            elif "bert" in source_model.lower():
                vocab = self._extract_bert_vocab(source_model, vocab_size)
            else:
                # Estratégia genérica
                vocab = self._extract_generic_vocab(source_model, vocab_size)

            # Adicionar metadados
            vocab['metadata'] = {
                'created_by': 'VocabularyMaker',
                'source_model': source_model,
                'creation_method': 'from_source_model',
                'created_at': datetime.now().isoformat(),
                'vocab_size': len(vocab.get('tokens', []))
            }

            # Registrar vocabulário criado
            vocab_info = {
                'id': f"vocab_{len(self.created_vocabularies)}",
                'type': 'from_source_model',
                'source': source_model,
                'size': len(vocab.get('tokens', [])),
                'created_at': datetime.now().isoformat(),
                'vocabulary': vocab
            }
            self.created_vocabularies.append(vocab_info)

            print(f"✅ Vocabulário criado: {vocab_info['size']} tokens do modelo {source_model}")
            return vocab

        except Exception as e:
            print(f"❌ Erro na criação do vocabulário: {e}")
            raise

    def create_semantic_vocab(self, base_words: List[str], expansion_factor: int = 2) -> Dict[str, Any]:
        """
        Cria vocabulário semântico expandido a partir de palavras base

        Args:
            base_words: Lista de palavras base
            expansion_factor: Fator de expansão semântica

        Returns:
            Vocabulário semântico criado
        """
        try:
            print(f"🔧 Criando vocabulário semântico: {len(base_words)} palavras base")

            # Criar vocabulário base
            vocab = {
                'tokens': base_words.copy(),
                'word_to_idx': {word: i for i, word in enumerate(base_words)},
                'idx_to_word': {i: word for i, word in enumerate(base_words)}
            }

            # Expansão semântica
            if expansion_factor > 1:
                expanded_words = self._expand_semantic_vocab(base_words, expansion_factor)
                vocab['tokens'].extend(expanded_words)

                # Recriar mapeamentos
                vocab['word_to_idx'] = {word: i for i, word in enumerate(vocab['tokens'])}
                vocab['idx_to_word'] = {i: word for i, word in enumerate(vocab['tokens'])}

            # Adicionar propriedades semânticas
            vocab['semantic_properties'] = self._analyze_semantic_properties(vocab['tokens'])

            # Metadados
            vocab['metadata'] = {
                'created_by': 'VocabularyMaker',
                'creation_method': 'semantic_expansion',
                'base_words_count': len(base_words),
                'expansion_factor': expansion_factor,
                'created_at': datetime.now().isoformat(),
                'vocab_size': len(vocab['tokens'])
            }

            # Registrar
            vocab_info = {
                'id': f"vocab_{len(self.created_vocabularies)}",
                'type': 'semantic',
                'base_words': len(base_words),
                'expansion': expansion_factor,
                'size': len(vocab['tokens']),
                'created_at': datetime.now().isoformat(),
                'vocabulary': vocab
            }
            self.created_vocabularies.append(vocab_info)

            print(f"✅ Vocabulário semântico criado: {vocab_info['size']} tokens")
            return vocab

        except Exception as e:
            print(f"❌ Erro na criação do vocabulário semântico: {e}")
            raise

    def create_quantum_vocab(self, quantum_features: torch.Tensor, vocab_size: int = 256) -> Dict[str, Any]:
        """
        Cria vocabulário baseado em features quânticas

        Args:
            quantum_features: Features quânticas [num_features, feature_dim]
            vocab_size: Tamanho desejado do vocabulário

        Returns:
            Vocabulário quântico criado
        """
        try:
            print(f"🔧 Criando vocabulário quântico: {quantum_features.shape[0]} features")

            # Gerar tokens baseados em padrões quânticos
            tokens = []
            quantum_patterns = self._extract_quantum_patterns(quantum_features)

            # Criar tokens representativos dos padrões
            for i, pattern in enumerate(quantum_patterns[:vocab_size]):
                # Representação simbólica do padrão quântico
                token = f"Ψ_{i:03d}_{pattern['complexity']:.2f}"
                tokens.append(token)

            # Criar vocabulário
            vocab = {
                'tokens': tokens,
                'word_to_idx': {token: i for i, token in enumerate(tokens)},
                'idx_to_word': {i: token for i, token in enumerate(tokens)},
                'quantum_patterns': quantum_patterns[:vocab_size]
            }

            # Adicionar propriedades quânticas
            vocab['quantum_properties'] = {
                'feature_dim': quantum_features.shape[1],
                'num_patterns': len(quantum_patterns),
                'complexity_distribution': [p['complexity'] for p in quantum_patterns[:vocab_size]]
            }

            # Metadados
            vocab['metadata'] = {
                'created_by': 'VocabularyMaker',
                'creation_method': 'quantum_based',
                'quantum_features_shape': list(quantum_features.shape),
                'created_at': datetime.now().isoformat(),
                'vocab_size': len(tokens)
            }

            # Registrar
            vocab_info = {
                'id': f"vocab_{len(self.created_vocabularies)}",
                'type': 'quantum',
                'features': quantum_features.shape[0],
                'size': len(tokens),
                'created_at': datetime.now().isoformat(),
                'vocabulary': vocab
            }
            self.created_vocabularies.append(vocab_info)

            print(f"✅ Vocabulário quântico criado: {vocab_info['size']} tokens")
            return vocab

        except Exception as e:
            print(f"❌ Erro na criação do vocabulário quântico: {e}")
            raise

    def create_hybrid_vocab(self, text_sources: List[str], quantum_features: Optional[torch.Tensor] = None,
                           target_size: int = 512) -> Dict[str, Any]:
        """
        Cria vocabulário híbrido combinando texto e features quânticas

        Args:
            text_sources: Lista de textos fonte
            quantum_features: Features quânticas (opcional)
            target_size: Tamanho alvo do vocabulário

        Returns:
            Vocabulário híbrido criado
        """
        try:
            print(f"🔧 Criando vocabulário híbrido: {len(text_sources)} fontes de texto")

            # Extrair vocabulário de texto
            text_vocab = self._extract_from_texts(text_sources, target_size // 2)

            # Adicionar componente quântica se disponível
            quantum_tokens = []
            if quantum_features is not None:
                quantum_vocab = self.create_quantum_vocab(quantum_features, target_size // 2)
                quantum_tokens = quantum_vocab['tokens']

            # Combinar vocabulários
            all_tokens = text_vocab['tokens'] + quantum_tokens

            # Remover duplicatas mantendo ordem
            seen = set()
            unique_tokens = []
            for token in all_tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)

            # Limitar ao tamanho alvo
            unique_tokens = unique_tokens[:target_size]

            # Criar vocabulário final
            vocab = {
                'tokens': unique_tokens,
                'word_to_idx': {token: i for i, token in enumerate(unique_tokens)},
                'idx_to_word': {i: token for i, token in enumerate(unique_tokens)},
                'components': {
                    'text_tokens': len(text_vocab['tokens']),
                    'quantum_tokens': len(quantum_tokens),
                    'total_unique': len(unique_tokens)
                }
            }

            # Metadados
            vocab['metadata'] = {
                'created_by': 'VocabularyMaker',
                'creation_method': 'hybrid',
                'text_sources_count': len(text_sources),
                'has_quantum_component': quantum_features is not None,
                'created_at': datetime.now().isoformat(),
                'vocab_size': len(unique_tokens)
            }

            # Registrar
            vocab_info = {
                'id': f"vocab_{len(self.created_vocabularies)}",
                'type': 'hybrid',
                'text_sources': len(text_sources),
                'quantum_features': quantum_features.shape[0] if quantum_features is not None else 0,
                'size': len(unique_tokens),
                'created_at': datetime.now().isoformat(),
                'vocabulary': vocab
            }
            self.created_vocabularies.append(vocab_info)

            print(f"✅ Vocabulário híbrido criado: {vocab_info['size']} tokens únicos")
            return vocab

        except Exception as e:
            print(f"❌ Erro na criação do vocabulário híbrido: {e}")
            raise

    def _extract_gpt2_vocab(self, model_name: str, vocab_size: int) -> Dict[str, Any]:
        """Extrai vocabulário de modelo GPT-2"""
        # Implementação simplificada - em produção usaria tokenizers reais
        base_tokens = [
            "<|endoftext|>", "<|padding|>", "the", "of", "and", "in", "to", "a", "is", "that",
            "for", "on", "with", "as", "by", "an", "this", "it", "from", "at", "are", "be",
            "have", "or", "was", "but", "not", "what", "all", "when", "can", "if", "there",
            "so", "out", "about", "who", "get", "which", "go", "me", "make", "good", "no",
            "will", "just", "think", "time", "see", "some", "other", "than", "then", "now"
        ]

        # Expandir para tamanho desejado
        while len(base_tokens) < vocab_size:
            # Adicionar variações e compostos
            for token in base_tokens[:vocab_size - len(base_tokens)]:
                if len(base_tokens) >= vocab_size:
                    break
                # Adicionar versão capitalizada
                if token[0].islower():
                    base_tokens.append(token.capitalize())

        tokens = base_tokens[:vocab_size]

        return {
            'tokens': tokens,
            'word_to_idx': {token: i for i, token in enumerate(tokens)},
            'idx_to_word': {i: token for i, token in enumerate(tokens)},
            'model_type': 'gpt2'
        }

    def _extract_bert_vocab(self, model_name: str, vocab_size: int) -> Dict[str, Any]:
        """Extrai vocabulário de modelo BERT"""
        # Implementação simplificada
        base_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "a", "an", "of", "to",
            "and", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "with",
            "as", "I", "his", "they", "be", "at", "one", "have", "this", "from", "or", "had",
            "by", "hot", "but", "some", "what", "there", "we", "can", "out", "other", "were"
        ]

        tokens = base_tokens[:vocab_size]

        return {
            'tokens': tokens,
            'word_to_idx': {token: i for i, token in enumerate(tokens)},
            'idx_to_word': {i: token for i, token in enumerate(tokens)},
            'model_type': 'bert'
        }

    def _extract_generic_vocab(self, model_name: str, vocab_size: int) -> Dict[str, Any]:
        """Extração genérica de vocabulário"""
        # Criar vocabulário genérico baseado em frequência
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them", "see"
        ]

        tokens = common_words[:vocab_size]

        return {
            'tokens': tokens,
            'word_to_idx': {token: i for i, token in enumerate(tokens)},
            'idx_to_word': {i: token for i, token in enumerate(tokens)},
            'model_type': 'generic'
        }

    def _expand_semantic_vocab(self, base_words: List[str], expansion_factor: int) -> List[str]:
        """Expande vocabulário semanticamente"""
        expanded = []

        # Regras de expansão simples
        prefixes = ["un", "in", "dis", "re", "pre", "post", "anti", "auto"]
        suffixes = ["ing", "ed", "er", "est", "ly", "ness", "ment", "tion", "sion"]

        target_expansions = int(len(base_words) * (expansion_factor - 1))

        for word in base_words:
            if len(expanded) >= target_expansions:
                break

            # Adicionar prefixos
            for prefix in prefixes:
                if len(expanded) >= target_expansions:
                    break
                expanded_word = prefix + word
                if expanded_word not in base_words and len(expanded_word) < 15:
                    expanded.append(expanded_word)

            # Adicionar sufixos
            for suffix in suffixes:
                if len(expanded) >= target_expansions:
                    break
                expanded_word = word + suffix
                if expanded_word not in base_words and len(expanded_word) < 15:
                    expanded.append(expanded_word)

        return expanded[:target_expansions]

    def _analyze_semantic_properties(self, tokens: List[str]) -> Dict[str, Any]:
        """Analisa propriedades semânticas do vocabulário"""
        properties = {
            'avg_length': np.mean([len(token) for token in tokens]),
            'has_capitalized': any(token[0].isupper() for token in tokens),
            'has_numbers': any(any(c.isdigit() for c in token) for token in tokens),
            'has_punctuation': any(any(not c.isalnum() for c in token) for token in tokens),
            'length_distribution': Counter(len(token) for token in tokens)
        }

        return properties

    def _extract_quantum_patterns(self, quantum_features: torch.Tensor) -> List[Dict[str, Any]]:
        """Extrai padrões de features quânticas"""
        patterns = []

        for i in range(min(len(quantum_features), 1000)):  # Limitar para performance
            feature = quantum_features[i]

            # Calcular propriedades do padrão quântico
            complexity = torch.std(feature).item()  # Variabilidade como complexidade
            energy = torch.sum(feature.abs() ** 2).item()
            coherence = torch.mean(torch.abs(feature)).item() / (torch.std(feature).item() + 1e-10)

            patterns.append({
                'index': i,
                'complexity': complexity,
                'energy': energy,
                'coherence': coherence,
                'feature_vector': feature.tolist()
            })

        # Ordenar por complexidade
        patterns.sort(key=lambda x: x['complexity'], reverse=True)

        return patterns

    def _extract_from_texts(self, texts: List[str], max_vocab: int) -> Dict[str, Any]:
        """Extrai vocabulário de textos"""
        # Contar frequência de palavras
        word_counts = Counter()

        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        # Selecionar palavras mais frequentes
        most_common = word_counts.most_common(max_vocab)
        tokens = [word for word, count in most_common]

        return {
            'tokens': tokens,
            'word_to_idx': {token: i for i, token in enumerate(tokens)},
            'idx_to_word': {i: token for i, token in enumerate(tokens)},
            'frequencies': dict(most_common)
        }

    def save_vocabulary(self, vocab: Dict[str, Any], vocab_path: str):
        """
        Salva vocabulário em arquivo

        Args:
            vocab: Vocabulário a salvar
            vocab_path: Caminho para salvar
        """
        try:
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False, default=str)

            print(f"💾 Vocabulário salvo: {vocab_path}")

        except Exception as e:
            print(f"❌ Erro ao salvar vocabulário: {e}")
            raise

    def load_vocabulary(self, vocab_path: str) -> Dict[str, Any]:
        """
        Carrega vocabulário salvo

        Args:
            vocab_path: Caminho do vocabulário

        Returns:
            Vocabulário carregado
        """
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)

            print(f"📂 Vocabulário carregado: {vocab_path}")
            return vocab

        except Exception as e:
            print(f"❌ Erro ao carregar vocabulário: {e}")
            raise

    def list_created_vocabularies(self) -> List[Dict[str, Any]]:
        """
        Lista vocabulários criados nesta sessão

        Returns:
            Lista de informações dos vocabulários criados
        """
        return self.created_vocabularies.copy()

    def validate_vocabulary(self, vocab: Dict[str, Any]) -> bool:
        """
        Valida estrutura do vocabulário

        Args:
            vocab: Vocabulário a validar

        Returns:
            True se válido
        """
        required_keys = ['tokens', 'word_to_idx', 'idx_to_word']

        for key in required_keys:
            if key not in vocab:
                print(f"❌ Chave obrigatória faltando: {key}")
                return False

        # Verificar consistência
        if len(vocab['tokens']) != len(vocab['word_to_idx']):
            print("❌ Inconsistência entre tokens e word_to_idx")
            return False

        if len(vocab['tokens']) != len(vocab['idx_to_word']):
            print("❌ Inconsistência entre tokens e idx_to_word")
            return False

        # Verificar mapeamentos
        for i, token in enumerate(vocab['tokens']):
            if vocab['word_to_idx'].get(token) != i:
                print(f"❌ Mapeamento word_to_idx incorreto para token '{token}'")
                return False

            if vocab['idx_to_word'].get(i) != token:
                print(f"❌ Mapeamento idx_to_word incorreto para índice {i}")
                return False

        return True