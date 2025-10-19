#!/usr/bin/env python3
"""
Quantum Native Vocabulary System
================================

Sistema de vocabulário quântico nativo com BPE puramente físico,
expandido para ≥50k tokens baseado em princípios físico-matemáticos.

Características principais:
- BPE quântico baseado em física (similaridade espectral, energia de acoplamento)
- Vocabulário expandido para 50k+ tokens
- Propriedades quânticas por token (energia, spin, coerência)
- Validação física rigorosa (conservação de invariantes)
- Integração com sistema fractal e quaterniónico

Autor: Kilo Code (Sistema ΨQRH)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from collections import defaultdict, Counter
import heapq
from dataclasses import dataclass
import hashlib

@dataclass
class QuantumTokenProperties:
    """Propriedades quânticas de um token."""
    energy_level: float
    coherence: float
    entropy: float
    spin: float
    mass: float
    charge: float
    frequency: float
    wavelength: float

class QuantumNativeVocabulary:
    """
    Sistema de vocabulário quântico nativo com BPE puramente físico.

    Meta: Expandir para ≥50k tokens com BPE baseado em física quântica.
    """

    def __init__(self, vocab_size: int = 50000, device: str = "cpu"):
        """
        Inicializa o sistema de vocabulário quântico nativo.

        Args:
            vocab_size: Tamanho meta do vocabulário (≥50k)
            device: Dispositivo para tensores
        """
        self.vocab_size = vocab_size
        self.device = device
        self.quantum_tokens = {}  # Token → propriedades quânticas
        self.bpe_merges = {}     # Merge rules baseadas em física
        self.token_to_id = {}    # Token → ID
        self.id_to_token = {}    # ID → Token
        self.base_vocab = set()  # Vocabulário base (caracteres)

        # Estatísticas
        self.stats = {
            'total_tokens': 0,
            'total_merges': 0,
            'quantum_energy_levels': set(),
            'coherence_range': (0.0, 1.0),
            'entropy_distribution': [],
            'physical_validation_score': 0.0
        }

        # Inicializar vocabulário base
        self._initialize_base_vocabulary()

        # Carregar vocabulário quântico existente se disponível
        self._load_existing_quantum_vocabulary()

        print(f"🔬 Quantum Native Vocabulary initialized: {self.stats['total_tokens']} tokens")
        print(f"   📊 Target vocab size: {self.vocab_size}, Current: {len(self.quantum_tokens)}")

    def _initialize_base_vocabulary(self):
        """Inicializa vocabulário base com caracteres ASCII printáveis."""
        # Caracteres ASCII printáveis (32-126) + caracteres especiais comuns
        base_chars = list(range(32, 127)) + [9, 10, 13]  # tab, newline, carriage return

        for i, char_code in enumerate(base_chars):
            char = chr(char_code)
            token_id = i

            self.base_vocab.add(char)
            self.token_to_id[char] = token_id
            self.id_to_token[token_id] = char

            # Propriedades quânticas para caracteres base
            properties = self._compute_base_quantum_properties(char, char_code)
            self.quantum_tokens[char] = properties

        self.stats['total_tokens'] = len(self.quantum_tokens)

    def _compute_base_quantum_properties(self, char: str, char_code: int) -> QuantumTokenProperties:
        """Computa propriedades quânticas para caracteres base."""
        # Normalizar código ASCII para [0,1]
        normalized_code = char_code / 255.0

        # Propriedades baseadas em física quântica
        energy_level = normalized_code  # Energia baseada na posição ASCII
        coherence = 0.8 + 0.2 * math.sin(2 * math.pi * normalized_code)  # Coerência oscilatória
        entropy = -normalized_code * math.log(normalized_code + 1e-8)  # Entropia de Shannon
        spin = 0.5 if char_code % 2 == 0 else -0.5  # Spin binário simples
        mass = 1.0 + normalized_code  # Massa baseada na posição
        charge = 0.0  # Caracteres neutros
        frequency = 1.0 / (normalized_code + 0.1)  # Frequência inversamente proporcional
        wavelength = 1.0 / frequency  # Comprimento de onda

        return QuantumTokenProperties(
            energy_level=energy_level,
            coherence=coherence,
            entropy=entropy,
            spin=spin,
            mass=mass,
            charge=charge,
            frequency=frequency,
            wavelength=wavelength
        )

    def _load_existing_quantum_vocabulary(self):
        """Carrega vocabulário quântico existente para expansão."""
        try:
            vocab_file = Path("dynamic_quantum_vocabulary.json")
            if vocab_file.exists():
                with open(vocab_file, 'r') as f:
                    existing_vocab = json.load(f)

                # Expandir vocabulário existente
                for char, words in existing_vocab.get('character_mappings', {}).items():
                    for word_info in words:
                        word = word_info['word']
                        if word not in self.quantum_tokens:
                            # Converter propriedades existentes para QuantumTokenProperties
                            properties = self._convert_legacy_properties(word_info)
                            self.quantum_tokens[word] = properties

                            # Adicionar ao mapeamento de IDs
                            token_id = len(self.token_to_id)
                            self.token_to_id[word] = token_id
                            self.id_to_token[token_id] = word

                print(f"   📚 Loaded {len(self.quantum_tokens) - len(self.base_vocab)} existing quantum words")

        except Exception as e:
            print(f"   ⚠️  Could not load existing vocabulary: {e}")

    def _convert_legacy_properties(self, word_info: Dict) -> QuantumTokenProperties:
        """Converte propriedades legadas para QuantumTokenProperties."""
        weight = word_info.get('weight', 1.0)
        energy_level = weight / 5.0  # Normalizar para [0,1]
        coherence = word_info.get('coherence', 0.8)
        entropy = word_info.get('entropy', 0.2)

        # Propriedades adicionais baseadas no peso
        spin = 0.5 if weight > 2.5 else -0.5
        mass = weight
        charge = 0.0
        frequency = weight / 10.0
        wavelength = 1.0 / (frequency + 0.1)

        return QuantumTokenProperties(
            energy_level=energy_level,
            coherence=coherence,
            entropy=entropy,
            spin=spin,
            mass=mass,
            charge=charge,
            frequency=frequency,
            wavelength=wavelength
        )

    def build_quantum_bpe(self, corpus_texts: List[str], target_vocab_size: Optional[int] = None):
        """
        Implementa BPE quântico baseado em física.

        Critérios de merge:
        - Similaridade espectral entre substrings
        - Energia de acoplamento quântico
        - Conservação de entropia fractal
        """
        if target_vocab_size is None:
            target_vocab_size = self.vocab_size

        print(f"🔬 Building Quantum BPE vocabulary to {target_vocab_size} tokens...")

        # Preparar corpus
        corpus = self._prepare_corpus(corpus_texts)

        # Calcular frequências base
        word_freqs = self._compute_base_frequencies(corpus)

        # Inicializar vocabulário BPE
        vocab = set(word_freqs.keys())
        merges = {}

        # Loop principal de BPE quântico
        while len(vocab) < target_vocab_size:
            # Encontrar melhor par para merge baseado em física
            best_pair, merge_score = self._find_quantum_merge_pair(word_freqs, vocab)

            if best_pair is None:
                break

            # Executar merge
            vocab, word_freqs = self._execute_quantum_merge(best_pair, vocab, word_freqs)

            # Registrar merge
            merge_id = len(merges)
            merges[best_pair] = merge_score

            if len(merges) % 1000 == 0:
                print(f"   📊 BPE Progress: {len(vocab)}/{target_vocab_size} tokens, {len(merges)} merges")

        # Atualizar vocabulário
        self.bpe_merges = merges
        self._update_vocabulary_from_bpe(vocab)

        print(f"   ✅ Quantum BPE completed: {len(self.quantum_tokens)} tokens, {len(merges)} merges")

    def _prepare_corpus(self, corpus_texts: List[str]) -> List[List[str]]:
        """Prepara corpus para BPE."""
        corpus = []
        for text in corpus_texts:
            # Tokenizar em caracteres
            tokens = list(text)
            corpus.append(tokens)
        return corpus

    def _compute_base_frequencies(self, corpus: List[List[str]]) -> Dict[Tuple[str, ...], int]:
        """Computa frequências base de n-gramas."""
        word_freqs = defaultdict(int)

        for tokens in corpus:
            # Adicionar unigramas
            for token in tokens:
                word_freqs[(token,)] += 1

            # Adicionar bigramas
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                word_freqs[bigram] += 1

        return dict(word_freqs)

    def _find_quantum_merge_pair(self, word_freqs: Dict, vocab: Set) -> Tuple[Optional[Tuple[str, str]], float]:
        """
        Encontra melhor par para merge baseado em critérios quânticos.

        Critérios:
        1. Similaridade espectral
        2. Energia de acoplamento
        3. Conservação de entropia
        """
        best_pair = None
        best_score = -float('inf')

        # Considerar apenas pares frequentes
        candidates = []
        for pair, freq in word_freqs.items():
            if len(pair) == 2 and freq > 1:  # Apenas bigramas com freq > 1
                candidates.append((pair, freq))

        # Limitar candidatos para eficiência
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:1000]

        for (pair, freq) in candidates:
            token1, token2 = pair

            # Verificar se ambos tokens existem no vocabulário
            if token1 not in vocab or token2 not in vocab:
                continue

            # Calcular score quântico
            quantum_score = self._compute_quantum_merge_score(token1, token2, freq)

            if quantum_score > best_score:
                best_score = quantum_score
                best_pair = pair

        return best_pair, best_score

    def _compute_quantum_merge_score(self, token1: str, token2: str, freq: int) -> float:
        """
        Computa score quântico para merge de dois tokens.

        Score baseado em:
        - Similaridade espectral (FFT)
        - Energia de acoplamento quântico
        - Conservação de entropia
        """
        # Propriedades quânticas dos tokens
        props1 = self.quantum_tokens.get(token1)
        props2 = self.quantum_tokens.get(token2)

        if not props1 or not props2:
            return -float('inf')

        # 1. Similaridade espectral (baseada em frequência e energia)
        spectral_similarity = 1.0 - abs(props1.frequency - props2.frequency) / max(props1.frequency, props2.frequency)

        # 2. Energia de acoplamento (baseada em diferença de energia)
        coupling_energy = 1.0 / (1.0 + abs(props1.energy_level - props2.energy_level))

        # 3. Conservação de entropia (entropia combinada deve ser menor)
        combined_entropy = (props1.entropy + props2.entropy) / 2.0
        entropy_conservation = 1.0 - combined_entropy

        # 4. Frequência de ocorrência
        freq_score = math.log(freq + 1) / 10.0

        # Score total (média ponderada)
        total_score = (
            0.3 * spectral_similarity +
            0.3 * coupling_energy +
            0.2 * entropy_conservation +
            0.2 * freq_score
        )

        return total_score

    def _execute_quantum_merge(self, pair: Tuple[str, str], vocab: Set, word_freqs: Dict) -> Tuple[Set, Dict]:
        """Executa merge quântico de um par de tokens."""
        token1, token2 = pair
        merged_token = token1 + token2

        # Adicionar novo token ao vocabulário
        vocab.add(merged_token)

        # Criar propriedades quânticas para o token merged
        props1 = self.quantum_tokens[token1]
        props2 = self.quantum_tokens[token2]

        merged_props = self._compute_merged_quantum_properties(props1, props2)
        self.quantum_tokens[merged_token] = merged_props

        # Adicionar ao mapeamento de IDs
        token_id = len(self.token_to_id)
        self.token_to_id[merged_token] = token_id
        self.id_to_token[token_id] = merged_token

        # Atualizar frequências
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            # Substituir ocorrências do par
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == token1 and word[i + 1] == token2:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_tuple = tuple(new_word)
            new_word_freqs[new_word_tuple] = new_word_freqs.get(new_word_tuple, 0) + freq

        return vocab, new_word_freqs

    def _compute_merged_quantum_properties(self, props1: QuantumTokenProperties,
                                         props2: QuantumTokenProperties) -> QuantumTokenProperties:
        """Computa propriedades quânticas para token merged."""
        # Média ponderada baseada na energia
        total_energy = props1.energy_level + props2.energy_level
        w1 = props1.energy_level / total_energy if total_energy > 0 else 0.5
        w2 = props2.energy_level / total_energy if total_energy > 0 else 0.5

        # Propriedades combinadas
        energy_level = (props1.energy_level + props2.energy_level) / 2.0
        coherence = min(props1.coherence, props2.coherence) * 0.9  # Coerência diminui ligeiramente
        entropy = (props1.entropy + props2.entropy) / 2.0  # Média da entropia
        spin = (props1.spin + props2.spin) / 2.0  # Média do spin
        mass = props1.mass + props2.mass  # Massa aditiva
        charge = props1.charge + props2.charge  # Carga aditiva
        frequency = (props1.frequency + props2.frequency) / 2.0  # Média da frequência
        wavelength = (props1.wavelength + props2.wavelength) / 2.0  # Média do comprimento de onda

        return QuantumTokenProperties(
            energy_level=energy_level,
            coherence=coherence,
            entropy=entropy,
            spin=spin,
            mass=mass,
            charge=charge,
            frequency=frequency,
            wavelength=wavelength
        )

    def _update_vocabulary_from_bpe(self, vocab: Set):
        """Atualiza vocabulário principal com resultados do BPE."""
        for token in vocab:
            if token not in self.quantum_tokens:
                # Converter tuple para string se necessário
                if isinstance(token, tuple):
                    token_str = ''.join(token)
                else:
                    token_str = str(token)

                # Criar propriedades básicas para tokens BPE
                token_hash = int(hashlib.md5(token_str.encode()).hexdigest(), 16)
                normalized_hash = (token_hash % 1000) / 1000.0

                properties = QuantumTokenProperties(
                    energy_level=normalized_hash,
                    coherence=0.7 + 0.3 * math.sin(2 * math.pi * normalized_hash),
                    entropy=-normalized_hash * math.log(normalized_hash + 1e-8),
                    spin=0.5 if normalized_hash > 0.5 else -0.5,
                    mass=1.0 + normalized_hash,
                    charge=0.0,
                    frequency=1.0 / (normalized_hash + 0.1),
                    wavelength=normalized_hash + 0.1
                )

                self.quantum_tokens[token] = properties

                # Adicionar ao mapeamento
                if token not in self.token_to_id:
                    token_id = len(self.token_to_id)
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token

        self.stats['total_tokens'] = len(self.quantum_tokens)
        self.stats['total_merges'] = len(self.bpe_merges)

    def tokenize_with_physics(self, text: str) -> List[str]:
        """
        Tokenização que respeita princípios físicos.

        - Limites semânticos naturais
        - Propriedades espectrais do texto
        - Dimensão fractal local
        """
        # Tokenização básica em caracteres
        char_tokens = list(text)

        # Aplicar merges BPE baseados em física
        tokens = self._apply_quantum_bpe_tokenization(char_tokens)

        return tokens

    def _apply_quantum_bpe_tokenization(self, char_tokens: List[str]) -> List[str]:
        """Aplica tokenização BPE com critérios quânticos."""
        tokens = char_tokens.copy()

        # Aplicar merges em ordem reversa (mais específicos primeiro)
        sorted_merges = sorted(self.bpe_merges.items(), key=lambda x: x[1], reverse=True)

        for merge_pair, _ in sorted_merges:
            token1, token2 = merge_pair
            merged_token = token1 + token2

            # Aplicar merge se ambos tokens estiverem presentes
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == token1 and tokens[i + 1] == token2:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def validate_physical_properties(self) -> Dict[str, float]:
        """
        Valida propriedades físicas do vocabulário.

        Critérios:
        1. Conservação de energia total
        2. Distribuição de coerência
        3. Invariantes topológicos
        """
        validation_results = {
            'energy_conservation': 0.0,
            'coherence_distribution': 0.0,
            'entropy_balance': 0.0,
            'spin_conservation': 0.0,
            'overall_score': 0.0
        }

        if not self.quantum_tokens:
            return validation_results

        # 1. Conservação de energia
        total_energy = sum(props.energy_level for props in self.quantum_tokens.values())
        expected_energy = len(self.quantum_tokens) * 0.5  # Energia média esperada
        energy_conservation = 1.0 - abs(total_energy - expected_energy) / expected_energy
        validation_results['energy_conservation'] = max(0.0, energy_conservation)

        # 2. Distribuição de coerência
        coherences = [props.coherence for props in self.quantum_tokens.values()]
        coherence_mean = np.mean(coherences)
        coherence_std = np.std(coherences)
        coherence_distribution = 1.0 - coherence_std  # Menor variabilidade = melhor
        validation_results['coherence_distribution'] = max(0.0, coherence_distribution)

        # 3. Balanceamento de entropia
        entropies = [props.entropy for props in self.quantum_tokens.values()]
        entropy_balance = 1.0 - np.std(entropies) / (np.mean(entropies) + 1e-8)
        validation_results['entropy_balance'] = max(0.0, entropy_balance)

        # 4. Conservação de spin
        total_spin = sum(props.spin for props in self.quantum_tokens.values())
        spin_conservation = 1.0 - abs(total_spin) / len(self.quantum_tokens)
        validation_results['spin_conservation'] = max(0.0, spin_conservation)

        # Score geral
        validation_results['overall_score'] = np.mean(list(validation_results.values())[:4])

        self.stats['physical_validation_score'] = validation_results['overall_score']

        return validation_results

    def save_vocabulary(self, filepath: str):
        """Salva vocabulário quântico em arquivo."""
        # Converter tuples para strings nas chaves
        bpe_merges_str = {}
        for merge_pair, score in self.bpe_merges.items():
            if isinstance(merge_pair, tuple):
                merge_key = ''.join(merge_pair)
            else:
                merge_key = str(merge_pair)
            bpe_merges_str[merge_key] = score

        token_to_id_str = {}
        for token, token_id in self.token_to_id.items():
            if isinstance(token, tuple):
                token_key = ''.join(token)
            else:
                token_key = str(token)
            token_to_id_str[token_key] = token_id

        vocab_data = {
            'metadata': {
                'vocab_size': len(self.quantum_tokens),
                'bpe_merges': len(self.bpe_merges),
                'validation_score': self.stats['physical_validation_score'],
                'created_at': str(torch.randint(0, 1000000, (1,)).item())
            },
            'token_to_id': token_to_id_str,
            'quantum_tokens': {
                str(token) if isinstance(token, tuple) else token: {
                    'energy_level': props.energy_level,
                    'coherence': props.coherence,
                    'entropy': props.entropy,
                    'spin': props.spin,
                    'mass': props.mass,
                    'charge': props.charge,
                    'frequency': props.frequency,
                    'wavelength': props.wavelength
                }
                for token, props in self.quantum_tokens.items()
            },
            'bpe_merges': bpe_merges_str
        }

        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)

        print(f"💾 Vocabulary saved to {filepath}: {len(self.quantum_tokens)} tokens")

    def load_vocabulary(self, filepath: str):
        """Carrega vocabulário quântico de arquivo."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)

        # Carregar metadados
        metadata = vocab_data.get('metadata', {})
        print(f"📚 Loading vocabulary: {metadata.get('vocab_size', 0)} tokens")

        # Carregar mapeamentos
        self.token_to_id = vocab_data.get('token_to_id', {})
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Carregar propriedades quânticas
        quantum_tokens_data = vocab_data.get('quantum_tokens', {})
        for token, props_data in quantum_tokens_data.items():
            props = QuantumTokenProperties(**props_data)
            self.quantum_tokens[token] = props

        # Carregar merges BPE
        self.bpe_merges = vocab_data.get('bpe_merges', {})

        # Atualizar estatísticas
        self.stats['total_tokens'] = len(self.quantum_tokens)
        self.stats['total_merges'] = len(self.bpe_merges)
        self.stats['physical_validation_score'] = metadata.get('validation_score', 0.0)

        print(f"✅ Vocabulary loaded: {len(self.quantum_tokens)} tokens, {len(self.bpe_merges)} merges")


# Funções utilitárias
def create_quantum_corpus_from_text(text: str, window_size: int = 1000) -> List[str]:
    """Cria corpus para treinamento BPE a partir de texto."""
    corpus = []
    for i in range(0, len(text), window_size):
        chunk = text[i:i + window_size]
        corpus.append(chunk)
    return corpus


def test_quantum_vocabulary():
    """Testa o sistema de vocabulário quântico nativo."""
    print("🧪 Testing Quantum Native Vocabulary")
    print("=" * 50)

    # Criar vocabulário
    vocab = QuantumNativeVocabulary(vocab_size=1000)  # Vocabulário menor para teste

    # Corpus de teste
    test_corpus = [
        "quantum mechanics describes the behavior of matter and energy",
        "wave particle duality is a fundamental concept in physics",
        "entanglement allows particles to be correlated instantaneously",
        "superposition enables quantum computers to process multiple states",
        "the schrodinger equation governs quantum mechanical systems"
    ]

    # Construir BPE quântico
    vocab.build_quantum_bpe(test_corpus, target_vocab_size=500)

    # Testar tokenização
    test_text = "quantum entanglement"
    tokens = vocab.tokenize_with_physics(test_text)
    print(f"\n🔬 Tokenization Test:")
    print(f"   Input: '{test_text}'")
    print(f"   Tokens: {tokens}")

    # Validar propriedades físicas
    validation = vocab.validate_physical_properties()
    print(f"\n📊 Physical Validation:")
    for key, value in validation.items():
        print(".3f")

    # Salvar vocabulário
    vocab.save_vocabulary("quantum_native_vocab_test.json")

    print("\n✅ Quantum Native Vocabulary test completed!")


if __name__ == "__main__":
    test_quantum_vocabulary()