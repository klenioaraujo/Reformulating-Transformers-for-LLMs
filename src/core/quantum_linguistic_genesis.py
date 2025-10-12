#!/usr/bin/env python3
"""
ΨQRH Quantum Linguistic Genesis System
======================================

Sistema de genesis linguístico quântico que codifica o alfabeto e numerais
como propriedades fundamentais dos estados quânticos.

Problema Resolvido:
- Sistema atual usa "ZERO FALLBACK" com inicialização aleatória
- Falta conhecimento linguístico básico na fundação quântica
- Sistema nasce "linguisticamente analfabeto"

Solução:
- Codificar alfabeto e numerais como estados quânticos fundamentais
- Inicialização não-aleatória baseada em propriedades linguísticas
- Parâmetros quânticos representam existência da linguagem
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class QuantumLinguisticGenesis:
    """
    Sistema de Genesis Linguístico Quântico

    Codifica alfabeto e numerais como propriedades fundamentais dos estados quânticos,
    resolvendo o problema de "analfabetismo linguístico quântico" do sistema atual.
    """

    def __init__(self, embed_dim: int = 64, device: str = 'cpu'):
        self.embed_dim = embed_dim
        self.device = device

        # Propriedades linguísticas fundamentais
        self.linguistic_properties = {
            'vowel_energy': 0.7,      # Energia mais alta para vogais
            'consonant_energy': 0.5,  # Energia média para consoantes
            'numeral_energy': 0.6,    # Energia para numerais
            'space_energy': 0.3,      # Energia para espaço
            'punctuation_energy': 0.4 # Energia para pontuação
        }

        # Mapa de calibração alfabética fundamental
        self.quantum_alphabet = self._create_quantum_alphabet()
        self.quantum_numerals = self._create_quantum_numerals()
        self.quantum_punctuation = self._create_quantum_punctuation()

        print("🧬 Quantum Linguistic Genesis System Initialized")
        print(f"   📊 Alphabet: {len(self.quantum_alphabet)} characters")
        print(f"   🔢 Numerals: {len(self.quantum_numerals)} digits")
        print(f"   📍 Punctuation: {len(self.quantum_punctuation)} symbols")

    def _create_quantum_alphabet(self) -> Dict[str, torch.Tensor]:
        """
        Cria estados quânticos para o alfabeto completo

        Cada caractere é representado como um estado quântico único
        baseado em propriedades fonéticas e posicionais.
        """
        alphabet = {}

        # Vogais (maior energia, maior estabilidade)
        vowels = 'aeiouAEIOU'
        vowel_energy = self.linguistic_properties['vowel_energy']

        for i, vowel in enumerate(vowels):
            # Estados quânticos para vogais: mais estáveis, maior amplitude
            state = self._create_vowel_state(vowel, i, len(vowels))
            alphabet[vowel] = state

        # Consoantes (energia média, propriedades específicas)
        consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
        consonant_energy = self.linguistic_properties['consonant_energy']

        for i, consonant in enumerate(consonants):
            # Estados quânticos para consoantes: propriedades específicas
            state = self._create_consonant_state(consonant, i, len(consonants))
            alphabet[consonant] = state

        return alphabet

    def _create_quantum_numerals(self) -> Dict[str, torch.Tensor]:
        """
        Cria estados quânticos para numerais (0-9)

        Cada numeral tem propriedades matemáticas fundamentais
        codificadas em seu estado quântico.
        """
        numerals = {}

        for digit in '0123456789':
            state = self._create_numeral_state(digit)
            numerals[digit] = state

        return numerals

    def _create_quantum_punctuation(self) -> Dict[str, torch.Tensor]:
        """
        Cria estados quânticos para pontuação básica
        """
        punctuation = {}

        symbols = ' .,!?;:\"\'()[]{}<>-–—=+*/'

        for symbol in symbols:
            state = self._create_punctuation_state(symbol)
            punctuation[symbol] = state

        return punctuation

    def _create_vowel_state(self, vowel: str, index: int, total: int) -> torch.Tensor:
        """
        Cria estado quântico para vogal

        Vogais têm maior energia e estabilidade no espaço linguístico.
        """
        # Base senoidal para estabilidade
        base_freq = 2.0 * math.pi * (index + 1) / total

        # Criar tensor quântico [embed_dim, 4]
        state = torch.zeros(self.embed_dim, 4, device=self.device)

        for i in range(self.embed_dim):
            # Componente real: seno com frequência base
            real = math.sin(base_freq * i) * self.linguistic_properties['vowel_energy']

            # Componente imaginária: cosseno com fase deslocada
            imag = math.cos(base_freq * i + math.pi/4) * self.linguistic_properties['vowel_energy']

            # Componente j: harmônica superior
            j_comp = math.sin(2 * base_freq * i) * (self.linguistic_properties['vowel_energy'] * 0.7)

            # Componente k: estabilidade
            k_comp = math.cos(2 * base_freq * i) * (self.linguistic_properties['vowel_energy'] * 0.7)

            state[i] = torch.tensor([real, imag, j_comp, k_comp], device=self.device)

        # Normalizar para conservação de energia
        state = self._normalize_quantum_state(state)

        return state

    def _create_consonant_state(self, consonant: str, index: int, total: int) -> torch.Tensor:
        """
        Cria estado quântico para consoante

        Consoantes têm propriedades específicas baseadas em:
        - Plosivas (p, t, k, b, d, g)
        - Fricativas (f, v, s, z, sh, zh)
        - Nasais (m, n, ng)
        - Líquidas (l, r)
        """
        # Classificar consoante por tipo
        consonant_type = self._classify_consonant(consonant)

        # Frequência base baseada no tipo
        if consonant_type == 'plosive':
            base_freq = 3.0 * math.pi * (index + 1) / total
            energy_factor = 0.8
        elif consonant_type == 'fricative':
            base_freq = 2.5 * math.pi * (index + 1) / total
            energy_factor = 0.7
        elif consonant_type == 'nasal':
            base_freq = 1.5 * math.pi * (index + 1) / total
            energy_factor = 0.9
        elif consonant_type == 'liquid':
            base_freq = 2.0 * math.pi * (index + 1) / total
            energy_factor = 0.85
        else:
            base_freq = 2.2 * math.pi * (index + 1) / total
            energy_factor = 0.75

        # Criar tensor quântico
        state = torch.zeros(self.embed_dim, 4, device=self.device)

        for i in range(self.embed_dim):
            # Componentes baseadas no tipo
            real = math.sin(base_freq * i) * self.linguistic_properties['consonant_energy'] * energy_factor
            imag = math.cos(base_freq * i) * self.linguistic_properties['consonant_energy'] * energy_factor

            # Componentes específicas do tipo
            if consonant_type == 'plosive':
                j_comp = math.sin(3 * base_freq * i) * 0.5
                k_comp = math.cos(3 * base_freq * i) * 0.5
            elif consonant_type == 'fricative':
                j_comp = math.sin(4 * base_freq * i) * 0.6
                k_comp = math.cos(4 * base_freq * i) * 0.6
            elif consonant_type == 'nasal':
                j_comp = math.sin(1.5 * base_freq * i) * 0.8
                k_comp = math.cos(1.5 * base_freq * i) * 0.8
            elif consonant_type == 'liquid':
                j_comp = math.sin(2.5 * base_freq * i) * 0.7
                k_comp = math.cos(2.5 * base_freq * i) * 0.7
            else:
                j_comp = math.sin(2 * base_freq * i) * 0.6
                k_comp = math.cos(2 * base_freq * i) * 0.6

            state[i] = torch.tensor([real, imag, j_comp, k_comp], device=self.device)

        state = self._normalize_quantum_state(state)
        return state

    def _create_numeral_state(self, digit: str) -> torch.Tensor:
        """
        Cria estado quântico para numeral

        Numerais têm propriedades matemáticas fundamentais codificadas.
        """
        value = int(digit)

        # Base matemática: propriedades do número
        prime_factors = self._get_prime_factors(value) if value > 1 else []

        # Criar tensor quântico
        state = torch.zeros(self.embed_dim, 4, device=self.device)

        for i in range(self.embed_dim):
            # Componente real: valor numérico normalizado
            real = (value / 9.0) * self.linguistic_properties['numeral_energy']

            # Componente imaginária: propriedades primas
            imag_factor = len(prime_factors) / 4.0 if prime_factors else 0.25
            imag = math.sin(math.pi * i / self.embed_dim) * imag_factor * self.linguistic_properties['numeral_energy']

            # Componente j: paridade
            j_comp = (1.0 if value % 2 == 0 else -1.0) * 0.3 * self.linguistic_properties['numeral_energy']

            # Componente k: estabilidade matemática
            k_comp = math.cos(math.pi * i / self.embed_dim) * 0.4 * self.linguistic_properties['numeral_energy']

            state[i] = torch.tensor([real, imag, j_comp, k_comp], device=self.device)

        state = self._normalize_quantum_state(state)
        return state

    def _create_punctuation_state(self, symbol: str) -> torch.Tensor:
        """
        Cria estado quântico para símbolo de pontuação
        """
        # Mapear símbolo para propriedade funcional
        if symbol == ' ':
            functional_energy = 0.3  # Espaço tem energia baixa
            base_freq = 0.5
        elif symbol in '.,':
            functional_energy = 0.5  # Pontuação básica
            base_freq = 1.0
        elif symbol in '!?':
            functional_energy = 0.7  # Pontuação expressiva
            base_freq = 1.5
        elif symbol in '()[]{}':
            functional_energy = 0.6  # Delimitadores
            base_freq = 1.2
        else:
            functional_energy = 0.4  # Outros símbolos
            base_freq = 0.8

        state = torch.zeros(self.embed_dim, 4, device=self.device)

        for i in range(self.embed_dim):
            real = math.sin(base_freq * i) * functional_energy
            imag = math.cos(base_freq * i) * functional_energy
            j_comp = math.sin(2 * base_freq * i) * (functional_energy * 0.5)
            k_comp = math.cos(2 * base_freq * i) * (functional_energy * 0.5)

            state[i] = torch.tensor([real, imag, j_comp, k_comp], device=self.device)

        state = self._normalize_quantum_state(state)
        return state

    def _classify_consonant(self, consonant: str) -> str:
        """Classifica consoante por tipo fonético"""
        consonant = consonant.lower()

        plosives = 'ptkbdg'
        fricatives = 'fvszh'
        nasals = 'mn'
        liquids = 'lr'

        if consonant in plosives:
            return 'plosive'
        elif consonant in fricatives:
            return 'fricative'
        elif consonant in nasals:
            return 'nasal'
        elif consonant in liquids:
            return 'liquid'
        else:
            return 'other'

    def _get_prime_factors(self, n: int) -> List[int]:
        """Retorna fatores primos de um número"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    def _normalize_quantum_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normaliza estado quântico para conservação de energia
        """
        # Calcular norma do estado
        norm = torch.norm(state, dim=(0, 1))

        # Normalizar se necessário
        if norm > 1e-6:
            state = state / norm

        return state

    def get_quantum_vocabulary(self) -> Dict[str, torch.Tensor]:
        """
        Retorna vocabulário quântico completo
        """
        quantum_vocab = {}
        quantum_vocab.update(self.quantum_alphabet)
        quantum_vocab.update(self.quantum_numerals)
        quantum_vocab.update(self.quantum_punctuation)

        return quantum_vocab

    def get_quantum_vocabulary_tensor(self) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Retorna vocabulário quântico como tensor e mapa de índices

        Returns:
            tensor: [vocab_size, embed_dim, 4] - Representações quânticas
            char_to_idx: Dict[str, int] - Mapeamento caractere para índice
        """
        quantum_vocab = self.get_quantum_vocabulary()

        # Criar lista ordenada de caracteres
        chars = sorted(quantum_vocab.keys())
        char_to_idx = {char: idx for idx, char in enumerate(chars)}

        # Stack representações quânticas
        quantum_representations = []
        for char in chars:
            quantum_representations.append(quantum_vocab[char])

        quantum_tensor = torch.stack(quantum_representations, dim=0)

        return quantum_tensor, char_to_idx

    def analyze_linguistic_properties(self, text: str) -> Dict[str, float]:
        """
        Analisa propriedades linguísticas do texto usando a fundação quântica
        """
        if not text:
            return {
                'vowel_ratio': 0.0,
                'consonant_ratio': 0.0,
                'numeral_ratio': 0.0,
                'space_ratio': 0.0,
                'linguistic_energy': 0.0,
                'quantum_coherence': 0.0
            }

        # Contar tipos de caracteres
        vowel_count = sum(1 for char in text if char.lower() in 'aeiou')
        consonant_count = sum(1 for char in text if char.isalpha() and char.lower() not in 'aeiou')
        numeral_count = sum(1 for char in text if char.isdigit())
        space_count = sum(1 for char in text if char.isspace())

        total_chars = len(text)

        # Calcular propriedades quânticas
        linguistic_energy = 0.0
        quantum_states = []

        for char in text:
            if char in self.quantum_alphabet:
                state = self.quantum_alphabet[char]
                quantum_states.append(state)
                linguistic_energy += torch.norm(state).item()
            elif char in self.quantum_numerals:
                state = self.quantum_numerals[char]
                quantum_states.append(state)
                linguistic_energy += torch.norm(state).item()
            elif char in self.quantum_punctuation:
                state = self.quantum_punctuation[char]
                quantum_states.append(state)
                linguistic_energy += torch.norm(state).item()

        # Calcular coerência quântica se temos estados suficientes
        quantum_coherence = 0.0
        if len(quantum_states) > 1:
            # Coerência média entre estados adjacentes
            coherence_sum = 0.0
            for i in range(len(quantum_states) - 1):
                state1 = quantum_states[i].flatten()
                state2 = quantum_states[i + 1].flatten()

                # Similaridade de cosseno
                similarity = torch.dot(state1, state2) / (torch.norm(state1) * torch.norm(state2) + 1e-8)
                coherence_sum += similarity.item()

            quantum_coherence = coherence_sum / (len(quantum_states) - 1)

        return {
            'vowel_ratio': vowel_count / total_chars if total_chars > 0 else 0.0,
            'consonant_ratio': consonant_count / total_chars if total_chars > 0 else 0.0,
            'numeral_ratio': numeral_count / total_chars if total_chars > 0 else 0.0,
            'space_ratio': space_count / total_chars if total_chars > 0 else 0.0,
            'linguistic_energy': linguistic_energy / len(quantum_states) if quantum_states else 0.0,
            'quantum_coherence': quantum_coherence
        }


# Função de conveniência para integração rápida
def create_quantum_linguistic_foundation(embed_dim: int = 64, device: str = 'cpu') -> QuantumLinguisticGenesis:
    """
    Cria sistema de genesis linguístico quântico

    Args:
        embed_dim: Dimensão do embedding quântico
        device: Dispositivo (cpu/cuda)

    Returns:
        QuantumLinguisticGenesis: Sistema de genesis linguístico
    """
    return QuantumLinguisticGenesis(embed_dim=embed_dim, device=device)


if __name__ == "__main__":
    # Teste do sistema de genesis linguístico
    genesis = QuantumLinguisticGenesis(embed_dim=32)

    # Obter vocabulário quântico
    quantum_vocab, char_to_idx = genesis.get_quantum_vocabulary_tensor()

    print(f"\n🧬 Quantum Linguistic Genesis Test")
    print(f"   📊 Vocabulary size: {len(quantum_vocab)}")
    print(f"   🔬 Tensor shape: {quantum_vocab.shape}")
    print(f"   📍 Character mapping: {len(char_to_idx)} characters")

    # Testar análise linguística
    test_text = "Hello World 123!"
    analysis = genesis.analyze_linguistic_properties(test_text)

    print(f"\n📊 Linguistic Analysis of '{test_text}':")
    for key, value in analysis.items():
        print(f"   {key}: {value:.4f}")

    # Verificar alguns estados quânticos
    print(f"\n🔬 Sample Quantum States:")
    for char in ['a', '1', ' ', '!']:
        if char in genesis.quantum_alphabet or char in genesis.quantum_numerals or char in genesis.quantum_punctuation:
            state = genesis.get_quantum_vocabulary()[char]
            print(f"   '{char}': norm={torch.norm(state):.4f}, shape={state.shape}")