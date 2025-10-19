
#!/usr/bin/env python3
"""
Dynamic Quantum Vocabulary System
==================================

Sistema de vocabulário quântico dinâmico com 1,346+ palavras em inglês,
referências quânticas e pesos baseados em múltiplos fatores.

Características principais:
- 1,346+ palavras em inglês com referências quânticas
- Pesos baseados em alinhamento de modelo, frequência, termos científicos
- Múltiplas palavras por caractere com seleção ponderada
- Fatores de influência quântica (nível de energia, coerência, entropia)
- Integração com sistema DCF para análise de tokens aprimorada

Autor: Kilo Code (Sistema ΨQRH)
"""

import torch
import json
import os
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
import numpy as np

class DynamicQuantumVocabulary:
    """
    Sistema de vocabulário quântico dinâmico com 1,346+ palavras.

    Fornece mapeamento entre caracteres e palavras quânticas com pesos
    baseados em fatores múltiplos para seleção otimizada de tokens.
    """

    def __init__(self, device: str = "cpu", vocab_size: int = 95):
        """
        Inicializa o sistema de vocabulário quântico dinâmico.

        Args:
            device: Dispositivo para tensores (cpu/cuda)
            vocab_size: Tamanho do vocabulário de caracteres (padrão: 95)
        """
        self.device = device
        self.vocab_size = vocab_size
        self.char_to_words = defaultdict(list)  # Mapeamento caractere -> lista de palavras
        self.word_to_quantum = {}  # Mapeamento palavra -> propriedades quânticas
        self.char_to_idx = {}  # Mapeamento caractere -> índice
        self.idx_to_char = {}  # Mapeamento índice -> caractere

        # Estatísticas do vocabulário
        self.stats = {
            'total_words': 0,
            'total_chars': 0,
            'avg_words_per_char': 0.0,
            'quantum_references': 0,
            'energy_levels': set(),
            'weight_range': (0.0, 5.0)
        }

        # Inicializar vocabulário básico
        self._initialize_basic_vocabulary()

        # Carregar vocabulário quântico expandido
        self._load_quantum_vocabulary()

        # Calcular estatísticas
        self._calculate_statistics()

        print(f"🔬 Dynamic Quantum Vocabulary initialized: {self.stats['total_words']} words, {self.stats['total_chars']} characters")
        print(f"   📊 Avg words per char: {self.stats['avg_words_per_char']:.1f}, Quantum refs: {self.stats['quantum_references']}")

    def _initialize_basic_vocabulary(self):
        """Inicializa vocabulário básico de caracteres ASCII printáveis."""
        # Caracteres ASCII printáveis (32-126)
        for i in range(32, 127):
            char = chr(i)
            idx = i - 32  # Mapeamento 0-based
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char

        self.stats['total_chars'] = len(self.char_to_idx)

    def _load_quantum_vocabulary(self):
        """Carrega o vocabulário quântico expandido com 1,346+ palavras."""

        # Vocabulário quântico expandido - 1,346+ palavras organizadas por temas
        quantum_vocabulary = {

            # ========== CONCEITOS FÍSICOS QUÂNTICOS ==========
            'quantum': {'word': 'quantum', 'weight': 5.0, 'energy_level': 'high_energy_state', 'coherence': 0.95, 'entropy': 0.1, 'scientific_term': True},
            'qubit': {'word': 'qubit', 'weight': 4.8, 'energy_level': 'excited_state', 'coherence': 0.92, 'entropy': 0.15, 'scientific_term': True},
            'superposition': {'word': 'superposition', 'weight': 4.7, 'energy_level': 'high_energy_state', 'coherence': 0.98, 'entropy': 0.05, 'scientific_term': True},
            'entanglement': {'word': 'entanglement', 'weight': 4.6, 'energy_level': 'high_energy_state', 'coherence': 0.96, 'entropy': 0.08, 'scientific_term': True},
            'wavefunction': {'word': 'wavefunction', 'weight': 4.5, 'energy_level': 'ground_state', 'coherence': 0.94, 'entropy': 0.12, 'scientific_term': True},
            'tunneling': {'word': 'tunneling', 'weight': 4.4, 'energy_level': 'excited_state', 'coherence': 0.89, 'entropy': 0.18, 'scientific_term': True},
            'coherence': {'word': 'coherence', 'weight': 4.3, 'energy_level': 'ground_state', 'coherence': 0.97, 'entropy': 0.06, 'scientific_term': True},
            'decoherence': {'word': 'decoherence', 'weight': 4.2, 'energy_level': 'unknown_state', 'coherence': 0.85, 'entropy': 0.22, 'scientific_term': True},
            'spin': {'word': 'spin', 'weight': 4.1, 'energy_level': 'ground_state', 'coherence': 0.91, 'entropy': 0.16, 'scientific_term': True},
            'orbital': {'word': 'orbital', 'weight': 4.0, 'energy_level': 'excited_state', 'coherence': 0.88, 'entropy': 0.19, 'scientific_term': True},
            'resonance': {'word': 'resonance', 'weight': 3.9, 'energy_level': 'excited_state', 'coherence': 0.93, 'entropy': 0.14, 'scientific_term': True},
            'interference': {'word': 'interference', 'weight': 3.8, 'energy_level': 'ground_state', 'coherence': 0.90, 'entropy': 0.17, 'scientific_term': True},
            'amplitude': {'word': 'amplitude', 'weight': 3.7, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.20, 'scientific_term': True},
            'phase': {'word': 'phase', 'weight': 3.6, 'energy_level': 'ground_state', 'coherence': 0.89, 'entropy': 0.18, 'scientific_term': True},
            'probability': {'word': 'probability', 'weight': 3.5, 'energy_level': 'unknown_state', 'coherence': 0.84, 'entropy': 0.23, 'scientific_term': True},
            'uncertainty': {'word': 'uncertainty', 'weight': 3.4, 'energy_level': 'unknown_state', 'coherence': 0.82, 'entropy': 0.25, 'scientific_term': True},
            'measurement': {'word': 'measurement', 'weight': 3.3, 'energy_level': 'unknown_state', 'coherence': 0.86, 'entropy': 0.21, 'scientific_term': True},
            'collapse': {'word': 'collapse', 'weight': 3.2, 'energy_level': 'unknown_state', 'coherence': 0.83, 'entropy': 0.24, 'scientific_term': True},
            'eigenstate': {'word': 'eigenstate', 'weight': 3.1, 'energy_level': 'ground_state', 'coherence': 0.95, 'entropy': 0.10, 'scientific_term': True},
            'eigenvalue': {'word': 'eigenvalue', 'weight': 3.0, 'energy_level': 'ground_state', 'coherence': 0.94, 'entropy': 0.11, 'scientific_term': True},
            'hamiltonian': {'word': 'hamiltonian', 'weight': 2.9, 'energy_level': 'ground_state', 'coherence': 0.92, 'entropy': 0.15, 'scientific_term': True},
            'schrodinger': {'word': 'schrodinger', 'weight': 2.8, 'energy_level': 'ground_state', 'coherence': 0.91, 'entropy': 0.16, 'scientific_term': True},
            'heisenberg': {'word': 'heisenberg', 'weight': 2.7, 'energy_level': 'unknown_state', 'coherence': 0.88, 'entropy': 0.19, 'scientific_term': True},
            'pauli': {'word': 'pauli', 'weight': 2.6, 'energy_level': 'ground_state', 'coherence': 0.89, 'entropy': 0.18, 'scientific_term': True},
            'dirac': {'word': 'dirac', 'weight': 2.5, 'energy_level': 'high_energy_state', 'coherence': 0.87, 'entropy': 0.20, 'scientific_term': True},
            'feynman': {'word': 'feynman', 'weight': 2.4, 'energy_level': 'excited_state', 'coherence': 0.85, 'entropy': 0.22, 'scientific_term': True},
            'bell': {'word': 'bell', 'weight': 2.3, 'energy_level': 'ground_state', 'coherence': 0.90, 'entropy': 0.17, 'scientific_term': True},
            'bohr': {'word': 'bohr', 'weight': 2.2, 'energy_level': 'ground_state', 'coherence': 0.88, 'entropy': 0.19, 'scientific_term': True},
            'planck': {'word': 'planck', 'weight': 2.1, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.21, 'scientific_term': True},
            'fermi': {'word': 'fermi', 'weight': 2.0, 'energy_level': 'excited_state', 'coherence': 0.84, 'entropy': 0.23, 'scientific_term': True},
            'boson': {'word': 'boson', 'weight': 1.9, 'energy_level': 'excited_state', 'coherence': 0.87, 'entropy': 0.20, 'scientific_term': True},
            'fermion': {'word': 'fermion', 'weight': 1.8, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.22, 'scientific_term': True},
            'photon': {'word': 'photon', 'weight': 1.7, 'energy_level': 'excited_state', 'coherence': 0.89, 'entropy': 0.18, 'scientific_term': True},
            'electron': {'word': 'electron', 'weight': 1.6, 'energy_level': 'ground_state', 'coherence': 0.88, 'entropy': 0.19, 'scientific_term': True},
            'proton': {'word': 'proton', 'weight': 1.5, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.21, 'scientific_term': True},
            'neutron': {'word': 'neutron', 'weight': 1.4, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.22, 'scientific_term': True},
            'atom': {'word': 'atom', 'weight': 1.3, 'energy_level': 'ground_state', 'coherence': 0.83, 'entropy': 0.24, 'scientific_term': True},
            'molecule': {'word': 'molecule', 'weight': 1.2, 'energy_level': 'ground_state', 'coherence': 0.82, 'entropy': 0.25, 'scientific_term': True},
            'crystal': {'word': 'crystal', 'weight': 1.1, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.20, 'scientific_term': True},
            'lattice': {'word': 'lattice', 'weight': 1.0, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.21, 'scientific_term': True},

            # ========== CONCEITOS MATEMÁTICOS ==========
            'fractal': {'word': 'fractal', 'weight': 4.2, 'energy_level': 'ground_state', 'coherence': 0.93, 'entropy': 0.14, 'scientific_term': True},
            'dimension': {'word': 'dimension', 'weight': 3.8, 'energy_level': 'ground_state', 'coherence': 0.91, 'entropy': 0.16, 'scientific_term': True},
            'topology': {'word': 'topology', 'weight': 3.6, 'energy_level': 'ground_state', 'coherence': 0.89, 'entropy': 0.18, 'scientific_term': True},
            'manifold': {'word': 'manifold', 'weight': 3.4, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.20, 'scientific_term': True},
            'tensor': {'word': 'tensor', 'weight': 3.2, 'energy_level': 'ground_state', 'coherence': 0.88, 'entropy': 0.19, 'scientific_term': True},
            'vector': {'word': 'vector', 'weight': 2.8, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.21, 'scientific_term': True},
            'matrix': {'word': 'matrix', 'weight': 2.6, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.22, 'scientific_term': True},
            'operator': {'word': 'operator', 'weight': 2.4, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.20, 'scientific_term': True},
            'function': {'word': 'function', 'weight': 2.2, 'energy_level': 'ground_state', 'coherence': 0.84, 'entropy': 0.23, 'scientific_term': True},
            'integral': {'word': 'integral', 'weight': 2.0, 'energy_level': 'ground_state', 'coherence': 0.83, 'entropy': 0.24, 'scientific_term': True},
            'derivative': {'word': 'derivative', 'weight': 1.8, 'energy_level': 'ground_state', 'coherence': 0.82, 'entropy': 0.25, 'scientific_term': True},
            'differential': {'word': 'differential', 'weight': 1.6, 'energy_level': 'ground_state', 'coherence': 0.81, 'entropy': 0.26, 'scientific_term': True},
            'equation': {'word': 'equation', 'weight': 1.4, 'energy_level': 'ground_state', 'coherence': 0.80, 'entropy': 0.27, 'scientific_term': True},
            'algorithm': {'word': 'algorithm', 'weight': 1.2, 'energy_level': 'unknown_state', 'coherence': 0.78, 'entropy': 0.29, 'scientific_term': True},
            'computation': {'word': 'computation', 'weight': 1.0, 'energy_level': 'unknown_state', 'coherence': 0.76, 'entropy': 0.31, 'scientific_term': True},

            # ========== CONCEITOS DE CONSCIÊNCIA ==========
            'consciousness': {'word': 'consciousness', 'weight': 4.0, 'energy_level': 'high_energy_state', 'coherence': 0.90, 'entropy': 0.17, 'scientific_term': True},
            'awareness': {'word': 'awareness', 'weight': 3.5, 'energy_level': 'excited_state', 'coherence': 0.88, 'entropy': 0.19, 'scientific_term': True},
            'cognition': {'word': 'cognition', 'weight': 3.2, 'energy_level': 'excited_state', 'coherence': 0.86, 'entropy': 0.21, 'scientific_term': True},
            'perception': {'word': 'perception', 'weight': 2.9, 'energy_level': 'ground_state', 'coherence': 0.84, 'entropy': 0.23, 'scientific_term': True},
            'intuition': {'word': 'intuition', 'weight': 2.6, 'energy_level': 'high_energy_state', 'coherence': 0.82, 'entropy': 0.25, 'scientific_term': True},
            'reasoning': {'word': 'reasoning', 'weight': 2.3, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.22, 'scientific_term': True},
            'logic': {'word': 'logic', 'weight': 2.0, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.20, 'scientific_term': True},
            'mind': {'word': 'mind', 'weight': 1.7, 'energy_level': 'high_energy_state', 'coherence': 0.83, 'entropy': 0.24, 'scientific_term': True},
            'thought': {'word': 'thought', 'weight': 1.4, 'energy_level': 'excited_state', 'coherence': 0.81, 'entropy': 0.26, 'scientific_term': True},
            'idea': {'word': 'idea', 'weight': 1.1, 'energy_level': 'excited_state', 'coherence': 0.79, 'entropy': 0.28, 'scientific_term': True},
            'concept': {'word': 'concept', 'weight': 0.8, 'energy_level': 'ground_state', 'coherence': 0.77, 'entropy': 0.30, 'scientific_term': True},
            'understanding': {'word': 'understanding', 'weight': 0.5, 'energy_level': 'ground_state', 'coherence': 0.75, 'entropy': 0.32, 'scientific_term': True},

            # ========== PALAVRAS COMUNS COM PESOS ADAPTADOS ==========
            'the': {'word': 'the', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.95, 'entropy': 0.10, 'scientific_term': False},
            'and': {'word': 'and', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.94, 'entropy': 0.11, 'scientific_term': False},
            'or': {'word': 'or', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.93, 'entropy': 0.12, 'scientific_term': False},
            'but': {'word': 'but', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.92, 'entropy': 0.13, 'scientific_term': False},
            'in': {'word': 'in', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.91, 'entropy': 0.14, 'scientific_term': False},
            'on': {'word': 'on', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.90, 'entropy': 0.15, 'scientific_term': False},
            'at': {'word': 'at', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.89, 'entropy': 0.16, 'scientific_term': False},
            'to': {'word': 'to', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.88, 'entropy': 0.17, 'scientific_term': False},
            'for': {'word': 'for', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.18, 'scientific_term': False},
            'of': {'word': 'of', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.19, 'scientific_term': False},
            'with': {'word': 'with', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.20, 'scientific_term': False},
            'by': {'word': 'by', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.84, 'entropy': 0.21, 'scientific_term': False},
            'an': {'word': 'an', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.83, 'entropy': 0.22, 'scientific_term': False},
            'a': {'word': 'a', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.82, 'entropy': 0.23, 'scientific_term': False},
            'is': {'word': 'is', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.81, 'entropy': 0.24, 'scientific_term': False},
            'are': {'word': 'are', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.80, 'entropy': 0.25, 'scientific_term': False},
            'was': {'word': 'was', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.79, 'entropy': 0.26, 'scientific_term': False},
            'were': {'word': 'were', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.78, 'entropy': 0.27, 'scientific_term': False},
            'be': {'word': 'be', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.77, 'entropy': 0.28, 'scientific_term': False},
            'been': {'word': 'been', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.76, 'entropy': 0.29, 'scientific_term': False},
            'being': {'word': 'being', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.75, 'entropy': 0.30, 'scientific_term': False},
            'have': {'word': 'have', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.74, 'entropy': 0.31, 'scientific_term': False},
            'has': {'word': 'has', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.73, 'entropy': 0.32, 'scientific_term': False},
            'had': {'word': 'had', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.72, 'entropy': 0.33, 'scientific_term': False},
            'do': {'word': 'do', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.71, 'entropy': 0.34, 'scientific_term': False},
            'does': {'word': 'does', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.70, 'entropy': 0.35, 'scientific_term': False},
            'did': {'word': 'did', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.69, 'entropy': 0.36, 'scientific_term': False},
            'will': {'word': 'will', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.68, 'entropy': 0.37, 'scientific_term': False},
            'would': {'word': 'would', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.67, 'entropy': 0.38, 'scientific_term': False},
            'can': {'word': 'can', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.66, 'entropy': 0.39, 'scientific_term': False},
            'could': {'word': 'could', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.65, 'entropy': 0.40, 'scientific_term': False},
            'should': {'word': 'should', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.64, 'entropy': 0.41, 'scientific_term': False},
            'may': {'word': 'may', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.63, 'entropy': 0.42, 'scientific_term': False},
            'might': {'word': 'might', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.62, 'entropy': 0.43, 'scientific_term': False},
            'must': {'word': 'must', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.61, 'entropy': 0.44, 'scientific_term': False},
            'shall': {'word': 'shall', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.60, 'entropy': 0.45, 'scientific_term': False},
            'this': {'word': 'this', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.59, 'entropy': 0.46, 'scientific_term': False},
            'that': {'word': 'that', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.58, 'entropy': 0.47, 'scientific_term': False},
            'these': {'word': 'these', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.57, 'entropy': 0.48, 'scientific_term': False},
            'those': {'word': 'those', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.56, 'entropy': 0.49, 'scientific_term': False},
            'i': {'word': 'i', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.55, 'entropy': 0.50, 'scientific_term': False},
            'you': {'word': 'you', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.54, 'entropy': 0.51, 'scientific_term': False},
            'he': {'word': 'he', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.53, 'entropy': 0.52, 'scientific_term': False},
            'she': {'word': 'she', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.52, 'entropy': 0.53, 'scientific_term': False},
            'it': {'word': 'it', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.51, 'entropy': 0.54, 'scientific_term': False},
            'we': {'word': 'we', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.50, 'entropy': 0.55, 'scientific_term': False},
            'they': {'word': 'they', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.49, 'entropy': 0.56, 'scientific_term': False},
            'me': {'word': 'me', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.48, 'entropy': 0.57, 'scientific_term': False},
            'him': {'word': 'him', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.47, 'entropy': 0.58, 'scientific_term': False},
            'her': {'word': 'her', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.46, 'entropy': 0.59, 'scientific_term': False},
            'us': {'word': 'us', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.45, 'entropy': 0.60, 'scientific_term': False},
            'them': {'word': 'them', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.44, 'entropy': 0.61, 'scientific_term': False},
            'my': {'word': 'my', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.43, 'entropy': 0.62, 'scientific_term': False},
            'your': {'word': 'your', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.42, 'entropy': 0.63, 'scientific_term': False},
            'his': {'word': 'his', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.41, 'entropy': 0.64, 'scientific_term': False},
            'its': {'word': 'its', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.40, 'entropy': 0.65, 'scientific_term': False},
            'our': {'word': 'our', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.39, 'entropy': 0.66, 'scientific_term': False},
            'their': {'word': 'their', 'weight': 0.1, 'energy_level': 'ground_state', 'coherence': 0.38, 'entropy': 0.67, 'scientific_term': False},
            'what': {'word': 'what', 'weight': 0.2, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.20, 'scientific_term': False},
            'when': {'word': 'when', 'weight': 0.2, 'energy_level': 'ground_state', 'coherence': 0.84, 'entropy': 0.21, 'scientific_term': False},
            'where': {'word': 'where', 'weight': 0.2, 'energy_level': 'ground_state', 'coherence': 0.83, 'entropy': 0.22, 'scientific_term': False},
            'why': {'word': 'why', 'weight': 0.2, 'energy_level': 'ground_state', 'coherence': 0.82, 'entropy': 0.23, 'scientific_term': False},
            'how': {'word': 'how', 'weight': 0.2, 'energy_level': 'ground_state', 'coherence': 0.81, 'entropy': 0.24, 'scientific_term': False},
            'which': {'word': 'which', 'weight': 0.2, 'energy_level': 'ground_state', 'coherence': 0.80, 'entropy': 0.25, 'scientific_term': False},
            'who': {'word': 'who', 'weight': 0.2, 'energy_level': 'ground_state', 'coherence': 0.79, 'entropy': 0.26, 'scientific_term': False},
            'color': {'word': 'color', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.88, 'entropy': 0.17, 'scientific_term': False},
            'sky': {'word': 'sky', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.18, 'scientific_term': False},
            'blue': {'word': 'blue', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.89, 'entropy': 0.16, 'scientific_term': False},
            'red': {'word': 'red', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.88, 'entropy': 0.17, 'scientific_term': False},
            'green': {'word': 'green', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.18, 'scientific_term': False},
            'yellow': {'word': 'yellow', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.19, 'scientific_term': False},
            'white': {'word': 'white', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.20, 'scientific_term': False},
            'black': {'word': 'black', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.84, 'entropy': 0.21, 'scientific_term': False},
            'light': {'word': 'light', 'weight': 0.4, 'energy_level': 'excited_state', 'coherence': 0.90, 'entropy': 0.15, 'scientific_term': False},
            'dark': {'word': 'dark', 'weight': 0.4, 'energy_level': 'ground_state', 'coherence': 0.82, 'entropy': 0.23, 'scientific_term': False},
            'bright': {'word': 'bright', 'weight': 0.4, 'energy_level': 'excited_state', 'coherence': 0.91, 'entropy': 0.14, 'scientific_term': False},
            'cloud': {'word': 'cloud', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.83, 'entropy': 0.22, 'scientific_term': False},
            'sun': {'word': 'sun', 'weight': 0.4, 'energy_level': 'high_energy_state', 'coherence': 0.92, 'entropy': 0.13, 'scientific_term': False},
            'moon': {'word': 'moon', 'weight': 0.4, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.20, 'scientific_term': False},
            'star': {'word': 'star', 'weight': 0.4, 'energy_level': 'high_energy_state', 'coherence': 0.93, 'entropy': 0.12, 'scientific_term': False},
            'space': {'word': 'space', 'weight': 0.4, 'energy_level': 'unknown_state', 'coherence': 0.80, 'entropy': 0.27, 'scientific_term': False},
            'time': {'word': 'time', 'weight': 0.5, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.19, 'scientific_term': False},
            'day': {'word': 'day', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.18, 'scientific_term': False},
            'night': {'word': 'night', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.81, 'entropy': 0.24, 'scientific_term': False},
            'morning': {'word': 'morning', 'weight': 0.3, 'energy_level': 'excited_state', 'coherence': 0.88, 'entropy': 0.17, 'scientific_term': False},
            'evening': {'word': 'evening', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.84, 'entropy': 0.21, 'scientific_term': False},
            'water': {'word': 'water', 'weight': 0.4, 'energy_level': 'ground_state', 'coherence': 0.89, 'entropy': 0.16, 'scientific_term': False},
            'fire': {'word': 'fire', 'weight': 0.4, 'energy_level': 'high_energy_state', 'coherence': 0.85, 'entropy': 0.20, 'scientific_term': False},
            'earth': {'word': 'earth', 'weight': 0.4, 'energy_level': 'ground_state', 'coherence': 0.87, 'entropy': 0.18, 'scientific_term': False},
            'air': {'word': 'air', 'weight': 0.4, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.19, 'scientific_term': False},
            'wind': {'word': 'wind', 'weight': 0.4, 'energy_level': 'excited_state', 'coherence': 0.83, 'entropy': 0.22, 'scientific_term': False},
            'rain': {'word': 'rain', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.82, 'entropy': 0.23, 'scientific_term': False},
            'snow': {'word': 'snow', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.88, 'entropy': 0.17, 'scientific_term': False},
            'ice': {'word': 'ice', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.89, 'entropy': 0.16, 'scientific_term': False},
            'heat': {'word': 'heat', 'weight': 0.4, 'energy_level': 'high_energy_state', 'coherence': 0.84, 'entropy': 0.21, 'scientific_term': False},
            'cold': {'word': 'cold', 'weight': 0.4, 'energy_level': 'ground_state', 'coherence': 0.81, 'entropy': 0.24, 'scientific_term': False},
            'hot': {'word': 'hot', 'weight': 0.4, 'energy_level': 'high_energy_state', 'coherence': 0.85, 'entropy': 0.20, 'scientific_term': False},
            'warm': {'word': 'warm', 'weight': 0.4, 'energy_level': 'excited_state', 'coherence': 0.86, 'entropy': 0.19, 'scientific_term': False},
            'cool': {'word': 'cool', 'weight': 0.4, 'energy_level': 'ground_state', 'coherence': 0.83, 'entropy': 0.22, 'scientific_term': False},
            'fast': {'word': 'fast', 'weight': 0.3, 'energy_level': 'excited_state', 'coherence': 0.87, 'entropy': 0.18, 'scientific_term': False},
            'slow': {'word': 'slow', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.80, 'entropy': 0.25, 'scientific_term': False},
            'quick': {'word': 'quick', 'weight': 0.3, 'energy_level': 'excited_state', 'coherence': 0.88, 'entropy': 0.17, 'scientific_term': False},
            'big': {'word': 'big', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.82, 'entropy': 0.23, 'scientific_term': False},
            'small': {'word': 'small', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.81, 'entropy': 0.24, 'scientific_term': False},
            'large': {'word': 'large', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.83, 'entropy': 0.22, 'scientific_term': False},
            'tiny': {'word': 'tiny', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.80, 'entropy': 0.25, 'scientific_term': False},
            'high': {'word': 'high', 'weight': 0.3, 'energy_level': 'excited_state', 'coherence': 0.84, 'entropy': 0.21, 'scientific_term': False},
            'low': {'word': 'low', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.79, 'entropy': 0.26, 'scientific_term': False},
            'deep': {'word': 'deep', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.82, 'entropy': 0.23, 'scientific_term': False},
            'shallow': {'word': 'shallow', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.81, 'entropy': 0.24, 'scientific_term': False},
            'hard': {'word': 'hard', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.85, 'entropy': 0.20, 'scientific_term': False},
            'soft': {'word': 'soft', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.83, 'entropy': 0.22, 'scientific_term': False},
            'strong': {'word': 'strong', 'weight': 0.3, 'energy_level': 'excited_state', 'coherence': 0.87, 'entropy': 0.18, 'scientific_term': False},
            'weak': {'word': 'weak', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.78, 'entropy': 0.27, 'scientific_term': False},
            'good': {'word': 'good', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.86, 'entropy': 0.19, 'scientific_term': False},
            'bad': {'word': 'bad', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.77, 'entropy': 0.28, 'scientific_term': False},
            'right': {'word': 'right', 'weight': 0.3, 'energy_level': 'ground_state', 'coherence': 0.84, 'entropy': 0.21, 'scientific_term': False},
            'wrong': {'word':