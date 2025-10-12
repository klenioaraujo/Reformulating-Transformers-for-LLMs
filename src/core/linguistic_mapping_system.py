#!/usr/bin/env python3
"""
Linguistic Mapping System - ΨQRH
=================================

Sistema de mapeamento linguístico que conecta caracteres individuais do Genesis
com tokens do vocabulário nativo, implementando compreensão linguística básica.

ESTRUTURA DE MAPEAMENTO:
linguistic_map = {
    "a": {
        "genesis_idx": 56,
        "native_token": 8,
        "spectral_coefficient": 0.85,
        "linguistic_energy": 0.7,
        "phonetic_type": "vowel"
    }
}

COMPONENTES:
- LinguisticMappingSystem: Mapeamento bidirecional
- LinguisticCoefficientCalibrator: Calibração automática de coeficientes
- Integração com auto-calibração: Calibração durante execução
- Integração com harmonização: Refinamento baseado em propriedades linguísticas
"""

import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os

class LinguisticMappingSystem:
    """
    Sistema de Mapeamento Linguístico ΨQRH

    Conecta caracteres individuais do Genesis com tokens do vocabulário nativo,
    implementando compreensão linguística básica através de coeficientes espectrais
    e propriedades fonéticas.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.linguistic_map = {}
        self.genesis_to_native = {}
        self.native_to_genesis = {}
        self.coefficient_matrix = None

        # Carregar vocabulários
        self._load_vocabularies()

        # Inicializar mapeamento
        self._initialize_linguistic_mapping()

        # Calibrar coeficientes
        self._calibrate_coefficients()

    def _load_vocabularies(self):
        """Carrega vocabulários Genesis e native"""
        try:
            # Carregar native vocabulary
            native_vocab_path = Path("data/native_vocab.json")
            if native_vocab_path.exists():
                with open(native_vocab_path, 'r', encoding='utf-8') as f:
                    native_data = json.load(f)
                self.native_vocab = native_data['tokens']
                self.native_vocab_size = native_data['vocab_size']
                print(f"📚 Native vocabulary loaded: {self.native_vocab_size} tokens")
            else:
                raise FileNotFoundError("Native vocabulary not found")

            # Carregar Genesis vocabulary
            from src.core.quantum_linguistic_genesis import QuantumLinguisticGenesis
            genesis = QuantumLinguisticGenesis(embed_dim=64, device=self.device)
            self.genesis_vocab, self.genesis_char_to_idx = genesis.get_quantum_vocabulary_tensor()
            self.genesis_idx_to_char = {v: k for k, v in self.genesis_char_to_idx.items()}
            print(f"🧬 Genesis vocabulary loaded: {len(self.genesis_idx_to_char)} characters")

        except Exception as e:
            print(f"⚠️  Error loading vocabularies: {e}")
            self.native_vocab = {}
            self.genesis_idx_to_char = {}

    def _initialize_linguistic_mapping(self):
        """Inicializa mapeamento linguístico completo para todos os caracteres do Genesis"""
        # Mapeamento completo: todos os 86 caracteres do Genesis para tokens do native_vocab
        character_mappings = {
            # Espaço e pontuação básica
            ' ': {'native_token': 1, 'phonetic_type': 'space', 'frequency': 0.150},
            '.': {'native_token': 13, 'phonetic_type': 'punctuation', 'frequency': 0.050},
            ',': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.030},
            '!': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.005},
            '?': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.005},
            ';': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.003},
            ':': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.003},
            '"': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.008},
            "'": {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.008},
            '(': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.004},
            ')': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.004},
            '[': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.002},
            ']': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.002},
            '{': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.002},
            '}': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.002},
            '<': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.002},
            '>': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.002},
            '-': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.010},
            '–': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.002},
            '—': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.001},
            '=': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.003},
            '+': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.003},
            '*': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.002},
            '/': {'native_token': 1, 'phonetic_type': 'punctuation', 'frequency': 0.003},

            # Numerais
            '0': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.010},
            '1': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.010},
            '2': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.008},
            '3': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.007},
            '4': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.006},
            '5': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.006},
            '6': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.005},
            '7': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.005},
            '8': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.005},
            '9': {'native_token': 1, 'phonetic_type': 'numeral', 'frequency': 0.005},

            # Letras minúsculas
            'a': {'native_token': 8, 'phonetic_type': 'vowel', 'frequency': 0.085},
            'b': {'native_token': 38, 'phonetic_type': 'consonant', 'frequency': 0.025},  # Mapeado para 'behavior'
            'c': {'native_token': 31, 'phonetic_type': 'consonant', 'frequency': 0.035},
            'd': {'native_token': 18, 'phonetic_type': 'consonant', 'frequency': 0.045},
            'e': {'native_token': 33, 'phonetic_type': 'vowel', 'frequency': 0.095},
            'f': {'native_token': 6, 'phonetic_type': 'consonant', 'frequency': 0.025},
            'g': {'native_token': 30, 'phonetic_type': 'consonant', 'frequency': 0.020},
            'h': {'native_token': 22, 'phonetic_type': 'consonant', 'frequency': 0.065},
            'i': {'native_token': 11, 'phonetic_type': 'vowel', 'frequency': 0.075},
            'j': {'native_token': 20, 'phonetic_type': 'consonant', 'frequency': 0.005},
            'k': {'native_token': 37, 'phonetic_type': 'consonant', 'frequency': 0.010},
            'l': {'native_token': 16, 'phonetic_type': 'consonant', 'frequency': 0.045},
            'm': {'native_token': 37, 'phonetic_type': 'consonant', 'frequency': 0.030},
            'n': {'native_token': 27, 'phonetic_type': 'consonant', 'frequency': 0.075},
            'o': {'native_token': 36, 'phonetic_type': 'vowel', 'frequency': 0.085},
            'p': {'native_token': 25, 'phonetic_type': 'consonant', 'frequency': 0.025},
            'q': {'native_token': 20, 'phonetic_type': 'consonant', 'frequency': 0.005},
            'r': {'native_token': 7, 'phonetic_type': 'consonant', 'frequency': 0.065},
            's': {'native_token': 17, 'phonetic_type': 'consonant', 'frequency': 0.075},
            't': {'native_token': 12, 'phonetic_type': 'consonant', 'frequency': 0.085},
            'u': {'native_token': 19, 'phonetic_type': 'vowel', 'frequency': 0.035},
            'v': {'native_token': 28, 'phonetic_type': 'consonant', 'frequency': 0.015},
            'w': {'native_token': 19, 'phonetic_type': 'consonant', 'frequency': 0.020},
            'x': {'native_token': 35, 'phonetic_type': 'consonant', 'frequency': 0.005},
            'y': {'native_token': 22, 'phonetic_type': 'consonant', 'frequency': 0.025},
            'z': {'native_token': 38, 'phonetic_type': 'consonant', 'frequency': 0.005},

            # Letras maiúsculas
            'A': {'native_token': 4, 'phonetic_type': 'vowel_upper', 'frequency': 0.015},
            'B': {'native_token': 1, 'phonetic_type': 'consonant_upper', 'frequency': 0.005},
            'C': {'native_token': 31, 'phonetic_type': 'consonant_upper', 'frequency': 0.008},
            'D': {'native_token': 18, 'phonetic_type': 'consonant_upper', 'frequency': 0.008},
            'E': {'native_token': 33, 'phonetic_type': 'vowel_upper', 'frequency': 0.012},
            'F': {'native_token': 6, 'phonetic_type': 'consonant_upper', 'frequency': 0.006},
            'G': {'native_token': 30, 'phonetic_type': 'consonant_upper', 'frequency': 0.005},
            'H': {'native_token': 22, 'phonetic_type': 'consonant_upper', 'frequency': 0.010},
            'I': {'native_token': 11, 'phonetic_type': 'vowel_upper', 'frequency': 0.012},
            'J': {'native_token': 20, 'phonetic_type': 'consonant_upper', 'frequency': 0.003},
            'K': {'native_token': 37, 'phonetic_type': 'consonant_upper', 'frequency': 0.004},
            'L': {'native_token': 16, 'phonetic_type': 'consonant_upper', 'frequency': 0.008},
            'M': {'native_token': 37, 'phonetic_type': 'consonant_upper', 'frequency': 0.008},
            'N': {'native_token': 27, 'phonetic_type': 'consonant_upper', 'frequency': 0.010},
            'O': {'native_token': 36, 'phonetic_type': 'vowel_upper', 'frequency': 0.010},
            'P': {'native_token': 25, 'phonetic_type': 'consonant_upper', 'frequency': 0.006},
            'Q': {'native_token': 20, 'phonetic_type': 'consonant_upper', 'frequency': 0.002},
            'R': {'native_token': 7, 'phonetic_type': 'consonant_upper', 'frequency': 0.010},
            'S': {'native_token': 17, 'phonetic_type': 'consonant_upper', 'frequency': 0.012},
            'T': {'native_token': 12, 'phonetic_type': 'consonant_upper', 'frequency': 0.012},
            'U': {'native_token': 19, 'phonetic_type': 'vowel_upper', 'frequency': 0.006},
            'V': {'native_token': 28, 'phonetic_type': 'consonant_upper', 'frequency': 0.004},
            'W': {'native_token': 19, 'phonetic_type': 'consonant_upper', 'frequency': 0.005},
            'X': {'native_token': 35, 'phonetic_type': 'consonant_upper', 'frequency': 0.002},
            'Y': {'native_token': 22, 'phonetic_type': 'consonant_upper', 'frequency': 0.004},
            'Z': {'native_token': 38, 'phonetic_type': 'consonant_upper', 'frequency': 0.002},
        }

        # Construir mapeamento linguístico
        for char, mapping in character_mappings.items():
            if char in self.genesis_char_to_idx:
                genesis_idx = self.genesis_char_to_idx[char]
                native_token = mapping['native_token']

                # Calcular coeficientes linguísticos
                spectral_coefficient = self._calculate_spectral_coefficient(char, mapping)
                linguistic_energy = self._calculate_linguistic_energy(char, mapping)

                self.linguistic_map[char] = {
                    'genesis_idx': genesis_idx,
                    'native_token': native_token,
                    'spectral_coefficient': spectral_coefficient,
                    'linguistic_energy': linguistic_energy,
                    'phonetic_type': mapping['phonetic_type'],
                    'frequency': mapping['frequency']
                }

                # Mapeamentos bidirecionais
                self.genesis_to_native[genesis_idx] = native_token
                self.native_to_genesis[native_token] = genesis_idx

        print(f"🗣️  Linguistic mapping initialized: {len(self.linguistic_map)} character mappings")

    def _calculate_spectral_coefficient(self, char: str, mapping: Dict) -> float:
        """Calcula coeficiente espectral baseado em propriedades fonéticas"""
        base_coefficient = 0.5

        # Ajuste baseado no tipo fonético
        phonetic_type = mapping['phonetic_type']
        if phonetic_type == 'vowel' or phonetic_type == 'vowel_upper':
            base_coefficient += 0.2  # Vogais têm maior energia espectral
        elif phonetic_type == 'consonant' or phonetic_type == 'consonant_upper':
            base_coefficient += 0.1  # Consoantes têm energia espectral moderada
        elif phonetic_type == 'numeral':
            base_coefficient += 0.15  # Numerais têm energia matemática
        elif phonetic_type == 'space':
            base_coefficient += 0.05  # Espaço tem energia baixa
        elif phonetic_type == 'punctuation':
            base_coefficient += 0.08  # Pontuação tem energia estrutural

        # Ajuste baseado na frequência
        frequency_factor = min(mapping['frequency'] * 10, 0.3)
        base_coefficient += frequency_factor

        # Caracteres mapeados para tokens específicos têm maior coeficiente
        if mapping['native_token'] != 1:  # Se não é <unk>
            base_coefficient += 0.2

        return min(base_coefficient, 1.0)

    def _calculate_linguistic_energy(self, char: str, mapping: Dict) -> float:
        """Calcula energia linguística baseada em propriedades do caractere"""
        base_energy = 0.3

        # Ajuste baseado no tipo fonético
        phonetic_type = mapping['phonetic_type']
        if phonetic_type == 'vowel' or phonetic_type == 'vowel_upper':
            base_energy += 0.3  # Vogais têm maior energia linguística
        elif phonetic_type == 'consonant' or phonetic_type == 'consonant_upper':
            base_energy += 0.2  # Consoantes têm energia linguística moderada
        elif phonetic_type == 'numeral':
            base_energy += 0.25  # Numerais têm energia matemática
        elif phonetic_type == 'space':
            base_energy += 0.1   # Espaço tem energia linguística baixa
        elif phonetic_type == 'punctuation':
            base_energy += 0.15  # Pontuação tem energia estrutural

        # Ajuste baseado na frequência de uso
        frequency_bonus = mapping['frequency'] * 2
        base_energy += frequency_bonus

        # Caracteres mapeados para tokens específicos têm maior energia
        if mapping['native_token'] != 1:  # Se não é <unk>
            base_energy += 0.2

        return min(base_energy, 1.0)

    def _calibrate_coefficients(self):
        """Calibra matriz de coeficientes linguísticos"""
        n_genesis = len(self.genesis_idx_to_char)
        n_native = self.native_vocab_size

        self.coefficient_matrix = torch.zeros(n_genesis, n_native, device=self.device)

        # Preencher matriz com coeficientes calculados
        for char, mapping in self.linguistic_map.items():
            genesis_idx = mapping['genesis_idx']
            native_token = mapping['native_token']
            coefficient = mapping['spectral_coefficient']

            self.coefficient_matrix[genesis_idx, native_token] = coefficient

        print(f"🔧 Linguistic coefficient matrix calibrated: {self.coefficient_matrix.shape}")

    def map_genesis_to_native(self, genesis_idx: int) -> Optional[int]:
        """Mapeia índice do Genesis para token nativo"""
        return self.genesis_to_native.get(genesis_idx)

    def map_native_to_genesis(self, native_token: int) -> Optional[int]:
        """Mapeia token nativo para índice do Genesis"""
        return self.native_to_genesis.get(native_token)

    def get_linguistic_coefficient(self, genesis_idx: int, native_token: int) -> float:
        """Retorna coeficiente linguístico para par genesis-native"""
        if self.coefficient_matrix is not None:
            return self.coefficient_matrix[genesis_idx, native_token].item()
        return 0.0

    def get_character_mapping(self, char: str) -> Optional[Dict]:
        """Retorna mapeamento completo para um caractere"""
        return self.linguistic_map.get(char)

    def save_mapping(self, filepath: str = "data/linguistic_mapping.json"):
        """Salva mapeamento linguístico em arquivo"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'linguistic_map': self.linguistic_map,
                    'genesis_to_native': self.genesis_to_native,
                    'native_to_genesis': self.native_to_genesis,
                    'coefficient_matrix_shape': list(self.coefficient_matrix.shape) if self.coefficient_matrix is not None else None
                }, f, indent=2, ensure_ascii=False)
            print(f"💾 Linguistic mapping saved to: {filepath}")
        except Exception as e:
            print(f"⚠️  Error saving linguistic mapping: {e}")

    def load_mapping(self, filepath: str = "data/linguistic_mapping.json"):
        """Carrega mapeamento linguístico do arquivo"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.linguistic_map = data.get('linguistic_map', {})
                self.genesis_to_native = data.get('genesis_to_native', {})
                self.native_to_genesis = data.get('native_to_genesis', {})
                print(f"📚 Linguistic mapping loaded from: {filepath}")
            else:
                print(f"⚠️  Linguistic mapping file not found: {filepath}")
        except Exception as e:
            print(f"⚠️  Error loading linguistic mapping: {e}")


class LinguisticCoefficientCalibrator:
    """
    Calibrador de Coeficientes Linguísticos

    Sistema automático para calibração de coeficientes linguísticos
    baseado em análise espectral e propriedades fonéticas.
    """

    def __init__(self, linguistic_system: LinguisticMappingSystem, device: str = "cpu"):
        self.linguistic_system = linguistic_system
        self.device = device

    def calibrate_from_text(self, text: str):
        """Calibra coeficientes baseado em análise de texto"""
        print(f"🔧 Calibrating linguistic coefficients from text: '{text[:50]}...'")

        # Análise básica do texto
        char_counts = {}
        for char in text:
            if char in self.linguistic_system.linguistic_map:
                char_counts[char] = char_counts.get(char, 0) + 1

        # Atualizar coeficientes baseado na frequência observada
        for char, count in char_counts.items():
            if char in self.linguistic_system.linguistic_map:
                # Ajuste baseado na frequência relativa
                relative_freq = count / len(text)
                current_coeff = self.linguistic_system.linguistic_map[char]['spectral_coefficient']

                # Atualizar coeficiente com média ponderada
                new_coeff = 0.7 * current_coeff + 0.3 * min(relative_freq * 5, 1.0)
                self.linguistic_system.linguistic_map[char]['spectral_coefficient'] = new_coeff

                # Atualizar matriz de coeficientes
                genesis_idx = self.linguistic_system.linguistic_map[char]['genesis_idx']
                native_token = self.linguistic_system.linguistic_map[char]['native_token']
                self.linguistic_system.coefficient_matrix[genesis_idx, native_token] = new_coeff

        print(f"✅ Linguistic coefficients calibrated for {len(char_counts)} characters")

    def adaptive_calibration(self, input_char: str, output_token: int, success: bool):
        """Calibração adaptativa baseada no sucesso da geração"""
        if input_char not in self.linguistic_system.linguistic_map:
            return

        mapping = self.linguistic_system.linguistic_map[input_char]
        expected_token = mapping['native_token']

        if success and output_token == expected_token:
            # Sucesso: aumentar coeficiente
            mapping['spectral_coefficient'] = min(mapping['spectral_coefficient'] * 1.1, 1.0)
        else:
            # Falha: diminuir coeficiente
            mapping['spectral_coefficient'] *= 0.9

        # Atualizar matriz
        genesis_idx = mapping['genesis_idx']
        native_token = mapping['native_token']
        self.linguistic_system.coefficient_matrix[genesis_idx, native_token] = mapping['spectral_coefficient']


def create_linguistic_mapping_system(device: str = "cpu") -> LinguisticMappingSystem:
    """Factory function para criar sistema de mapeamento linguístico"""
    return LinguisticMappingSystem(device=device)


def create_linguistic_coefficient_calibrator(linguistic_system: LinguisticMappingSystem,
                                           device: str = "cpu") -> LinguisticCoefficientCalibrator:
    """Factory function para criar calibrador de coeficientes"""
    return LinguisticCoefficientCalibrator(linguistic_system, device=device)


if __name__ == "__main__":
    # Teste do sistema
    print("🗣️  Testing Linguistic Mapping System...")

    system = create_linguistic_mapping_system()
    calibrator = create_linguistic_coefficient_calibrator(system)

    # Teste de mapeamento
    test_chars = ['a', 'b', 'e', 't', 'h']
    for char in test_chars:
        mapping = system.get_character_mapping(char)
        if mapping:
            print(f"  '{char}' → native_token: {mapping['native_token']}, spectral_coeff: {mapping['spectral_coefficient']:.3f}")
        else:
            print(f"  '{char}' → no mapping found")

    # Teste de calibração
    calibrator.calibrate_from_text("hello world this is a test")

    print("✅ Linguistic Mapping System test completed!")