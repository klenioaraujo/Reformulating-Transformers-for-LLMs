#!/usr/bin/env python3
"""
Pipeline Científico Integrado
=============================
Combina ΨQRH transform com processamento espectral científico
Abordagem: Texto → ΨQRH → Espectro → Características → Texto
"""

import torch
import torch.fft as fft
import math
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Importar componentes do processamento científico
from examples.scientific_spectral_processor import (
    SpectralCharacteristicAnalyzer,
    LinguisticCharacterMapper,
    ContextProcessor,
    ScientificSpectralProcessor
)


def text_to_spectral_representation(text: str, embed_dim: int = 64) -> torch.Tensor:
    """
    Converte texto para representação espectral usando ΨQRH principles
    """
    print(f"📝 Convertendo texto para representação espectral: {len(text)} caracteres")

    # Converter texto para valores numéricos
    ascii_values = [ord(char) for char in text]
    seq_len = len(ascii_values)

    # Criar representação espectral baseada em características linguísticas
    spectral_sequence = []

    for i, ascii_val in enumerate(ascii_values):
        char = chr(ascii_val)

        # Criar espectro baseado em características fonéticas
        spectrum = torch.zeros(embed_dim)

        # Características baseadas no tipo de caractere
        if char.lower() in 'aeiou':
            # Vogais: harmônicos fortes, fundamental baixo
            fundamental = 0.2 + (ascii_val % 5) * 0.05
            for k in range(1, 6):
                freq_pos = int(k * fundamental * embed_dim)
                if freq_pos < embed_dim:
                    amplitude = 0.8 / k  # Decaimento harmônico
                    spectrum[freq_pos] = amplitude

        elif char in ' bcdfghjklmnpqrstvwxyz':
            # Consoantes: energia mais distribuída
            fundamental = 0.4 + (ascii_val % 7) * 0.03
            spread = 0.3 + (ascii_val % 3) * 0.1

            for j in range(embed_dim):
                freq = j / embed_dim
                distance = abs(freq - fundamental)
                amplitude = math.exp(-distance / spread) * 0.6
                spectrum[j] = amplitude

        elif char in '.,!?;:':
            # Pontuação: energia concentrada em baixas frequências
            for j in range(min(10, embed_dim)):
                spectrum[j] = 0.3 * (1 - j/10)

        else:
            # Outros caracteres: padrão genérico
            for j in range(embed_dim):
                phase = (ascii_val + j) * 2 * math.pi / 256
                spectrum[j] = 0.5 * math.sin(phase) + 0.5

        spectral_sequence.append(spectrum)

    spectral_tensor = torch.stack(spectral_sequence)
    spectral_tensor = spectral_tensor.unsqueeze(0)  # Adicionar dimensão de batch

    print(f"   ✅ Representação espectral criada: shape {spectral_tensor.shape}")
    return spectral_tensor


def apply_psiqrh_spectral_transform(spectral_sequence: torch.Tensor) -> torch.Tensor:
    """
    Aplica transformação ΨQRH no domínio espectral
    """
    print("🌀 Aplicando transformação ΨQRH espectral...")

    batch_size, seq_len, embed_dim = spectral_sequence.shape

    # FFT ao longo da dimensão sequencial
    spectral_fft = fft.fft(spectral_sequence, dim=1)

    # Filtro espectral baseado em ΨQRH
    freqs = fft.fftfreq(seq_len)
    k = 2 * math.pi * freqs.view(1, seq_len, 1)

    # Filtro: F(k) = exp(iα · arctan(ln(|k| + ε)))
    alpha = 1.0
    epsilon = 1e-10
    k_mag = torch.abs(k) + epsilon
    log_k = torch.log(k_mag)
    phase = torch.atan(log_k)
    filter_response = torch.exp(1j * alpha * phase)

    # Aplicar filtro
    filtered_fft = spectral_fft * filter_response

    # FFT inversa
    transformed_spectrum = fft.ifft(filtered_fft, dim=1).real

    print(f"   ✅ Transformação ΨQRH aplicada: shape {transformed_spectrum.shape}")
    return transformed_spectrum


class IntegratedScientificPipeline:
    """
    Pipeline Científico Integrado
    Combina ΨQRH com processamento espectral baseado em características
    """

    def __init__(self):
        self.spectral_processor = ScientificSpectralProcessor()

    def process_text(self, text: str) -> Dict:
        """
        Processa texto completo através do pipeline integrado
        """
        print(f"🔧 Iniciando pipeline integrado para texto de {len(text)} caracteres")

        # 1. Texto → Representação Espectral
        print("\n📊 ETAPA 1: Texto → Representação Espectral")
        spectral_representation = text_to_spectral_representation(text)

        # 2. Aplicar ΨQRH Transform
        print("\n🌀 ETAPA 2: Aplicar ΨQRH Transform")
        transformed_spectrum = apply_psiqrh_spectral_transform(spectral_representation)

        # 3. Processamento Científico: Espectro → Texto
        print("\n🔬 ETAPA 3: Processamento Científico")
        reconstructed_text = self.spectral_processor.spectrum_to_text(transformed_spectrum)

        # 4. Análise de Resultados
        print("\n📈 ETAPA 4: Análise de Resultados")
        results = self._analyze_results(text, reconstructed_text)

        return results

    def _analyze_results(self, original: str, reconstructed: str) -> Dict:
        """Analisa resultados do pipeline"""
        min_len = min(len(original), len(reconstructed))

        if min_len == 0:
            return {'accuracy': 0.0, 'analysis': {}}

        # Contar correspondências
        matches = sum(1 for i in range(min_len) if original[i] == reconstructed[i])
        accuracy = matches / min_len

        # Análise linguística detalhada
        analysis = {
            'vowel_accuracy': self._calculate_vowel_accuracy(original, reconstructed),
            'consonant_accuracy': self._calculate_consonant_accuracy(original, reconstructed),
            'space_accuracy': self._calculate_space_accuracy(original, reconstructed),
            'word_structure': self._analyze_word_structure(original, reconstructed)
        }

        return {
            'original_text': original,
            'reconstructed_text': reconstructed,
            'accuracy': accuracy,
            'matches': matches,
            'total_chars': min_len,
            'analysis': analysis
        }

    def _calculate_vowel_accuracy(self, original: str, reconstructed: str) -> float:
        """Calcula precisão para vogais"""
        vowels = 'aeiouAEIOU'
        original_vowels = [c for c in original if c in vowels]
        reconstructed_vowels = [c for c in reconstructed if c in vowels]

        min_len = min(len(original_vowels), len(reconstructed_vowels))
        if min_len == 0:
            return 0.0

        matches = sum(1 for i in range(min_len)
                     if original_vowels[i] == reconstructed_vowels[i])
        return matches / min_len

    def _calculate_consonant_accuracy(self, original: str, reconstructed: str) -> float:
        """Calcula precisão para consoantes"""
        consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'
        original_consonants = [c for c in original if c in consonants]
        reconstructed_consonants = [c for c in reconstructed if c in consonants]

        min_len = min(len(original_consonants), len(reconstructed_consonants))
        if min_len == 0:
            return 0.0

        matches = sum(1 for i in range(min_len)
                     if original_consonants[i] == reconstructed_consonants[i])
        return matches / min_len

    def _calculate_space_accuracy(self, original: str, reconstructed: str) -> float:
        """Calcula precisão para espaços"""
        original_spaces = [i for i, c in enumerate(original) if c == ' ']
        reconstructed_spaces = [i for i, c in enumerate(reconstructed) if c == ' ']

        if not original_spaces:
            return 1.0 if not reconstructed_spaces else 0.0

        # Verificar se espaços estão em posições similares
        matches = 0
        for pos in original_spaces:
            if pos < len(reconstructed) and reconstructed[pos] == ' ':
                matches += 1

        return matches / len(original_spaces)

    def _analyze_word_structure(self, original: str, reconstructed: str) -> Dict:
        """Analisa estrutura de palavras"""
        original_words = original.split()
        reconstructed_words = reconstructed.split()

        return {
            'original_word_count': len(original_words),
            'reconstructed_word_count': len(reconstructed_words),
            'avg_word_length_original': np.mean([len(w) for w in original_words]) if original_words else 0,
            'avg_word_length_reconstructed': np.mean([len(w) for w in reconstructed_words]) if reconstructed_words else 0
        }


def run_integrated_pipeline():
    """
    Executa o pipeline científico integrado completo
    """
    print("=" * 80)
    print("🧪 PIPELINE CIENTÍFICO INTEGRADO - ΨQRH + PROCESSAMENTO ESPECTRAL")
    print("=" * 80)
    print("Abordagem: Texto → ΨQRH → Espectro → Características → Texto")
    print()

    # Texto de teste
    test_texts = [
        "The quick brown fox",
        "Hello world",
        "Natural language processing",
        "Quantum spectral transform"
    ]

    pipeline = IntegratedScientificPipeline()

    for i, test_text in enumerate(test_texts, 1):
        print(f"\n🔬 TESTE {i}: '{test_text}'")
        print("-" * 50)

        results = pipeline.process_text(test_text)

        # Exibir resultados
        print(f"\n📊 RESULTADOS DO TESTE {i}:")
        print(f"   Original:      '{results['original_text']}'")
        print(f"   Reconstruído:  '{results['reconstructed_text']}'")
        print(f"   Precisão:      {results['accuracy']:.1%} ({results['matches']}/{results['total_chars']})")

        # Análise detalhada
        analysis = results['analysis']
        print(f"\n🔍 ANÁLISE DETALHADA:")
        print(f"   Precisão vogais:    {analysis['vowel_accuracy']:.1%}")
        print(f"   Precisão consoantes: {analysis['consonant_accuracy']:.1%}")
        print(f"   Precisão espaços:   {analysis['space_accuracy']:.1%}")

        word_analysis = analysis['word_structure']
        print(f"   Palavras originais: {word_analysis['original_word_count']}")
        print(f"   Palavras reconstruídas: {word_analysis['reconstructed_word_count']}")

    print("\n" + "=" * 80)
    print("🎯 PIPELINE INTEGRADO CONCLUÍDO")
    print("=" * 80)


if __name__ == "__main__":
    run_integrated_pipeline()