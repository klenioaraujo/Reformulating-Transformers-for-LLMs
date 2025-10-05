#!/usr/bin/env python3
"""
Scientific Spectral Processor
=============================
Abordagem científica: Processamento de padrões espectrais em vez de conversão direta
Baseado na análise: espectro → padrões → características → texto
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


class SpectralCharacteristicAnalyzer:
    """
    Analisa padrões espectrais para extrair características linguísticas
    """

    def analyze(self, spectrum: torch.Tensor) -> Dict[str, float]:
        """
        Analisa um espectro e extrai características relevantes para linguística
        """
        # Converter para magnitude se necessário
        if spectrum.is_complex():
            magnitude = torch.abs(spectrum)
        else:
            magnitude = spectrum

        # Normalizar
        magnitude = magnitude / (torch.max(magnitude) + 1e-8)

        characteristics = {}

        # 1. Frequência Fundamental (aproximada)
        characteristics['fundamental_freq'] = self._find_fundamental_frequency(magnitude)

        # 2. Razões Harmônicas
        characteristics['harmonic_ratios'] = self._compute_harmonic_ratios(magnitude,
                                                                         characteristics['fundamental_freq'])

        # 3. Medidas de Energia
        characteristics['spectral_centroid'] = self._compute_spectral_centroid(magnitude)
        characteristics['spectral_spread'] = self._compute_spectral_spread(magnitude)
        characteristics['spectral_flatness'] = self._compute_spectral_flatness(magnitude)

        # 4. Padrões de Fase (se disponível)
        if spectrum.is_complex():
            characteristics['phase_coherence'] = self._compute_phase_coherence(spectrum)
        else:
            characteristics['phase_coherence'] = 0.5  # Valor padrão

        return characteristics

    def _find_fundamental_frequency(self, magnitude: torch.Tensor) -> float:
        """Encontra a frequência fundamental aproximada"""
        # Encontrar pico principal
        peak_idx = torch.argmax(magnitude).item()
        fundamental = peak_idx / len(magnitude)
        return min(fundamental, 1.0)

    def _compute_harmonic_ratios(self, magnitude: torch.Tensor, fundamental: float) -> List[float]:
        """Calcula razões entre harmônicos"""
        ratios = []

        if fundamental > 0:
            # Calcular posições dos harmônicos
            harmonic_positions = []
            for k in range(1, 6):  # Primeiros 5 harmônicos
                pos = int(k * fundamental * len(magnitude))
                if pos < len(magnitude):
                    harmonic_positions.append(pos)

            # Calcular razões entre harmônicos consecutivos
            for i in range(len(harmonic_positions) - 1):
                if harmonic_positions[i] < len(magnitude) and harmonic_positions[i+1] < len(magnitude):
                    ratio = magnitude[harmonic_positions[i+1]] / (magnitude[harmonic_positions[i]] + 1e-8)
                    ratios.append(min(ratio.item(), 2.0))  # Limitar razão

        # Preencher com valores padrão se necessário
        while len(ratios) < 4:
            ratios.append(1.0)

        return ratios[:4]  # Retornar apenas as 4 primeiras razões

    def _compute_spectral_centroid(self, magnitude: torch.Tensor) -> float:
        """Calcula o centróide espectral"""
        frequencies = torch.linspace(0, 1, len(magnitude))
        weighted_sum = torch.sum(frequencies * magnitude)
        total_energy = torch.sum(magnitude) + 1e-8
        return (weighted_sum / total_energy).item()

    def _compute_spectral_spread(self, magnitude: torch.Tensor) -> float:
        """Calcula o spread espectral"""
        centroid = self._compute_spectral_centroid(magnitude)
        frequencies = torch.linspace(0, 1, len(magnitude))
        variance = torch.sum(((frequencies - centroid) ** 2) * magnitude)
        total_energy = torch.sum(magnitude) + 1e-8
        spread = torch.sqrt(variance / total_energy).item()
        return min(spread, 1.0)

    def _compute_spectral_flatness(self, magnitude: torch.Tensor) -> float:
        """Calcula a planicidade espectral"""
        # Evitar zeros para cálculo logarítmico
        magnitude_safe = magnitude + 1e-8

        geometric_mean = torch.exp(torch.mean(torch.log(magnitude_safe)))
        arithmetic_mean = torch.mean(magnitude_safe)

        flatness = geometric_mean / (arithmetic_mean + 1e-8)
        return flatness.item()

    def _compute_phase_coherence(self, spectrum: torch.Tensor) -> float:
        """Calcula coerência de fase (simplificado)"""
        phases = torch.angle(spectrum)
        phase_diff = torch.diff(phases)
        coherence = 1.0 - torch.std(phase_diff).item() / math.pi
        return max(coherence, 0.0)


class LinguisticCharacterMapper:
    """
    Mapeia características espectrais para caracteres usando regras linguísticas
    """

    def __init__(self):
        # Frequências linguísticas do inglês
        self.vowel_frequency = {'e': 12.7, 'a': 8.2, 'i': 7.0, 'o': 7.5, 'u': 2.8}
        self.consonant_frequency = {
            't': 9.1, 'n': 6.7, 's': 6.3, 'h': 6.1, 'r': 6.0,
            'd': 4.3, 'l': 4.0, 'c': 2.8, 'm': 2.4, 'w': 2.4,
            'f': 2.2, 'g': 2.0, 'y': 2.0, 'p': 1.9, 'b': 1.5,
            'v': 1.0, 'k': 0.8, 'j': 0.15, 'x': 0.15, 'q': 0.10, 'z': 0.07
        }

    def map(self, characteristics: Dict[str, float]) -> str:
        """
        Mapeia características espectrais para um caractere usando regras fonéticas
        """
        fundamental = characteristics['fundamental_freq']
        centroid = characteristics['spectral_centroid']
        spread = characteristics['spectral_spread']
        flatness = characteristics['spectral_flatness']
        harmonic_ratios = characteristics['harmonic_ratios']

        # Regras baseadas em fonética acústica

        # Vogais: fundamental baixo, harmônicos fortes, centróide médio
        if (fundamental < 0.3 and
            len(harmonic_ratios) > 0 and harmonic_ratios[0] > 0.7 and
            centroid > 0.3 and centroid < 0.7):
            return self._select_vowel(characteristics)

        # Consoantes sonoras: fundamental médio, spread moderado
        elif (0.3 <= fundamental <= 0.6 and
              spread > 0.4 and spread < 0.8):
            return self._select_voiced_consonant(characteristics)

        # Consoantes surdas: fundamental alto, spread alto
        elif (fundamental > 0.6 and
              spread > 0.6):
            return self._select_voiceless_consonant(characteristics)

        # Espaços/pontuação: energia concentrada em baixas frequências
        elif (centroid < 0.2 and
              flatness > 0.8):
            return self._select_punctuation(characteristics)

        # Caractere genérico (fallback)
        else:
            return self._select_generic(characteristics)

    def _select_vowel(self, characteristics: Dict) -> str:
        """Seleciona vogal baseada em características"""
        centroid = characteristics['spectral_centroid']

        # Vogais anteriores (i, e): centróide mais alto
        if centroid > 0.6:
            return 'e' if np.random.random() < 0.6 else 'i'
        # Vogais centrais (a): centróide médio
        elif centroid > 0.4:
            return 'a'
        # Vogais posteriores (o, u): centróide mais baixo
        else:
            return 'o' if np.random.random() < 0.6 else 'u'

    def _select_voiced_consonant(self, characteristics: Dict) -> str:
        """Seleciona consoante sonora"""
        spread = characteristics['spectral_spread']

        # Consoantes nasais (m, n): spread baixo
        if spread < 0.5:
            return 'n' if np.random.random() < 0.6 else 'm'
        # Consoantes líquidas (l, r): spread médio
        elif spread < 0.7:
            return 'r' if np.random.random() < 0.6 else 'l'
        # Consoantes plosivas sonoras (b, d, g): spread alto
        else:
            choices = ['b', 'd', 'g']
            return np.random.choice(choices)

    def _select_voiceless_consonant(self, characteristics: Dict) -> str:
        """Seleciona consoante surda"""
        flatness = characteristics['spectral_flatness']

        # Consoantes fricativas (s, f): planicidade alta
        if flatness > 0.7:
            return 's' if np.random.random() < 0.7 else 'f'
        # Consoantes plosivas surdas (p, t, k): planicidade baixa
        else:
            choices = ['p', 't', 'k']
            return np.random.choice(choices)

    def _select_punctuation(self, characteristics: Dict) -> str:
        """Seleciona pontuação ou espaço"""
        # Prioridade para espaço
        return ' '

    def _select_generic(self, characteristics: Dict) -> str:
        """Seleção genérica baseada em frequência linguística"""
        # Combinar todas as probabilidades
        all_chars = list(self.vowel_frequency.keys()) + list(self.consonant_frequency.keys())
        all_freqs = list(self.vowel_frequency.values()) + list(self.consonant_frequency.values())

        # Normalizar probabilidades
        total = sum(all_freqs)
        probabilities = [f/total for f in all_freqs]

        return np.random.choice(all_chars, p=probabilities)


class ContextProcessor:
    """
    Aplica regras contextuais e linguísticas à sequência de caracteres
    """

    def __init__(self):
        self.word_patterns = self._load_word_patterns()

    def apply_linguistic_rules(self, char_sequence: List[str]) -> str:
        """
        Aplica regras linguísticas para melhorar a coerência do texto
        """
        text = ''.join(char_sequence)

        # 1. Corrigir sequências de vogais/consoantes
        text = self._fix_vowel_consonant_patterns(text)

        # 2. Garantir espaçamento adequado
        text = self._fix_spacing(text)

        # 3. Aplicar regras básicas de capitalização
        text = self._apply_capitalization(text)

        return text

    def _load_word_patterns(self) -> Dict[str, float]:
        """Carrega padrões de palavras comuns (simplificado)"""
        # Padrões básicos do inglês
        return {
            'the': 0.05, 'and': 0.03, 'ing': 0.02, 'ion': 0.02,
            'ent': 0.01, 'ate': 0.01, 'ous': 0.01, 'ive': 0.01
        }

    def _fix_vowel_consonant_patterns(self, text: str) -> str:
        """Corrige padrões inválidos de vogais/consoantes"""
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'

        chars = list(text)

        for i in range(1, len(chars) - 1):
            # Evitar 3 vogais consecutivas
            if (chars[i-1] in vowels and chars[i] in vowels and chars[i+1] in vowels):
                chars[i] = np.random.choice(list(consonants))

            # Evitar 4 consoantes consecutivas
            if (i >= 3 and
                all(c in consonants for c in chars[i-3:i+1])):
                chars[i-1] = np.random.choice(list(vowels))

        return ''.join(chars)

    def _fix_spacing(self, text: str) -> str:
        """Garante espaçamento adequado"""
        # Garantir que espaços não estejam muito próximos
        chars = list(text)
        space_count = 0

        for i in range(len(chars)):
            if chars[i] == ' ':
                space_count += 1
                # Se muitos espaços consecutivos, remover alguns
                if space_count > 2:
                    chars[i] = np.random.choice(list('aeiourtns'))
            else:
                space_count = 0

        return ''.join(chars)

    def _apply_capitalization(self, text: str) -> str:
        """Aplica capitalização básica"""
        if len(text) > 0:
            # Capitalizar primeira letra
            text = text[0].upper() + text[1:]

            # Capitalizar após ponto final
            for i in range(1, len(text) - 1):
                if text[i] == '.' and i + 1 < len(text):
                    text = text[:i+1] + text[i+1].upper() + text[i+2:]

        return text


class ScientificSpectralProcessor:
    """
    Processador Espectral Científico
    Abordagem: espectro → padrões → características → texto
    """

    def __init__(self):
        self.characteristic_analyzer = SpectralCharacteristicAnalyzer()
        self.linguistic_mapper = LinguisticCharacterMapper()
        self.context_processor = ContextProcessor()

    def spectrum_to_text(self, spectral_sequence: torch.Tensor) -> str:
        """
        Converte sequência espectral para texto usando processamento científico
        """
        print(f"🔬 Processando {spectral_sequence.shape[1]} frames espectrais...")

        characteristics_sequence = []

        # 1. Extrair características de cada frame espectral
        for i in range(spectral_sequence.shape[1]):
            spectrum = spectral_sequence[0, i]  # Remover dimensão de batch
            characteristics = self.characteristic_analyzer.analyze(spectrum)
            characteristics_sequence.append(characteristics)

        print(f"   ✅ Características extraídas: {len(characteristics_sequence)} frames")

        # 2. Mapear características para caracteres
        char_sequence = []
        for i, characteristics in enumerate(characteristics_sequence):
            char = self.linguistic_mapper.map(characteristics)
            char_sequence.append(char)

        print(f"   🔤 Caracteres mapeados: {len(char_sequence)} caracteres")

        # 3. Aplicar processamento contextual
        text = self.context_processor.apply_linguistic_rules(char_sequence)

        print(f"   📝 Texto processado: {len(text)} caracteres finais")

        return text


def run_scientific_pipeline():
    """
    Executa o pipeline científico de processamento espectral
    """
    print("=" * 80)
    print("🧪 PIPELINE CIENTÍFICO - PROCESSAMENTO ESPECTRAL")
    print("=" * 80)
    print("Abordagem: espectro → padrões → características → texto")
    print()

    # Texto de teste
    test_text = "The quick brown fox jumps over the lazy dog"
    print(f"📝 Texto de teste: '{test_text}'")
    print(f"📊 Comprimento: {len(test_text)} caracteres")
    print()

    # Criar espectro simulado (para demonstração)
    # Em uma implementação real, isso viria do ΨQRH transform
    print("🔮 Gerando espectro simulado...")
    spectral_sequence = torch.randn(1, len(test_text), 64)  # [batch, seq_len, features]
    print(f"   ✅ Espectro gerado: shape {spectral_sequence.shape}")
    print()

    # Processar com abordagem científica
    processor = ScientificSpectralProcessor()

    print("🔄 Executando processamento científico...")
    reconstructed_text = processor.spectrum_to_text(spectral_sequence)
    print()

    # Análise de resultados
    print("📊 RESULTADOS:")
    print(f"   Texto original:  '{test_text}'")
    print(f"   Texto reconstruído: '{reconstructed_text}'")

    # Métricas básicas
    min_len = min(len(test_text), len(reconstructed_text))
    if min_len > 0:
        matches = sum(1 for i in range(min_len) if test_text[i] == reconstructed_text[i])
        accuracy = matches / min_len
        print(f"   Precisão de caracteres: {accuracy:.1%} ({matches}/{min_len})")

    print()
    print("🔍 ANÁLISE LINGUÍSTICA:")
    print(f"   Vogais no original: {sum(1 for c in test_text if c in 'aeiouAEIOU')}")
    print(f"   Vogais no reconstruído: {sum(1 for c in reconstructed_text if c in 'aeiouAEIOU')}")
    print(f"   Espaços no original: {test_text.count(' ')}")
    print(f"   Espaços no reconstruído: {reconstructed_text.count(' ')}")

    print("\n" + "=" * 80)
    print("🎯 PROCESSAMENTO CIENTÍFICO CONCLUÍDO")
    print("=" * 80)


if __name__ == "__main__":
    run_scientific_pipeline()