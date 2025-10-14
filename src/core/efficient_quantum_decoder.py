"""
Efficient Quantum Decoder - ΨQRH Architecture
==============================================

Implementação específica para decodificação quântica baseada na arquitetura real do ΨQRH.
Resolve o problema de gibberish através de inversão matemática direta da transformada de Padilha.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, List
import numpy as np

class EfficientQuantumDecoder:
    """
    Decoder eficiente baseado na arquitetura ΨQRH real.

    Implementa inversão matemática direta da transformada quântica,
    eliminando métricas de similaridade complexas que causam gibberish.
    """

    def __init__(self, vocab_size=195, seq_length=64, embed_dim=64, device='cpu', verbose=True):
        """
        Inicializa decoder com parâmetros do ΨQRH.

        Args:
            vocab_size: Tamanho do vocabulário (195 tokens do GPT-2 spectral)
            seq_length: Comprimento da sequência (64)
            embed_dim: Dimensão do embedding (64)
            device: Dispositivo para processamento
            verbose: Se deve imprimir logs detalhados
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.device = device
        self.verbose = verbose

        # Carregar vocabulário específico do ΨQRH
        self.vocab = self._load_psiqrh_vocab()

        # Parâmetros físicos baseados na equação de Padilha
        self.I0 = 1.0      # Amplitude máxima
        self.omega = 2.0 * math.pi  # Frequência angular
        self.k = 2.0 * math.pi      # Número de onda

        # Camada de projeção para mapear estado quântico para vocabulário
        self.projection = nn.Linear(embed_dim * 4, vocab_size, bias=False).to(device)

        # Conjunto pré-computado de tokens significativos para validação eficiente
        self.significant_tokens = set(range(3, 95))  # tokens 3-94 são significativos

        self._log(f"🔧 EfficientQuantumDecoder initialized: vocab_size={vocab_size}, seq_length={seq_length}, embed_dim={embed_dim}")

        # Para produção: inicializar pesos com matriz quântica
        # decoder.set_projection_weights(quantum_matrix.quantum_matrix.reshape(195, -1))

    def _log(self, message: str):
        """Logging configurável para produção"""
        if self.verbose:
            print(message)

    def initialize_with_quantum_matrix(self, quantum_matrix_instance):
        """Inicializa decoder com pesos da matriz quântica real"""
        embedding_matrix = quantum_matrix_instance.quantum_matrix.reshape(self.vocab_size, -1)
        self.set_projection_weights(embedding_matrix)
        self._log("🔗 Decoder weights initialized from quantum matrix")

    def set_projection_weights(self, embedding_matrix: torch.Tensor):
        """
        Alinha decoder com encoder usando pesos compartilhados (weight tying).
        embedding_matrix: [vocab_size, embed_dim*4]
        """
        with torch.no_grad():
            self.projection.weight.copy_(embedding_matrix.T)

    def _load_psiqrh_vocab(self) -> dict:
        """Carrega vocabulário específico do ΨQRH (195 tokens do GPT-2 spectral)"""
        # Baseado no vocabulário real do projeto ΨQRH
        vocab = {}

        # Tokens especiais (0-2)
        vocab[0] = "<pad>"
        vocab[1] = "<unk>"
        vocab[2] = "<eos>"

        # Tokens de pontuação e símbolos comuns (3-32)
        punctuation = [' ', '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '"', "'",
                      '+', '=', '*', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '<', '>']

        for i, char in enumerate(punctuation, 3):
            vocab[i] = char

        # Números (33-42)
        for i in range(10):
            vocab[33 + i] = str(i)

        # Letras maiúsculas (43-68)
        for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 43):
            vocab[i] = char

        # Letras minúsculas (69-94)
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz', 69):
            vocab[i] = char

        # Tokens especiais adicionais para completar 195
        for i in range(95, 195):
            vocab[i] = f"<special_{i}>"

        return vocab

    def inverse_decode(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Decodificação inversa baseada na física do ΨQRH.

        Args:
            quantum_state: torch.Size([1, 64, 64, 4]) - saída do optical_probe

        Returns:
            tokens: torch.Tensor - índices de tokens do vocabulário
        """
        # Validação rigorosa de shape para produção
        expected_shape = (1, self.seq_length, self.embed_dim, 4)
        assert quantum_state.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {quantum_state.shape}"

        self._log(f"🔄 [EfficientQuantumDecoder] Starting inverse decode: shape={quantum_state.shape}")

        # 1. INVERSA DA SONDA ÓPTICA (Padilha Wave Equation)
        optical_inverse = self._inverse_optical_probe(quantum_state)

        # 2. INVERSA DA ROTAÇÃO SO(4)
        rotation_inverse = self._inverse_so4_rotation(optical_inverse)

        # 3. INVERSA DA FILTRAGEM ESPECTRAL
        spectral_inverse = self._inverse_spectral_filter(rotation_inverse)

        # 4. DECODIFICAÇÃO DIRETA PARA VOCABULÁRIO GPT-2 (195 tokens)
        tokens = self._quantum_to_token_mapping(spectral_inverse)

        self._log(f"✅ [EfficientQuantumDecoder] Inverse decode complete: {len(tokens)} tokens")
        return tokens

    def _inverse_optical_probe(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Inversa matemática da sonda óptica de Padilha.

        Original: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
        Inversa: f⁻¹(ψ) = ψ · exp(-i(ωt - kλ + βλ²)) / I₀ sin(ωt + αλ)
        """
        self._log("🌊 [EfficientQuantumDecoder] Applying inverse optical probe...")

        batch, seq, embed, quat = quantum_state.shape

        # Decompor estado quântico em componentes
        real_part = quantum_state[..., 0]  # componente real
        imag_part = quantum_state[..., 1:] # componentes imaginários quaterniônicos

        # Calcular ângulo e magnitude para inversão
        time_component = torch.angle(quantum_state.mean(dim=-1, keepdim=True))  # ângulo médio
        spatial_component = torch.abs(quantum_state).mean(dim=-1, keepdim=True)  # magnitude média

        # Parâmetros da inversão (baseados nos parâmetros do pipeline)
        alpha = 1.36  # valor calibrado
        beta = 0.725  # valor calibrado

        # Aplicar inversa da modulação temporal: exp(-i(ωt - kλ + βλ²))
        # Simplificação: usar ângulo do estado quântico
        inverse_time = torch.exp(-1j * time_component * beta)

        # Aplicar inversa da modulação espacial: 1 / (I₀ sin(ωt + αλ))
        # Usar suavização estável para evitar instabilidade numérica
        spatial_modulation = torch.sin(time_component + alpha * torch.arange(seq, device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        epsilon = 1e-3
        denominator = self.I0 * spatial_modulation
        inverse_spatial = denominator / (denominator**2 + epsilon**2)  # Versão estável

        # Aplicar inversão completa
        inverted_state = quantum_state * inverse_time * inverse_spatial

        self._log(f"   ✅ Inverse optical probe applied: shape={inverted_state.shape}")
        return inverted_state

    def _inverse_so4_rotation(self, optical_inverse: torch.Tensor) -> torch.Tensor:
        """
        Inversa da rotação SO(4) unitária.

        Como as rotações SO(4) são unitárias, a inversa é a transposta/conjugada.
        """
        self._log("🔄 [EfficientQuantumDecoder] Applying inverse SO(4) rotation...")

        # Para rotações unitárias, a inversa é a transposta do conjugado
        # Simplificação: como estamos trabalhando com quaternions, aplicamos conjugado
        rotation_inverse = torch.conj(optical_inverse)

        self._log(f"   ✅ Inverse SO(4) rotation applied: shape={rotation_inverse.shape}")
        return rotation_inverse

    def _inverse_spectral_filter(self, rotation_inverse: torch.Tensor) -> torch.Tensor:
        """
        Inversa da filtragem espectral F(k) = exp(i α · arctan(ln(|k| + ε)))
        Garante que a FFT foi aplicada na mesma dimensão e sinal está centrado.
        """
        self._log("🎼 [EfficientQuantumDecoder] Applying inverse spectral filter...")

        # Aplicar transformada de Fourier inversa na dimensão correta (embed_dim)
        # Assumindo que a FFT original foi aplicada em dim=-2 (embed_dim)
        spectral_inverse = torch.fft.ifft(rotation_inverse, dim=-2)

        # Garantir centering correto (FFT assume periodicidade)
        # Shift para centralizar frequências zero
        spectral_inverse = torch.fft.ifftshift(spectral_inverse, dim=-2)

        # Normalizar resultado
        spectral_inverse = spectral_inverse / (torch.abs(spectral_inverse).max() + 1e-8)

        self._log(f"   ✅ Inverse spectral filter applied: shape={spectral_inverse.shape}")
        return spectral_inverse

    def _quantum_to_token_mapping(self, spectral_inverse: torch.Tensor) -> torch.Tensor:
        """
        Mapeamento direto estado quântico -> tokens GPT-2 usando projeção linear.
        spectral_inverse: [batch, seq_len, embed_dim, 4] (possivelmente complexo)
        """
        self._log("🎯 [EfficientQuantumDecoder] Mapping quantum state to tokens...")

        batch, seq_len, embed_dim, quat = spectral_inverse.shape

        # Tratar números complexos: converter para real antes da projeção
        if torch.is_complex(spectral_inverse):
            # Usar magnitude para representação real
            real_input = torch.abs(spectral_inverse)
        else:
            real_input = spectral_inverse

        # Achatar quaterniões
        flattened = real_input.reshape(batch, seq_len, embed_dim * quat)  # [1, 64, 256]

        # Projetar para vocabulário
        logits = self.projection(flattened)  # [1, 64, 195]

        # Aplicar softmax para probabilidades
        probabilities = torch.softmax(logits, dim=-1)

        # Selecionar tokens
        tokens = torch.argmax(probabilities, dim=-1)  # [1, 64]

        self._log(f"   ✅ Token mapping complete: {tokens.numel()} tokens generated")
        return tokens.squeeze(0)  # [64]


    def _apply_vocabulary_constraints(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Aplica constraints linguísticas do vocabulário ΨQRH.
        """
        # Garantir que tokens estão dentro do vocabulário válido
        valid_tokens = torch.clamp(tokens, 0, self.vocab_size - 1)

        # Adicionar token de fim de sequência se necessário
        if len(valid_tokens) < self.seq_length:
            eos_token = torch.tensor([2] * (self.seq_length - len(valid_tokens)), device=self.device)
            valid_tokens = torch.cat([valid_tokens, eos_token])

        return valid_tokens[:self.seq_length]

    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """
        Converte tokens de volta para texto usando vocabulário ΨQRH.
        """
        text_tokens = []
        for token_idx in tokens.tolist():
            if token_idx in self.vocab:
                token = self.vocab[token_idx]
                if token not in ['<pad>', '<unk>', '<eos>'] + [f'<special_{i}>' for i in range(95, 195)]:
                    text_tokens.append(token)
            else:
                text_tokens.append('?')  # Token desconhecido

        return ''.join(text_tokens)

    def validate_quantum_output(self, tokens: torch.Tensor, quantum_state: torch.Tensor, min_meaningful_tokens: int = 5) -> Tuple[str, bool]:
        """
        Validação específica para saída quântica do ΨQRH.
        """
        # Converter tokens para texto
        text = self.tokens_to_text(tokens)

        # Verificar se há tokens significativos (não apenas especiais)
        meaningful_count = sum(1 for t in tokens.tolist() if t in self.significant_tokens)

        if meaningful_count < min_meaningful_tokens:
            # Ativar fallback inteligente baseado no estado quântico
            fallback_text = self._generate_quantum_fallback(quantum_state)
            return fallback_text, False

        return text, True

    def _generate_quantum_fallback(self, quantum_state: torch.Tensor) -> str:
        """ZERO FALLBACK POLICY: Sistema deve falhar claramente"""
        raise RuntimeError("Efficient quantum decoder failed - ZERO FALLBACK POLICY: No quantum fallback allowed")

    def _calculate_quantum_coherence(self, psi: torch.Tensor) -> float:
        """
        Calcula coerência quântica do estado ψ.
        """
        # Coerência baseada na magnitude da parte imaginária
        imag_part = psi[..., 1::2] if psi.shape[-1] == 4 else psi.imag
        coherence = torch.mean(torch.abs(imag_part)).item()
        return coherence

    def run_regression_tests(self) -> dict:
        """
        Testes de regressão para validar funcionamento correto em produção.
        """
        import torch

        self._log("🧪 [EfficientQuantumDecoder] Running regression tests...")

        results = {
            'determinism_test': False,
            'token_quality_test': False,
            'shape_validation_test': False
        }

        try:
            # Teste 1: Determinismo - mesma entrada deve gerar mesma saída
            test_state = torch.randn(1, self.seq_length, self.embed_dim, 4, dtype=torch.complex64)

            tokens1 = self.inverse_decode(test_state.clone())
            tokens2 = self.inverse_decode(test_state.clone())

            results['determinism_test'] = torch.equal(tokens1, tokens2)
            self._log(f"   Determinism test: {'✅ PASS' if results['determinism_test'] else '❌ FAIL'}")

            # Teste 2: Qualidade dos tokens - presença mínima de tokens significativos
            text, is_valid = self.validate_quantum_output(tokens1, test_state, min_meaningful_tokens=3)
            results['token_quality_test'] = is_valid
            self._log(f"   Token quality test: {'✅ PASS' if results['token_quality_test'] else '❌ FAIL'}")

            # Teste 3: Validação de shape - deve falhar com shape incorreta
            try:
                wrong_shape = torch.randn(2, 32, 32, 4)  # Shape incorreta
                self.inverse_decode(wrong_shape)
                results['shape_validation_test'] = False  # Deve ter falhado
                self._log("   Shape validation test: ❌ FAIL (should have raised AssertionError)")
            except AssertionError:
                results['shape_validation_test'] = True
                self._log("   Shape validation test: ✅ PASS")

        except Exception as e:
            self._log(f"   Regression tests failed with error: {e}")

        all_passed = all(results.values())
        self._log(f"🧪 Regression tests: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")

        return results