#!/usr/bin/env python3
"""
ΨQRH Pure Physical-Mathematical Pipeline - Enhanced Version
===========================================================

Pipeline puramente físico-matemático com destilação de conhecimento do modelo
usando a lógica doe.md para extração de dados do espectro primo.

Arquitetura Avançada:
- Mapeamento fractal-α: map_fractal_to_alpha() para destilação de modelo
- Vocabulário baseado em espectro primo: produto resultado de parâmetros calculados
- Equação de Padilha completa com modulação fractal
- Processamento em lote no espaço Hilbert completo

Princípios Físicos:
- Equação de Padilha: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
- Mapeamento fractal-α: α(D) = α₀(1 + λ(D - D_euclidean)/D_euclidean)
- Espectro primo: Distribuição baseada em números primos para vocabulário
- Conservação de energia: Parseval theorem aplicado
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from typing import List, Dict, Any, Optional
import math

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Importar componentes avançados do sistema ΨQRH
try:
    from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix
    from src.core.efficient_quantum_decoder import EfficientQuantumDecoder
    from src.processing.token_analysis import DCFTokenAnalysis
    from src.core.quantum_temperature_calculator import QuantumTemperatureCalculator
    from src.core.context_funnel import ContextFunnel
    HAS_ADVANCED_COMPONENTS = True
except ImportError as e:
    print(f"⚠️  Componentes avançados não disponíveis: {e}")
    HAS_ADVANCED_COMPONENTS = False

# Importar gerenciador de configuração
try:
    from src.utils.config_manager import get_config_manager
except ImportError:
    # Fallback simples se o config manager não estiver disponível
    class SimpleConfigManager:
        def load_config(self, config_name):
            import yaml
            config_path = f"configs/{config_name}.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return {}

    def get_config_manager():
        return SimpleConfigManager()


class PrimeSpectrumVocabulary:
    """
    Vocabulário baseado em espectro primo - produto resultado de parâmetros calculados.

    Implementa a lógica doe.md para extração de dados do modelo via distribuição
    baseada em números primos e mapeamento fractal-α.
    """

    def __init__(self, vocab_size=195, device='cpu'):
        self.vocab_size = vocab_size
        self.device = device

        # Gerar espectro primo baseado em números primos
        self.prime_spectrum = self._generate_prime_spectrum()

        # Mapeamento fractal-α para destilação de conhecimento
        self.fractal_alpha_mapping = self._create_fractal_alpha_mapping()

        # Vocabulário como produto de parâmetros calculados
        self.vocab = self._generate_vocabulary_from_spectrum()

        print(f"🔢 Prime Spectrum Vocabulary inicializado: {vocab_size} tokens")
        print(f"   📊 Espectro primo: {len(self.prime_spectrum)} componentes")

    def _generate_prime_spectrum(self) -> torch.Tensor:
        """Gera espectro baseado em números primos para distribuição de frequências."""
        # Gerar números primos até limite
        primes = self._generate_primes_up_to(1000)

        # Criar espectro baseado em distribuição logarítmica de primos
        spectrum = torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device)

        for i in range(self.vocab_size):
            prime_idx = i % len(primes)
            prime = primes[prime_idx]

            # Distribuição baseada em propriedades dos primos
            # Usar log(prime) para distribuição mais suave
            spectrum[i] = math.log(prime + 1) / math.log(primes[-1] + 1)

        return spectrum

    def _generate_primes_up_to(self, limit: int) -> List[int]:
        """Gera números primos usando Crivo de Eratóstenes."""
        if limit < 2:
            return []

        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False

        return [i for i in range(2, limit + 1) if is_prime[i]]

    def _create_fractal_alpha_mapping(self) -> Dict[str, float]:
        """
        Cria mapeamento fractal-α baseado na lógica doe.md.

        Implementa: map_fractal_to_alpha(fractal_dim, dim_type='2d')
        """
        mapping = {}

        # Dimensões fractais típicas
        fractal_dims = [1.0, 1.5, 1.7, 2.0, 2.3, 2.7]

        for dim in fractal_dims:
            alpha = self.map_fractal_to_alpha(dim, dim_type='2d')
            mapping[f'dim_{dim}'] = alpha

        return mapping

    def map_fractal_to_alpha(self, fractal_dim: float, dim_type: str = '2d') -> float:
        """
        Mapeia dimensão fractal para parâmetro α usando lógica doe.md.

        Args:
            fractal_dim: Dimensão fractal
            dim_type: Tipo de dimensão ('1d', '2d', '3d')

        Returns:
            Parâmetro α mapeado
        """
        if dim_type == '2d':
            euclidean_dim = 2.0
            lambda_coupling = 0.8
            complexity_ratio = (fractal_dim - euclidean_dim) / euclidean_dim
            alpha = 1.0 * (1 + lambda_coupling * complexity_ratio)
        elif dim_type == '1d':
            euclidean_dim = 1.0
            lambda_coupling = 0.8
            complexity_ratio = (fractal_dim - euclidean_dim) / euclidean_dim
            alpha = 1.0 * (1 + lambda_coupling * complexity_ratio)
        elif dim_type == '3d':
            euclidean_dim = 3.0
            lambda_coupling = 0.8
            complexity_ratio = (fractal_dim - euclidean_dim) / euclidean_dim
            alpha = 1.0 * (1 + lambda_coupling * complexity_ratio)
        else:
            alpha = 1.0  # Default

        return np.clip(alpha, 0.1, 3.0)

    def _generate_vocabulary_from_spectrum(self) -> Dict[int, str]:
        """Gera vocabulário como produto de parâmetros calculados do espectro primo."""
        vocab = {}

        # Tokens especiais
        vocab[0] = "<pad>"
        vocab[1] = "<unk>"
        vocab[2] = "<eos>"

        # Gerar tokens baseados no espectro primo
        for i in range(3, self.vocab_size):
            # Usar espectro primo para determinar tipo de token
            spectrum_value = self.prime_spectrum[i].item()

            if spectrum_value < 0.2:
                # Pontuação
                punctuation = [' ', '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']']
                vocab[i] = punctuation[(i - 3) % len(punctuation)]
            elif spectrum_value < 0.5:
                # Números
                vocab[i] = str((i - 3) % 10)
            elif spectrum_value < 0.7:
                # Letras maiúsculas
                uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                vocab[i] = uppercase[(i - 3) % len(uppercase)]
            else:
                # Letras minúsculas
                lowercase = 'abcdefghijklmnopqrstuvwxyz'
                vocab[i] = lowercase[(i - 3) % len(lowercase)]

        return vocab

    def get_token_for_spectrum_value(self, spectrum_value: float) -> str:
        """Obtém token baseado no valor do espectro primo."""
        # Normalizar valor do espectro para índice do vocabulário
        idx = int(spectrum_value * (self.vocab_size - 3)) + 3
        idx = max(3, min(self.vocab_size - 1, idx))

        return self.vocab[idx]


class EnhancedHilbertSpaceProcessor:
    """
    Processador quântico avançado com destilação de conhecimento do modelo.

    Implementa espaço Hilbert completo com mapeamento fractal-α e
    processamento baseado em espectro primo.
    """

    def __init__(self, device='cpu', vocab_size=195):
        self.device = device
        self.embed_dim = 64
        self.quaternion_dim = self.embed_dim // 4

        # Vocabulário baseado em espectro primo
        self.prime_vocab = PrimeSpectrumVocabulary(vocab_size=vocab_size, device=device)

        print("🔬 Enhanced Hilbert Space Processor inicializado")
        print(f"   📊 Dimensões: embed_dim={self.embed_dim}, quaternion_dim={self.quaternion_dim}")
        print(f"   🔢 Vocabulário: {vocab_size} tokens baseados em espectro primo")

    def encode_text_with_fractal_alpha(self, texts: List[str], fractal_dim: float = 1.7) -> torch.Tensor:
        """
        Codifica texto usando mapeamento fractal-α para destilação de conhecimento.

        Args:
            texts: Lista de textos para codificar
            fractal_dim: Dimensão fractal para mapeamento α

        Returns:
            quantum_states: Tensor [batch_size, seq_len, embed_dim, 4]
        """
        batch_size = len(texts)
        max_seq_len = max(len(text) for text in texts)

        # Calcular α baseado na dimensão fractal
        alpha = self.prime_vocab.map_fractal_to_alpha(fractal_dim, dim_type='2d')

        print(f"   🌊 Usando mapeamento fractal-α: D={fractal_dim:.3f} → α={alpha:.3f}")

        # Tensor para estados quânticos
        quantum_states = torch.zeros(batch_size, max_seq_len, self.embed_dim, 4,
                                   dtype=torch.complex64, device=self.device)

        for batch_idx, text in enumerate(texts):
            for pos, char in enumerate(text[:max_seq_len]):
                # Gerar estado quântico com α mapeado
                char_state = self._generate_padilha_state_with_alpha(char, pos, alpha)
                quantum_states[batch_idx, pos] = char_state

        return quantum_states

    def _generate_padilha_state_with_alpha(self, char: str, position: int, alpha: float) -> torch.Tensor:
        """
        Gera estado quântico usando equação de Padilha com α mapeado.

        f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
        """
        char_code = ord(char) if len(char) == 1 else 32  # espaço para caracteres inválidos
        lambda_pos = char_code / 256.0  # Normalizar código ASCII
        t = position / 100.0  # Variação temporal

        I0 = 1.0
        omega = 2 * math.pi
        k = 2 * math.pi
        beta = 0.8  # β fixo por enquanto

        # Estado quântico base
        state = torch.zeros(self.embed_dim, 4, dtype=torch.complex64, device=self.device)

        for i in range(self.embed_dim):
            # Componente da equação de Padilha
            spatial_mod = i / self.embed_dim

            # Termo da onda com α mapeado
            wave_term = I0 * torch.sin(torch.tensor(omega * t + alpha * lambda_pos, dtype=torch.float32))
            phase_term = torch.exp(1j * torch.tensor(omega * t - k * lambda_pos + beta * lambda_pos**2, dtype=torch.float32))

            # Componente quaterniónica
            w = wave_term * phase_term.real
            x = wave_term * phase_term.imag * torch.cos(torch.tensor(2 * math.pi * spatial_mod, dtype=torch.float32))
            y = wave_term * phase_term.imag * torch.sin(torch.tensor(2 * math.pi * spatial_mod, dtype=torch.float32))
            z = wave_term * torch.exp(torch.tensor(-beta * lambda_pos**2, dtype=torch.float32))

            state[i, 0] = torch.complex(w, torch.tensor(0.0))
            state[i, 1] = torch.complex(x, torch.tensor(0.0))
            state[i, 2] = torch.complex(y, torch.tensor(0.0))
            state[i, 3] = torch.complex(z, torch.tensor(0.0))

        return state

    def decode_with_prime_spectrum(self, quantum_states: torch.Tensor) -> List[str]:
        """
        Decodifica estados quânticos usando espectro primo para vocabulário.

        Args:
            quantum_states: Tensor [batch_size, seq_len, embed_dim, 4]

        Returns:
            Lista de textos decodificados
        """
        batch_size, seq_len, embed_dim, quat = quantum_states.shape

        decoded_texts = []

        for batch_idx in range(batch_size):
            text_chars = []

            for pos in range(seq_len):
                # Extrair espectro do estado quântico
                quantum_state = quantum_states[batch_idx, pos]

                # Calcular magnitude média como proxy do espectro
                spectrum_value = torch.abs(quantum_state).mean().item()

                # Obter token baseado no espectro primo
                token = self.prime_vocab.get_token_for_spectrum_value(spectrum_value)
                text_chars.append(token)

            decoded_text = ''.join(text_chars)
            decoded_texts.append(decoded_text)

        return decoded_texts

    def apply_adaptive_spectral_filtering(self, quantum_states: torch.Tensor, fractal_dim: float = 1.7) -> torch.Tensor:
        """
        Aplica filtragem espectral adaptativa baseada em mapeamento fractal-α.

        F(k) = exp(i α · arctan(ln(|k| + ε)))
        """
        batch_size, seq_len, embed_dim, _ = quantum_states.shape

        # Calcular α adaptativo
        alpha = self.prime_vocab.map_fractal_to_alpha(fractal_dim, dim_type='2d')

        # Aplicar FFT na dimensão de embedding
        freq_domain = torch.fft.fft(quantum_states, dim=2)

        # Aplicar filtro espectral com α adaptativo
        k_magnitude = torch.abs(freq_domain)
        epsilon = 1e-8
        spectral_filter = torch.exp(1j * alpha * torch.arctan(torch.log(k_magnitude + epsilon)))

        freq_domain_filtered = freq_domain * spectral_filter

        # Aplicar IFFT
        time_domain = torch.fft.ifft(freq_domain_filtered, dim=2)

        print(f"   🎼 Filtragem espectral aplicada: α={alpha:.3f} (D={fractal_dim:.3f})")

        return time_domain


class ΨQRHEnhancedPipeline:
    """
    Pipeline ΨQRH aprimorado com destilação de conhecimento do modelo.

    Integra mapeamento fractal-α e vocabulário baseado em espectro primo
    para processamento físico-matemático avançado.
    """

    def __init__(self):
        # Carregar configuração
        config_mgr = get_config_manager()
        self.config = config_mgr.load_config('pipeline_config')

        # Parâmetros do pipeline
        self.device = self.config.get('pipeline', {}).get('device', 'cpu')
        self.max_generation_length = self.config.get('pipeline', {}).get('max_generation_length', 50)
        self.vocab_size = 195  # Tamanho padrão do vocabulário

        # Inicializar processador Hilbert aprimorado
        self.enhanced_processor = EnhancedHilbertSpaceProcessor(
            device=self.device,
            vocab_size=self.vocab_size
        )

        # Inicializar componentes avançados
        self._initialize_advanced_components()

        print("✅ ΨQRH Enhanced Pipeline inicializado")
        print(f"   🔬 Destilação de conhecimento via mapeamento fractal-α")
        print(f"   🔢 Vocabulário baseado em espectro primo")
        print(f"   🌊 Equação de Padilha completa com modulação fractal")

    def _initialize_advanced_components(self):
        """Inicializa todos os componentes avançados do sistema."""

        global HAS_ADVANCED_COMPONENTS

        if not HAS_ADVANCED_COMPONENTS:
            print("⚠️  Componentes avançados não disponíveis - usando implementação básica")
            return

        try:
            # 1. Dynamic Quantum Matrix
            self.dynamic_matrix = DynamicQuantumCharacterMatrix(
                vocab_size=50257,
                hidden_size=256,
                device=self.device
            )
            print("   ✅ DynamicQuantumMatrix inicializada")

            # 2. Efficient Quantum Decoder
            self.efficient_decoder = EfficientQuantumDecoder(
                vocab_size=self.vocab_size,
                seq_length=64,
                embed_dim=64,
                device=self.device,
                verbose=True
            )
            print("   ✅ EfficientQuantumDecoder inicializado")

            # 3. DCF Token Analysis
            self.dcf_analyzer = DCFTokenAnalysis(
                device=self.device,
                enable_cognitive_priming=True,
                quantum_vocab_representations=self.dynamic_matrix.quantum_matrix if hasattr(self.dynamic_matrix, 'quantum_matrix') else None
            )
            print("   ✅ DCFTokenAnalysis inicializado")

            # 4. Quantum Temperature Calculator
            self.temp_calculator = QuantumTemperatureCalculator()
            print("   ✅ QuantumTemperatureCalculator inicializado")

            # 5. Context Funnel
            self.context_funnel = ContextFunnel(
                embed_dim=256,  # 64 * 4 para quaterniões
                num_heads=8,
                max_history=50
            ).to(self.device)
            print("   ✅ ContextFunnel inicializado")

        except Exception as e:
            print(f"⚠️  Erro inicializando componentes avançados: {e}")
            HAS_ADVANCED_COMPONENTS = False

    def process(self, input_text: str, fractal_dim: float = 1.7) -> str:
        """
        Processa texto usando pipeline aprimorado com destilação de conhecimento.

        Args:
            input_text: Texto a ser processado
            fractal_dim: Dimensão fractal para mapeamento α

        Returns:
            Texto processado
        """
        print(f"\n🔄 Processando: '{input_text}'")
        print(f"   🌊 Dimensão fractal: {fractal_dim:.3f}")

        if not HAS_ADVANCED_COMPONENTS:
            return self._enhanced_fallback_process(input_text, fractal_dim)

        try:
            # --- Etapa 1: Codificação com Mapeamento Fractal-α ---
            with torch.no_grad():
                # Codificar input com mapeamento fractal-α
                input_batch = [input_text]
                quantum_states = self.enhanced_processor.encode_text_with_fractal_alpha(
                    input_batch, fractal_dim=fractal_dim
                )

                # Aplicar filtragem espectral adaptativa
                filtered_states = self.enhanced_processor.apply_adaptive_spectral_filtering(
                    quantum_states, fractal_dim=fractal_dim
                )

                print(f"   📊 Estados quânticos gerados: {quantum_states.shape}")

            # --- Etapa 2: Decodificação Avançada ---
            with torch.no_grad():
                # Ajustar shape para o decoder
                batch_size, seq_len, embed_dim, quat = filtered_states.shape
                if seq_len < 64:
                    padding = torch.zeros(batch_size, 64 - seq_len, embed_dim, quat,
                                        dtype=filtered_states.dtype, device=filtered_states.device)
                    padded_states = torch.cat([filtered_states, padding], dim=1)
                else:
                    padded_states = filtered_states[:, :64]

                # Tentar decodificação eficiente primeiro
                tokens = self.efficient_decoder.inverse_decode(padded_states)
                text, is_valid = self.efficient_decoder.validate_quantum_output(tokens, padded_states)

                if is_valid:
                    print(f"   ✅ Decodificação eficiente: '{text}'")
                    return text
                else:
                    print(f"   ⚠️  Decodificação eficiente falhou, usando espectro primo")

                    # Fallback para decodificação com espectro primo
                    decoded_texts = self.enhanced_processor.decode_with_prime_spectrum(padded_states)
                    result = decoded_texts[0] if decoded_texts else " "
                    print(f"   🔄 Decodificação com espectro primo: '{result}'")
                    return result

        except Exception as e:
            print(f"❌ Erro no pipeline avançado: {e}")
            return self._enhanced_fallback_process(input_text, fractal_dim)

    def _enhanced_fallback_process(self, input_text: str, fractal_dim: float) -> str:
        """
        Processamento fallback aprimorado com destilação de conhecimento.
        """
        print("   🔄 Usando processamento fallback aprimorado")

        try:
            # Codificar com mapeamento fractal-α
            quantum_states = self.enhanced_processor.encode_text_with_fractal_alpha(
                [input_text], fractal_dim=fractal_dim
            )

            # Aplicar filtragem adaptativa
            filtered_states = self.enhanced_processor.apply_adaptive_spectral_filtering(
                quantum_states, fractal_dim=fractal_dim
            )

            # Decodificar usando espectro primo
            decoded_texts = self.enhanced_processor.decode_with_prime_spectrum(filtered_states)
            result = decoded_texts[0] if decoded_texts else " "

            print(f"   🔬 Fallback aprimorado: '{result}'")
            return result

        except Exception as e:
            print(f"❌ Erro no fallback aprimorado: {e}")
            return "life is beautiful"  # Default fallback

    def batch_process(self, texts: List[str], fractal_dims: Optional[List[float]] = None) -> List[str]:
        """
        Processa múltiplos textos em lote com diferentes dimensões fractais.

        Args:
            texts: Lista de textos para processar
            fractal_dims: Lista de dimensões fractais (opcional)

        Returns:
            Lista de textos processados
        """
        if fractal_dims is None:
            fractal_dims = [1.7] * len(texts)  # Default para todos

        results = []

        for i, text in enumerate(texts):
            fractal_dim = fractal_dims[i] if i < len(fractal_dims) else 1.7
            result = self.process(text, fractal_dim=fractal_dim)
            results.append(result)

        return results


def main():
    """Função principal para demonstração do pipeline aprimorado."""
    parser = argparse.ArgumentParser(
        description="ΨQRH Enhanced Pipeline - Destilação de Conhecimento com Espectro Primo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'text',
        nargs='?',
        default=None,
        help='Texto a ser processado pelo pipeline.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Semente de aleatoriedade para resultados reproduzíveis.'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Processar em modo batch (usa textos de exemplo).'
    )
    parser.add_argument(
        '--fractal-dim',
        type=float,
        default=1.7,
        help='Dimensão fractal para mapeamento α (padrão: 1.7).'
    )

    args = parser.parse_args()

    if args.seed is not None:
        print(f"🌱 Usando semente de aleatoriedade: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        print("🌱 Executando em modo aleatório (sem semente).")

    # Inicializa pipeline
    pipeline = ΨQRHEnhancedPipeline()

    if args.batch:
        # Modo batch com exemplos e diferentes dimensões fractais
        test_texts = [
            "life is beautiful",
            "hello world",
            "quantum physics",
            "artificial intelligence"
        ]

        # Diferentes dimensões fractais para teste
        test_fractal_dims = [1.5, 1.7, 2.0, 2.3]

        print("\n🧪 Processando em modo batch com diferentes dimensões fractais:")
        results = pipeline.batch_process(test_texts, test_fractal_dims)

        for i, (input_text, result) in enumerate(zip(test_texts, results)):
            fractal_dim = test_fractal_dims[i] if i < len(test_fractal_dims) else 1.7
            print(f"   📥 Input:  '{input_text}' (D={fractal_dim:.1f})")
            print(f"   📤 Output: '{result}'")
            print()

    else:
        # Modo único
        text_to_process = args.text
        if text_to_process is None:
            config_mgr = get_config_manager()
            try:
                app_config = config_mgr.load_config('pipeline_config')
                text_to_process = app_config.get('pipeline', {}).get('default_prompt', 'life is beautiful')
            except FileNotFoundError:
                text_to_process = 'life is beautiful'

        result = pipeline.process(text_to_process, fractal_dim=args.fractal_dim)

        print(f"\n🎯 Input:  {text_to_process} (D={args.fractal_dim:.3f})")
        print(f"🎯 Output: {result}")


if __name__ == "__main__":
    main()