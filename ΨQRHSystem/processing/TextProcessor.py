#!/usr/bin/env python3
"""
TextProcessor - Converte texto em sinais fractais e analisa suas propriedades.
"""

import torch
from typing import Dict, Any, Optional, Tuple

class TextProcessor:
    """
    Responsável pela primeira etapa do pipeline ΨQRH: conversão de texto para sinal fractal.
    """
    def __init__(self, device: str = 'cpu'):
        """
        Inicializa o processador de texto.

        Args:
            device: O dispositivo computacional (ex: 'cpu', 'cuda').
        """
        self.device = device
        print("✅ TextProcessor inicializado.")

    def text_to_fractal_signal(self, text: str, embed_dim: int, proc_params: Dict[str, Any] = None) -> torch.Tensor:
        """
        Converte texto para sinal fractal sequencial (doe.md 2.9.1: Fractal Embedding)

        Produz representação sequencial [seq_len, features] onde seq_len = len(text),
        permitindo processamento token-a-token em vez de representação global.
        
        (Lógica migrada de psiqrh.py: _text_to_fractal_signal)
        """
        seq_len = len(text)

        # Criar representação sequencial: cada caractere mapeado para um vetor de features
        signal_features = []
        for char in text:
            # Análise espectral básica do caractere
            char_value = torch.tensor([ord(char) / 127.0], dtype=torch.float32)

            # Criar representação multidimensional via análise de frequência simples
            base_features = torch.randn(embed_dim, device=self.device) * 0.1
            base_features[0] = char_value

            char_idx = ord(char.lower()) - ord('a') if char.isalpha() else 26
            if 0 <= char_idx < 27:
                base_features[1] = char_idx / 26.0

            base_features[2] = 1.0 if char.isupper() else 0.0
            base_features[3] = 1.0 if char.isdigit() else 0.0
            base_features[4] = 1.0 if char.isspace() else 0.0
            base_features[5] = 1.0 if char in 'aeiouAEIOU' else 0.0

            signal_features.append(base_features)

        signal = torch.stack(signal_features, dim=0)

        if proc_params and 'input_window' in proc_params:
            window_type = proc_params['input_window']
            window = torch.ones(seq_len, device=self.device)
            if window_type == 'hann':
                window = torch.hann_window(seq_len, device=self.device)
            elif window_type == 'hamming':
                window = torch.hamming_window(seq_len, device=self.device)
            signal = signal * window.unsqueeze(-1)

        return signal.to(self.device)

    def calculate_fractal_dimension(self, signal: torch.Tensor) -> float:
        """
        Calcula dimensão fractal via power-law fitting (doe.md 2.9.1)

        P(k) ~ k^(-β) → D = (3 - β) / 2
        
        (Lógica migrada de psiqrh.py: _calculate_fractal_dimension)
        """
        if signal.shape[0] < 2: # A análise de espectro precisa de pelo menos 2 pontos
            return 1.5 # Retorna um valor médio se o sinal for muito curto

        spectrum = torch.fft.fft(signal, dim=0)
        power_spectrum = torch.abs(spectrum) ** 2
        power_spectrum = power_spectrum.mean(dim=1) # Média através das features

        k = torch.arange(1, len(power_spectrum) + 1, dtype=torch.float32, device=self.device)

        log_k = torch.log(k.clamp(min=1e-9))
        log_P = torch.log(power_spectrum.clamp(min=1e-9))

        # Regressão linear para encontrar o expoente β
        n = len(log_k)
        sum_x = log_k.sum()
        sum_y = log_P.sum()
        sum_xy = (log_k * log_P).sum()
        sum_x2 = (log_k ** 2).sum()

        denominator = (n * sum_x2 - sum_x ** 2)
        if torch.abs(denominator) < 1e-9:
            return 1.5 # Evita divisão por zero

        beta = (n * sum_xy - sum_x * sum_y) / denominator

        D = (3.0 - beta.item()) / 2.0
        D = max(1.0, min(D, 2.0)) # Clamping para valores físicos

        return D

    def process(self, text: str, embed_dim: int, proc_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, float]:
        """
        Executa o processamento completo de texto para sinal e dimensão fractal.

        Args:
            text: O texto de entrada.
            embed_dim: A dimensão do embedding para o sinal.
            proc_params: Parâmetros de processamento adicionais.

        Returns:
            Uma tupla contendo o sinal fractal e a dimensão fractal.
        """
        fractal_signal = self.text_to_fractal_signal(text, embed_dim, proc_params)
        fractal_dimension = self.calculate_fractal_dimension(fractal_signal)
        return fractal_signal, fractal_dimension

# Exemplo de uso
if __name__ == '__main__':
    processor = TextProcessor(device='cpu')
    input_text = "Prove that √2 is irrational"
    embedding_dimension = 64

    signal, dimension = processor.process(input_text, embedding_dimension)

    print(f"Texto de entrada: '{input_text}'")
    print(f"Sinal Fractal gerado com shape: {signal.shape}")
    print(f"Dimensão Fractal calculada: {dimension:.4f}")
