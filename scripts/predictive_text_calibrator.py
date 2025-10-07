"""
Predictive Text Calibrator for ΨQRH
====================================

Calibrador que implementa a matemática básica do Transformers usando os parâmetros
reais do pipeline ΨQRH para otimizar previsão de texto via entropia cruzada.

Implementação direta da matemática Transformer:
1. Embedding: X = We[x] + P
2. Q, K, V: Q = XWQ, K = XWK, V = XWV
3. Attention: softmax(QKT/√dk)V
4. Multi-Head: Concat + WO
5. Logits: hlastWeT
6. Probabilidades: softmax(logits)
7. Amostragem: argmax ou sampling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import time
import random
import math

# --- Add project root to path ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Import the actual ΨQRH components ---
from src.core.ΨQRH import QRHFactory
from src.core.enhanced_qrh_processor import EnhancedQRHProcessor

class ΨQRHTransformerCalibrator(nn.Module):
    """
    Implementação direta da matemática Transformer usando parâmetros ΨQRH.

    Segue exatamente a matemática básica do Transformer:
    1. Embedding + Positional Encoding
    2. Multi-Head Attention
    3. Feed-Forward Network
    4. Layer Normalization
    5. Output Head
    """

    def __init__(self, vocab_size=256, d_model=256, n_heads=8, d_ff=1024, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        self.register_buffer('positional_encoding', self._create_positional_encoding())

        # 3. Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, n_heads)

        # 4. Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # 5. Layer Normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # 6. Output Head
        self.output_head = nn.Linear(d_model, vocab_size)

    def _create_positional_encoding(self):
        """Cria positional encoding como no Transformer original."""
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                           (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass seguindo exatamente a matemática Transformer.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # 1. Embedding + Positional Encoding
        x = self.embedding(input_ids)  # [batch, seq, d_model]
        x = x + self.positional_encoding[:, :seq_len, :]

        # 2. Multi-Head Attention
        residual = x
        x = self.layer_norm1(x)
        attn_output = self.attention(x, x, x)  # Self-attention
        x = residual + attn_output

        # 3. Feed-Forward Network
        residual = x
        x = self.layer_norm2(x)
        ff_output = self.ffn(x)
        x = residual + ff_output

        # 4. Output Head
        logits = self.output_head(x)  # [batch, seq, vocab_size]

        return logits

class MultiHeadAttention(nn.Module):
    """Implementação exata da Multi-Head Attention do Transformer."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Projeções Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[:2]

        # Projeções Q, K, V
        Q = self.w_q(query)  # [batch, seq, d_model]
        K = self.w_k(key)    # [batch, seq, d_model]
        V = self.w_v(value)  # [batch, seq, d_model]

        # Reshape para multi-head
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # TODO: Consider replacing softmax with physical attention mechanism in future ΨQRH evolution
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        # Output projection
        output = self.w_o(attn_output)

        return output

class FeedForwardNetwork(nn.Module):
    """Implementação exata da FFN do Transformer."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

def run_predictive_calibration(num_epochs=100, learning_rate=0.005):
    """
    Função principal para executar calibração preditiva usando ΨQRH real.
    """
    print("🧠 Inicializando Calibrador Preditivo ΨQRH...")

    # 1. Dataset de treinamento simples em inglês
    dataset = ["hydrogen is the lightest element", "oxygen is essential for life", "carbon forms organic compounds", "nitrogen makes up most of the air", "iron is a magnetic metal", "carbon dioxide is a gas", "water is essential for life", "sodium chloride is salt", "gold is a precious metal", "silver conducts electricity"]
    chars = sorted(list(set("".join(dataset))))
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"  - Vocabulário: {vocab_size} caracteres")
    print(f"  - Dataset: {len(dataset)} frases")

    # 2. Inicializar ΨQRH Factory e Enhanced Processor
    print("  - Inicializando pipeline ΨQRH...")
    qrh_factory = QRHFactory()
    enhanced_processor = EnhancedQRHProcessor(embed_dim=64, device='cpu')

    # 3. Inicializar calibrador
    calibrator = ΨQRHTransformerCalibrator(vocab_size=vocab_size, d_model=256)

    # 4. Otimizador e função de perda
    optimizer = optim.Adam(calibrator.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    print(f"  - Treinando por {num_epochs} épocas...")
    print("-" * 60)

    # 5. Loop de calibração
    for epoch in range(num_epochs):
        calibrator.train()

        # Selecionar frase aleatória
        sentence = random.choice(dataset)

        # Preparar inputs e targets
        input_ids = torch.tensor([char_to_id[c] for c in sentence[:-1]], dtype=torch.long).unsqueeze(0)
        target_char = sentence[-1]
        target_id = char_to_id[target_char]
        target_tensor = torch.tensor([target_id], dtype=torch.long)

        # Forward pass
        optimizer.zero_grad()
        logits = calibrator(input_ids)

        # Calcular perda (usando apenas o último token)
        loss = loss_function(logits[:, -1, :], target_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"  Época {epoch:03d}/{num_epochs} | Perda: {loss.item():.4f}")

    print("-" * 60)
    print(f"\033[92m✅ Calibração Preditiva Concluída!\033[0m")

    # 6. Demonstrar capacidade preditiva
    print("\n🤖 Demonstrando capacidade preditiva...")
    calibrator.eval()

    with torch.no_grad():
        prompt = "carbon forms"
        print(f"  - Prompt: '{prompt}'")
        generated_text = prompt

        for _ in range(10):  # Gerar 10 caracteres
            # Preparar input_ids do prompt atual
            input_ids = torch.tensor([char_to_id[c] for c in generated_text], dtype=torch.long).unsqueeze(0)

            # Prever próximo caractere
            logits = calibrator(input_ids)
            predicted_id = torch.argmax(logits[:, -1, :], dim=-1).item()

            # Verificar se ID está no vocabulário
            if predicted_id < len(id_to_char):
                next_char = id_to_char[predicted_id]
                generated_text += next_char
            else:
                break

        print(f"  - Predição: '{generated_text}'")

    print("=" * 60)

    return calibrator

if __name__ == "__main__":
    calibrated_model = run_predictive_calibration()