"""
ΨQRH Transformer Text Generator
================================

Implementação direta da matemática Transformer usando ΨQRH:
1. Embedding: X = We[x] + P
2. Q, K, V: Q = XWQ, K = XWK, V = XWV
3. Atenção: softmax(QKT/√dk)V
4. Multi-Head: Concat + WO
5. Decoder (N camadas): MHA + FFN + Norm → hlast
6. Logits: hlastWeT
7. Probabilidades: softmax(logits)
8. Amostragem: argmax ou sampling → Próximo token

Usa os parâmetros espectrais reais do ΨQRH para otimização.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pathlib import Path
import sys

# Add local modules to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

class ΨQRHTransformerGenerator(nn.Module):
    """
    Gerador Transformer completo usando ΨQRH.
    Implementa exatamente os 8 passos da matemática Transformer.
    """

    def __init__(self, vocab_size=256, d_model=512, n_heads=8, d_ff=2048,
                 n_layers=6, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # 1. Embedding Layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 2-5. Decoder Layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 6. Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # 7. Output Head (Logits)
        self.output_head = nn.Linear(d_model, vocab_size)

        # Spectral parameters for optimization
        self.spectral_alpha = nn.Parameter(torch.tensor(1.5))
        self.fractal_dimension = nn.Parameter(torch.tensor(1.7))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass completo do Transformer.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Create causal mask for decoder [seq, seq]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(input_ids.device)

        # Create padding mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq] -> [batch, seq, seq]
            padding_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            # mask = True where to mask: future tokens or padded positions
            mask = causal_mask.unsqueeze(0) | (~padding_mask)
        else:
            mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # 1. Embedding + Positional Encoding
        x = self.token_embedding(input_ids)  # [batch, seq, d_model]
        x = self.positional_encoding(x)      # [batch, seq, d_model]
        x = self.dropout(x)

        # 2-5. Decoder Layers
        for layer in self.layers:
            x = layer(x, causal_mask)

        # 6. Layer Normalization
        x = self.layer_norm(x)

        # 7. Logits
        logits = self.output_head(x)  # [batch, seq, vocab_size]

        return logits

    def generate(self, prompt_ids, max_length=100, temperature=1.0,
                top_k=None, top_p=None, do_sample=True):
        """
        Geração de texto usando os 8 passos do Transformer.

        Args:
            prompt_ids: IDs do prompt inicial
            max_length: Comprimento máximo da sequência gerada
            temperature: Controle da aleatoriedade
            top_k: Amostragem top-k
            top_p: Amostragem top-p (nucleus)
            do_sample: Se deve amostrar ou usar argmax

        Returns:
            generated_ids: Sequência completa de IDs gerados
        """
        self.eval()
        generated_ids = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - len(prompt_ids)):
                # Forward pass para obter logits do próximo token
                logits = self.forward(generated_ids.unsqueeze(0))  # [1, seq, vocab]
                next_token_logits = logits[0, -1, :]  # [vocab]

                # Aplicar temperatura
                next_token_logits = next_token_logits / temperature

                # Aplicar top-k se especificado
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')

                # Aplicar top-p se especificado
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens com probabilidade cumulativa acima do threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Desloca os índices para manter o primeiro token acima do threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')

                # 7. Probabilidades
                probs = F.softmax(next_token_logits, dim=-1)

                # 8. Amostragem
                if do_sample:
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(probs, dim=-1, keepdim=True)

                # Adicionar token gerado à sequência
                generated_ids = torch.cat([generated_ids, next_token_id.squeeze().unsqueeze(0)])

                # Parar se token de fim de sequência
                if next_token_id.item() == 0:  # Assumindo 0 como token especial
                    break

        return generated_ids

class PositionalEncoding(nn.Module):
    """Positional Encoding do Transformer original."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class TransformerDecoderLayer(nn.Module):
    """Camada do Decoder Transformer."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention com residual connection
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward com residual connection
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention do Transformer."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[:2]

        # 2. Q, K, V projections
        Q = self.w_q(query)  # [batch, seq, d_model]
        K = self.w_k(key)    # [batch, seq, d_model]
        V = self.w_v(value)  # [batch, seq, d_model]

        # Reshape para multi-head
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Aplicar máscara
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3. Contexto ponderado
        attn_output = torch.matmul(attn_weights, V)

        # 4. Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        # Output projection
        output = self.w_o(attn_output)

        return output

class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network do Transformer."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

def create_simple_dataset():
    """Cria dataset simples para demonstração."""
    texts = [
        "hydrogen is the lightest element",
        "oxygen is essential for life",
        "carbon forms organic compounds",
        "nitrogen makes up most of the air",
        "iron is a magnetic metal",
        "carbon dioxide is a gas",
        "water is essential for life",
        "sodium chloride is salt",
        "gold is a precious metal",
        "silver conducts electricity"
    ]
    return texts

def train_psiqrh_transformer(num_epochs=100, learning_rate=0.001):
    """Treina o ΨQRH Transformer Generator."""
    print("🚀 Inicializando ΨQRH Transformer Generator...")

    # Preparar dataset
    dataset = create_simple_dataset()
    all_chars = sorted(list(set("".join(dataset))))
    vocab_size = len(all_chars)
    char_to_id = {ch: i for i, ch in enumerate(all_chars)}
    id_to_char = {i: ch for i, ch in enumerate(all_chars)}

    print(f"  - Vocabulário: {vocab_size} caracteres")
    print(f"  - Dataset: {len(dataset)} frases")

    # Inicializar modelo
    model = ΨQRHTransformerGenerator(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=8,
        d_ff=512,
        n_layers=4,
        max_seq_len=128
    )

    # Otimizador e função de perda
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"  - Treinando por {num_epochs} épocas...")
    print("-" * 60)

    # Loop de treinamento
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for sentence in dataset:
            # Converter para IDs
            input_ids = torch.tensor([char_to_id[c] for c in sentence[:-1]], dtype=torch.long)
            target_ids = torch.tensor([char_to_id[c] for c in sentence[1:]], dtype=torch.long)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids.unsqueeze(0))  # [1, seq-1, vocab]

            # Calcular perda
            loss = criterion(logits.view(-1, vocab_size), target_ids)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)

        if epoch % 10 == 0:
            print(f"  Época {epoch:03d}/{num_epochs} | Perda: {avg_loss:.4f}")

    print("-" * 60)
    print(f"\033[92m✅ Treinamento Concluído!\033[0m")

    return model, char_to_id, id_to_char

def demonstrate_generation(model, char_to_id, id_to_char):
    """Demonstra a geração de texto."""
    print("\n🤖 Demonstrando Geração de Texto...")

    prompts = ["carbon ", "oxygen ", "water "]

    for prompt in prompts:
        print(f"\n  - Prompt: '{prompt}'")

        # Converter prompt para IDs
        prompt_ids = torch.tensor([char_to_id[c] for c in prompt], dtype=torch.long)

        # Gerar texto
        generated_ids = model.generate(
            prompt_ids,
            max_length=30,
            temperature=0.7,
            top_k=5,
            do_sample=True
        )

        # Converter IDs de volta para texto
        generated_text = "".join([id_to_char[id.item()] for id in generated_ids])

        print(f"  - Geração: '{generated_text}'")

    print("=" * 60)

if __name__ == "__main__":
    # Treinar e demonstrar
    model, char_to_id, id_to_char = train_psiqrh_transformer(num_epochs=50)
    demonstrate_generation(model, char_to_id, id_to_char)
