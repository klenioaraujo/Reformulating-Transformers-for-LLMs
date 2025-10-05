"""
Teste Direto com Modelo Pequeno ΨQRH
=====================================

Teste direto do pipeline completo com modelo pequeno e dados simples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dataset simples
sentences = [
    "hydrogen is the lightest element",
    "oxygen is essential for life",
    "carbon forms organic compounds",
    "nitrogen makes up most of the air",
    "iron is a magnetic metal"
]

# Criar vocabulário
all_chars = sorted(list(set("".join(sentences))))
vocab_size = len(all_chars)
char_to_id = {ch: i for i, ch in enumerate(all_chars)}
id_to_char = {i: ch for i, ch in enumerate(all_chars)}

print(f"Vocabulário: {vocab_size} caracteres")
print(f"Caracteres: {all_chars}")

# Modelo Transformer simples direto
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Criar modelo
model = SimpleTransformer(vocab_size, d_model=32, nhead=2, num_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print(f"\nTreinando modelo pequeno...")

# Treinamento rápido
for epoch in range(100):
    model.train()
    total_loss = 0

    for sentence in sentences:
        # Preparar dados
        input_ids = torch.tensor([char_to_id[c] for c in sentence[:-1]], dtype=torch.long)
        target_ids = torch.tensor([char_to_id[c] for c in sentence[1:]], dtype=torch.long)

        # Forward
        optimizer.zero_grad()
        logits = model(input_ids.unsqueeze(0))
        loss = criterion(logits.view(-1, vocab_size), target_ids)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Época {epoch}: perda = {total_loss/len(sentences):.4f}")

print(f"\nTreinamento concluído!")

# Teste de geração
print(f"\nTestando geração...")
model.eval()

with torch.no_grad():
    prompts = ["carbon ", "oxygen ", "iron "]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        # Converter para IDs
        input_ids = torch.tensor([char_to_id[c] for c in prompt], dtype=torch.long)
        generated = input_ids.clone()

        # Gerar 10 caracteres
        for _ in range(10):
            logits = model(generated.unsqueeze(0))
            next_token_logits = logits[0, -1, :]

            # Amostragem com temperatura
            probs = F.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token.squeeze().unsqueeze(0)])

        # Converter de volta para texto
        generated_text = "".join([id_to_char[id.item()] for id in generated])
        print(f"Geração: '{generated_text}'")