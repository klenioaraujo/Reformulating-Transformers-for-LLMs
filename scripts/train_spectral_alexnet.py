#!/usr/bin/env python3
"""
Treinamento do Spectral AlexNet
================================

Treina a rede neural espectral de 3 camadas usando o banco de aprendizagem.

Processo de Aprendizagem:
1. Carregar banco espectral (Ψtws)
2. Para cada exemplo:
   - Camada 1: Buscar padrão no banco
   - Camada 2: Converter texto para espectro
   - Camada 3: Processar com SO(4) e autoacoplagem
3. Comparar as 3 camadas
4. Atualizar banco com novos padrões aprendidos

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.architecture.spectral_alexnet import SpectralAlexNet
from scripts.create_spectral_learning_bank import SpectralLearningBank


def load_training_data(bank: SpectralLearningBank, max_samples: int = 20):
    """Carrega dados de treinamento do banco."""

    print("\n📚 Carregando dados de treinamento...")

    # Pegar todas as entradas
    all_entries = list(bank.index["entries"].keys())[:max_samples]

    training_data = []
    for entry_hash in all_entries:
        data = bank.load_spectral_learning(entry_hash)

        # Usar campo espectral como input
        spectral_input = data['spectral_field']

        # Target: campo evoluído
        spectral_target = torch.fft.fft(data['evolved_field'].mean(dim=-1))

        training_data.append({
            'input': spectral_input,
            'target': spectral_target,
            'text': data['text'],
            'fci': data['fci'],
            'fractal_D': data['fractal_D']
        })

    print(f"   ✓ Carregados {len(training_data)} exemplos")
    return training_data


def train_epoch(
    model: SpectralAlexNet,
    training_data: list,
    optimizer: torch.optim.Optimizer,
    epoch: int
):
    """Treina uma época."""

    model.train()
    total_loss = 0.0
    total_resonance = 0.0
    total_coherence = 0.0

    print(f"\n{'='*70}")
    print(f"ÉPOCA {epoch}")
    print(f"{'='*70}")

    for i, sample in enumerate(training_data, 1):
        optimizer.zero_grad()

        # Input e target
        x = sample['input'].unsqueeze(0).to(model.device)  # [1, input_dim]
        target = sample['target'].unsqueeze(0).to(model.device)

        # Forward pass
        output = model(x, return_layer_outputs=True)

        # Loss: distância espectral + termo de ressonância
        spectral_loss = torch.abs(output['output'] - target).mean()

        # Termo de ressonância: queremos maximizar (1 - resonance para minimizar)
        resonance_term = 1.0 - output['comparison']['resonance']

        # Termo de coerência: queremos maximizar (1 - coherence para minimizar)
        coherence_term = 1.0 - output['comparison']['coherence']

        # Loss total
        loss = spectral_loss + 0.1 * resonance_term + 0.1 * coherence_term

        # Backward
        loss.backward()
        optimizer.step()

        # Estatísticas
        total_loss += loss.item()
        total_resonance += output['comparison']['resonance'].item()
        total_coherence += output['comparison']['coherence'].item()

        # Log
        if i % 5 == 0 or i == len(training_data):
            print(f"[{i}/{len(training_data)}] Loss: {loss.item():.6f} | "
                  f"Resonance: {output['comparison']['resonance'].item():.4f} | "
                  f"Coherence: {output['comparison']['coherence'].item():.4f}")

    avg_loss = total_loss / len(training_data)
    avg_resonance = total_resonance / len(training_data)
    avg_coherence = total_coherence / len(training_data)

    print(f"\n{'='*70}")
    print(f"RESUMO ÉPOCA {epoch}")
    print(f"{'='*70}")
    print(f"Loss média: {avg_loss:.6f}")
    print(f"Ressonância média: {avg_resonance:.4f}")
    print(f"Coerência média: {avg_coherence:.4f}")

    return avg_loss, avg_resonance, avg_coherence


def evaluate(model: SpectralAlexNet, training_data: list):
    """Avalia o modelo."""

    model.eval()

    print(f"\n{'='*70}")
    print("AVALIAÇÃO")
    print(f"{'='*70}")

    with torch.no_grad():
        resonances = []
        coherences = []

        for sample in training_data:
            x = sample['input'].unsqueeze(0).to(model.device)
            output = model(x)

            resonances.append(output['comparison']['resonance'].item())
            coherences.append(output['comparison']['coherence'].item())

        avg_resonance = np.mean(resonances)
        avg_coherence = np.mean(coherences)

        print(f"\nRessonância média: {avg_resonance:.4f}")
        print(f"Coerência média: {avg_coherence:.4f}")

        # Análise por faixa de ressonância
        high_res = sum(1 for r in resonances if r > 0.7)
        med_res = sum(1 for r in resonances if 0.4 <= r <= 0.7)
        low_res = sum(1 for r in resonances if r < 0.4)

        print(f"\nDistribuição de ressonância:")
        print(f"  Alta (>0.7): {high_res} ({100*high_res/len(resonances):.1f}%)")
        print(f"  Média (0.4-0.7): {med_res} ({100*med_res/len(resonances):.1f}%)")
        print(f"  Baixa (<0.4): {low_res} ({100*low_res/len(resonances):.1f}%)")


def update_bank_with_learned_patterns(
    model: SpectralAlexNet,
    bank: SpectralLearningBank,
    training_data: list,
    learning_rate: float = 0.01
):
    """Atualiza banco espectral com padrões aprendidos."""

    print(f"\n{'='*70}")
    print("ATUALIZANDO BANCO ESPECTRAL")
    print(f"{'='*70}")

    model.eval()
    new_patterns = []

    with torch.no_grad():
        for sample in training_data:
            x = sample['input'].unsqueeze(0).to(model.device)
            output = model(x)

            # Pegar output da camada de interpretação
            new_patterns.append(output['output'].squeeze(0))

    # Empilhar padrões
    new_patterns_tensor = torch.stack(new_patterns)

    # Atualizar banco do modelo
    model.update_spectral_bank(new_patterns_tensor, learning_rate=learning_rate)

    print(f"✓ Banco atualizado com {len(new_patterns)} novos padrões")
    print(f"  Taxa de aprendizagem: {learning_rate}")


def save_model(model: SpectralAlexNet, save_path: Path, metadata: dict):
    """Salva modelo e metadados."""

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Salvar estado do modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, save_path)

    print(f"\n💾 Modelo salvo: {save_path}")


def main():
    print("="*70)
    print("TREINAMENTO: Spectral AlexNet")
    print("="*70)

    # Configuração
    device = 'cpu'
    num_epochs = 10
    learning_rate = 0.001
    bank_update_lr = 0.01

    # Carregar banco de aprendizagem
    bank_path = "/home/padilha/trabalhos/QRH2/Reformulating-Transformers-for-LLMs/data/Ψtws"
    bank = SpectralLearningBank(bank_path)

    # Carregar dados de treinamento
    training_data = load_training_data(bank, max_samples=20)

    # Criar modelo
    model = SpectralAlexNet(
        input_dim=training_data[0]['input'].shape[0],
        hidden_dims=[256, 512, 256],
        spectral_vocab_size=256,
        logistic_r=3.8,
        device=device
    )
    model.to(device)

    # Otimizador (apenas parâmetros treináveis, não o banco)
    trainable_params = [
        p for name, p in model.named_parameters()
        if 'spectral_bank' not in name
    ]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Histórico de treinamento
    history = {
        'loss': [],
        'resonance': [],
        'coherence': []
    }

    # Treinamento
    for epoch in range(1, num_epochs + 1):
        loss, resonance, coherence = train_epoch(
            model, training_data, optimizer, epoch
        )

        history['loss'].append(loss)
        history['resonance'].append(resonance)
        history['coherence'].append(coherence)

        # Avaliar a cada 3 épocas
        if epoch % 3 == 0:
            evaluate(model, training_data)

        # Atualizar banco a cada 5 épocas
        if epoch % 5 == 0:
            update_bank_with_learned_patterns(
                model, bank, training_data, learning_rate=bank_update_lr
            )

    # Avaliação final
    evaluate(model, training_data)

    # Salvar modelo
    metadata = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'bank_update_lr': bank_update_lr,
        'num_samples': len(training_data),
        'final_loss': history['loss'][-1],
        'final_resonance': history['resonance'][-1],
        'final_coherence': history['coherence'][-1],
        'history': history,
        'timestamp': datetime.now().isoformat()
    }

    save_path = Path("models/spectral_alexnet_trained.pt")
    save_model(model, save_path, metadata)

    # Exportar estatísticas
    stats_path = Path("models/spectral_alexnet_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("✅ TREINAMENTO CONCLUÍDO")
    print(f"{'='*70}")
    print(f"Modelo salvo: {save_path}")
    print(f"Estatísticas: {stats_path}")


if __name__ == "__main__":
    main()
