#!/usr/bin/env python3
"""
Treinamento com Evolução Quântica Estável
==========================================

Implementação do treinamento usando Prime Resonant Filtering + Leech Lattice Embedding
para resolver problemas de instabilidade numérica e colapso de similaridade.

Este treinamento substitui o FFT padrão por técnicas de estabilização baseadas
em princípios matemáticos avançados.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import argparse

# Import ΨQRH components
from psiqrh import ΨQRHPipeline
from src.core.losses import QuantumContrastiveLoss
from src.core.prime_resonant_filter import StableQuantumEvolution


class StableEvolutionTrainer:
    """
    Treinador para evolução quântica estável usando filtragem ressonante
    e embedding em Leech Lattice.
    """

    def __init__(self, embed_dim: int = 64, device: str = 'cpu'):
        """
        Inicializa o treinador.

        Args:
            embed_dim: Dimensão do embedding
            device: Dispositivo para computação
        """
        self.embed_dim = embed_dim
        self.device = device

        # Componentes principais
        self.stable_evolution = StableQuantumEvolution(embed_dim=embed_dim, device=device)
        self.contrastive_loss = QuantumContrastiveLoss(margin=0.5)

        # Otimizadores
        self.optimizer = optim.AdamW(
            self.stable_evolution.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        # Scheduler para decaimento de learning rate
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )

        print("🎓 Stable Evolution Trainer initialized")
        print(f"   📐 Embed dim: {embed_dim}")
        print(f"   🔧 Device: {device}")

    def prepare_training_data(self, vocab_size: int = 256) -> DataLoader:
        """
        Prepara dados de treinamento usando caracteres ASCII.

        Args:
            vocab_size: Tamanho do vocabulário

        Returns:
            DataLoader com pares de treinamento
        """
        print(f"📚 Preparando dados de treinamento (vocab_size={vocab_size})...")

        # Criar pares de caracteres similares e diferentes
        training_pairs = []

        # Caracteres similares (mesma categoria)
        similar_pairs = [
            ('a', 'A'), ('e', 'E'), ('i', 'I'), ('o', 'O'), ('u', 'U'),  # Vogais
            ('b', 'B'), ('c', 'C'), ('d', 'D'), ('f', 'F'), ('g', 'G'),  # Consoantes
            ('1', 'l'), ('0', 'O'), ('2', 'Z'), ('5', 'S'), ('8', 'B'),  # Visualmente similares
        ]

        # Caracteres diferentes (para contraste)
        all_chars = [chr(i) for i in range(32, 127)]  # Printable ASCII

        for char1, char2 in similar_pairs:
            if char1 in all_chars and char2 in all_chars:
                # Criar embeddings quânticos
                psi1 = self._char_to_quantum(char1)
                psi2 = self._char_to_quantum(char2)

                # Escolher caractere negativo aleatório
                negative_char = np.random.choice([c for c in all_chars if c not in [char1, char2]])
                psi_negative = self._char_to_quantum(negative_char)

                training_pairs.append((psi1, psi2, psi_negative))

        # Adicionar mais pares aleatórios para aumentar o dataset
        for _ in range(1000):
            char1, char2 = np.random.choice(all_chars, 2, replace=False)
            psi1 = self._char_to_quantum(char1)
            psi2 = self._char_to_quantum(char2)

            negative_char = np.random.choice([c for c in all_chars if c not in [char1, char2]])
            psi_negative = self._char_to_quantum(negative_char)

            training_pairs.append((psi1, psi2, psi_negative))

        # Converter para tensores
        contexts = torch.stack([p[0] for p in training_pairs])
        positives = torch.stack([p[1] for p in training_pairs])
        negatives = torch.stack([p[2] for p in training_pairs])

        # Criar dataset e dataloader
        dataset = TensorDataset(contexts, positives, negatives)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        print(f"   ✅ Dados preparados: {len(training_pairs)} pares de treinamento")
        return dataloader

    def _char_to_quantum(self, char: str) -> torch.Tensor:
        """
        Converte caractere para representação quântica.

        Args:
            char: Caractere a converter

        Returns:
            Tensor quântico [embed_dim, 4]
        """
        # Usar codificação simples baseada em ASCII
        ascii_val = ord(char)
        psi = torch.zeros(self.embed_dim, 4, dtype=torch.float32, device=self.device)

        for j in range(self.embed_dim):
            # Codificação determinística baseada no caractere
            phase = (ascii_val + j) * 2 * np.pi / 256.0
            amplitude = (ascii_val / 255.0) * (j / self.embed_dim)

            psi[j, 0] = amplitude * np.cos(phase)  # w (real)
            psi[j, 1] = amplitude * np.sin(phase)  # x (i)
            psi[j, 2] = 0.1 * amplitude  # y (j) - reduzido
            psi[j, 3] = 0.1 * amplitude  # z (k) - reduzido

        return psi

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Executa uma época de treinamento.

        Args:
            dataloader: DataLoader com dados de treinamento

        Returns:
            Métricas da época
        """
        self.stable_evolution.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (contexts, positives, negatives) in enumerate(dataloader):
            # Mover para dispositivo
            contexts = contexts.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)

            # Aplicar evolução estável aos contextos
            evolved_contexts = self.stable_evolution(contexts)

            # Calcular perda de contraste
            loss = self.contrastive_loss(evolved_contexts, positives, negatives)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"   📊 Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")

        # Atualizar scheduler
        self.scheduler.step()

        avg_loss = epoch_loss / num_batches
        return {'loss': avg_loss}

    def train(self, num_epochs: int = 10, save_path: str = 'models/stable_evolution'):
        """
        Executa treinamento completo.

        Args:
            num_epochs: Número de épocas
            save_path: Caminho para salvar checkpoints
        """
        print("🚀 Iniciando treinamento com evolução quântica estável...")
        print(f"   🎯 Épocas: {num_epochs}")
        print(f"   💾 Checkpoint path: {save_path}")

        # Preparar dados
        dataloader = self.prepare_training_data()

        # Criar diretório de checkpoints
        os.makedirs(save_path, exist_ok=True)

        # Histórico de treinamento
        training_history = []

        for epoch in range(num_epochs):
            print(f"\n🎯 Epoch {epoch+1}/{num_epochs}")

            # Treinar época
            start_time = time.time()
            metrics = self.train_epoch(dataloader)
            epoch_time = time.time() - start_time

            # Registrar métricas
            epoch_data = {
                'epoch': epoch + 1,
                'loss': metrics['loss'],
                'time': epoch_time,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_data)

            print(f"   ✅ Epoch concluída: Loss={metrics['loss']:.4f}, Time={epoch_time:.2f}s")

            # Salvar checkpoint a cada 5 épocas
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(checkpoint_path)
                print(f"   💾 Checkpoint salvo: {checkpoint_path}")

        # Salvar modelo final
        final_path = os.path.join(save_path, 'final_model.pt')
        self.save_checkpoint(final_path)
        print(f"   🎉 Modelo final salvo: {final_path}")

        # Salvar histórico
        history_path = os.path.join(save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"   📊 Histórico salvo: {history_path}")

        return training_history

    def save_checkpoint(self, path: str):
        """Salva checkpoint do modelo."""
        checkpoint = {
            'stable_evolution_state_dict': self.stable_evolution.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'embed_dim': self.embed_dim
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Carrega checkpoint do modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.stable_evolution.load_state_dict(checkpoint['stable_evolution_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Checkpoint carregado: {path}")

    def evaluate_stability(self) -> Dict[str, float]:
        """
        Avalia métricas de estabilidade do sistema treinado.

        Returns:
            Dicionário com métricas de estabilidade
        """
        print("🔬 Avaliando estabilidade do sistema...")

        # Obter métricas de estabilidade dos componentes
        stability_metrics = self.stable_evolution.get_stability_metrics()

        # Teste adicional: verificar preservação de energia
        test_input = torch.randn(1, self.embed_dim, 4, device=self.device)
        test_output = self.stable_evolution(test_input)

        energy_preservation = torch.norm(test_output) / torch.norm(test_input)
        energy_error = abs(energy_preservation.item() - 1.0)

        # Teste de similaridade: verificar se caracteres similares permanecem próximos
        char_a = self._char_to_quantum('a').unsqueeze(0)
        char_A = self._char_to_quantum('A').unsqueeze(0)
        char_z = self._char_to_quantum('z').unsqueeze(0)

        evolved_a = self.stable_evolution(char_a)
        evolved_A = self.stable_evolution(char_A)
        evolved_z = self.stable_evolution(char_z)

        # Calcular similaridades
        sim_a_A = torch.cosine_similarity(evolved_a.flatten(), evolved_A.flatten(), dim=0)
        sim_a_z = torch.cosine_similarity(evolved_a.flatten(), evolved_z.flatten(), dim=0)

        similarity_preservation = sim_a_A > sim_a_z  # Caracteres similares devem ser mais próximos

        evaluation_results = {
            **stability_metrics,
            'energy_preservation': energy_preservation.item(),
            'energy_error': energy_error,
            'similarity_preservation': float(similarity_preservation),
            'similarity_a_A': sim_a_A.item(),
            'similarity_a_z': sim_a_z.item()
        }

        print("   📊 Métricas de estabilidade:")
        for key, value in evaluation_results.items():
            print(".4f")

        return evaluation_results


def main():
    """Função principal para treinamento."""
    parser = argparse.ArgumentParser(description='Treinamento com Evolução Quântica Estável')
    parser.add_argument('--embed-dim', type=int, default=64, help='Dimensão do embedding')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
    parser.add_argument('--device', type=str, default='cpu', help='Dispositivo (cpu/cuda)')
    parser.add_argument('--save-path', type=str, default='models/stable_evolution', help='Caminho para salvar')
    parser.add_argument('--load-checkpoint', type=str, help='Carregar checkpoint existente')

    args = parser.parse_args()

    # Verificar dispositivo
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA não disponível, usando CPU")
        args.device = 'cpu'

    # Inicializar treinador
    trainer = StableEvolutionTrainer(embed_dim=args.embed_dim, device=args.device)

    # Carregar checkpoint se especificado
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    # Executar treinamento
    try:
        history = trainer.train(num_epochs=args.epochs, save_path=args.save_path)

        # Avaliar estabilidade final
        stability_metrics = trainer.evaluate_stability()

        # Salvar resultados finais
        final_results = {
            'training_history': history,
            'final_stability_metrics': stability_metrics,
            'config': {
                'embed_dim': args.embed_dim,
                'epochs': args.epochs,
                'device': args.device
            }
        }

        results_path = os.path.join(args.save_path, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n🎉 Treinamento concluído com sucesso!")
        print(f"   📊 Loss final: {history[-1]['loss']:.4f}")
        print(f"   🔬 Erro de unitariedade: {stability_metrics['unitarity_error']:.6f}")
        print(f"   💾 Resultados salvos em: {results_path}")

    except Exception as e:
        print(f"❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())